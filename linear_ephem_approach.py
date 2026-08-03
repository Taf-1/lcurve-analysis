from astropy.timeseries import BoxLeastSquares
import argparse as ap
from logger import sf_logging
import logging
import os
import numpy as np
from astropy.time import Time
from astropy.io import fits
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 

def arg_parse():
    p = ap.ArgumentParser(description="Linear Ephemeris Approach for Transit Timing Variations")
    p.add_argument("data_root", help="Root directory for all target data")
    p.add_argument("ngts_dat",  help="FITS file containing the NGTS data for the target")
    p.add_argument("ultracam_dat", help="FITS file containing the ULTRACAM data")
    p.add_argument("name",      help="Name of target")
    p.add_argument("t0",        help="t0 of binary system (BJD TDB)")
    return p.parse_args()

def int_day_detection(period):
    """Distance of a period from the nearest integer number of days.
    (Remove this if you already have your own implementation.)"""
    return np.abs(period - np.round(period))

def fit_single_eclipse(time, flux, flux_err,
                       tc_pred, duration, depth,
                       window=None,
                       min_points=15,
                       min_in_eclipse=5,
                       min_side_oot=5,
                       min_side_ecl=3):

    if window is None:
        window = max(3*duration, 0.02)

    m = np.abs(time - tc_pred) < window
    in_ecl = np.abs(time - tc_pred) < 0.5 * duration

    if m.sum() < min_points or in_ecl.sum() < min_in_eclipse:
        return None

    p0 = [tc_pred, depth, 0.5 * duration, 2.0 * duration, 1.0, 0.0]
    bounds = (
        [tc_pred - 0.5*window, 0.0,  1e-4, 1e-4,           0.8, -5.0],
        [tc_pred + 0.5*window, 1.0,  6.0*duration, 8.0*duration, 1.2,  5.0]
    )

    try:
        popt, pcov = curve_fit(
            trapezoid_model,
            time[m],
            flux[m],
            p0=p0,
            sigma=flux_err[m],
            absolute_sigma=True,
            bounds=bounds,
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return None

    err = np.sqrt(pcov[0, 0])

    if not np.isfinite(err):
        return None

    if np.abs(popt[0] - tc_pred) > 0.45 * window:
        return None
    tmin_fit, w_fit = popt[0], popt[3]
    tt = time[m]

    pre_oot  = np.sum(tt <  tmin_fit - 0.5 * w_fit)            # baseline before
    post_oot = np.sum(tt >  tmin_fit + 0.5 * w_fit)            # baseline after
    ingress  = np.sum((tt >= tmin_fit - 0.5 * w_fit) & (tt < tmin_fit))
    egress   = np.sum((tt >  tmin_fit) & (tt <= tmin_fit + 0.5 * w_fit))

    if min(pre_oot, post_oot) < min_side_oot or min(ingress, egress) < min_side_ecl:
        return None

    return popt[0], err, popt[3]

def trapezoid_model(t, t0, depth, t_full, t_total, oot, slope):
    x = np.abs(t - t0)
    hf, ht = 0.5 * t_full, 0.5 * t_total
    ingress = np.clip((ht - x) / max(ht - hf, 1e-8), 0.0, 1.0)
    return oot + slope * (t - t0) - depth * ingress

def measure_minima(time, flux, flux_err, period, t0, duration, depth, logger,
                   min_points=15, min_in_eclipse=5):
    """Fit each observed eclipse individually and return the times of minimum,
    their uncertainties, and the corresponding cycle numbers."""
    e_min = int(np.floor((time.min() - t0) / period))
    e_max = int(np.ceil((time.max() - t0) / period))
    window = max(1.5 * duration, 0.02)

    tmins, tmin_errs, epochs, widths = [], [], [], []
    for e in range(e_min, e_max + 1):

        tc = t0 + e*period

        result = fit_single_eclipse(
            time,
            flux,
            flux_err,
            tc,
            duration,
            depth,
            window=max(1.5*duration, 0.02),
            min_points=min_points,
            min_in_eclipse=min_in_eclipse,
        )

        if result is None:
            continue

        tmin, err, width = result

        tmins.append(tmin)
        tmin_errs.append(err)
        epochs.append(e)
        widths.append(width)

    logger.info(f"Measured {len(tmins)} individual times of minimum")

    if len(widths) > 0:
        widths = np.array(widths)
        w_lo, w_med, w_hi = np.percentile(widths, [16, 50, 84])
        logger.info(f"Fitted eclipse widths (t_total): median = {w_med*1440:.2f} min "
                    f"(16-84%: {w_lo*1440:.2f} - {w_hi*1440:.2f} min); "
                    f"bounds were {1e-4*1440:.2f} - {8*duration*1440:.2f} min")
        at_upper = np.mean(widths > 0.95 * 8.0 * duration)
        if at_upper > 0.2:
            logger.warning(f"{at_upper*100:.0f}% of fitted widths are pinned near the "
                           f"upper bound (8 x duration) -- widen the width bounds!")

    return np.array(tmins), np.array(tmin_errs), np.array(epochs, dtype=int)

def measure_single_minimum(time, flux, flux_err,
                           predicted_t0,
                           duration,
                           depth,
                           logger):

    result = fit_single_eclipse(
        time,
        flux,
        flux_err,
        predicted_t0,
        duration,
        depth,
    )

    if result is None:
        raise RuntimeError("Unable to measure eclipse.")

    tmin, err, width = result

    logger.info(
        f"Measured eclipse at {tmin:.10f} ± {err*86400:.2f} s "
        f"(fitted width {width*1440:.2f} min)"
    )

    return tmin, err

def _fit_linear(tmins, tmin_errs, epochs):
    e0  = int(np.round(np.median(epochs)))
    E   = (epochs - e0).astype(float)
    w   = 1.0 / tmin_errs**2
    A   = np.vstack([np.ones_like(E), E]).T
    Aw  = A * np.sqrt(w)[:, None]
    yw  = tmins * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)
    T0, P   = coef
    T0_err  = np.sqrt(cov[0, 0])
    P_err   = np.sqrt(cov[1, 1])
    oc      = tmins - (T0 + P * E)
    return T0, T0_err, P, P_err, E, oc, e0

def fit_linear_ephemeris(tmins, tmin_errs, epochs, logger, nsig=5.0, max_iter=5):
    keep = np.ones(len(tmins), dtype=bool)
    T0 = T0_err = P = P_err = E = oc = None
    for it in range(max_iter):
        T0, T0_err, P, P_err, E, oc, e0 = _fit_linear(
            tmins[keep], tmin_errs[keep], epochs[keep])
        e0_all = epochs - e0
        oc_all = tmins - (T0 + P * e0_all)
        bad = (np.abs(oc_all / tmin_errs) > nsig) & keep
        if not bad.any():
            break
        for i in np.where(bad)[0]:
            logger.warning(f"Sigma-clipping timing at epoch {epochs[i]}: "
                           f"O-C = {oc_all[i]*86400:+.1f} s "
                           f"({oc_all[i]/tmin_errs[i]:+.1f} sigma)")
        keep &= ~bad
    logger.info(f"Kept {keep.sum()}/{len(tmins)} timings after clipping")
    return T0, T0_err, P, P_err, E, oc

def fit_quadratic_ephemeris(tmins, tmin_errs, epochs, chi2_linear, logger):
    """Fit a quadratic ephemeris and test the significance of the period
    derivative. Returns (dP/dt, err, significance in sigma)."""
    e0 = int(np.round(np.median(epochs)))
    E = (epochs - e0).astype(float)

    w = 1.0 / tmin_errs**2
    A = np.vstack([np.ones_like(E), E, 0.5 * E**2]).T
    Aw = A * np.sqrt(w)[:, None]
    yw = tmins * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)
    T0q, Pq, dPdE = coef
    dPdE_err = np.sqrt(cov[2, 2])

    oc_q = tmins - A @ coef
    chi2_q = np.sum((oc_q / tmin_errs) ** 2)
    dof_q = max(len(tmins) - 3, 1)
    delta_chi2 = chi2_linear - chi2_q

    # dP/dE (days per cycle) -> dP/dt (dimensionless, days per day)
    pdot = dPdE / Pq
    pdot_err = dPdE_err / Pq
    sig = np.abs(dPdE) / dPdE_err if dPdE_err > 0 else 0.0

    # In convenient units: seconds per year
    to_s_per_yr = 86400.0 * 365.25
    logger.info("Quadratic ephemeris test:")
    logger.info(f"  dP/dE = {dPdE:.3e} +/- {dPdE_err:.3e} d/cycle")
    logger.info(f"  dP/dt = {pdot:.3e} +/- {pdot_err:.3e} (dimensionless)")
    logger.info(f"          = {pdot*to_s_per_yr:.4f} +/- {pdot_err*to_s_per_yr:.4f} s/yr")
    logger.info(f"  significance of curvature: {sig:.2f} sigma "
                f"(delta chi2 vs linear = {delta_chi2:.2f} for 1 extra dof)")
    logger.info(f"  quadratic reduced chi2 = {chi2_q/dof_q:.2f}")
    if sig < 3.0:
        logger.info(f"  -> no significant period change; 3-sigma upper limit "
                    f"|dP/dt| < {3.0*pdot_err*to_s_per_yr:.4f} s/yr")
    else:
        logger.info(f"  -> possible period change detected at {sig:.1f} sigma -- "
                    f"verify against systematics before claiming a detection")
    return pdot, pdot_err, sig

def bin_data_on_phase(phase: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, binfact: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if binfact is None:
        binfact = 12
    if len(phase) <= 11:
        binfact = len(phase)
    n_binned = int(len(phase) / binfact)
    binned_len = int(n_binned * binfact)
    temp = zip(phase[:binned_len], flux[:binned_len], flux_err[:binned_len])
    temp = sorted(temp)
    phase_s, flux_s, flux_err_s = map(np.array, zip(*temp))
    phase_bin     = np.average(phase_s.reshape(n_binned, binfact),     axis=1)
    flux_bin      = np.average(flux_s.reshape(n_binned, binfact),      axis=1)
    flux_err_bin  = np.average(flux_err_s.reshape(n_binned, binfact),  axis=1)
    return phase_bin, flux_bin, flux_err_bin

def run(cfg: dict, logger: logging.Logger):

    logger.info(f"Running linear ephemeris approach for target {cfg['name']}")
    logger.info(f"Loading NGTS data from {cfg['ngts_dat']}")

    with fits.open(cfg["ngts_dat"]) as f:
        data = f[1].data

    time = Time(
        data["bjd_mid"],
        format="jd",
        scale="tdb"
    ).mjd

    flux = np.asarray(data["flux_3"], dtype=float)
    flux_err = np.asarray(data["fluxerr_3"], dtype=float)

    oot = flux > 0.85 * np.nanmedian(flux)
    norm = np.nanmean(flux[oot])

    flux /= norm
    flux_err /= norm
    logger.info("Running BLS")

    bls = BoxLeastSquares(time, flux)

    period_grid = np.linspace(0.005, 5.0, 40000)
    duration_grid = np.array([0.0025])

    results = bls.power(
        period_grid,
        duration_grid,
        oversample=10
    )

    for idx in np.argsort(results.power)[::-1]:

        if (
            (0.99 <= results.period[idx] <= 1.044)
            or (27 <= results.period[idx] <= 30)
            or (int_day_detection(results.period[idx]) < 1e-4)
        ):
            continue

        index = idx
        break


    period = results.period[index]
    t0 = results.transit_time[index]
    depth = results.depth[index]
    duration = results.duration[index]

    logger.info(
        f"BLS: P={period:.8f} d   T0={t0:.8f} BMJD"
    )

    tmins, tmin_errs, epochs = measure_minima(
        time,
        flux,
        flux_err,
        period,
        t0,
        duration,
        depth,
        logger
    )

    if len(tmins) < 3:
        logger.error("Too few NGTS eclipse timings.")
        return

    logger.info("Fitting initial ephemeris")

    T0, T0_err, P, P_err, E, oc = fit_linear_ephemeris(
        tmins,
        tmin_errs,
        epochs,
        logger
    )

    # FIX: fit_linear_ephemeris re-zeros the cycle count internally, so the
    # returned T0 corresponds to cycle 0 of the *renumbered* frame E, not of
    # the original BLS-based `epochs`. Adopt E as the epoch frame from here
    # on, otherwise ULTRACAM cycle numbers (computed relative to T0) are
    # offset by ~(T0 - t0_BLS)/P cycles and wreck the combined fit.
    epochs = E.astype(int)

    # FIX: rescale the NGTS timing errors so the NGTS-only fit has
    # reduced chi2 = 1. Otherwise over/under-estimated per-point errors
    # give NGTS and ULTRACAM the wrong relative weights in the joint fit.
    chi2_red_ngts = np.sum((oc / tmin_errs) ** 2) / max(len(tmins) - 2, 1)
    if chi2_red_ngts > 0:
        scale = np.sqrt(chi2_red_ngts)
        logger.info(f"Rescaling NGTS timing errors by x{scale:.3f} "
                    f"(NGTS-only reduced chi2 = {chi2_red_ngts:.3f})")
        tmin_errs = tmin_errs * scale

    logger.info(
        f"Loading ULTRACAM data from {cfg['ultracam_dat']}"
    )

    with fits.open(cfg["ultracam_dat"]) as f:
        data = f[2].data


    u_time = np.asarray(
        data["BMJD(TDB)"],
        dtype=float
    )

    u_flux = np.asarray(
        data["Flux"],
        dtype=float
    )

    u_flux_err = np.asarray(
        data["Flux_err"],
        dtype=float
    )


    oot = u_flux > 0.85*np.median(u_flux)
    norm = np.mean(u_flux[oot])

    u_flux /= norm
    u_flux_err /= norm

    logger.info("Searching ULTRACAM for eclipses")


    e_start = int(np.floor((u_time.min() - T0)/P)) - 1
    e_end   = int(np.ceil((u_time.max() - T0)/P)) + 1


    for e in range(e_start, e_end + 1):

        predicted = T0 + e*P

        if (
            predicted < u_time.min() - duration
            or predicted > u_time.max() + duration
        ):
            continue


        logger.info(
            f"ULTRACAM eclipse epoch {e}, "
            f"prediction={predicted:.8f}"
        )
        try:

            t_new, t_new_err = measure_single_minimum(
                u_time,
                u_flux,
                u_flux_err,
                predicted,
                duration,
                depth,
                logger
            )

            if (
                not np.isfinite(t_new_err)
                or t_new_err > 5e-4       # >43 seconds
                or abs(t_new - predicted) > 0.02
            ):
                logger.warning(
                    f"Rejecting ULTRACAM eclipse epoch {e}: "
                    f"offset={(t_new-predicted)*86400:.1f}s "
                    f"err={t_new_err*86400:.1f}s"
                )
                continue


        except Exception:

            continue


        if not np.isfinite(t_new_err):
            continue


        logger.info(
            f"Measured ULTRACAM eclipse: "
            f"{t_new:.8f} +/- {t_new_err*86400:.2f} s"
        )


        tmins = np.append(tmins, t_new)
        tmin_errs = np.append(tmin_errs, t_new_err)
        epochs = np.append(epochs, e)


    logger.info(
        f"Total eclipse timings: {len(tmins)}"
    )

    logger.info("Refitting final ephemeris")

    T0, T0_err, P, P_err, E, oc = fit_linear_ephemeris(
        tmins,
        tmin_errs,
        epochs,
        logger
    )

    oc_paper, n_cyc, t0_paper = None, None, None
    try:
        t0_paper = float(cfg["t0"]) - 2400000.5   # BJD TDB -> BMJD TDB
        n_cyc = np.round((t0_paper - T0) / P)
        oc_paper = t0_paper - (T0 + n_cyc * P)
        logger.info(
            f"Literature t0 = {t0_paper:.8f} BMJD TDB is {abs(n_cyc):.0f} "
            f"cycles from new T0; O-C of paper epoch = {oc_paper*86400:.2f} s "
            f"(phase {oc_paper/P:+.5f})"
        )
    except (TypeError, ValueError):
        logger.warning("Could not parse supplied t0; skipping literature comparison")

    chi2_linear = np.sum((oc / tmin_errs) ** 2)
    pdot, pdot_err, pdot_sig = fit_quadratic_ephemeris(
        tmins,
        tmin_errs,
        epochs,
        chi2_linear,
        logger
    )

    oc_path = os.path.join(
        cfg["data_root"],
        cfg["name"],
        "oc_table.csv"
    )

    np.savetxt(
        oc_path,
        np.column_stack([
            E,
            tmins,
            tmin_errs,
            oc*86400
        ]),
        delimiter=",",
        header="cycle,tmin_bmjd_tdb,tmin_err_d,oc_seconds",
        comments=""
    )


    logger.info(
        f"O-C table written to {oc_path}"
    )

    to_s_per_yr = 86400.0 * 365.25
    eph_path = os.path.join(cfg["data_root"], cfg["name"], "ephemeris.txt")
    with open(eph_path, "w") as fh:
        fh.write(f"# Final linear ephemeris for {cfg['name']}\n")
        fh.write(f"# T_min(E) = T0 + P * E   (E = 0 at mid-dataset)\n")
        fh.write(f"T0_BMJD_TDB   = {T0:.10f} +/- {T0_err:.10f}\n")
        fh.write(f"T0_BJD_TDB    = {T0 + 2400000.5:.10f} +/- {T0_err:.10f}\n")
        fh.write(f"P_days        = {P:.10f} +/- {P_err:.10f}\n")
        fh.write(f"N_timings     = {len(tmins)}\n")
        fh.write(f"rms_OC_s      = {np.std(oc)*86400:.2f}\n")
        fh.write(f"red_chi2      = {chi2_linear / max(len(tmins)-2, 1):.3f}\n")
        fh.write(f"Pdot          = {pdot:.4e} +/- {pdot_err:.4e}  (dimensionless dP/dt)\n")
        fh.write(f"Pdot_s_per_yr = {pdot*to_s_per_yr:.4f} +/- {pdot_err*to_s_per_yr:.4f}\n")
        fh.write(f"Pdot_signif   = {pdot_sig:.2f} sigma\n")
        if oc_paper is not None:
            fh.write(f"paper_t0_BMJD = {t0_paper:.10f}\n")
            fh.write(f"paper_OC_s    = {oc_paper*86400:.2f}  ({abs(n_cyc):.0f} cycles from T0)\n")
    logger.info(f"Ephemeris written to {eph_path}")
    
    logger.info("Phase folding NGTS using new ephemeris")
    t_fold = ((time - T0) % P) / P
    loc = np.where(t_fold > 0.5)[0]
    t_fold[loc] -= 1
    t_fold += 0.5
    phase_bin, flux_bin, flux_err_bin = bin_data_on_phase(
            t_fold, flux, flux_err, 15
        )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(phase_bin, flux_bin, yerr=flux_err_bin, fmt='o', markersize=3, alpha=0.7,
                    zorder=3)
    ax.set_ylabel("Normalized Flux")
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Phase ($\phi$)")
    eph_phot_path = os.path.join(cfg["data_root"], cfg["name"], "eph_phased_lc.png")
    plt.savefig(eph_phot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"Phase-folded light curve written to {eph_phot_path}")

if __name__ == "__main__":
    args = arg_parse()
    pathname = f"{args.data_root}/{args.name}"
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    eph_logger = sf_logging.setup_logger(f"{pathname}/linear_ephem_approach.log", "linear_ephem_approach")
    run(
        cfg={"data_root": args.data_root,
             "ngts_dat": args.ngts_dat,
             "ultracam_dat": args.ultracam_dat,
             "name": args.name,
             "t0": args.t0},
        logger=eph_logger)