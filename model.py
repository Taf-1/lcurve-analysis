import argparse as ap
import logging
import numpy as np
import os
import sys
import emcee
import corner
import uuid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from astropy import constants as const
import astropy.units as u
from logger import sf_logging
from lcurve_commands import lcurve
from lcurve_model_file import adjust_parameters
from lcurve_stats import Jacobian, q_i_degeneracy, parabola
from lcurve_rv_calc import xshooter_params
from scipy.optimize import curve_fit
from plotting import lcurve_model_plot, stats_plots
import plotting

def arg_parse() -> ap.Namespace:
    p = ap.ArgumentParser()
    p.add_argument("data_root", help="Root directory for all target data")
    p.add_argument("gaia_id",   help="Gaia ID number of the WD")
    p.add_argument("binfact",   default=100, help="Bin factor")
    p.add_argument("a_r_sun",   help="The semi-major axis in units of R_sun")
    return p.parse_args()

def read_ephemeris(config: str) -> tuple[str, str]:
    period, t0 = [], []
    with open(config, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "=":
                if parts[0] == "period":
                    period.append(parts[2])
                elif parts[0] == "t0":
                    t0.append(parts[2])
    return period[0], t0[0]

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

def change_params(logger, config: str, path: str, band_idx: int, rv_config: str | None, method: str, fix_geometry: bool = False) -> str:
    new_config = f"{path}/{method}_model_ultracam_model_file_{band_idx}"
    geom = "0" if fix_geometry else "1"

    ldc_gravity_fixes = []
    with open(config, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and parts[1] == "=":
                name = parts[0]
                if name.startswith("ldc") or name.startswith("gravity_"):
                    ldc_gravity_fixes.append((name, "0", 5))

    changes = [
        ("period", "1.0", 2), ("period", "0", 5),
        ("t0", "0.5", 2), ("t0", "0.005", 3), ("t0", "1", 5),
        ("q", "0", 5),
        ("absorb", "1", 5),
        ("gravity_dark2", "1", 5),
        *ldc_gravity_fixes,
        ("tperiod", "1", 2),
        ("iangle", geom, 5),
        ("r1",     geom, 5),
        ("r2",     geom, 5),
        ("t1",     geom, 5),
        ("t2",     geom, 5),
        ("iscale", "0" if not fix_geometry else "1", 2),
    ]

    if not fix_geometry:
        """changes += [
            ("iangle", "0.1",   3), ("iangle", "0.05",  4),
            ("r1",     "0.0001", 3), ("r1",     "0.0005", 4),
            ("r2",     "0.001", 3), ("r2",     "0.001",  4),
            ("t1",     "10",    3), ("t1",     "5",      4),
            ("t2",     "50",    3), ("t2",     "25",     4),
        ]"""
        if rv_config is not None:
            _, _, vs_value, _ = xshooter_params(logger, rv_config, []).q_n_velocityscale()
            changes += [("velocity_scale", f"{vs_value:.15e}", 2), ("velocity_scale", "0", 5)]
        else:
            changes += [("velocity_scale", "0", 5)]
    else:
        changes += [("velocity_scale", "0", 5)]

    names   = [c[0] for c in changes]
    values  = [c[1] for c in changes]
    indices = [c[2] for c in changes]
    return adjust_parameters(logger, config, new_config, names, values, indices, set(names)).change_config()

def adjusted_config(logger, base: str, config: str, path: str, band_idx: int, method: str) -> str:
    new_config = f"{path}/{method}_adjusted_config_{band_idx}"
    prefix = ("ldc", "gravity", "beam", "absorb")
    values = {}

    with open(base, "r") as f:
        for line in f:
            if "=" in line:
                name, rest = line.split("=", 1)
                name = name.strip()
                if name == "wavelength" or name.startswith(prefix):
                    values[name] = rest.strip().split()[0]

    names   = list(values.keys())
    vals    = list(values.values())
    indices = [2] * len(names)
    return adjust_parameters(logger, config, new_config, names, vals, indices, set(names)).change_config()

def vary_mass_ratio(logger, config: str, path: str, band_idx: int) -> str:
    new_config = f"{path}/model_vary_q_{band_idx}"
    return adjust_parameters(logger, config, new_config, ["q"], ["1"], [5], {"q"}).change_config()

def run_simplex_model(logger, filename: str, band_index: int, pathname: str, tar_name: str, rv_config: str | None, reference_model: str | None = None) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if reference_model is None:
        base_model = f"{pathname}/{tar_name}_ultracam_model_file_2"
        fix_geometry = False
        logger.info("Fitting geometry + temperatures")
    else:
        base = f"{pathname}/{tar_name}_ultracam_model_file_{band_index}"
        base_model = adjusted_config(logger, base, reference_model, pathname, band_index, "simplex")
        fix_geometry = True
        logger.info("Geometry fixed, fitting only temperatures")

    model_config = change_params(logger, base_model, pathname, band_index, rv_config, "simplex", fix_geometry)

    ultracam_prelim = f"{pathname}/{tar_name}_model_simplex_model_{band_index}"
    lcurve(logger, model_config, filename, ultracam_prelim).simplex()

    model_dat = f"{pathname}/simplex_model_dat_{band_index}"
    lcurve(logger, ultracam_prelim, filename, model_dat).lroche()

    phase, flux_norm, flux_err_norm = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
    phase_model, flux_model, _      = np.loadtxt(model_dat, usecols=(0, 2, 3), unpack=True)
    return ultracam_prelim, phase, flux_norm, flux_err_norm, phase_model, flux_model

def run_levmarq_model(logger, filename: str, band_index: int, pathname: str, tar_name: str, rv_config: str | None, reference_model: str | None = None) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if reference_model is None:
        base_model = f"{pathname}/{tar_name}_model_simplex_model_2"
        fix_geometry = False
        logger.info("Fitting geometry + temperatures")
    else:
        base = f"{pathname}/{tar_name}_model_simplex_model_{band_index}"
        base_model = adjusted_config(logger, base, reference_model, pathname, band_index, "levmarq")
        fix_geometry = True
        logger.info("Geometry fixed, fitting only temperatures")

    model_config = change_params(logger, base_model, pathname, band_index, rv_config, "levmarq", fix_geometry)

    ultracam_prelim = f"{pathname}/{tar_name}_model_levmarq_model_{band_index}"
    lcurve(logger, model_config, filename, ultracam_prelim).levmarq()

    model_dat = f"{pathname}/levmarq_model_dat_{band_index}"
    lcurve(logger, ultracam_prelim, filename, model_dat).lroche()

    phase, flux_norm, flux_err_norm = np.loadtxt(filename, usecols=(0, 2, 3), unpack=True)
    phase_model, flux_model, _      = np.loadtxt(model_dat, usecols=(0, 2, 3), unpack=True)
    return ultracam_prelim, phase, flux_norm, flux_err_norm, phase_model, flux_model

def log_prior(params: np.ndarray, names: list[str], rv_config: str | None, a_r_sun: float, period: float, logger: logging.Logger) -> float:
    if params is None or not np.all(np.isfinite(params)):
        return -np.inf

    p  = dict(zip(names, params))
    lp = 0.0

    if rv_config is not None:
        q0, q_err, _, _ = xshooter_params(logger, rv_config, []).q_n_velocityscale()
        lp += -0.5 * ((p["q"] - q0) / q_err) ** 2

        rvs       = xshooter_params(logger, rv_config, []).load_rv_parameters()
        k2_obs    = rvs["K2"][0]
        k2_err    = rvs["K2"][1]
        a_m       = a_r_sun * const.R_sun.value
        P_s       = period * 86400.0
        i_rad     = np.radians(p["iangle"])
        k2_model  = 2 * np.pi * a_m * np.sin(i_rad) / (P_s * (1 + p["q"])) / 1e3
        lp += -0.5 * ((k2_model - k2_obs) / k2_err) ** 2

    if not (75    < p["iangle"] < 90):    return -np.inf
    if not (0.009 < p["r1"]     < 0.03):  return -np.inf
    if not (0.09  < p["r2"]     < 0.3):   return -np.inf
    if not (9500  < p["t1"]     < 11600): return -np.inf
    if not (2200  < p["t2"]     < 3000):  return -np.inf

    return lp

def log_likelihood(params: np.ndarray, names: list[str], data_dir: str, rv_config: str | None, gaia_id: str, band_order: list[int], logger: logging.Logger) -> float:
    p_dict   = dict(zip(names, params))
    rvs      = xshooter_params(logger, rv_config, []).load_rv_parameters()
    vs_value = rvs["K2"][0] * (1 + p_dict["q"])
    total_ll = 0.0

    for band_idx in band_order:
        base      = f"{data_dir}/{gaia_id}_model_simplex_model_{band_idx}"
        orig_data = f"{data_dir}/{gaia_id}_phase_data_file_{band_idx}"
        _, flux, flux_errors, weights = np.loadtxt(orig_data, usecols=(0, 2, 3, 4), unpack=True)

        walker_id = uuid.uuid4().hex[:8]
        mcmc_file = f"/tmp/w{walker_id}_mcmc_model"
        mcmc_dat  = f"/tmp/w{walker_id}_mcmc_dat"

        param_names = list(names) + ["velocity_scale", "quad"]
        param_vals  = [f"{v:.15e}" for v in params] + [f"{vs_value:.15e}", "0"]
        param_idx   = [2] * len(param_names)

        try:
            adjust_parameters(
                logger, base, mcmc_file, param_names, param_vals, param_idx, set(param_names)
            ).change_config()
        except Exception as e:
            logger.info(f"Error writing model: {e}")
            return -np.inf

        try:
            lcurve(logger, mcmc_file, orig_data, mcmc_dat).lroche()
        except Exception as e:
            logger.info(f"lroche failed: {e}")
            try:
                os.remove(mcmc_file)
            except Exception:
                pass
            return -np.inf

        try:
            mcmc_flux = np.loadtxt(mcmc_dat, usecols=(2,), unpack=True)
        except Exception as e:
            logger.info(f"Error reading model data: {e}")
            return -np.inf
        finally:
            for tmp in [mcmc_file, mcmc_dat]:
                try:
                    os.remove(tmp)
                except Exception:
                    pass

        if mcmc_flux.shape != flux.shape or not np.all(np.isfinite(mcmc_flux)):
            return -np.inf

        chi2 = np.sum(weights * ((flux - mcmc_flux) / flux_errors) ** 2)
        if not np.isfinite(chi2):
            return -np.inf
        total_ll += -0.5 * chi2

    return total_ll

def log_probability(params: np.ndarray, names: list[str], data_dir: str, rv_config: str | None, gaia_id: str, band_order: list[int], a_r_sun: float, period: float, logger: logging.Logger) -> float:
    lp = log_prior(params, names, rv_config, a_r_sun, period, logger)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, names, data_dir, rv_config, gaia_id, band_order, logger)

def plot_ellipsoidal_signal(logger: logging.Logger, fig_name: str, results: dict, band_idx: int = 3) -> None:
    """Zoom plot of the out-of-eclipse i-band region to reveal ellipsoidal modulation."""
    res         = results[band_idx]
    phase       = res['phase']
    flux        = res['flux']
    flux_err    = res['flux_err']
    phase_model = res['phase_model']
    flux_model  = res['flux_model']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})
    ax1.plot(phase_model, flux_model, 'k-', lw=2, zorder=5, label='Model')
    ax1.errorbar(phase, flux, yerr=flux_err, fmt='o', color='red',
                markersize=3, alpha=0.7, zorder=3, label='i-band data')

    oot_flux = flux[flux > 0.85]
    oot_mask = flux > 0.85
    if len(oot_flux) > 0:
        ax1.set_ylim(oot_flux.min() - 0.02, oot_flux.max() + 0.02)
    rse = np.sqrt(np.sum((flux[oot_mask] - flux_model[oot_mask])**2) /
                            (len(flux_model[oot_mask]) - 2))
    residuals_sigma = (flux - flux_model) / rse
    ax2.errorbar(phase, residuals_sigma,
                        color='red', fmt='o', markersize=3, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax2.axhline(2.5, color='black', linestyle='--', alpha=0.3)
    ax2.axhline(-2.5, color='black', linestyle='--', alpha=0.3)
    y_med = np.median(residuals_sigma)
    y_mad = np.median(np.abs(residuals_sigma - y_med))
    ax2.set_ylim(y_med - 8 * y_mad, y_med + 8 * y_mad)
    ax2.set_ylabel(r"Residuals ($\sigma$)")
    ax2.set_xlabel(r"Phase ($\phi$)")
    ax1.set_ylabel("Normalized Flux")
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{fig_name}.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{fig_name}.pdf", bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {fig_name}.png/.pdf")

def run(cfg: dict, logger: logging.Logger) -> None:
    data_root = cfg['data_root']
    gaia_id   = cfg['gaia_id']
    binfact   = int(cfg.get('binfact', 100))
    a_r_sun   = float(cfg['a_r_sun'])

    data_dir  = f"{data_root}/{gaia_id}"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    rv_config = f"{data_root}/{gaia_id}_rv_config"
    if not os.path.exists(rv_config):
        rv_config = None

    eph_config = f"{data_dir}/best_fit_ephemeris_model"
    period, t0 = read_ephemeris(eph_config)
    period = float(period)
    t0     = float(t0)
    logger.info(f"Ephemeris: period={period:.10f}, t0={t0:.10f}")

    band_order   = [2, 1, 3]
    band_names   = {1: 'u', 2: 'g', 3: 'i'}
    band_colours = {1: 'blue', 2: 'green', 3: 'red'}

    phase_data    = {}
    flux_data     = {}
    flux_err_data = {}

    band_binfacts = {1: 2, 2: 3, 3: 3}
    n_bands = len(band_order)

    fig = plt.figure(figsize=(10, 12))
    gs  = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.05)

    for band_idx in band_order:
        data_file = f"{data_dir}/{gaia_id}_ultracam_data_file_{band_idx}"
        time, flux, flux_err = np.loadtxt(data_file, usecols=(0, 2, 3), unpack=True)

        t_fold = ((time - t0) % period) / period
        loc = np.where(t_fold > 0.5)[0]
        t_fold[loc] -= 1
        t_fold += 0.5

        eclipse_phase = t_fold[np.argmin(flux)]
        logger.info(f"Band {band_idx}: eclipse at phase {eclipse_phase:.4f}")
        t0_adj = t0 + eclipse_phase * period
        logger.info(f"Band {band_idx}: adjusted t0 = {t0_adj:.10f}")

        t_fold = ((time - t0_adj) % period) / period
        phase_bin, flux_bin, flux_err_bin = bin_data_on_phase(t_fold, flux, flux_err, band_binfacts[band_idx])

        orig_dat  = f"{data_dir}/{gaia_id}_phase_data_file_{band_idx}"
        bin_width = 1.0 / len(phase_bin)

        oot_mask  = flux_bin > 0.925
        ie_mask   = flux_bin < 0.5
        ingr_mask = ~oot_mask & ~ie_mask
        categories = [oot_mask, ~oot_mask]   # OOT vs everything inside eclipse
        n_nonempty = sum(1 for m in categories if np.sum(m) > 0)
        weights    = np.zeros(len(flux_bin))
        for m in categories:
            n = np.sum(m)
            if n > 0:
                weights[m] = 1.0 / (n_nonempty * n * n_bands)
        logger.info(
            f"Band {band_idx} weights: oot={np.sum(oot_mask)}, "
            f"ingr/egr={np.sum(ingr_mask)}, in={np.sum(ie_mask)}, "
            f"band_total={weights.sum():.6f} (1/{n_bands} of global)"
        )

        with open(orig_dat, "w") as f:
            f.write(f"# Written by model.py for the {band_names[band_idx]}-band\n")
            f.write("# EB contamination removed\n")
            f.write(f"# Balanced weights: ingr/egr=in-eclipse=oot, bin width = {bin_width:.6f}\n")
            for ph, fl, fe, w in zip(phase_bin, flux_bin, flux_err_bin, weights):
                f.write(f"{ph:.8f} {bin_width:.8f} {fl:.6f} {fe:.6f} {w:.8f} 1\n")
        logger.info(f"Saved: {orig_dat}")

        phase_data[band_idx]    = phase_bin
        flux_data[band_idx]     = flux_bin
        flux_err_data[band_idx] = flux_err_bin

        ax = fig.add_subplot(gs[band_idx - 1, 0])
        ax.errorbar(phase_bin, flux_bin, yerr=flux_err_bin,
                    color=band_colours[band_idx], fmt='o', markersize=3, alpha=0.7,
                    zorder=3, label=f"{band_names[band_idx]}-band")
        ax.set_ylabel("Normalized Flux")
        ax.set_xlim(0, 1)
        ax.legend(loc='upper right', fontsize=12)
        if band_idx < 3:
            plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xlabel(r"Phase ($\phi$)")
    plt.savefig(f"{data_dir}/phased_lc.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {data_dir}/phased_lc.png")

    results_simplex       = {}
    reference_model_simplex = None

    for band_idx in band_order:
        filename = f"{data_dir}/{gaia_id}_phase_data_file_{band_idx}"
        simplex_model_file, phase, flux_norm, flux_err_norm, phase_model, flux_model = run_simplex_model(
            logger, filename, band_idx, data_dir, gaia_id, rv_config, reference_model_simplex
        )
        results_simplex[band_idx] = {
            'phase':       phase,
            'flux':        flux_norm,
            'flux_err':    flux_err_norm,
            'phase_model': phase_model,
            'flux_model':  flux_model,
            't0_nearby':   0.0,
        }
        if band_idx == 2:
            reference_model_simplex = simplex_model_file
            logger.info("Using g-band geometry for subsequent simplex fits")

    lcurve_model_plot(logger, f"{data_dir}/{gaia_id}_model_simplex", results_simplex).lc_with_model(use_phase=True)
    plot_ellipsoidal_signal(logger, f"{data_dir}/{gaia_id}_ellipsoidal_simplex", results_simplex)

    """results_levmarq       = {}
    reference_model_levmarq = None

    for band_idx in band_order:
        filename = f"{data_dir}/{gaia_id}_phase_data_file_{band_idx}"
        levmarq_model_file, phase, flux_norm, flux_err_norm, phase_model, flux_model = run_levmarq_model(
            logger, filename, band_idx, data_dir, gaia_id, rv_config, reference_model_levmarq
        )
        results_levmarq[band_idx] = {
            'phase':       phase,
            'flux':        flux_norm,
            'flux_err':    flux_err_norm,
            'phase_model': phase_model,
            'flux_model':  flux_model,
            't0_nearby':   0.0,
        }
        if band_idx == 2:
            reference_model_levmarq = levmarq_model_file
            logger.info("Using g-band geometry for subsequent levmarq fits")

    lcurve_model_plot(logger, f"{data_dir}/{gaia_id}_model_levmarq", results_levmarq).lc_with_model(use_phase=True)
    plot_ellipsoidal_signal(logger, f"{data_dir}/{gaia_id}_ellipsoidal_levmarq", results_levmarq)
    """
    results = {}
    for band_idx in band_order:
        con_lev    = f"{data_dir}/{gaia_id}_model_simplex_model_{band_idx}"
        new_config = vary_mass_ratio(logger, con_lev, data_dir, band_idx)
        names, p_best, steps = adjust_parameters(
            logger, new_config, new_config, [], [], [], {"q", "iangle", "r1", "r2", "t1", "t2"}
        ).load_mcmc_params()
        results[band_idx] = {"names": names, "p_best": p_best, "steps": steps}

    os.chdir(data_dir)

    param_label_map = {
        "q":      r"$q$",
        "iangle": r"$i\ [^\circ]$",
        "r1":     r"$r_1/a$",
        "r2":     r"$r_2/a$",
        "t1":     r"$T_1\ [\mathrm{K}]$",
        "t2":     r"$T_2\ [\mathrm{K}]$",
        "t0":     r"$t_0$",
    }
    param_label_map_short = {
        "q":      r"$q$",
        "iangle": r"$i$",
        "r1":     r"$r_1/a$",
        "r2":     r"$r_2/a$",
        "t1":     r"$T_1$",
        "t2":     r"$T_2$",
        "t0":     r"$t_0$",
    }
    plot_labels = [param_label_map_short.get(n, n) for n in results[band_order[0]]["names"]]

    for band_idx in band_order:
        new_config = f"{data_dir}/model_vary_q_{band_idx}"
        orig_dat   = f"{data_dir}/{gaia_id}_phase_data_file_{band_idx}"
        res        = results_simplex[band_idx]

        jac    = Jacobian(logger, gaia_id, orig_dat, new_config,
                          results[band_idx]["p_best"], results[band_idx]["steps"],
                          results[band_idx]["names"], res["phase"], res["flux_err"])
        C, c_err = jac.compute_covariance()

        residuals = res["flux"] - res["flux_model"]
        chi2_val  = np.sum((residuals / res["flux_err"]) ** 2)
        dof       = len(res["flux"]) - len(results[band_idx]["p_best"])
        chi2_red  = chi2_val / dof
        logger.info(f"Band {band_idx} reduced chi2: {chi2_red:.4f}")

        if len(C) == 0:
            logger.info("Covariance matrix empty — check fitting")
            sys.exit(1)

        C *= chi2_red
        corr = C / np.outer(c_err, c_err)
        np.fill_diagonal(corr, 1.0)
        mask        = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_masked = np.where(mask, np.nan, corr)

        stats_plots(
            logger, corr_masked, plot_labels, band_idx,
            [], [], [], 0, 0, [], 0, 0, gaia_id
        ).correlation()

    band_idx_chi2 = 2
    chi2_config   = f"{data_dir}/model_vary_q_{band_idx_chi2}"
    q_vals, i_vals, delta_q, delta_chi2, rv_q_mean, rv_q_sigma, p_best_chi2, i_best = q_i_degeneracy(
        logger, f"{data_dir}/{gaia_id}", results, band_idx_chi2, chi2_config, rv_config
    ).chi_squared_method()

    names_2 = results[2]["names"]
    q_best  = p_best_chi2[names_2.index("q")]

    sp = stats_plots(
        logger, None, plot_labels, band_idx_chi2,
        q_vals, i_vals, delta_chi2, q_best, i_best,
        delta_q, rv_q_mean, rv_q_sigma, gaia_id
    )
    sp.chi2_2d_map()
    sp.marginalised_profile()

    chi2_marginal_i = np.min(delta_chi2, axis=0)
    delta_i         = chi2_marginal_i - chi2_marginal_i.min()
    try:
        p0_i = [1.0, i_vals[np.argmin(delta_i)], 0.0]
        popt_i, _ = curve_fit(parabola, i_vals, delta_i, p0=p0_i)
        i_mean_prior  = float(popt_i[1])
        i_sigma_prior = 1.0 / np.sqrt(abs(popt_i[0]))
        i_prior = (i_mean_prior, i_sigma_prior)
        logger.info(f"Inclination prior from chi2 map: i = {i_mean_prior:.3f} ± {i_sigma_prior:.3f} deg")
    except Exception as e:
        logger.info(f"Could not derive inclination prior from chi2 map: {e} — using box prior only")
        i_prior = None

    names   = results[band_order[0]]["names"]
    p_best  = results[2]["p_best"]
    mcmc_steps  = 20000
    steps_mcmc  = 0.01 * np.abs(p_best)
    steps_mcmc  = np.where(steps_mcmc < 1e-8, 1e-8, steps_mcmc)
    ndim        = len(p_best)
    nwalkers    = max(4 * ndim, 32)
    p0          = p_best + steps_mcmc * np.random.randn(nwalkers, ndim)

    with Pool(processes=32) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(names, data_dir, rv_config, gaia_id, band_order, a_r_sun, period, logger),
            pool=pool
        )
        sampler.run_mcmc(p0, mcmc_steps, progress=True)

    chain = sampler.get_chain()
    nsteps, nwalkers_out, ndim_out = chain.shape

    for i in range(ndim):
        fig, ax = plt.subplots(figsize=(10, 5))
        for j in range(nwalkers_out):
            ax.plot(chain[:, j, i], alpha=0.5)
        ax.set_title(f"Walker trajectories for {names[i]}")
        ax.set_xlabel("Step")
        ax.set_ylabel(names[i])
        ax.set_xlim(0, mcmc_steps)
        plt.tight_layout()
        plt.savefig(f"{data_dir}/{mcmc_steps}_{names[i]}_walker_traj.png", dpi=300)
        plt.close(fig)
        logger.info(f"Saved: {data_dir}/{mcmc_steps}_{names[i]}_walker_traj.png")

    #discard = int(input("Enter number of burn-in steps to discard: "))
    discard = 5000

    samples   = sampler.get_chain(discard=discard, thin=10, flat=True)
    param_med = np.median(samples, axis=0)

    tau = sampler.get_autocorr_time(tol=0)
    logger.info(f"Autocorrelation time: {tau}")

    for name, med in zip(names, param_med):
        logger.info(f"{name} = {med:.6f}")

    q0, q_err, vs0, vs_err = xshooter_params(logger, rv_config, []).q_n_velocityscale()
    i_samples  = samples[:, names.index("iangle")]
    q_samples  = samples[:, names.index("q")]                        
    vs_ms      = vs0 * 1e3
    P_s        = (period * u.day).to(u.s).value
    sin_i      = np.sin(np.radians(i_samples))

    a_m        = vs_ms * P_s / (2*np.pi * sin_i)    
    m_total_kg = (4*np.pi**2 * a_m**3) / (const.G.value * P_s**2)
    m_total    = m_total_kg / const.M_sun.value
    m1_samples = m_total / (1 + q_samples)
    m2_samples = q_samples * m1_samples

    q_col = names.index("q")
    samples_extended = np.hstack([m1_samples[:, None], m2_samples[:, None], samples])
    samples_extended = np.delete(samples_extended, 2 + q_col, axis=1)

    corner_labels = (
        [r'$M_{\rm WD}\ [M_\odot]$', r'$M_{\rm comp}\ [M_\odot]$']
        + [param_label_map.get(n, n) for n in names if n != "q"]
    )
    param_med_corner = np.median(samples_extended, axis=0)
    cornerfig = corner.corner(
        samples_extended, labels=corner_labels, truths=param_med_corner,
        show_titles=True, title_fmt=".2f",
        label_kwargs={"fontsize": 13}, title_kwargs={"fontsize": 13},
    )
    for ax in cornerfig.get_axes():
        ax.tick_params(labelsize=18)
    cornerfig.savefig(f"{data_dir}/model_corner_plot.png", dpi=300)
    cornerfig.savefig(f"{data_dir}/model_corner_plot.pdf", dpi=300)
    logger.info(f"Saved: {data_dir}/model_corner_plot.png/.pdf")
    plt.close(cornerfig)

    cornerfig_t = corner.corner(
        samples_extended, labels=corner_labels, truths=param_med_corner,
        show_titles=True, title_fmt=".2f", color='white',
        hist_kwargs={'color': 'white'}, contour_kwargs={'colors': 'grey'},
        label_kwargs={"fontsize": 18}, title_kwargs={"fontsize": 18},
        fig=plt.figure(figsize=(7, 7)),
    )
    for ax in cornerfig_t.get_axes():
        ax.tick_params(labelsize=18)
    plotting._apply_transparent_style(cornerfig_t)
    cornerfig_t.savefig(f"{data_dir}/model_corner_plot_transparent.png", dpi=600, transparent=True)
    cornerfig_t.savefig(f"{data_dir}/model_corner_plot_transparent.pdf", dpi=600, transparent=True)
    logger.info(f"Saved: {data_dir}/model_corner_plot_transparent.png/.pdf")
    plt.close(cornerfig_t)

    percent = np.percentile(samples_extended, [16, 50, 84], axis=0)
    medians = percent[1]
    lowers  = medians - percent[0]
    uppers  = percent[2] - medians

    txt_file = f"{data_dir}/{gaia_id}_model_params.txt"
    with open(txt_file, "w") as of:
        of.write(f"{'Parameter':<30} {'Median':>12} {'−Error':>12} {'+Error':>12}\n")
        of.write("-" * 68 + "\n")
        for label, med, lo, hi in zip(corner_labels, medians, lowers, uppers):
            of.write(f"{label:<30} {med:>12.4f} {lo:>12.4f} {hi:>12.4f}\n")
    logger.info(f"Saved: {txt_file}")

    results_best = {}
    for band_idx in band_order:
        best_model  = f"{data_dir}/best_fit_model_{band_names[band_idx]}"
        base_config = f"{data_dir}/{gaia_id}_model_simplex_model_{band_idx}"
        orig_dat    = f"{data_dir}/{gaia_id}_phase_data_file_{band_idx}"

        bfit_names  = list(names) + ["quad"]
        bfit_vals   = [f"{v:.15e}" for v in param_med] + ["0"]
        bfit_idx    = [2] * len(bfit_names)
        adjust_parameters(
            logger, base_config, best_model, bfit_names, bfit_vals, bfit_idx, set(bfit_names)
        ).change_config()

        best_dat = f"{data_dir}/best_data_{band_names[band_idx]}"
        lcurve(logger, best_model, orig_dat, best_dat).lroche()

        phase_model, flux_model, _ = np.loadtxt(best_dat, usecols=(0, 2, 3), unpack=True)
        phase_obs, flux_obs, flux_err_obs = np.loadtxt(orig_dat, usecols=(0, 2, 3), unpack=True)

        results_best[band_idx] = {
            'phase':       phase_obs,
            'flux':        flux_obs,
            'flux_err':    flux_err_obs,
            'phase_model': phase_model,
            'flux_model':  flux_model,
            't0_nearby':   0.0,
        }

    lcurve_model_plot(logger, f"{data_dir}/lc_with_model", results_best).lc_with_model(use_phase=True)
    plot_ellipsoidal_signal(logger, f"{data_dir}/{gaia_id}_ellipsoidal_bestfit", results_best)

if __name__ == "__main__":
    args   = arg_parse()
    d_dir  = f"{args.data_root}/{args.gaia_id}"
    if not os.path.exists(d_dir):
        os.mkdir(d_dir)
    logger = sf_logging("model", f"{d_dir}/{args.gaia_id}.log").setup_logger()
    run({
        'data_root': args.data_root,
        'gaia_id':   args.gaia_id,
        'binfact':   args.binfact,
        'a_r_sun':   args.a_r_sun,
    }, logger)
