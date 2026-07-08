from __future__ import annotations
import argparse as ap
import logging
import numpy as np
import os
import sys
import emcee
import re
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
from lcurve_wd_tracks import WDTrackInterpolator, CompTrackInterpolator
from scipy.optimize import curve_fit
from plotting import lcurve_model_plot, stats_plots
import plotting
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
})

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
    inc, inc_err = [], []
    with open(config, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and parts[1] == "=":
                name = parts[0]
                if name.startswith("ldc") or name.startswith("gravity_"):
                    ldc_gravity_fixes.append((name, "0", 5))
                elif name == "iangle":
                    inc.append(float(parts[2]))
                    inc_err.append(float(parts[3]))

    t0_val = "0.5"
    if not fix_geometry:
        ref_file = f"{path}/{method}_model_ultracam_model_file_2"
        try:
            with open(ref_file, "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 6 and parts[1] == "=" and parts[0] == "t0":
                        t0_val = parts[2]
                        break
        except FileNotFoundError:
            pass

    changes = [
        ("period", "1", 2), ("period", "0", 5),
        ("t0", t0_val, 2), ("t0", "0.005", 3), ("t0", geom, 5),
        ("q", "0", 5),
        ("absorb", "0", 5),
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
        if rv_config is not None:
            _, _, vs_value, _ = xshooter_params(logger, rv_config, inc[0], inc_err[0]).q_n_velocityscale()
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

def nsub_for(mask, target_substeps=33):
    n = int(np.sum(mask))
    if n == 0:
        return 1
    return max(1, int(np.ceil(target_substeps / n)))

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

def eggleton_rl(q: float) -> float:
    q23 = q ** (2.0 / 3.0)
    return 0.49 * q23 / (0.6 * q23 + np.log(1.0 + q ** (1.0 / 3.0)))


def split_normal_logpdf(x: float, mu: float, sig_lo: float, sig_hi: float) -> float:
    sig = sig_hi if x >= mu else sig_lo
    return -0.5 * ((x - mu) / sig) ** 2


def log_prior(params: np.ndarray, names: list[str], rv_config: str | None,
              period: float, logger: logging.Logger,
              wd_interp: "WDTrackInterpolator | None" = None,
              comp_interp: "CompTrackInterpolator | None" = None,
              spec_priors: dict | None = None,
              roche_fill_max: float = 1.0) -> float:
    if params is None or not np.all(np.isfinite(params)):
        return -np.inf

    p = dict(zip(names, params))

    if not (79 < p["iangle"] < 90):    return -np.inf
    if not (0.0075 < p["r1"]     < 0.057): return -np.inf
    if not (0.075  < p["r2"]     < 0.5):   return -np.inf
    if not (4000   < p["t1"]     < 24000): return -np.inf
    if not (900    < p["t2"]     < 3900):  return -np.inf
    if "t0" in p and not (0.45 < p["t0"] < 0.55): return -np.inf

    if "q" in p and p["q"] > 0:
        r_lobe = eggleton_rl(p["q"])
        if p["r2"] > roche_fill_max * r_lobe:
            return -np.inf
    else:
        return -np.inf

    lp = 0.0

    if rv_config is not None:
        q0, q_err, vs_value, vs_err = \
            xshooter_params(logger, rv_config, p["iangle"]).q_n_velocityscale()
        lp += -0.5 * ((p["q"] - q0) / q_err) ** 2
    else:
        vs_value = None

    # Calculate K_emis = K2 * (1- f(1+q)r2) where f is between 0-1 with 1 being the most extreme cause and 0 is emission being unitform
    # 0.5 - Parsons et al. 2026 - if there is not much information about the origin of the H-alpha emission

    if spec_priors is not None:
        if "t1" in spec_priors:
            mu, lo, hi = spec_priors["t1"]
            lp += split_normal_logpdf(p["t1"], mu, lo, hi)
        if "t2" in spec_priors:
            mu, lo, hi = spec_priors["t2"]
            lp += split_normal_logpdf(p["t2"], mu, lo, hi)

    if (vs_value is not None
            and {"r1", "t1", "q", "iangle"}.issubset(p)):

        sin_i = np.sin(np.radians(p["iangle"]))
        P_s   = period * 86400.0

        a_m = (vs_value * 1.0e3) * P_s / (2.0 * np.pi)
        a_r_sun_step = a_m / const.R_sun.value

        M_tot_kg   = 4.0 * np.pi**2 * a_m**3 / (const.G.value * P_s**2)
        M_tot_msun = M_tot_kg / const.M_sun.value
        m1_msun    = M_tot_msun / (1.0 + p["q"])

        if m1_msun > 0:
            if wd_interp is not None:
                R1_obs  = p["r1"] * a_r_sun_step
                R1_pred = wd_interp.predict_radius(p["t1"], m1_msun)
                if np.isfinite(R1_pred):
                    lp += -0.5 * ((R1_obs - R1_pred) / wd_interp.sigma_R) ** 2

                if wd_interp.logg1_ref is not None and wd_interp.logg1_err is not None:
                    logg_pred = wd_interp.predict_logg(p["t1"], m1_msun)
                    if np.isfinite(logg_pred):
                        lp += -0.5 * ((logg_pred - wd_interp.logg1_ref)
                                      / wd_interp.logg1_err) ** 2

            if comp_interp is not None and "r2" in p:
                m2_msun = p["q"] * m1_msun
                R2_obs  = p["r2"] * a_r_sun_step
                R2_pred = comp_interp.predict_radius(m2_msun)
                if np.isfinite(R2_pred):
                    sigma_R2 = comp_interp.sigma_R_frac * R2_pred
                    lp += -0.5 * ((R2_obs - R2_pred) / sigma_R2) ** 2

    return lp

def log_likelihood(params: np.ndarray, names: list[str], data_dir: str, rv_config: str | None, gaia_id: str, band_order: list[int], logger: logging.Logger) -> float:
    p_dict   = dict(zip(names, params))
    rvs      = xshooter_params(logger, rv_config).load_rv_parameters()
    vs_value = (rvs["K1"][0] + rvs["K2"][0]) / np.sin(np.radians(p_dict["iangle"])) if rv_config is not None else None
    total_ll = 0.0

    for band_idx in band_order:
        base      = f"{data_dir}/{gaia_id}_model_simplex_model_{band_idx}"
        orig_data = f"{data_dir}/{gaia_id}_phase_data_file_{band_idx}"
        _, flux, flux_errors, weights = np.loadtxt(orig_data, usecols=(0, 2, 3, 4), unpack=True)

        walker_id = uuid.uuid4().hex[:8]
        mcmc_file = f"/tmp/w{walker_id}_mcmc_model"
        mcmc_dat  = f"/tmp/w{walker_id}_mcmc_dat"

        param_names = list(names) + ["velocity_scale", "quad"]
        param_vals  = [f"{v:.15e}" for v in params] + [f"{vs_value:.15e}", "0"] if vs_value is not None else [f"{v:.15e}" for v in params] + ["0"]
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

def log_probability(params, names, data_dir, rv_config, gaia_id, band_order,
                    a_r_sun, period, logger,
                    wd_interp=None, comp_interp=None,
                    spec_priors=None, roche_fill_max=1.0):
    lp = log_prior(params, names, rv_config, period, logger,
                   wd_interp, comp_interp, spec_priors, roche_fill_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, names, data_dir, rv_config,
                               gaia_id, band_order, logger)

def plot_ellipsoidal_signal(logger: logging.Logger, fig_name: str, results: dict, band_idx: int = 3) -> None:
    """Zoom plot of the out-of-eclipse i-band region to reveal ellipsoidal modulation."""
    res         = results[band_idx]
    phase       = res['phase']
    flux        = res['flux']
    flux_err    = res['flux_err']
    phase_model = res['phase_model']
    flux_model  = res['flux_model']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})
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
    spec_priors = {}
    if 't1_spec' in cfg:
        spec_priors['t1'] = (float(cfg['t1_spec']),
                             float(cfg['t1_low']), float(cfg['t1_up']))
    if 't2_spec' in cfg:
        spec_priors['t2'] = (float(cfg['t2_spec']),
                             float(cfg['t2_low']), float(cfg['t2_up']))
    spec_priors = spec_priors or None
    logger.info(f"Spectroscopic temperature priors: {spec_priors}")

    comp_track_file = cfg.get('comp_tracks',
                              f"{data_root}/BHAC15_tracks_structure.txt")
    try:
        comp_interp = CompTrackInterpolator(logger, comp_track_file,
                                            sigma_R_frac=float(cfg.get('sigma_r2_frac', 0.06)))
    except Exception as e:
        logger.warning(f"Could not build companion track interpolator ({e}); "
                       f"companion M–R prior disabled")
        comp_interp = None

    data_dir  = f"{data_root}/{gaia_id}"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    rv_config = f"{data_root}/{gaia_id}_rv_config"
    if not os.path.exists(rv_config):
        rv_config = None

    eph_config = f"{data_dir}/ephemeris.txt"
    eph = {}
    with open(eph_config) as fh:
        for line in fh:
            if line.startswith('#') or '=' not in line:
                continue
            key, _, rest = line.partition('=')
            eph[key.strip()] = float(rest.split()[0])
    period  = eph['P_days']
    t0_ngts = eph['T0_BMJD_TDB']
    logger.info(f"Ephemeris from {eph_config}: P={period:.10f} d, T0={t0_ngts:.10f} BMJD TDB")

    band_order   = [2, 1, 3]
    band_names   = {1: 'u', 2: 'g', 3: 'i'}
    band_colours = {1: 'blue', 2: 'green', 3: 'red'}

    phase_data    = {}
    flux_data     = {}
    flux_err_data = {}

    band_binfacts = {1: 2, 2: 2, 3: 3}
    n_bands = len(band_order)

    with open(f"{data_dir}/{gaia_id}_ultracam_data_file_2") as f:
        for line in f:
            if "t0_nearby" in line:
                match = re.search(r"t0_nearby\s*=\s*([0-9.eE+-]+)", line)
                if match:
                    ultracam_t0_nearby = float(match.group(1))
                    break

    fig = plt.figure(figsize=(10, 12))
    gs  = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.05)

    for band_idx in band_order:
        data_file = f"{data_dir}/{gaia_id}_ultracam_data_file_{band_idx}"
        time, flux, flux_err = np.loadtxt(data_file, usecols=(0, 2, 3), unpack=True)
        time += ultracam_t0_nearby
        t_fold = ((np.float64(time) - np.float64(t0_ngts)) % np.float64(period)) / np.float64(period)
        loc = np.where(t_fold > 0.5)[0]
        t_fold[loc] -= 1
        t_fold += 0.5
        print(t0_ngts, period, time[0], t_fold[0], t_fold[-1])

        phase_bin, flux_bin, flux_err_bin = bin_data_on_phase(
            t_fold, flux, flux_err, band_binfacts[band_idx]
        )

        orig_dat  = f"{data_dir}/{gaia_id}_phase_data_file_{band_idx}"
        bin_width = 1.0 / len(phase_bin)

        config = f"{data_dir}/{gaia_id}_ultracam_model_file_{band_idx}"

        extract_params = {"r1", "r2", "iangle"}
        values = {}
        with open(config, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and parts[1] == "=":
                    name = parts[0]
                    if name in extract_params:
                        values[name] = float(parts[2])

        r1    = values["r1"]
        r2    = values["r2"]
        inc   = np.radians(values["iangle"])
        cos_i = np.cos(inc)
        sin_i = np.sin(inc)

        d1   = r1 + r2
        d2   = abs(r1 - r2)
        arg1 = d1**2 - cos_i**2
        arg2 = d2**2 - cos_i**2

        phi_contact1 = 0.5 + np.arcsin(np.sqrt(arg1) / sin_i) / (2 * np.pi)
        phi1 = 0.5 - phi_contact1
        phi4 = 0.5 + phi_contact1

        partial = arg2 < 0  # b > |r1-r2|: no totality, partial eclipse only
        if not partial:
            phi_contact2 = 0.5 + np.arcsin(np.sqrt(arg2) / sin_i) / (2 * np.pi)
            phi2 = 0.5 - phi_contact2
            phi3 = 0.5 + phi_contact2
        else:
            phi2 = phi3 = 0.5

        logger.info(f"Band {band_idx}: phi1={phi1:.4f} phi2={phi2:.4f} phi3={phi3:.4f} phi4={phi4:.4f} partial={partial}")

        oot_mask  = (phase_bin < phi1) | (phase_bin > phi4)
        ingr_mask = ((phase_bin >= phi1) & (phase_bin < phi2)) | ((phase_bin > phi3) & (phase_bin <= phi4))
        ie_mask   = (phase_bin >= phi2) & (phase_bin <= phi3)

        categories = [oot_mask, ingr_mask] if partial else [oot_mask, ingr_mask, ie_mask]

        # Per-category boost factors applied on top of 1/n.
        # lcurve forms chi**2 per point as weight / ferr**2 ,
        # so column 5 is a pure relative multiplier — ferr stays in column 4.
        BOOST_OOT  = 1.0
        BOOST_INGR = 2.0   # ingress/egress constrain geometry most
        BOOST_IE   = 6.0   # in-eclipse (totality) constrains flux ratio

        boosts = [BOOST_OOT, BOOST_INGR] if partial else [BOOST_OOT, BOOST_INGR, BOOST_IE]

        weights = np.ones(len(flux_bin))
        for m, b in zip(categories, boosts):
            n = np.sum(m)
            if n > 0:
                weights[m] = b / n

        # lcurve data file: 6 columns
        #   time, exposure, flux, ferr, weight, nsub
        exposure = np.full(len(phase_bin), bin_width)
        # Finer sub-stepping where the model changes fastest (contacts).
        nsub = np.ones(len(phase_bin), dtype=int)
        ns_ingr = max(nsub_for(ingr_mask), 3)
        nsub[ingr_mask] = ns_ingr
        if not partial:
            ns_ie = nsub_for(ie_mask, target_substeps=11)
            nsub[ie_mask] = ns_ie

        logger.info(f"Band {band_idx}: nsub ingress={ns_ingr}"
                    + ("" if partial else f" in-eclipse={ns_ie}"))

        out = np.column_stack([
            phase_bin,
            exposure,
            flux_bin,
            flux_err_bin,
            weights,
            nsub,
        ])

        np.savetxt(
            orig_dat,
            out,
            fmt=["%.10f", "%.10f", "%.6e", "%.6e", "%.6e", "%d"],
        )
        logger.info(f"Band {band_idx}: wrote {len(phase_bin)} points to {orig_dat}")

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
    results_simplex = {}
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

    results = {}
    for band_idx in band_order:
        con_lev    = f"{data_dir}/{gaia_id}_model_simplex_model_{band_idx}"
        new_config = vary_mass_ratio(logger, con_lev, data_dir, band_idx)
        names, p_best, steps = adjust_parameters(
            logger, new_config, new_config, [], [], [], {"q", "iangle", "r1", "r2", "t1", "t2"}
        ).load_mcmc_params()
        results[band_idx] = {"names": names, "p_best": p_best, "steps": steps}

    os.chdir(data_dir)

    spec_type  = cfg.get('wdtype', 'DA')
    sigma_r_wd = float(cfg.get('sigma_r_wd', 0.003))
    logg1_ref  = float(cfg['logg1'])      if 'logg1'     in cfg else None
    logg1_err  = float(cfg['logg1_err'])  if 'logg1_err' in cfg else None
    try:
        wd_interp = WDTrackInterpolator(logger, spec_type=spec_type, sigma_R=sigma_r_wd,
                                        logg1_ref=logg1_ref, logg1_err=logg1_err)
        logger.info(f"WD track prior enabled: sigma_R={wd_interp.sigma_R} Rsun, "
                    f"logg1={logg1_ref}±{logg1_err}")
    except Exception as e:
        logger.warning(f"Could not build WD track interpolator ({e}); track priors disabled")
        wd_interp = None

    names   = results[band_order[0]]["names"]
    p_best  = results[2]["p_best"]
    ndim     = len(p_best)
    mcmc_steps  = 20000
    frac = 0.01
    steps_mcmc = frac * np.abs(p_best)
    abs_floor = np.full(ndim, 1e-6)
    for k, nm in enumerate(names):
        if nm == "iangle": abs_floor[k] = 0.2        
        if nm in ("r1", "r2"): abs_floor[k] = 1e-3      
        if nm in ("t1",): abs_floor[k] = 200.0         
        if nm in ("t2",): abs_floor[k] = 50.0
    steps_mcmc = np.maximum(steps_mcmc, abs_floor)
    nwalkers = max(4 * ndim, 32)
    p0       = p_best + steps_mcmc * np.random.randn(nwalkers, ndim)

    with Pool(processes=32) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(names, data_dir, rv_config, gaia_id, band_order, a_r_sun, period, logger, wd_interp, comp_interp, spec_priors),
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
    discard = 2500

    samples   = sampler.get_chain(discard=discard, thin=10, flat=True)
    param_med = np.median(samples, axis=0)

    if "t0" in names:
        t0_idx = names.index("t0")
        samples[:, t0_idx] = ultracam_t0_nearby + (samples[:, t0_idx] - 0.5)*period

    tau = sampler.get_autocorr_time(tol=0)
    logger.info(f"Autocorrelation time: {tau}")

    for name, med in zip(names, param_med):
        logger.info(f"{name} = {med:.6f}")

    q_samples = samples[:, names.index("q")]
    i_samples = samples[:, names.index("iangle")] 
    rvs      = xshooter_params(logger, rv_config).load_rv_parameters()
    P_s = (period * u.day).to(u.s)
    sin_i = np.sin(np.radians(i_samples))

    k1_k2 = ( rvs["K1"][0] +  rvs["K2"][0]) * u.km / u.s
    a_m = (k1_k2 * P_s / (2.0 * np.pi * sin_i)).to(u.m)

    a_med = np.median(a_m.value)

    a_rsun = (a_med * u.m).to(u.R_sun).value
    m_total_kg = ((4 * np.pi**2 * a_m**3) / (const.G * P_s**2)).value
    m_total_msun = m_total_kg / const.M_sun.value
    m1_samples = m_total_msun / (1 + q_samples)
    m2_samples = q_samples * m1_samples

    q_col = names.index("q")
    samples_extended = np.hstack([m1_samples[:, None], m2_samples[:, None], samples])
    samples_extended = np.delete(samples_extended, 2 + q_col, axis=1)

    corner_labels    = [
        r'$M_{\rm WD}\ [M_\odot]$', r'$M_{\rm comp}\ [M_\odot]$',
        r'$i\ [^\circ]$', r'$r_1/a$', r'$r_2/a$', r'$T_1\ [K]$', r'$T_2\ [K]$'
    ]
    param_med_corner = np.median(samples_extended, axis=0)
    cornerfig = corner.corner(
        samples_extended, labels=corner_labels, truths=param_med_corner,
        show_titles=True, title_fmt=".2f",
        label_kwargs={"fontsize": 13}, title_kwargs={"fontsize": 13},
    )
    for ax in cornerfig.get_axes():
        ax.tick_params(labelsize=13)
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
        of.write(f"{a_med:<30.6e} m {a_rsun:>12.4f} R_sun "
         f"0 0\n")
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
            't0':   0.5,
        }

    lcurve_model_plot(logger, f"{data_dir}/lc_with_model", results_best).lc_with_model(use_phase=True)
    plot_ellipsoidal_signal(logger, f"{data_dir}/{gaia_id}_ellipsoidal_bestfit", results_best)
    logger.info(f"{np.mean(sampler.acceptance_fraction)}")

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
