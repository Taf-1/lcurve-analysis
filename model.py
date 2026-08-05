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
from sed_prior import ClaretSEDInterp, compute_a_over_ebv, extract_wd_obs_flux, sed_log_prior_terms
import plotting
import matplotlib as mpl
from astropy.io import fits as astrofits

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
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
    p.add_argument("logg1",     help="log g of the WD")
    p.add_argument("logg1_err",     help="error of log g of the WD")
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

def read_param(config, name):
    with open(config) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and p[0] == name and p[1] == "=":
                return p[2]
    raise KeyError(name)

def bin_data_on_phase(phase, flux, flux_err, binfact=None):
    if binfact is None:
        binfact = 12
    if len(phase) <= 11:
        binfact = len(phase)
    n_binned   = int(len(phase) / binfact)
    binned_len = n_binned * binfact

    idx = np.argsort(phase)
    phase, flux, flux_err = phase[idx], flux[idx], flux_err[idx]
    p = phase[:binned_len].reshape(n_binned, binfact)
    f = flux[:binned_len].reshape(n_binned, binfact)
    e = flux_err[:binned_len].reshape(n_binned, binfact)

    w = 1.0 / e**2                                 
    wsum = w.sum(axis=1)
    phase_bin    = (w * p).sum(axis=1) / wsum
    flux_bin     = (w * f).sum(axis=1) / wsum
    flux_err_bin = 1.0 / np.sqrt(wsum)              

    return phase_bin, flux_bin, flux_err_bin

def change_params(logger, config: str, path: str, band_idx: int, rv_config: str | None, tar_name: str | None, method: str | None, fix_geometry: bool = False) -> str:
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

    t0_val = "0.5"
    vs_change = []
    if fix_geometry:
        s_con = f"{path}/{tar_name}_model_simplex_model_2"
        cur_vs = read_param(s_con, "velocity_scale")
        vs_change = [("velocity_scale", cur_vs, 2)]
    changes = [
        ("period", "1", 2), ("period", "0", 5),
        ("t0", t0_val, 2), ("t0", "0.005", 3), ("t0", "0", 5),
        ("q", "0", 5),
        *vs_change,
        ("velocity_scale", geom, 5),
        ("absorb", "1" if band_idx == 3 else "0", 5),
        *ldc_gravity_fixes,
        ("tperiod", "1", 2),
        ("iangle", geom, 5), 
        ("r1",     geom, 5), 
        ("r2",     geom, 5), 
        ("t1",     geom, 5), 
        ("t2",     geom, 5), 
        ("iscale", "0" if not fix_geometry else "1", 2),
    ]

    names   = [c[0] for c in changes]
    values  = [c[1] for c in changes]
    indices = [c[2] for c in changes]
    return adjust_parameters(logger, config, new_config, names, values, indices, set(names)).change_config()

def adjusted_config(logger, base: str, config: str, path: str, band_idx: int, method: str) -> str:
    new_config = f"{path}/{method}_adjusted_config_{band_idx}"
    prefix = ("ldc", "gravity", "beam", "absorb", "velocity_scale")
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

    model_config = change_params(logger, base_model, pathname, band_index, rv_config, tar_name, "simplex", fix_geometry)

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

def split_normal_logpdf(x, mu, sig_lo, sig_hi):
    if x < mu:
        sig = sig_lo
    else:
        sig = sig_hi
    return -0.5 * ((x - mu) / sig) ** 2

def log_prior(params: np.ndarray, names: list[str], rv_config: str | None,
              period: float, wd_logg: float | None, wd_logg_err: float | None,
              logger: logging.Logger,
              wd_interp: "WDTrackInterpolator | None" = None,
              comp_interp: "CompTrackInterpolator | None" = None,
              spec_priors: dict | None = None,
              roche_fill_max: float = 1.0,
              sed_data: dict | None = None) -> float:
    if params is None or not np.all(np.isfinite(params)):
        return -np.inf

    p = dict(zip(names, params))

    if not (79 < p["iangle"] < 89.99):      return -np.inf
    if not (0.01 < p["r1"]  < 0.036):   return -np.inf
    if not (0.2  < p["r2"]  < 0.385):    return -np.inf
    if not (9000  < p["t1"]  < 18000):   return -np.inf
    if not (2000  < p["t2"]  < 4000):    return -np.inf
    if "ebv" in p and "parallax" in p:
        if not (0.0 <= p["ebv"] < 0.1):      return -np.inf   
        if not (0.0 < p["parallax"] < 50.0): return -np.inf    

    if "q" in p and p["q"] > 0:
        r_lobe = eggleton_rl(p["q"])
        if p["r2"] > roche_fill_max * r_lobe:
            return -np.inf
    else:
        return -np.inf

    lp = 0.0

    lp_tspec = 0.0
    lp_t1_spec = 0.0
    lp_t2_spec = 0.0

    if spec_priors is not None:
        if "t1" in spec_priors:
            mu, lo, hi = spec_priors["t1"]
            lp_t1_spec = split_normal_logpdf(p["t1"], mu, lo, hi)
            lp_tspec += lp_t1_spec
        else:
            lp_t1_spec = 0.0

        if "t2" in spec_priors:
            mu, lo, hi = spec_priors["t2"]
            lp_t2_spec = split_normal_logpdf(p["t2"], mu, lo, hi)
            lp_tspec += lp_t2_spec
        else:
            lp_t2_spec = 0.0
    t1_val = p.get("t1", None)
    t2_val = p.get("t2", None)
    logger.debug(f"Spectroscopic temperature priors: t1={t1_val:3f}, t2={t2_val:.3f}, lp_t1_spec={lp_t1_spec:.3f}, lp_t2_spec={lp_t2_spec:.3f}, total={lp_tspec:.3f}")
    lp += lp_tspec

    if not {"velocity_scale", "q", "iangle", "r1", "t1"}.issubset(p):
        return lp

    P_s    = period * 86400.0
    sin_i  = np.sin(np.radians(p["iangle"]))
    a_m    = p["velocity_scale"] * 1.0e3 * P_s / (2.0 * np.pi)
    a_rsun = a_m / const.R_sun.value

    M_tot_kg   = 4.0 * np.pi**2 * a_m**3 / (const.G.value * P_s**2)
    M_tot_msun = M_tot_kg / const.M_sun.value
    m1_msun    = M_tot_msun / (1.0 + p["q"])
    if m1_msun <= 0:
        return -np.inf

    m1_kg    = m1_msun * const.M_sun.value
    R1_m     = p["r1"] * a_m
    g_si     = const.G.value * m1_kg / R1_m**2
    logg_obs = np.log10(g_si * 100.0)   # cgs

    # --- RV constraints ---
    if rv_config is not None:
        xs     = xshooter_params(logger, rv_config, p["iangle"])
        rvs    = xs.load_rv_parameters()
        """q_pred = rvs["K1"][0] / rvs["K2"][0]
        q_pred_err = np.sqrt((rvs["K1"][1] / rvs["K1"][0])**2 + (rvs["K2"][1] / rvs["K2"][0]**2)**2) * q_pred
        lp += -0.5 * ((p["q"] - q_pred) / q_pred_err) ** 2
        logger.debug(f"RV prior: q_obs = {p['q']:.3f}, q_pred = {q_pred:.3f}, lp_q={-0.5 * ((p['q'] - q_pred) / q_pred_err) ** 2:.3f}, total={lp:.3f}")
        """
        Kscale = p["velocity_scale"] * sin_i / (1 + p["q"])
        lp += -0.5 * ((Kscale - rvs["K2"][0]) / rvs["K2"][1]) ** 2
        lp += -0.5 * ((Kscale * p["q"] - rvs["K1"][0]) / rvs["K1"][1]) ** 2
        logger.debug(f"RV prior: Kscale = {Kscale:.3f}, K2_pred = {rvs['K2'][0]:.3f}, K1_pred = {rvs['K1'][0]:.3f}, lp_K={-0.5 * ((Kscale - rvs['K2'][0]) / rvs['K2'][1]) ** 2 + -0.5 * ((Kscale * p['q'] - rvs['K1'][0]) / rvs['K1'][1]) ** 2:.3f}, total={lp:.3f}")

    # --- WD surface gravity prior ---
    if wd_logg is not None and wd_logg_err is not None:
        lp += -0.5 * ((logg_obs - wd_logg) / wd_logg_err) ** 2
        lp_logg = -0.5 * ((logg_obs - wd_logg) / wd_logg_err) ** 2
        logger.debug(f"WD logg prior: log g = {logg_obs:.3f}, lp_logg={lp_logg:.3f}, total={lp_logg:.3f}")

    # --- WD mass-radius relation prior ---
    if wd_interp is not None:
        R1_obs  = p["r1"] * a_rsun
        R1_pred = wd_interp.predict_radius(p["t1"], m1_msun)
        if np.isfinite(R1_pred):
            lp += -0.5 * ((R1_obs - R1_pred) / (wd_interp.sigma_R_frac * R1_pred)) ** 2
            lp_mr = -0.5 * ((R1_obs - R1_pred) / (wd_interp.sigma_R_frac * R1_pred)) ** 2
            logger.debug(f"WD M-R prior: rad1 = {R1_obs:.3f}, lp_mr={lp_mr:.3f}, total={lp_mr:.3f}")

    # --- Companion mass-radius relation prior ---
    if comp_interp is not None and "r2" in p:
        m2_msun = p["q"] * m1_msun
        R2_obs  = p["r2"] * a_rsun
        R2_pred = comp_interp.predict_radius(m2_msun, logg=5.0)
        if np.isfinite(R2_pred):
            lp += -0.5 * ((R2_obs - R2_pred) / (comp_interp.sigma_R_frac * R2_pred)) ** 2
            logger.debug(f"Companion M-R prior: rad2 = {R2_obs:.3f}, lp_comp_mr={-0.5 * ((R2_obs - R2_pred) / (comp_interp.sigma_R_frac * R2_pred)) ** 2:.3f}, total={lp:.3f}")

    # --- SED + parallax + extinction prior ---
    if sed_data is not None and "parallax" in p and "ebv" in p:
        try:
            sed_lp = sed_log_prior_terms(logger, p, a_m, logg_obs, sed_data)
        except Exception as e:
            logger.warning(f"SED prior raised {type(e).__name__}: {e}")
            return -np.inf
        if not np.isfinite(sed_lp):
            return -np.inf
        lp += sed_lp
        logger.debug(f"SED prior: lp_sed={sed_lp:.3f}, total={lp:.3f}")

    return lp

def log_likelihood(params: np.ndarray, names: list[str], data_dir: str, rv_config: str | None, gaia_id: str, band_order: list[int], logger: logging.Logger) -> float:
    p_dict   = dict(zip(names, params))
    total_ll = 0.0
    for band_idx in band_order:
        base      = f"{data_dir}/{gaia_id}_model_simplex_model_{band_idx}"
        orig_data = f"{data_dir}/{gaia_id}_phase_data_file_{band_idx}"
        _, flux, flux_errors, weights = np.loadtxt(orig_data, usecols=(0, 2, 3, 4), unpack=True)
        walker_id = uuid.uuid4().hex[:8]
        mcmc_file = f"{data_dir}/w{walker_id}_mcmc_model"
        mcmc_dat  = f"{data_dir}/w{walker_id}_mcmc_dat"

        param_names = list(names) + ["quad"]
        param_vals  = [f"{v:.15e}" for v in params] + ["0"]
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
        ll = -0.5 * chi2
        #logger.info(f"Band {band_idx}: chi2={chi2:.3f}, ll={ll:.3f}, total_ll={total_ll:.3f}")

    return total_ll

def log_probability(params, names, data_dir, rv_config, gaia_id, band_order,
                    a_r_sun, period, wd_logg, wd_logg_err, logger,
                    wd_interp=None, comp_interp=None,
                    spec_priors=None, roche_fill_max=1.0, sed_data=None):
    lp = log_prior(params, names, rv_config, period, wd_logg, wd_logg_err, logger,
                   wd_interp, comp_interp, spec_priors, roche_fill_max, sed_data)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, names, data_dir, rv_config,
                               gaia_id, band_order, logger)

def plot_ellipsoidal_signal(logger: logging.Logger, fig_name: str, results: dict,
                           band_idx: int = 3, oot_norms_jy: dict | None = None) -> None:
    """Zoom plot of the out-of-eclipse i-band region to reveal ellipsoidal modulation."""
    res  = results[band_idx]
    norm = oot_norms_jy[band_idx] * 1e3 if (oot_norms_jy and band_idx in oot_norms_jy) else None

    phase       = res['phase']
    flux        = res['flux']        if norm is None else res['flux']        * norm
    flux_err    = res['flux_err']    if norm is None else res['flux_err']    * norm
    flux_model  = res['flux_model']  if norm is None else res['flux_model']  * norm
    phase_model = res['phase_model']
    ylabel      = "Normalized Flux" if norm is None else r"Flux (mJy)"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4),
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})
    ax1.plot(phase_model, flux_model, 'k-', lw=2, zorder=5, label='Model')
    ax1.errorbar(phase, flux, yerr=flux_err, fmt='o', color='red',
                 markersize=3, alpha=0.7, zorder=3, label='i-band data')

    oot_mask = res['flux'] > 0.85
    oot_flux = flux[oot_mask]
    if len(oot_flux) > 0:
        ax1.set_ylim(oot_flux.min() - (0.02 if norm is None else 0.02*norm),
                     oot_flux.max() + (0.02 if norm is None else 0.02*norm))
    model_at_data   = np.interp(phase, phase_model, flux_model)
    residuals_sigma = (flux - model_at_data) / flux_err
    ax2.errorbar(phase, residuals_sigma, color='red', fmt='o', markersize=3, alpha=0.7)
    ax2.axhline(0,    color='black', linestyle='--', alpha=0.3)
    ax2.axhline(2.5,  color='black', linestyle='--', alpha=0.3)
    ax2.axhline(-2.5, color='black', linestyle='--', alpha=0.3)
    y_med = np.median(residuals_sigma)
    y_mad = np.median(np.abs(residuals_sigma - y_med))
    ax2.set_ylim(y_med - 8*y_mad, y_med + 8*y_mad)
    ax2.set_ylabel(r"Residuals ($\sigma$)")
    ax2.set_xlabel(r"Phase ($\phi$)")
    ax1.set_ylabel(ylabel)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    plt.tight_layout()
    ax1.legend()
    plt.savefig(f"{fig_name}.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{fig_name}.pdf", bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {fig_name}.png/.pdf")

def run(cfg: dict, logger: logging.Logger) -> None:
    data_root = cfg['data_root']
    gaia_id   = cfg['gaia_id']
    binfact   = int(cfg.get('binfact', 100))
    a_r_sun   = float(cfg['a_r_sun'])
    wd_logg  = float(cfg['logg1'])
    wd_logg_err  = float(cfg['logg1_err'])
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
                              f"{data_root}/data/cooling_tracks/Baraffe/baraffe.dat")
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

    band_binfacts = {1: 2, 2: 2, 3: 2}
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
        logger.info(f"Band {band_idx}: fold check t0={t0_ngts} P={period} "
                    f"phase range [{t_fold.min():.4f}, {t_fold.max():.4f}]")

        phase_bin, flux_bin, flux_err_bin = bin_data_on_phase(
            t_fold, flux, flux_err, band_binfacts[band_idx]
        )

        orig_dat = f"{data_dir}/{gaia_id}_phase_data_file_{band_idx}"

        edges = np.empty(len(phase_bin) + 1)
        edges[1:-1] = 0.5 * (phase_bin[1:] + phase_bin[:-1])
        edges[0]    = phase_bin[0]  - 0.5 * (phase_bin[1] - phase_bin[0])
        edges[-1]   = phase_bin[-1] + 0.5 * (phase_bin[-1] - phase_bin[-2])
        exposure = np.diff(edges)

        config = f"{data_dir}/{gaia_id}_ultracam_model_file_{band_idx}"
        extract_params = {"r1", "r2", "iangle"}
        values = {}
        with open(config, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and parts[1] == "=":
                    if parts[0] in extract_params:
                        values[parts[0]] = float(parts[2])

        r1    = values["r1"]
        r2    = values["r2"]
        inc   = np.radians(values["iangle"])
        cos_i = np.cos(inc)
        sin_i = np.sin(inc)

        d1   = r1 + r2
        d2   = abs(r1 - r2)
        arg1 = d1**2 - cos_i**2
        arg2 = d2**2 - cos_i**2

        half_width1 = np.arcsin(np.sqrt(arg1) / sin_i) / (2 * np.pi)
        phi1 = 0.5 - half_width1
        phi4 = 0.5 + half_width1

        partial = arg2 < 0 
        if not partial:
            half_width2 = np.arcsin(np.sqrt(arg2) / sin_i) / (2 * np.pi)
            phi2 = 0.5 - half_width2
            phi3 = 0.5 + half_width2
        else:
            phi2 = phi3 = 0.5

        logger.info(f"Band {band_idx}: phi1={phi1:.4f} phi2={phi2:.4f} "
                    f"phi3={phi3:.4f} phi4={phi4:.4f} partial={partial}")

        ingr_mask = (((phase_bin >= phi1) & (phase_bin < phi2)) |
                    ((phase_bin > phi3) & (phase_bin <= phi4)))
        ie_mask   = (phase_bin >= phi2) & (phase_bin <= phi3)

        # --- statistical weights: NATURAL inverse-variance weighting ---
        # Every point contributes chi2 = weight * (resid/ferr)**2 with weight = 1,
        # so the information balance between phase regions is set by the data,
        # not imposed. Any per-region reweighting is equivalent to silently
        # rescaling error bars and corrupts both chi2 and the parameter
        # uncertainties derived from it.
        weights = np.ones(len(flux_bin))

        nsub = np.ones(len(phase_bin), dtype=int)
        ns_ingr = max(nsub_for(ingr_mask), 7)
        nsub[ingr_mask] = ns_ingr
        if not partial:
            ns_ie = nsub_for(ie_mask, target_substeps=11)
            nsub[ie_mask] = ns_ie

        logger.info(f"Band {band_idx}: nsub ingress={ns_ingr}"
                    + ("" if partial else f" in-eclipse={ns_ie}"))

        # lcurve data file: time, exposure, flux, ferr, weight, nsub
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
            logger, new_config, new_config, [], [], [], {"q", "iangle", "r1", "r2", "t1", "t2", "velocity_scale"}
        ).load_mcmc_params()
        results[band_idx] = {"names": names, "p_best": p_best, "steps": steps}

    os.chdir(data_dir)

    spec_type  = cfg.get('wdtype', 'DA')
    sigma_r_wd = float(cfg.get('sigma_r_wd', 0.05))
    logg1_ref  = float(cfg['logg1'])      if 'logg1'     in cfg else None
    logg1_err  = float(cfg['logg1_err'])  if 'logg1_err' in cfg else None
    try:
        wd_interp = WDTrackInterpolator(logger, spec_type=spec_type, sigma_R=sigma_r_wd,
                                        logg1_ref=logg1_ref, logg1_err=logg1_err)
        logger.info(f"WD track prior enabled: sigma_R_frac={wd_interp.sigma_R_frac:.1%}, "
                    f"logg1={logg1_ref}±{logg1_err}")
    except Exception as e:
        logger.warning(f"Could not build WD track interpolator ({e}); track priors disabled")
        wd_interp = None

    # --- SED + parallax + extinction setup ---
    fits_path    = cfg['ultracam_dat']
    gaia_plx     = float(cfg.get('gaia_parallax',     0.0))
    gaia_plx_err = float(cfg.get('gaia_parallax_err', 1.0))
    ebv_map      = float(cfg.get('ebv',     0.0))
    ebv_err      = float(cfg.get('ebv_err', 0.01))
    sed_data     = None
    oot_norms_jy = {}   # {band_idx: OOT flux in Jy}  — used for mJy plots

    t0_for_sed = eph.get('T0_BMJD_TDB', float(cfg.get('t0', 0.0)))
    try:
        wd_flux_raw = extract_wd_obs_flux(fits_path, t0_for_sed, period)
        oot_norms_jy = {bi: wd_flux_raw[bi][2] for bi in band_order}
        logger.info(f"OOT norms (mJy): " +
                    ", ".join(f"band{bi}={v*1e3:.3f}" for bi, v in oot_norms_jy.items()))
    except Exception as e:
        logger.warning(f"Could not read OOT norms for mJy plots: {e}")

    if gaia_plx > 0:
        try:
            logger.info("Building Claret SED interpolator (queries Vizier) ...")
            claret_sed   = ClaretSEDInterp(logger, wdtype=cfg.get('wdtype', 'DA'))
            A_over_EBV   = {bi: compute_a_over_ebv(bi, T_source=float(cfg.get('teff1', 10500.0)))
                            for bi in band_order}
            for bi in band_order:
                fw, fe, _ = wd_flux_raw[bi]
                logger.info(f"Band {bi}: WD depth = {fw*1e3:.4f} ± {fe*1e3:.4f} mJy  "
                            f"A/E(B-V) = {A_over_EBV[bi]:.3f}")
            ldc_per_band = {}
            for bi in band_order:
                # try claret limb darkening - switch to new model file bad corner
                mf = f"{data_dir}/ultracam_model_file_{bi}"
                try:
                    a1 = float(read_param(mf, "ldc1_1"))
                    a2 = float(read_param(mf, "ldc1_2"))
                    a3 = float(read_param(mf, "ldc1_3"))
                    a4 = float(read_param(mf, "ldc1_4"))
                    ldc_per_band[bi] = (a1, a2, a3, a4)
                    logger.info(f"Band {bi} LDC from model file: "
                                f"a1={a1:.4f} a2={a2:.4f} a3={a3:.4f} a4={a4:.4f}")
                except (KeyError, FileNotFoundError, OSError) as exc:
                    logger.warning(f"Could not read LDC for band {bi} from {mf}: {exc}")
            sed_data = {
                'claret_interp': claret_sed,
                'F_WD_OBS':      {bi: wd_flux_raw[bi][0] for bi in band_order},
                'F_WD_OBS_ERR':  {bi: wd_flux_raw[bi][1] for bi in band_order},
                'A_OVER_EBV':    A_over_EBV,
                'band_order':    band_order,
                'gaia_plx':      gaia_plx,
                'gaia_plx_err':  gaia_plx_err,
                'ebv_map':       ebv_map,
                'ebv_err':       max(ebv_err, 1e-4),
                'ldc_per_band':  ldc_per_band,
            }
            logger.info(f"SED prior enabled: plx={gaia_plx}±{gaia_plx_err} mas, "
                        f"E(B-V)={ebv_map}±{ebv_err}")
        except Exception as e:
            logger.warning(f"SED prior disabled: {e}")
            sed_data = None
    else:
        logger.info("gaia_parallax not set in config; SED prior disabled")

    names   = list(results[2]["names"])
    p_best  = results[2]["p_best"].copy()

    if sed_data is not None:
        names  = names + ["parallax", "ebv"]
        p_best = np.append(p_best, [gaia_plx, ebv_map])

    logger.info(f"MCMC parameter names: {names}")
    ndim        = len(p_best)
    mcmc_steps  = 1000
    steps_mcmc = 0.01 * np.abs(p_best)
    abs_floor  = np.full(ndim, 1e-6)
    for k, nm in enumerate(names):
        if nm == "t1":       abs_floor[k] = 20.0
        if nm == "t2":       abs_floor[k] = 10.0
        if nm == "parallax": abs_floor[k] = gaia_plx_err
        if nm == "ebv":      abs_floor[k] = max(ebv_err, 1e-4)
    steps_mcmc = np.maximum(steps_mcmc, abs_floor)

    FIXED_ABS_STEP = {"iangle": 0.03, "r1": 3e-4, "r2": 1.5e-4}
    for k, nm in enumerate(names):
        if nm in FIXED_ABS_STEP:
            steps_mcmc[k] = FIXED_ABS_STEP[nm]

    nwalkers = max(4 * ndim, 32)
    rng      = np.random.default_rng()
    p0       = p_best + steps_mcmc * rng.standard_normal((nwalkers, ndim))

    if "iangle" in names:
        i_idx = names.index("iangle")
        p0[:, i_idx] = np.clip(p0[:, i_idx], None, 89.99)

    logger.info(f"Walker init steps: { {nm: f'{steps_mcmc[k]:.4g}' for k, nm in enumerate(names)} }")
    logger.info(f"Running MCMC with {nwalkers} walkers, {ndim} parameters, {mcmc_steps} steps")

    moves = [
        (emcee.moves.StretchMove(), 0.8),
        (emcee.moves.DEMove(), 0.2),
    ]

    with Pool(processes=32) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(names, data_dir, rv_config, gaia_id, band_order, a_r_sun, period,
                wd_logg, wd_logg_err, logger, wd_interp, comp_interp,
                spec_priors, 1.0, sed_data),
            pool=pool,
            moves=moves
        )
        logger.info("Starting MCMC sampling...")
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

    tau     = sampler.get_autocorr_time(tol=0)
    logger.info(f"Autocorrelation time: {tau}")
    discard = int(3 * np.nanmax(tau))
    thin    = max(1, int(0.5 * np.nanmin(tau)))

    log_prob_chain = sampler.get_log_prob()           # (nsteps, nwalkers)
    mean_lp = np.mean(log_prob_chain[discard:], axis=0)
    lp_med  = np.median(mean_lp)
    lp_mad  = np.median(np.abs(mean_lp - lp_med))
    good_walkers = mean_lp > lp_med - 5 * lp_mad
    n_bad = (~good_walkers).sum()
    logger.info(f"Walker sigma-clip: removing {n_bad}/{nwalkers_out} outlier walkers "
                f"(threshold={lp_med - 5*lp_mad:.1f}, median={lp_med:.1f})")

    logger.info(f"Discarding {discard} burn-in steps, thinning by {thin}")

    full_chain = sampler.get_chain(discard=discard, thin=thin)   # (steps, walkers, ndim)
    samples    = full_chain[:, good_walkers, :].reshape(-1, ndim_out)
    param_med = np.median(samples, axis=0)

    if "t0" in names:
        t0_idx = names.index("t0")
        samples[:, t0_idx] = ultracam_t0_nearby + (samples[:, t0_idx] - 0.5)*period

    for name, med in zip(names, param_med):
        logger.info(f"{name} = {med:.6f}")

    q_samples  = samples[:, names.index("q")]
    vs_samples = samples[:, names.index("velocity_scale")]
    P_s = (period * u.day).to(u.s)
    a_m = (vs_samples * 1e3 * P_s.value / (2.0 * np.pi)) * u.m

    a_med = np.median(a_m.value)

    a_rsun = (a_med * u.m).to(u.R_sun).value
    m_total_kg = ((4 * np.pi**2 * a_m**3) / (const.G * P_s**2)).value
    m_total_msun = m_total_kg / const.M_sun.value
    m1_samples = m_total_msun / (1 + q_samples)
    m2_samples = q_samples * m1_samples

    _CORNER_LABEL = {
        "iangle":         r"$i\ [^\circ]$",
        "r1":             r"$r_1/a$",
        "r2":             r"$r_2/a$",
        "t1":             r"$T_1\ [K]$",
        "t2":             r"$T_2\ [K]$",
        "velocity_scale": r"$V_{\rm s}\ [\rm km\,s^{-1}]$",
        "parallax":       r"$\varpi\ [\rm mas]$",
        "ebv":            r"$E(B-V)$",
    }
    # Drop q; keep everything else (incl. parallax/ebv if present)
    keep_cols        = [i for i, n in enumerate(names) if n != "q"]
    phys_names       = [names[i] for i in keep_cols]
    samples_phys     = samples[:, keep_cols]
    samples_extended = np.hstack([m1_samples[:, None], m2_samples[:, None], samples_phys])
    corner_labels    = (
        [r'$M_{\rm WD}\ [M_\odot]$', r'$M_{\rm comp}\ [M_\odot]$']
        + [_CORNER_LABEL.get(n, n) for n in phys_names]
    )
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

    lcurve_model_plot(logger, f"{data_dir}/lc_with_model", results_best).lc_with_model(
        use_phase=True, oot_norms_jy=None
    )
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
        'logg1':     args.logg1,
        'logg1_err':     args.logg1_err,
    }, logger)
