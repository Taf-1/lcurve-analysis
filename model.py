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
from scipy.optimize import curve_fit
from plotting import lcurve_model_plot, stats_plots
import plotting
import matplotlib as mpl
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from astroquery.vizier import Vizier

#os.nice(19)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
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
        ("t0", t0_val, 2), ("t0", "0.001", 3), ("t0", "0", 5),
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

def eggleton_rl(q: float) -> float:
    if q <= 0:
        return np.nan
    q23 = q ** (2.0 / 3.0)
    return 0.49 * q23 / (0.6 * q23 + np.log(1.0 + q ** (1.0 / 3.0)))

def split_normal_logpdf(x, mu, sig_lo, sig_hi):
    if x < mu:
        sig = sig_lo
    else:
        sig = sig_hi
    return -0.5 * ((x - mu) / sig) ** 2

def _decode_geometry(p: dict) -> dict:
    if "cos_i" not in p:
        return p
    cos_i = p.pop("cos_i")
    b2 = cos_i ** 2
    if "w" in p:
        w = p.pop("w")
        z = p.pop("z")
        if w <= 0.0 or z <= 0.0 or z >= w:
            raise ValueError(f"invalid: cos_i={cos_i}, w={w}, z={z}")
        r_sum  = np.sqrt(w + b2)
        r_diff = np.sqrt(z + b2)
    else:
        r_sum  = p.pop("r_sum")
        r_diff = p.pop("r_diff")
        if r_sum <= 0.0 or r_diff <= 0.0 or r_diff >= r_sum:
            raise ValueError(f"invalid: cos_i={cos_i}, r_sum={r_sum}, r_diff={r_diff}")
    if not (0.0 < cos_i < 1.0):
        raise ValueError(f"invalid: cos_i={cos_i}")
    p["iangle"] = np.degrees(np.arccos(cos_i))
    p["r1"]     = (r_sum - r_diff) / 2.0
    p["r2"]     = (r_sum + r_diff) / 2.0
    return p

def query_vizier(vizier_catalogue: str) -> Table:
    Vizier.ROW_LIMIT = 30000
    cat = Vizier.get_catalogs(vizier_catalogue)
    return cat[0]

def interp_param(col, filt_data, points, query):
    values = np.array(filt_data[col], dtype=float)
    val = LinearNDInterpolator(points, values)(query)[0]
    if np.isnan(val):
        val = NearestNDInterpolator(points, values)(query)[0]
    return float(val)

def itp(filt_data, logg, temp):
    points = np.column_stack([np.array(filt_data['logg']), np.array(filt_data['Teff'])])
    query  = np.array([[logg, temp]])
    return interp_param('u', filt_data, points, query)

def comp_linear_ldc(logg: float, temp: float, filt: str) -> tuple[float, float, float, float]:
    data = query_vizier("J/A+A/546/A14/limb1-4")
    mask = (data["Filt"] == filt) & (data["Mod"] == 'qs') & (data['Met'] == 'L')
    filt_data = data[mask]
    u_ms = itp(filt_data, logg, temp)
    return u_ms

def ellips_modu_coeff(flux_model: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, config: str, logg: float, temp: float, filt: str) -> float:
    f_ratio = (np.max(flux_model) - 1) / 1
    max_idx = np.argmax(flux_model)
    max_flux = flux[max_idx]
    max_flux_err = flux_err[max_idx]
    u_ms = comp_linear_ldc(logg, temp, filt)
    with open(config, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "=":
                if parts[0] == "gravity_dark2":
                    b_ms = float(parts[2])
                    break
    return f_ratio, max_flux_err, u_ms, b_ms

def grav_red_prior(rv_config: str) -> tuple[float, float]:
    if rv_config is None:
        return 0.0, 0.0
    xs = xshooter_params(logger, rv_config)
    rvs = xs.load_rv_parameters()
    v_sys1, vsy2 = rvs["vsys1"][0], rvs["vsys2"][0]
    v_sys_err1, v_sys_err2 = rvs["vsys1"][1], rvs["vsys2"][1]
    grav_red = v_sys1 - vsy2
    grav_red_err = np.sqrt(v_sys_err1**2 + v_sys_err2**2)
    return grav_red, grav_red_err

def log_prior(params: np.ndarray, names: list[str], rv_config: str | None,
              period: float, wd_logg: float | None, wd_logg_err: float | None,
              logger: logging.Logger,
              spec_priors: dict | None = None,
              roche_fill_max: float = 1.0,
              K_total: float = 0.0, f_ratio_obs=None, f_ratio_err=None, u_ms=None, b_ms=None) -> float:
    if params is None or not np.all(np.isfinite(params)):
        return -np.inf

    p = dict(zip(names, params))
    try:
        _decode_geometry(p)
    except ValueError:
        return -np.inf

    lp = 0.0

    if not (78 < p["iangle"] < 90):      return -np.inf
    if not (0.001 < p["r1"]  < 0.05):    return -np.inf
    if not (0.01 < p["r2"]  < 0.5):   return -np.inf
    if not (9500  < p["t1"]  < 12750):   return -np.inf
    if not (2000  < p["t2"]  < 3000):    return -np.inf
    if "q" not in p or p["q"] <= 0:
        return -np.inf

    if not {"q", "iangle", "r1", "t1"}.issubset(p):
        return lp
    if K_total <= 0.0:
        return lp

    if rv_config is not None:
        xs     = xshooter_params(logger, rv_config, p["iangle"])
        rvs    = xs.load_rv_parameters()
        Kscale = K_total / (1.0 + p["q"])
        vs  = K_total / np.sin(np.radians(p["iangle"]))
        P_s = (period * u.day).to(u.s)
        a_m = (vs * 1e3 * P_s.value / (2.0 * np.pi)) * u.m
        a_med  = np.median(a_m.value)
        a_rsun = (a_med * u.m).to(u.R_sun).value
        lp += -0.5 * ((Kscale - rvs["K2"][0]) / rvs["K2"][1]) ** 2
        lp += -0.5 * ((Kscale * p["q"] - rvs["K1"][0]) / rvs["K1"][1]) ** 2
        logger.debug(f"RV prior: Kscale = {Kscale:.3f}, K2_pred = {rvs['K2'][0]:.3f}, K1_pred = {rvs['K1'][0]:.3f}, lp_K={-0.5 * ((Kscale - rvs['K2'][0]) / rvs['K2'][1]) ** 2 + -0.5 * ((Kscale * p['q'] - rvs['K1'][0]) / rvs['K1'][1]) ** 2:.3f}, total={lp:.3f}")

        # remove degeneracy between scaled radii and i 
        grav_red, grav_red_err = grav_red_prior(rv_config)
        r_sin3i = (P_s * 0.635 * rvs["K2"][0] * K_total**2) / (2 * np.pi * grav_red * const.G.value)
        A = K_total**2
        A_err = 2 * K_total * (np.sqrt(rvs["K1"][1]**2 + rvs["K2"][1]**2))
        B = rvs["K2"][0] * A
        B_err = B * np.sqrt((rvs["K2"][1]/rvs["K2"][0])**2 + (A_err/A)**2)
        C = B / grav_red
        C_err = C * np.sqrt((B_err/B)**2 + (grav_red_err/grav_red)**2)
        r_sin3i_err = C_err * (P_s * 0.635) / (2 * np.pi * const.G.value)
        lp += -0.5 * ((p["r1"] * a_rsun * np.sin(np.radians(p["iangle"]))**3 - r_sin3i) / r_sin3i_err) ** 2

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

        if "r1" in spec_priors:
            mu, sigma = spec_priors["r1"]
            lp_tspec += -0.5 * ((p["r1"] - (mu/a_rsun)) / (sigma/a_rsun)) ** 2

        if "r2" in spec_priors:
            mu, sigma = spec_priors["r2"]
            lp_tspec += -0.5 * ((p["r2"] - (mu/a_rsun)) / (sigma/a_rsun)) ** 2

    lp += lp_tspec

    if f_ratio_obs is not None and f_ratio_err is not None:
        a_coeff = (0.15 * (15.0 + u_ms) * (1.0 + b_ms)) / (3.0 - u_ms)
        f_ratio_pred = a_coeff * (p["r2"]) ** 3 * (rvs["K2"][0] / rvs["K1"][0]) * np.sin(np.radians(p["iangle"])) ** 2
        lp += -0.5 * ((f_ratio_pred - f_ratio_obs) / f_ratio_err) ** 2 + np.log(2*np.pi*f_ratio_err**2)

    return lp

def log_likelihood(params: np.ndarray, names: list[str], data_dir: str, rv_config: str | None, gaia_id: str, band_order: list[int], logger: logging.Logger) -> float:
    p_dict = dict(zip(names, params))
    try:
        _decode_geometry(p_dict)
    except ValueError:
        return -np.inf
    decoded_names = list(p_dict.keys())
    decoded_vals  = [p_dict[n] for n in decoded_names]
    total_ll = 0.0
    for band_idx in band_order:
        base      = f"{data_dir}/{gaia_id}_model_simplex_model_{band_idx}"
        orig_data = f"{data_dir}/{gaia_id}_phase_data_file_{band_idx}"
        _, flux, flux_errors, weights = np.loadtxt(orig_data, usecols=(0, 2, 3, 4), unpack=True)
        walker_id = uuid.uuid4().hex[:8]
        mcmc_file = f"{data_dir}/w{walker_id}_mcmc_model"
        mcmc_dat  = f"{data_dir}/w{walker_id}_mcmc_dat"

        param_names = decoded_names + ["quad"]
        param_vals  = [f"{v:.15e}" for v in decoded_vals] + ["0"]
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

    return total_ll

def log_probability(params, names, data_dir, rv_config, gaia_id, band_order,
                    a_r_sun, period, wd_logg, wd_logg_err, logger,
                    spec_priors=None, roche_fill_max=1.0,
                    K_total=0.0, f_ratio=None, max_flux_err=None, u_ms=None, b_ms=None) -> float:
    lp = log_prior(params, names, rv_config, period, wd_logg, wd_logg_err, logger,
                   spec_priors, roche_fill_max, K_total, f_ratio, max_flux_err, u_ms, b_ms)
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})
    ax1.plot(phase_model, flux_model, 'k-', lw=2, zorder=5)
    ax1.errorbar(phase, flux, yerr=flux_err, fmt='o', color='red',
                 markersize=3, alpha=0.7, zorder=3)

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
    comp_logg = float(cfg.get('logg2', 5.0))
    spec_priors = {}
    if 't1_spec' in cfg:
        spec_priors['t1'] = (float(cfg['t1_spec']),
                             float(cfg['t1_low']), float(cfg['t1_up']))
    if 't2_spec' in cfg:
        spec_priors['t2'] = (float(cfg['t2_spec']),
                             float(cfg['t2_low']), float(cfg['t2_up']))
    if 'r1_spec' in cfg and 'r1_spec_err' in cfg:
        spec_priors['r1'] = (float(cfg['r1_spec']),
                             float(cfg['r1_spec_err']))
    if 'r2_spec' in cfg and 'r2_spec_err' in cfg:
        spec_priors['r2'] = (float(cfg['r2_spec']),
                             float(cfg['r2_spec_err']))
    spec_priors = spec_priors or None
    logger.info(f"Spectroscopic priors: {spec_priors}")

    data_dir  = f"{data_root}/{gaia_id}"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    rv_config = f"{data_root}/{gaia_id}_rv_config"
    if not os.path.exists(rv_config):
        rv_config = None

    K_total = 0.0
    if rv_config is not None:
        _rvs    = xshooter_params(logger, rv_config).load_rv_parameters()
        K_total = _rvs["K1"][0] + _rvs["K2"][0]
        logger.info(f"RV: K1={_rvs['K1'][0]:.2f}, K2={_rvs['K2'][0]:.2f}, "
                    f"K1+K2={K_total:.2f} km/s")

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
        weights = np.ones(len(flux_bin))

        nsub = np.ones(len(phase_bin), dtype=int)
        ns_ingr = max(nsub_for(ingr_mask), 7)
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

    mcmc_allowed = {"q", "iangle", "r1", "r2", "t1", "t2"}
    with open(reference_model_simplex, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and parts[1] == "=":
                if parts[0] == "t2":
                    t2_ref = float(parts[2])
                    break
    f_ratio, max_flux_err, u_ms, b_ms = ellips_modu_coeff(results_simplex[3]['flux_model'], results_simplex[3]["flux"], results_simplex[3]['flux_err'], reference_model_simplex, comp_logg, t2_ref, "i'")

    results = {}
    for band_idx in band_order:
        con_lev    = f"{data_dir}/{gaia_id}_model_simplex_model_{band_idx}"
        new_config = vary_mass_ratio(logger, con_lev, data_dir, band_idx)
        names, p_best, steps = adjust_parameters(
            logger, new_config, new_config, [], [], [], mcmc_allowed
        ).load_mcmc_params()
        results[band_idx] = {"names": names, "p_best": p_best, "steps": steps}

    os.chdir(data_dir)

    names   = list(results[2]["names"])
    p_best  = results[2]["p_best"].copy()

    if "iangle" in names:
        i_idx   = names.index("iangle")
        r1_idx  = names.index("r1")
        r2_idx  = names.index("r2")
        iangle0 = p_best[i_idx]
        r1_0    = p_best[r1_idx]
        r2_0    = p_best[r2_idx]
        cos_i0  = np.cos(np.radians(iangle0))
        b0      = cos_i0
        r_sum0  = r1_0 + r2_0
        r_diff0 = r2_0 - r1_0
        w0      = r_sum0**2 - b0**2
        z0      = r_diff0**2 - b0**2
        names[i_idx]   = "cos_i"; p_best[i_idx]  = cos_i0
        names[r1_idx]  = "w";     p_best[r1_idx] = w0
        names[r2_idx]  = "z";     p_best[r2_idx] = z0
        logger.info(f"Geometry reparametrisation: cos_i={cos_i0:.6f}, "
                    f"w={w0:.6f}, z={z0:.6f} "
                    f"(r_sum={r_sum0:.6f}, r_diff={r_diff0:.6f})")

    logger.info(f"MCMC parameter names: {names}")
    ndim       = len(p_best)
    steps_mcmc = 0.001 * np.abs(p_best)
    if "cos_i" in names:
        steps_mcmc[i_idx] = np.sin(np.radians(iangle0)) * np.radians(0.01)

    nwalkers   = 4 * ndim
    mcmc_steps = 1000
    rng = np.random.default_rng()
    p0 = p_best + steps_mcmc * rng.standard_normal((nwalkers, ndim))

    if "cos_i" in names:
 
        sigma_i_deg = 0.01
        sigma_r1    = 1e-4
        sigma_r2    = 1e-3

        i_p0  = np.clip(iangle0 + sigma_i_deg * rng.standard_normal(nwalkers), 82.0, 90)
        r2_p0 = r2_0 + sigma_r2 * rng.standard_normal(nwalkers)
        r1_p0 = r1_0 + sigma_r1 * rng.standard_normal(nwalkers)

        cos_i_p0  = np.cos(np.radians(i_p0))
        r_sum_p0  = r1_p0 + r2_p0
        r_diff_p0 = r2_p0 - r1_p0
        p0[:, i_idx]  = cos_i_p0
        p0[:, r1_idx] = np.maximum(r_sum_p0**2 - cos_i_p0**2, 1e-8)
        p0[:, r2_idx] = np.maximum(r_diff_p0**2 - cos_i_p0**2, 1e-8)
        logger.info(f"  p0 geometry: iangle={iangle0:.3f}°±{sigma_i_deg:.3f}°, "
                    f"r1={r1_0:.5f}±{sigma_r1:.5f}, r2={r2_0:.5f}±{sigma_r2:.5f}")

    logger.info(f"Walker init spread: { {nm: f'{steps_mcmc[k]:.4g}' for k, nm in enumerate(names)} }")
    logger.info(f"Running MCMC with {nwalkers} walkers, {ndim} parameters, {mcmc_steps} steps")

    moves = [
            (emcee.moves.DEMove(),        0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ]

    with Pool(processes=40) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(names, data_dir, rv_config, gaia_id, band_order, a_r_sun, period,
                wd_logg, wd_logg_err, logger, spec_priors, 1.0, K_total, f_ratio, max_flux_err, u_ms, b_ms),
            pool=pool,
            moves=moves
        )
        logger.info("Starting MCMC sampling...")
        sampler.run_mcmc(p0, mcmc_steps, progress=True)

    acc = sampler.acceptance_fraction
    logger.info(f"Acceptance fraction: mean={acc.mean():.3f}, "
                f"min={acc.min():.3f}, max={acc.max():.3f}")

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

    discard = 150
    tau = sampler.get_autocorr_time(discard=discard, tol=50, quiet=True)
    logger.info(f"Autocorrelation time: {tau}")
    #discard = min(int(5 * np.nanmax(tau)), max(0, mcmc_steps - 50))
    thin = 10
    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    if "cos_i" in names:
        cos_i_s = samples[:, i_idx].copy()
        b2_s    = cos_i_s ** 2
        if "w" in names:
            w_s      = samples[:, r1_idx].copy()
            z_s      = samples[:, r2_idx].copy()
            r_sum_s  = np.sqrt(np.maximum(w_s + b2_s, 0.0))
            r_diff_s = np.sqrt(np.maximum(z_s + b2_s, 0.0))
        else:
            r_sum_s  = samples[:, r1_idx].copy()
            r_diff_s = samples[:, r2_idx].copy()
        samples[:, i_idx]  = np.degrees(np.arccos(np.clip(cos_i_s, -1.0, 1.0)))
        samples[:, r1_idx] = (r_sum_s - r_diff_s) / 2.0
        samples[:, r2_idx] = (r_sum_s + r_diff_s) / 2.0
        names[i_idx]  = "iangle"
        names[r1_idx] = "r1"
        names[r2_idx] = "r2"

    param_med = np.median(samples, axis=0)

    if "t0" in names:
        t0_idx = names.index("t0")
        samples[:, t0_idx] = ultracam_t0_nearby + (samples[:, t0_idx] - 0.5)*period

    for name, med in zip(names, param_med):
        logger.info(f"{name} = {med:.6f}")

    q_samples  = samples[:, names.index("q")]
    P_s = (period * u.day).to(u.s)
    if rv_config is not None:
        rvs_post       = xshooter_params(logger, rv_config).load_rv_parameters()
        K_total        = rvs_post["K1"][0] + rvs_post["K2"][0]
        iangle_samples = samples[:, names.index("iangle")]
        vs_samples     = K_total / np.sin(np.radians(iangle_samples))
        logger.info(f"Post-MCMC mass: (K1+K2)={K_total:.2f} km/s / sin(i) per sample")
    else:
        vs_geom    = 2.0 * np.pi * a_r_sun * const.R_sun.value / (period * 86400.0) / 1.0e3
        vs_samples = np.full(len(samples), vs_geom)
        logger.info(f"Post-MCMC mass: no rv_config, using geometric velocity_scale={vs_geom:.4f} km/s")
    a_m = (vs_samples * 1e3 * P_s.value / (2.0 * np.pi)) * u.m
    a_med  = np.median(a_m.value)
    a_rsun = (a_med * u.m).to(u.R_sun).value
    m_total_kg = ((4 * np.pi**2 * a_m**3) / (const.G * P_s**2)).value
    m_total_msun = m_total_kg / const.M_sun.value
    m1_samples = m_total_msun / (1 + q_samples)
    m2_samples = q_samples * m1_samples

    _CORNER_LABEL = {
        "iangle":          r"$i\ (^\circ)$",
        "r1":              r"$r_1/a$",
        "r2":              r"$r_2/a$",
        "t1":              r"$T_1\ (K)$",
        "t2":              r"$T_2\ (K)$",
    }
    keep_cols        = [j for j, n in enumerate(names) if n != "q"]
    corner_names     = [names[j] for j in keep_cols]
    samples_phys     = samples[:, keep_cols]
    samples_extended = np.hstack([m1_samples[:, None], m2_samples[:, None], samples_phys])
    corner_labels    = (
        [r'$M_{\rm WD}\ (M_\odot)$', r'$M_{\rm comp}\ (M_\odot)$']
        + [_CORNER_LABEL.get(n, n) for n in corner_names]
    )
    param_med_corner = np.median(samples_extended, axis=0)
    ndim = samples_extended.shape[1]
    rng = [tuple(np.percentile(samples_extended[:, k], [0.05, 99.95]))
        for k in range(ndim)]

    levels = 1 - np.exp(-0.5 * np.arange(1, 4)**2)
    cornerfig = corner.corner(
        samples_extended, labels=corner_labels,
        levels=levels,
        bins=20,                  # default; no smoothing
        plot_datapoints=True,
        plot_density=True,
        quantiles=[0.16, 0.5, 0.84],
        plot_contours=True,
        fill_contours=False,
        data_kwargs={"alpha": 0.5, "ms": 3},  # darker, larger scatter
        label_kwargs={"fontsize": 26},
        fig=plt.figure(figsize=(20, 20)),
    )
    for ax in cornerfig.get_axes():
        ax.tick_params(labelsize=18)

    cornerfig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.99)
    cornerfig.savefig(f"{data_dir}/model_corner_plot.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    cornerfig.savefig(f"{data_dir}/model_corner_plot.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
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
        of.write(f"{'a [m]':<30} {a_med:>12.6e} {'a [R_sun]':>12} {a_rsun:>12.4f}\n")
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
