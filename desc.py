import argparse as ap
import logging
import os
import numpy as np
from logger import sf_logging
from lcurve_commands import lcurve
from lcurve_data_files import create_data_file, save_data_file
from lcurve_model_file import claret_tables_interp, effective_wavelength, adjust_parameters, roche_distortion
from plotting import lcurve_model_plot
from astropy.time import Time
from lcurve_rv_calc import xshooter_params
import tqdm as tqdm

BASELINE_GRID = {"nlat1f": 50, "nlat1c": 50, "nlat2f": 150, "nlat2c": 150}
GRID_KEYS = ("nlat1f", "nlat1c", "nlat2f", "nlat2c")

def arg_parse() -> ap.Namespace:
    p = ap.ArgumentParser()
    p.add_argument("data_root",   help="Root directory for all target data")
    p.add_argument("ultracam_dat", help="FITS file containing the ULTRACAM data")
    p.add_argument("target",      help="Name of target")
    p.add_argument("period",      help="Orbital period of binary system in days")
    p.add_argument("t0",          help="t0 of binary system in BJD")
    p.add_argument("teff1",       help="Effective temperature of the WD")
    p.add_argument("logg1",       help="log g of the WD")
    p.add_argument("logg1_err",   help="Error in log g of the WD")
    p.add_argument("wdtype",      help="WD type: DA or DB")
    p.add_argument("teff2",       help="Effective temperature of the companion")
    p.add_argument("logg2",       help="log g of the companion")
    p.add_argument("example_model", help="Base lcurve model file used as initial template")
    p.add_argument("wd_model_path", help="SPEEDYFIT KOESTER wd model txt file")
    p.add_argument("comp_model_path", help="SPEEDYFIT NextGEN/Phoenix comp model txt file")
    return p.parse_args()

def read_param(config, name):
    with open(config) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and p[0] == name and p[1] == "=":
                return p[2]
    raise KeyError(name)

def change_params(logger: logging.Logger, config: str, path: str, period: float, t0: float, band_index: int,
                  wd_ldc: list[float], comp_ldc: list[float], wd_gdc: float, comp_gdc: float, wavelength: float,
                  bf1: float, bf2: float, rv_config: str | None = None, fix_geometry: bool = False,
                  grid_res: dict | None = None) -> str:
    new_config = f"{path}/ultracam_model_file_{band_index}"
    a1,  a2,  a3,  a4  = wd_ldc
    a1c, a2c, a3c, a4c = comp_ldc
    y1 = wd_gdc
    y2 = comp_gdc
    period = float(period)
    t0 = float(t0)
    geom = "0" if fix_geometry else "1"
    grid = dict(BASELINE_GRID) if grid_res is None else dict(grid_res)

    rv_value_changes = []
    if rv_config is not None:
        rvs = xshooter_params(logger, rv_config).load_rv_parameters()
        vs_value = rvs["K1"][0] + rvs["K2"][0]
        q_val = rvs["K1"][0] / rvs["K2"][0]
        rv_value_changes = [("q", f"{q_val}", 2), ("velocity_scale", f"{vs_value}", 2)]
    
    r1_init = float(read_param(config, "r1"))
    r2_init = float(read_param(config, "r2"))
    logger.info(f"Band {band_index}: Initial r1 = {r1_init:.6f}, r2 = {r2_init:.6f}")

    if not fix_geometry:
        r2 = float(roche_distortion(r2_init, q_val))
        logger.info(f"Band {band_index}: Roche-distorted r2 = {r2:.6f} (q = {q_val:.4f})")
    else:
        r2 = r2_init
        logger.info(f"Band {band_index}: Using fixed r2 = {r2:.6f} (q = {q_val:.4f})")

    phase1 = (np.arcsin(r1_init + r2) / (2 * np.pi))+0.001
    logger.info(f"Band {band_index}: Calculated phase1 = {phase1:.6f}")
    phase2 = 0.5 - phase1
    logger.info(f"Band {band_index}: Calculated phase2 = {phase2:.6f}")

    changes = [
        # Always-fixed parameters
        *rv_value_changes,
        ("q", "0", 5),
        ("velocity_scale", geom, 5),
        # Geometry-dependent fit flags
        ("iangle", geom, 5),
        ("r1", geom, 5),
        ("r2", geom, 5),
        ("t1", geom, 5),
        ("t2", geom, 5),
        ("absorb", "0.5", 2), ("absorb", geom, 5),
        # WD limb darkening: set value and lock
        ("ldc1_1", f"{a1}", 2), ("ldc1_1", geom, 5),
        ("ldc1_2", f"{a2}", 2), ("ldc1_2", geom, 5),
        ("ldc1_3", f"{a3}", 2), ("ldc1_3", geom, 5),
        ("ldc1_4", f"{a4}", 2), ("ldc1_4", geom, 5),
        # Companion limb darkening: set value and lock
        ("ldc2_1", f"{a1c}", 2), ("ldc2_1", geom, 5),
        ("ldc2_2", f"{a2c}", 2), ("ldc2_2", geom, 5),
        ("ldc2_3", f"{a3c}", 2), ("ldc2_3", geom, 5),
        ("ldc2_4", f"{a4c}", 2), ("ldc2_4", geom, 5),
        # Gravity darkening: set value and lock
        ("gravity_dark1", geom, 5), ("gravity_dark1", f"{y1}", 2), 
        ("gravity_dark2", f"{y2}", 2), ("gravity_dark2", geom, 5),
        # 3-part lines: wavelength and beam factors 
        ("wavelength", f"{wavelength}", 2),
        ("beam_factor1", "0", 5), #("beam_factor1", f"{bf1}", 2), 
        ("beam_factor2", "0", 5), #("beam_factor2", f"{bf2}", 2), 
        ("iscale", "0" if not fix_geometry else "1", 2),
        ("phase1", f"{phase1:.6f}", 2),
        ("phase2", f"{phase2:.6f}", 2),
        # Grid resolution — written explicitly so it is never silently inherited
        ("nlat1f", str(grid["nlat1f"]), 2),
        ("nlat1c", str(grid["nlat1c"]), 2),
        ("nlat2f", str(grid["nlat2f"]), 2),
        ("nlat2c", str(grid["nlat2c"]), 2),
    ]

    if fix_geometry: 
        cur_period = read_param(config, "period")
        cur_vs = read_param(config, "velocity_scale")  
        changes += [
            ("period",  cur_period, 2), ("period",  "0", 5),
            ("tperiod", cur_period, 2), ("t0", "0", 5),
            ("velocity_scale", cur_vs, 2),  
        ]
    else:
        changes += [
            ("period",  f"{period}", 2), ("period", "1", 5),
            ("tperiod", f"{period}", 2),                  
            ("t0", f"0.0", 2), ("t0", "1", 5), ("r2", f"{r2:.6f}", 2),
        ]

    names   = [c[0] for c in changes]
    values  = [c[1] for c in changes]
    indices = [c[2] for c in changes]

    return adjust_parameters(
        logger, config, new_config, names, values, indices, set(names)
    ).change_config()

def run_model(logger: logging.Logger, filename: str, band_index: int, t0: float, pathname: str, tar_name: str, period: float,
              wd_temp: float, wd_logg: float, wd_type: str, comp_temp: float, comp_logg: float, filt: str,
              reference_model: str | None = None, example_model: str | None = None, wd_model_path: str | None = None,
              comp_model_path: str | None = None, rv_config: str | None = None,
              grid_res: dict | None = None) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:

    time, exp_time, flux_norm, flux_err_norm, t0_nearby, oot_norm_jy = create_data_file(
        logger, pathname, filename, band_index, t0, period
    ).extract_ultracam_data()
    logger.info(f"Band {band_index}: OOT norm = {oot_norm_jy*1e3:.4f} mJy")

    output = f"{pathname}/{tar_name}_ultracam_data_file_{band_index}"
    save_data_file(
        logger, tar_name, band_index, time, exp_time,
        flux_norm, flux_err_norm, t0_nearby, output
    ).write_data_file()

    claret = claret_tables_interp(logger, wd_temp, wd_logg, wd_type, comp_temp, comp_logg, filt)
    wd_ldc = claret.wd_limb_darkening()
    comp_ldc = claret.comp_limb_darkening()
    wd_gdc = claret.wd_gravity_darkening()
    comp_gdc = claret.comp_gravity_darkening()

    ew = effective_wavelength(logger, band_index)
    wavelength = ew.pivot_wave()

    wd_wave, wd_flux = effective_wavelength.load_speedyfit_spectrum(wd_model_path)
    comp_wave, comp_flux = effective_wavelength.blackbody_spectrum(comp_temp)
    bf1 = ew.beam_factor(wd_wave, wd_flux)
    bf2 = ew.beam_factor(comp_wave, comp_flux)

    fix_geometry = reference_model is not None
    base_model = reference_model if fix_geometry else example_model
    if base_model is None:
        raise ValueError("example_model must be provided in the config for the first band fit")
    if fix_geometry:
        logger.info(f"Band {band_index}: geometry fixed, fitting temperatures only")
    else:
        logger.info(f"Band {band_index}: fitting geometry + temperatures")

    model_config = change_params(
        logger, base_model, pathname, period, t0_nearby, band_index,
        wd_ldc, comp_ldc, wd_gdc, comp_gdc, wavelength, bf1, bf2, rv_config, fix_geometry,
        grid_res=grid_res
    )

    ultracam_prelim = f"{pathname}/{tar_name}_ultracam_model_file_{band_index}"
    lcurve(logger, model_config, output, ultracam_prelim).simplex()

    model_dat = f"{pathname}/desc_model_{band_index}"
    lcurve(logger, ultracam_prelim, output, model_dat).lroche()

    time_model, flux_model, _ = np.loadtxt(model_dat, usecols=(0, 2, 3), unpack=True)
    return ultracam_prelim, time, flux_norm, flux_err_norm, time_model, flux_model, t0_nearby, oot_norm_jy

def jitter_grid_search(logger: logging.Logger, model_file: str, data_file: str,
                       pathname: str, n_trials: int = 20, half_width: int = 5) -> dict:
    flux_data = np.loadtxt(data_file, usecols=2)
    flux_err  = np.loadtxt(data_file, usecols=3)

    best_chi2 = np.inf
    best_grid = dict(BASELINE_GRID)

    for i in range(n_trials):
        grid = {
            k: max(1, int(np.random.randint(BASELINE_GRID[k] - half_width,
                                            BASELINE_GRID[k] + half_width + 1)))
            for k in GRID_KEYS
        }
        names   = list(GRID_KEYS)
        values  = [str(grid[k]) for k in GRID_KEYS]
        indices = [2] * len(GRID_KEYS)

        trial_model = f"{pathname}/jitter_trial_{i}"
        trial_out   = f"{pathname}/jitter_out_{i}"
        cfg_path = adjust_parameters(
            logger, model_file, trial_model, names, values, indices, set(names)
        ).change_config()
        lcurve(logger, cfg_path, data_file, trial_out).lroche()

        flux_model = np.loadtxt(trial_out, usecols=2)
        chi2 = float(np.sum(((flux_data - flux_model) / flux_err) ** 2))
        logger.info(f"Jitter trial {i+1}/{n_trials}: {grid} -> chi2 = {chi2:.4f}")

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_grid = dict(grid)

    # clean up temporary files
    for i in range(n_trials):
        for f in (f"{pathname}/jitter_trial_{i}", f"{pathname}/jitter_out_{i}"):
            if os.path.exists(f):
                os.remove(f)

    logger.info(f"Best grid from jitter search: {best_grid} (chi2 = {best_chi2:.4f})")
    return best_grid

def log_g_search(logger: logging.Logger, model_file: str, data_file: str, pathname: str, 
        wd_temp: float, wd_logg: float, wd_logg_err: float, wd_type: str, comp_temp: float, 
        comp_logg: float, filt: str, period: float, t0_nearby: float, rv_config: str, n_trials=1000) -> dict:
    flux_data = np.loadtxt(data_file, usecols=2)
    flux_err  = np.loadtxt(data_file, usecols=3)

    best_chi2 = np.inf
    best_logg = wd_logg

    upper_limit = wd_logg + wd_logg_err
    lower_limit = wd_logg - wd_logg_err

    for i in range(n_trials):
        new_logg = np.random.uniform(lower_limit, upper_limit)
        claret = claret_tables_interp(logger, wd_temp, new_logg, wd_type, comp_temp, comp_logg, filt)
        new_wd_ldc = claret.wd_limb_darkening()
        comp_ldc = claret.comp_limb_darkening()
        wd_gdc = claret.wd_gravity_darkening()
        comp_gdc = claret.comp_gravity_darkening()

        names = [f"ldc1_1", f"ldc1_2", f"ldc1_3", f"ldc1_4", f"ldc2_1", f"ldc2_2", f"ldc2_3", f"ldc2_4", "gravity_dark1", "gravity_dark2"]
        values = [new_wd_ldc[0], new_wd_ldc[1], new_wd_ldc[2], new_wd_ldc[3], comp_ldc[0], comp_ldc[1], comp_ldc[2], comp_ldc[3], wd_gdc, comp_gdc]
        indices = [2] * len(names)
        ew = effective_wavelength(logger, 3)
        wavelength = ew.pivot_wave()
        i_band_config = change_params(
                logger, model_file, pathname, period, t0_nearby, 3,
                new_wd_ldc, comp_ldc, wd_gdc, comp_gdc, wavelength, "0", "0", rv_config, True,
                grid_res=None
            )
        trial_model = f"{pathname}/logg_trial_{i}"
        trial_out   = f"{pathname}/logg_out_{i}"
        cfg_path = adjust_parameters( 
            logger, i_band_config, trial_model, names, values, indices, set(names)
        ).change_config()
        lcurve(logger, cfg_path, data_file, trial_out).lroche()

        flux_model = np.loadtxt(trial_out, usecols=2)
        chi2 = float(np.sum(((flux_data - flux_model) / flux_err) ** 2))
        logger.info(f"Log g trial {i+1}/{n_trials}: {new_logg} -> chi2 = {chi2:.4f}")

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_logg = new_logg

    for i in range(n_trials):
        for f in (f"{pathname}/logg_trial_{i}", f"{pathname}/logg_out_{i}"):
            if os.path.exists(f):
                os.remove(f)

    logger.info(f"Best log g from search: {best_logg} (chi2 = {best_chi2:.4f})")
    return best_logg


def run(cfg: dict, logger: logging.Logger) -> None:
    data_root     = cfg['data_root']
    filename      = cfg['ultracam_dat']
    tar_name      = cfg['name']
    period        = float(cfg['period'])
    t0            = cfg['t0']
    wd_temp       = float(cfg['teff1'])
    wd_logg       = float(cfg['logg1'])
    wd_logg_err   = float(cfg['logg1_err'])
    wd_type       = cfg['wdtype']
    comp_temp     = float(cfg['teff2'])
    comp_logg     = float(cfg['logg2'])
    example_model = cfg['example_model']
    wd_model_path = cfg['wd_model_path']
    comp_model_path = cfg['comp_model_path']
    pathname      = f"{data_root}/{tar_name}"

    if not os.path.exists(pathname):
        os.makedirs(pathname)
    
    rv_config = f"{data_root}/{tar_name}_rv_config"
    if not os.path.exists(rv_config):
        rv_config = None

    band_order = [2, 1, 3]
    band_names = {1: "u'", 2: "g'", 3: "i'"}

    def _run(band_idx, ref_model, start_model, grid, new_logg):
        return run_model(
            logger, filename, band_idx, t0, pathname, tar_name, period,
            wd_temp, new_logg, wd_type, comp_temp, comp_logg,
            band_names[band_idx], ref_model, start_model, wd_model_path, comp_model_path, rv_config,
            grid_res=grid,
        )

    results = {}
    reference_model = None
    best_grid = None
    best_logg = None
    for band_idx in band_order:
        model_file, time, flux_norm, flux_err_norm, time_model, flux_model, t0_nearby, oot_norm_jy = \
            _run(band_idx, reference_model, example_model, best_grid, new_logg=best_logg if best_logg is not None else wd_logg)

        if band_idx == 2:
            # search for best grid using the converged g-band result
            data_file_g = f"{pathname}/{tar_name}_ultracam_data_file_3"
            best_logg = log_g_search(logger, model_file, data_file_g, pathname, wd_temp, wd_logg, wd_logg_err, wd_type, comp_temp, comp_logg, "i'", period, t0_nearby, rv_config)
            logger.info(f"Re-running g-band simplex with best log g: {best_logg}")
            best_grid = jitter_grid_search(logger, model_file, data_file_g, pathname)
            logger.info(f"Re-running g-band simplex with best grid: {best_grid}")
            model_file, time, flux_norm, flux_err_norm, time_model, flux_model, t0_nearby, oot_norm_jy = \
                _run(band_idx, None, model_file, best_grid, best_logg)
            reference_model = model_file
            logger.info("Using g-band geometry (best grid) for subsequent fits")

        results[band_idx] = {
            'time':        time,
            'flux':        flux_norm,
            'flux_err':    flux_err_norm,
            'time_model':  time_model,
            'flux_model':  flux_model,
            't0_nearby':   t0_nearby,
            'oot_norm_jy': oot_norm_jy,
        }
    logger.info(f"All bands processed using log g = {best_logg}. Generating plots...")
    lcurve_model_plot(logger, f"{pathname}/ultracam_model", results).lc_with_model(use_phase=False)

if __name__ == "__main__":
    args     = arg_parse()
    pathname = f"{args.data_root}/{args.target}"
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    logger = sf_logging("desc", f"{pathname}/{args.target}.log").setup_logger()
    run({
        'data_root':    args.data_root,
        'ultracam_dat': args.ultracam_dat,
        'name':         args.target,
        'period':       args.period,
        't0':           args.t0,
        'teff1':        args.teff1,
        'logg1':        args.logg1,
        'logg1_err':   args.logg1_err,
        'wdtype':       args.wdtype,
        'teff2':        args.teff2,
        'logg2':        args.logg2,
        'example_model': args.example_model,
        'wd_model_path': args.wd_model_path,
        'comp_model_path': args.comp_model_path,
    }, logger)