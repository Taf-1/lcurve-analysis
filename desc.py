import argparse as ap
import logging
import os
import numpy as np
from logger import sf_logging
from lcurve_commands import lcurve
from lcurve_data_files import create_data_file, save_data_file
from lcurve_model_file import claret_tables_interp, effective_wavelength, adjust_parameters
from plotting import lcurve_model_plot
from astropy.time import Time

def arg_parse() -> ap.Namespace:
    p = ap.ArgumentParser()
    p.add_argument("data_root",   help="Root directory for all target data")
    p.add_argument("ultracam_dat", help="FITS file containing the ULTRACAM data")
    p.add_argument("target",      help="Name of target")
    p.add_argument("period",      help="Orbital period of binary system in days")
    p.add_argument("t0",          help="t0 of binary system in BJD")
    p.add_argument("teff1",       help="Effective temperature of the WD")
    p.add_argument("logg1",       help="log g of the WD")
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
                  bf1: float, bf2: float, fix_geometry: bool = False) -> str:
    new_config = f"{path}/ultracam_model_file_{band_index}"
    a1,  a2,  a3,  a4  = wd_ldc
    a1c, a2c, a3c, a4c = comp_ldc
    y1 = wd_gdc
    y2 = comp_gdc
    period = float(period)
    t0 = float(t0)
    geom = "0" if fix_geometry else "1"

    changes = [
        # Always-fixed parameters
        ("q", "0", 5),
        ("velocity_scale", "0", 5),
        # Geometry-dependent fit flags
        ("iangle", geom, 5),
        ("r1", geom, 5),
        ("r2", geom, 5),
        ("t1", geom, 5),
        ("t2", geom, 5),
        ("absorb", geom, 5),
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
        ("gravity_dark1", f"{y1}", 2), ("gravity_dark1", geom, 5),
        ("gravity_dark2", f"{y2}", 2), ("gravity_dark2", geom, 5),
        # 3-part lines: wavelength and beam factors 
        ("wavelength", f"{wavelength}", 2),
        ("beam_factor1", f"{bf1}", 2), ("beam_factor2", "0", 5),
        ("beam_factor2", f"{bf2}", 2), ("beam_factor2", "0", 5), 
        ("iscale", "0" if not fix_geometry else "1", 2)
    ]

    if fix_geometry: 
        cur_period = read_param(config, "period")  
        changes += [
            ("period",  cur_period, 2), ("period",  "0", 5),
            ("tperiod", cur_period, 2), ("t0", "0", 5),         
        ]
    else:
        changes += [
            ("period",  f"{period}", 2), ("period", "1", 5),
            ("tperiod", f"{period}", 2),                  
            ("t0", f"0.0", 2), ("t0", "1", 5),
        ]

    names   = [c[0] for c in changes]
    values  = [c[1] for c in changes]
    indices = [c[2] for c in changes]

    return adjust_parameters(
        logger, config, new_config, names, values, indices, set(names)
    ).change_config()

def run_model(logger: logging.Logger, filename: str, band_index: int, t0: float, pathname: str, tar_name: str, period: float,
              wd_temp: float, wd_logg: float, wd_type: str, comp_temp: float, comp_logg: float, filt: str,
              reference_model: str | None = None, example_model: str | None = None, wd_model_path: str | None = None, comp_model_path: str | None = None) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:

    time, exp_time, flux_norm, flux_err_norm, t0_nearby = create_data_file(
        logger, pathname, filename, band_index, t0, period
    ).extract_ultracam_data()

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
        wd_ldc, comp_ldc, wd_gdc, comp_gdc, wavelength, bf1, bf2, fix_geometry
    )

    ultracam_prelim = f"{pathname}/{tar_name}_ultracam_model_file_{band_index}"
    lcurve(logger, model_config, output, ultracam_prelim).simplex()

    model_dat = f"{pathname}/desc_model_{band_index}"
    lcurve(logger, ultracam_prelim, output, model_dat).lroche()

    time_model, flux_model, _ = np.loadtxt(model_dat, usecols=(0, 2, 3), unpack=True)
    return ultracam_prelim, time, flux_norm, flux_err_norm, time_model, flux_model, t0_nearby

def run(cfg: dict, logger: logging.Logger) -> None:
    data_root     = cfg['data_root']
    filename      = cfg['ultracam_dat']
    tar_name      = cfg['name']
    period        = float(cfg['period'])
    t0            = cfg['t0']
    wd_temp       = float(cfg['teff1'])
    wd_logg       = float(cfg['logg1'])
    wd_type       = cfg['wdtype']
    comp_temp     = float(cfg['teff2'])
    comp_logg     = float(cfg['logg2'])
    example_model = cfg['example_model']
    wd_model_path = cfg['wd_model_path']
    comp_model_path = cfg['comp_model_path']
    pathname      = f"{data_root}/{tar_name}"

    if not os.path.exists(pathname):
        os.makedirs(pathname)

    band_order = [2, 1, 3]
    band_names = {1: "u'", 2: "g'", 3: "i'"}

    results = {}
    reference_model = None
    for band_idx in band_order:
        model_file, time, flux_norm, flux_err_norm, time_model, flux_model, t0_nearby = run_model(
            logger, filename, band_idx, t0, pathname, tar_name, period,
            wd_temp, wd_logg, wd_type, comp_temp, comp_logg,
            band_names[band_idx], reference_model, example_model, wd_model_path, comp_model_path
        )
        results[band_idx] = {
            'time':       time,
            'flux':       flux_norm,
            'flux_err':   flux_err_norm,
            'time_model': time_model,
            'flux_model': flux_model,
            't0_nearby':  t0_nearby,
        }
        if band_idx == 2:
            reference_model = model_file
            logger.info("Using g-band geometry for subsequent fits")

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
        'wdtype':       args.wdtype,
        'teff2':        args.teff2,
        'logg2':        args.logg2,
        'example_model': args.example_model,
        'wd_model_path': args.wd_model_path,
        'comp_model_path': args.comp_model_path,
    }, logger)