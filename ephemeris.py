# Using the ultracam simplex model, fix all values except period and t0
# Change the NGTS data to suit the format of simplex and run this to determine the ephemeris
# MCMC for period and t0 determination

import argparse as ap
import logging
import os
import numpy as np
import emcee
from emcee import utils
import corner
import uuid
import matplotlib.pyplot as plt
from multiprocessing import Pool
from logger import sf_logging
from lcurve_commands import lcurve
from lcurve_data_files import create_data_file, save_data_file, adjust_data_file
from lcurve_model_file import adjust_parameters

def arg_parse() -> ap.Namespace:
    p = ap.ArgumentParser()
    p.add_argument("data_root", help="Root directory for all target data")
    p.add_argument("NGTS_dat",  help="FITS file containing the NGTS data for the target")
    p.add_argument("target",    help="Name of target")
    p.add_argument("t0",        help="t0 of binary system (BJD TDB)")
    return p.parse_args()

def load_model_params(logger: logging.Logger, config: str, tar: str, path: str) -> str:
    new_config = f"{path}/{tar}_fix_ultracam_model_file"
    all_names = []
    with open(config, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and parts[1] == "=":
                all_names.append(parts[0])

    names   = all_names + ["period", "t0", "period", "t0", "period", "t0", "t0"]
    values  = ["0"] * len(all_names) + ["1", "1", "0.0001", "0.0001", "0.0001", "0.0001", "0"]
    indices = [5] * len(all_names) + [5, 5, 4, 4, 3, 3, 2]

    return adjust_parameters(
        logger, config, new_config, names, values, indices, set(names)
    ).change_config()

def log_prior(params: np.ndarray, names: list[str]) -> float:
    if params is None or not np.all(np.isfinite(params)):
        return -np.inf

    p = dict(zip(names, params))

    if not (0.035 < p["period"] < 0.095):
        return -np.inf
    if not (-1 < p["t0"] < 1):
        return -np.inf

    return 0.0

def log_likelihood(params: np.ndarray, names: list[str], t_rel: np.ndarray, flux: np.ndarray, flux_errors: np.ndarray, orig_data: str, base_lines: list[str], logger: logging.Logger) -> float:
    walker_id = uuid.uuid4().hex[:8]
    mcmc_file = f"/tmp/w{walker_id}_mcmc_model"
    mcmc_dat  = f"/tmp/w{walker_id}_mcmc_dat"

    try:
        with open(mcmc_file, "w") as f:
            for line in base_lines:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "=":
                    for p, name in zip(params, names):
                        if parts[0] == name:
                            parts[2] = f"{p:.15e}"
                            line = " ".join(parts) + "\n"
                            break
                f.write(line)
    except Exception as e:
        logger.info(f"Error writing model: {e}")
        return -np.inf

    try:
        lcurve(logger, mcmc_file, orig_data, mcmc_dat).lroche()
    except Exception as e:
        logger.info(f"lroche failed: {e}")
        return -np.inf
    finally:
        try:
            if os.path.exists(mcmc_file):
                os.remove(mcmc_file)
        except Exception:
            pass

    try:
        mcmc_flux = np.loadtxt(mcmc_dat, usecols=(2,), unpack=True)
    except Exception as e:
        logger.info(f"Error reading model data: {e}")
        return -np.inf
    finally:
        try:
            if os.path.exists(mcmc_dat):
                os.remove(mcmc_dat)
        except Exception:
            pass

    if mcmc_flux.shape != flux.shape or not np.all(np.isfinite(mcmc_flux)):
        return -np.inf

    chi2 = np.sum(((flux - mcmc_flux) / flux_errors) ** 2)
    if not np.isfinite(chi2):
        return -np.inf

    return -0.5 * chi2

def log_probability(params: np.ndarray, names: list[str], t_rel: np.ndarray, flux: np.ndarray, flux_errors: np.ndarray, orig_data: str, base_lines: list[str], logger: logging.Logger) -> float:
    lp = log_prior(params, names)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, names, t_rel, flux, flux_errors, orig_data, base_lines, logger)

def run(cfg: dict, logger: logging.Logger) -> None:
    data_root = cfg['data_root']
    filename  = cfg['ngts_dat']
    tar_name  = cfg['name']
    t0_input  = cfg['t0']
    pathname  = f"{data_root}/{tar_name}"

    if not os.path.exists(pathname):
        os.makedirs(pathname)

    config = f"{pathname}/{tar_name}_ultracam_model_file_3"
    with open(config, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "period" and parts[1] == "=":
                period_val = float(parts[2])
                break
    logger.info(f"Period from Ultracam model: {period_val:.10f} days")

    time, cadence, flux, flux_err, t0_nearby = create_data_file(
        logger, pathname, filename, 1, t0_input, period_val
    ).extract_ngts_data()

    time, flux, flux_err = adjust_data_file(time, flux, flux_err, 8).bin_data_on_time()
    cadence *= 8
    logger.info(f"Binned NGTS data by factor 8: {len(time)} points remaining")

    t_mid = 0.5 * (time.min() + time.max())
    n_mid = round((t_mid - t0_nearby) / period_val)
    t0_mid = t0_nearby + n_mid * period_val
    logger.info(f"Recentred reference epoch: t0_nearby={t0_nearby:.10f} -> t0_mid={t0_mid:.10f} (cycle {n_mid})")

    output = f"{pathname}/{tar_name}_ngts_data_file"
    save_data_file(
        logger, tar_name, 1, time,
        np.full(len(time), cadence), flux, flux_err, t0_mid, output
    ).write_data_file()

    new_model = load_model_params(logger, config, tar_name, pathname)
    logger.info(f"Initialised t0 = 0 in t0_mid frame (t0_mid={t0_mid:.10f})")

    ngts_prelim = f"{pathname}/{tar_name}_ngts_model_file"
    lcurve(logger, new_model, output, ngts_prelim).simplex()

    names, p_best, steps = adjust_parameters(
        logger, ngts_prelim, ngts_prelim, [], [], [], {"period", "t0"}
    ).load_mcmc_params()
    logger.info(f"Best-fit parameters: { {n: v for n, v in zip(names, p_best)} }")

    with open(ngts_prelim, "r") as f:
        base_lines = f.readlines()

    mcmc_steps    = 20000
    ndim          = len(p_best)
    nwalkers      = 4 * ndim
    initial_spread = 0.01 * steps
    p0             = utils.sample_ball(p_best, initial_spread, size=nwalkers)

    with Pool(processes=nwalkers) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(names, time - t0_nearby, flux, flux_err, output, base_lines, logger),
            pool=pool
        )
        sampler.run_mcmc(p0, mcmc_steps, progress=True)

    chain = sampler.get_chain()
    nsteps, nw, nd = chain.shape
    for i in range(nd):
        fig, ax = plt.subplots(figsize=(10, 5))
        for j in range(nw):
            ax.plot(chain[:, j, i], alpha=0.5)
        ax.set_title(f"Walker trajectories for {names[i]}")
        ax.set_xlabel("Step")
        ax.set_ylabel(names[i])
        ax.set_xlim(0, mcmc_steps)
        plt.tight_layout()
        plt.savefig(f"{pathname}/fix_walker_{names[i]}.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {pathname}/fix_walker_{names[i]}.png")

    """discard = int(input("\nEnter number of burn-in steps to discard: "))"""
    discard = 5000

    samples = sampler.get_chain(discard=discard, thin=15, flat=True)
    logger.info(f"Final samples: {samples.shape[0]} (discard={discard}, thin=15)")

    tau = sampler.get_autocorr_time(tol=0)
    logger.info(f"Autocorrelation time: {tau}")

    param_med = np.median(samples, axis=0)
    param_std = np.std(samples, axis=0)
    for name, med, std in zip(names, param_med, param_std):
        logger.info(f"{name} = {med} ± {std}")

    cornerfig = corner.corner(
        samples, labels=names, truths=param_med,
        show_titles=True, title_fmt=".5f", label_kwargs={"fontsize": 16}
    )
    cornerfig.savefig(f"{pathname}/fix_corner_plot.png", dpi=150)
    plt.close(cornerfig)
    logger.info(f"Saved: {pathname}/fix_corner_plot.png")

    if "t0" in names:
        t0_idx = names.index("t0")
        t0_abs = param_med[t0_idx] + t0_mid
        t0_abs_err = param_std[t0_idx]
        logger.info(f"Absolute t0 = {t0_abs:.10f} ± {t0_abs_err:.10f} BMJD (TDB)")

    percent = np.percentile(samples, [16, 50, 84], axis=0)
    medians = percent[1]
    lowers  = medians - percent[0]
    uppers  = percent[2] - medians
    txt_file = f"{pathname}/fix_mcmc_vals.txt"
    with open(txt_file, "w") as fo:
        fo.write(f"{'Parameter':<30} {'Median':>12} {'−Error':>12} {'+Error':>12}\n")
        fo.write("-" * 68 + "\n")
        for label, med, lo, hi in zip(names, medians, lowers, uppers):
            if label == "t0":
                fo.write(f"{label:<30} {med + t0_mid:>12.10f} {lo:>12.10f} {hi:>12.10f}\n")
            else:
                fo.write(f"{label:<30} {med:>12.10f} {lo:>12.10f} {hi:>12.10f}\n")
    logger.info(f"Saved: {txt_file}")

    best_model = f"{pathname}/best_fit_ephemeris_model"
    adjust_parameters(
        logger, ngts_prelim, best_model,
        list(names), [f"{v:.15e}" for v in param_med], [2] * len(names), set(names)
    ).change_config()
    logger.info(f"Saved best-fit model: {best_model}")

    model_dat = f"{pathname}/ephemeris_fit_model"
    lcurve(logger, best_model, output, model_dat).lroche()

    time_model, flux_model, _ = np.loadtxt(model_dat, usecols=(0, 2, 3), unpack=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_model, flux_model, 'r-', lw=2, label='Best-fit model', zorder=5)
    ax.errorbar(time - t0_mid, flux, yerr=flux_err,
                fmt='k.', ms=2, alpha=0.5, label='NGTS data', zorder=3)
    ax.set_xlabel(f"Time from t0_mid = {t0_mid:.6f} (days)")
    ax.set_ylabel("Normalized Flux")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{pathname}/ephemeris_fit.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {pathname}/ephemeris_fit.png")


if __name__ == "__main__":
    args     = arg_parse()
    pathname = f"{args.data_root}/{args.target}"
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    logger = sf_logging("ephemeris", f"{pathname}/{args.target}.log").setup_logger()
    run({
        'data_root': args.data_root,
        'ngts_dat':  args.NGTS_dat,
        'name':      args.target,
        't0':        args.t0,
    }, logger)
