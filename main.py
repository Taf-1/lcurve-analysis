import argparse
import configparser
import logging
import os
from logger import sf_logging
import desc
import ephemeris
import model

def arg_parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Binary star light curve modelling pipeline: DESC → EPHEMERIS → MODEL"
    )
    p.add_argument("config", help="Path to the pipeline configuration (.ini) file")
    p.add_argument(
        "--stages", nargs="+",
        choices=["desc", "ephemeris", "model"],
        default=["desc", "ephemeris", "model"],
        help="Pipeline stages to run (default: all three)",
    )
    return p.parse_args()

def load_config(config_path: str) -> dict[str, str]:
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    flat = {}
    for section in cfg.sections():
        for key, val in cfg.items(section):
            flat[key] = val
    return flat

if __name__ == "__main__":
    args = arg_parse()
    cfg  = load_config(args.config)

    data_root = cfg["data_root"]
    name      = cfg["name"]
    pathname  = f"{data_root}/{name}"

    if not os.path.exists(pathname):
        os.makedirs(pathname)

    logger = sf_logging("pipeline", f"{pathname}/{name}_pipeline.log").setup_logger()
    logger.info(f"Config: {args.config}")
    logger.info(f"Target: {name}  |  Data root: {data_root}")
    logger.info(f"Stages: {args.stages}")

    if "desc" in args.stages:
        logger.info("=" * 60)
        logger.info("Stage 1 — DESC: initial ULTRACAM light curve model")
        logger.info("=" * 60)
        desc.run(cfg, logger)
        logger.info("DESC stage complete.")

    if "ephemeris" in args.stages:
        logger.info("=" * 60)
        logger.info("Stage 2 — EPHEMERIS: period and t0 from NGTS data")
        logger.info("=" * 60)
        ephemeris.run(cfg, logger)
        logger.info("EPHEMERIS stage complete.")

    if "model" in args.stages:
        logger.info("=" * 60)
        logger.info("Stage 3 — MODEL: full MCMC light curve modelling")
        logger.info("=" * 60)
        model.run(cfg, logger)
        logger.info("MODEL stage complete.")

    logger.info("Pipeline finished.")
