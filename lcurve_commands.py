import os
import subprocess
import logging

_ENV = {**os.environ, "PGPLOT_DEV": "/null"}

NMAX = "200000"
DELTA = "0.01"
LMAX = "2e10"
SCALE = "yes"
NOISE = "0.0"
SEED = "57565"
NFILE = "1"
DEVICE = "/null"
FTOL = "1e-09"
NMAX_simplex = "100000"
FTOL_simplex = "1e-05"
ALINE = '1'
TAU = '0.01'
EFAC = '0.6'
NPHASE = '400'
PHASE1 = '0.0'
PHASE2 = '1.0'
ALIGN = 'no'

class lcurve:
    def __init__(self, logger: logging.Logger, model: str, data: str, output: str) -> None:
        self.logger = logger
        self.MODEL = model
        self.DATA = data
        self.OUTPUT = output

    def simplex(self) -> None:
        try:
            subprocess.run(["simplex", self.MODEL, self.DATA, FTOL_simplex, NMAX_simplex, self.OUTPUT], check=True, env=_ENV)
        except subprocess.CalledProcessError as e:
            self.logger.info(f"simplex failed: {e}")
            raise RuntimeError(f"simplex failed: {e}")

    def levmarq(self) -> None:
        try:
            subprocess.run(["levmarq", self.MODEL, self.DATA, NMAX, DELTA, LMAX, SCALE, self.OUTPUT], check=True, env=_ENV)
        except subprocess.CalledProcessError as e:
            self.logger.info(f"levmarq failed with code {e.returncode}: {e.cmd}")
            raise RuntimeError(f"levmarq failed with code {e.returncode}: {e.cmd}")

    def lroche(self) -> None:
        try:
            subprocess.run(["lroche", self.MODEL, self.DATA, NOISE, SEED, NFILE, self.OUTPUT, DEVICE, SCALE], check=True)
        except subprocess.CalledProcessError as e:
            self.logger.info(f"lroche failed: {e}")
            raise RuntimeError(f"lroche failed: {e}")