import subprocess
import logging

NMAX = "5000"
DELTA = "0.01"
LMAX = "2e10"
SCALE = "yes"
NOISE = "0.0"
SEED = "12345"
NFILE = "1"
DEVICE = "/null"
FTOL = "1e-012"

class lcurve:
    def __init__(self, logger: logging.Logger, model: str, data: str, output: str) -> None:
        self.logger = logger
        self.MODEL = model
        self.DATA = data
        self.OUTPUT = output

    def simplex(self) -> None:
        try:
            subprocess.run(["simplex", self.MODEL, self.DATA, FTOL, NMAX, self.OUTPUT], check=True)
        except subprocess.CalledProcessError as e:
            self.logger.info(f"simplex failed: {e}")
            raise RuntimeError(f"simplex failed: {e}")

    def levmarq(self) -> None:
        try:
            subprocess.run(["levmarq", self.MODEL, self.DATA, NMAX, DELTA, LMAX, SCALE, self.OUTPUT], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            self.logger.info(f"levmarq failed with code {e.returncode}: {e.cmd}")
            raise RuntimeError(f"levmarq failed with code {e.returncode}: {e.cmd}")

    def lroche(self) -> None:
        try:
            subprocess.run(["lroche", self.MODEL, self.DATA, NOISE, SEED, NFILE, self.OUTPUT, DEVICE, SCALE],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            self.logger.info(f"lroche failed: {e}")
            raise RuntimeError(f"lroche failed: {e}")