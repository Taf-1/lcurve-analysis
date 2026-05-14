import logging
import numpy as np
import os
import sys
import uuid
from lcurve_commands import lcurve
from lcurve_rv_calc import xshooter_params
from lcurve_model_file import adjust_parameters
from scipy.optimize import curve_fit

def parabola(x: np.ndarray, a: float, x0: float, c: float) -> np.ndarray:
        return a * (x - x0)**2 + c

class Jacobian:
    def __init__(self, logger: logging.Logger, gaia_id: str, orig_data: str, base: str, p_best: np.ndarray, steps: np.ndarray, names: list[str], time: np.ndarray, flux_err: np.ndarray) -> None:
        self.logger = logger
        self.gaia_id = gaia_id
        self.orig_data = orig_data
        self.base = base
        self.p_best = p_best
        self.steps = steps
        self.names = names
        self.time = time
        self.flux_err = flux_err

    def compute_jacobian(self) -> np.ndarray:
        self.logger.info("Compute the Jacobian")
        N = len(self.p_best)
        M = len(self.time)
        J = np.zeros((M, N))
        for j in range(N):
            dp = self.steps[j]
            p_plus = self.p_best.copy()
            p_minus = self.p_best.copy()
            p_plus[j] += dp
            p_minus[j] -= dp
            p_plus_model = "p_plus"
            p_minus_model = "p_minus"
            adjust_parameters(
                self.logger, self.base, p_plus_model,
                self.names, p_plus, [2]*N, set(self.names)
            ).change_config()
            adjust_parameters(
                self.logger, self.base, p_minus_model,
                self.names, p_minus, [2]*N, set(self.names)
            ).change_config()
            plus_synth = "plus_synth"
            minus_synth = "minus_synth"
            try:
                lcurve(self.logger, p_plus_model, self.orig_data, plus_synth).lroche()
                lcurve(self.logger, p_minus_model, self.orig_data, minus_synth).lroche()
            except Exception as e:
                self.logger.info(f"Failed to run lroche - {e}")
                sys.exit(0)
            f_plus = np.loadtxt(plus_synth, usecols=(2), unpack=True)
            f_minus = np.loadtxt(minus_synth, usecols=(2), unpack=True)
            if len(f_plus) != len(f_minus):
                self.logger.info("Mismatch in lengths")
                sys.exit(1)
            for f in [plus_synth, minus_synth, p_plus_model, p_minus_model]:
                if os.path.exists(f):
                    os.remove(f)
            J[:,j] = (f_plus - f_minus) / (2 * dp)
        self.logger.info(f"Jacobian - {J}")
        return J

    def compute_covariance(self) -> tuple[np.ndarray, np.ndarray]:
        self.logger.info("Computing the covariance using the Jacobian")
        J = self.compute_jacobian()
        W = np.diag(1.0/self.flux_err**2)
        JTJ = J.T @ W @ J
        C = np.linalg.pinv(JTJ)
        self.logger.info(f"Covariance - {C} \n And its error - {np.sqrt(np.diag(C))}")
        return C, np.sqrt(np.diag(C))

class q_i_degeneracy:
    def __init__(self, logger: logging.Logger, gaia_id: str, results: dict, band_index: int, config: str, rv_config: str) -> None:
        self.logger = logger
        self.gaia_id = gaia_id
        self.results = results
        self.band_index = band_index
        self.config = config
        self.rv_config = rv_config

    def chi_squared_method(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, float]:
        self.logger.info("Looking at the q-i degeneracy present in LCURVE")
        p_best = self.results[2]['p_best']
        names = self.results[2]['names']
        iq = names.index("q")
        ii = names.index("iangle")
        q_best = p_best[iq]
        i_best = p_best[ii]
        q_vals = np.linspace(0.12, 0.2, 100)
        i_vals = np.linspace(80, 89.9, 80)
        chi2 = np.zeros((len(q_vals), len(i_vals)))
        for iq_idx, q in enumerate(q_vals):
            for ii_idx, inc in enumerate(i_vals):
                id = uuid.uuid4().hex
                orig_data = f"{self.gaia_id}_phase_data_file_{self.band_index}"
                new_con = f"chi2_{id}_{self.band_index}"
                adjust_parameters(
                    self.logger, self.config, new_con,
                    ["q", "iangle"], [q, inc], [2, 2], {"q", "iangle"}
                ).change_config()
                chi_data = f"chi2_{id}_data_{self.band_index}"
                try:
                    lcurve(self.logger, new_con, orig_data, chi_data).lroche()
                except Exception as e:
                    self.logger.info(f"Failed to run lroche - {e}")
                    sys.exit(0)
                chi_time, chi_flux, chi_flux_err = np.loadtxt(chi_data, usecols=(0,2,3), unpack=True)
                time, flux, flux_err = np.loadtxt(orig_data, usecols=(0,2,3), unpack=True)
                try:
                    os.remove(chi_data); os.remove(new_con)
                except Exception:
                    pass
                chi2[iq_idx, ii_idx] = np.sum(((flux - chi_flux) / flux_err)**2)
        chi2_min = np.min(chi2)
        delta_chi2 = chi2 - chi2_min
        chi2_marginal_q = np.min(chi2, axis=1)
        rv_params = xshooter_params(logger=self.logger, rv_config=self.rv_config, phase=[])
        rv_q_mean, rv_q_sigma, _, _ = rv_params.q_n_velocityscale()
        delta_q_raw = chi2_marginal_q - chi2_marginal_q.min()
        p0 = [1000, q_vals[np.argmin(delta_q_raw)], 0]
        popt, _ = curve_fit(parabola, q_vals, delta_q_raw, p0=p0)
        delta_q = parabola(q_vals, *popt)
        delta_q -= delta_q.min()
        return q_vals, i_vals, delta_q, delta_chi2, rv_q_mean, rv_q_sigma, p_best, i_best
