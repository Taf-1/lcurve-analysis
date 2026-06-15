"""
For substellar companion, Claret et al. 2012 gives J/A+A/546/A14/limb6 for limb darkening coeffients via 4-term law:
    a1, a2, a3, a4 for filters - 
        Kp = Kepler
        C = CoRoT
        S1 = Spitzer filter 1, 3.6um
        S2 = Spitzer filter 2, 4.5um
        S3 = Spitzer filter 3, 5.8um
        S4 = Spitzer filter 4, 8.0um
        uvby = Stroemgren uvby filters
        UBVRIJHK = Johnson-Cousins UBVRIJHK filters
        u'g'r'i'z' = SDSS u'g'r'i'z' filters

For White Dwarf, Claret et al. 2020, gives J/A+A/634/A93/tablea4 for the limb darkening for compact stars DA, DB, DBA eclipsing white dwarfs via 4-term law:
    a1, a2, a3, a4 for filters - 
            Kp = Kepler
            C = CoRoT
            S1 = Spitzer filter 1, 3.6um
            S2 = Spitzer filter 2, 4.5um
            S3 = Spitzer filter 3, 5.8um
            S4 = Spitzer filter 4, 8.0um
            uvby = Stroemgren uvby filters
            UBVRIJHK = Johnson-Cousins UBVRIJHK filters
            u'g'r'i'z' = SDSS u'g'r'i'z' filters

For the gravity darkening 1 & 2, Claret et al. 2020 gives J/A+A/634/A93/tabley for the two-term gravity darkening coefficients:
    y1, y2 for filters - 
        Kp = Kepler
        C = CoRoT
        S1 = Spitzer filter 1, 3.6um
        S2 = Spitzer filter 2, 4.5um
        S3 = Spitzer filter 3, 5.8um
        S4 = Spitzer filter 4, 8.0um
        uvby = Stroemgren uvby filters
        UBVRIJHK = Johnson-Cousins UBVRIJHK filters
        u'g'r'i'z' = SDSS u'g'r'i'z' filters        
"""
import logging
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from astroquery.vizier import Vizier
from astropy.table import Table
from scipy.optimize import minimize

class claret_tables_interp:
    def __init__(self, logger: logging.Logger, wd_temp: float, wd_logg: float, wdtype: str, comp_temp: float, comp_logg: float, filt: str) -> None:
        self.logger = logger
        self.wd_temp = wd_temp
        self.wd_logg = wd_logg
        self.wdtype = wdtype
        self.comp_temp = comp_temp 
        self.comp_logg = comp_logg
        self.filt = filt

    @staticmethod
    def query_vizier(logger: logging.Logger, vizier_catalogue: str) -> Table:
        Vizier.ROW_LIMIT = 30000
        cat = Vizier.get_catalogs(vizier_catalogue)
        logger.info(f"Queried the {vizier_catalogue} catalogue ...")
        return cat[0]

    @staticmethod
    def itp(logger: logging.Logger, filt_data: Table, logg: float, temp: float, coef: str) -> tuple[float, ...]:
        points = np.column_stack([np.array(filt_data['logg']), np.array(filt_data['Teff'])])
        query  = np.array([[logg, temp]])

        def interp_param(col: str) -> float:
            values = np.array(filt_data[col], dtype=float)
            val = LinearNDInterpolator(points, values)(query)[0]
            if np.isnan(val):
                val = NearestNDInterpolator(points, values)(query)[0]
                logger.info(f"{col}: outside convex hull, using nearest neighbour")
            return float(val)

        if coef == "ldc":
            a1, a2, a3, a4 = (interp_param(p) for p in ['a1', 'a2', 'a3', 'a4'])
            logger.info(f"Interpolated to get the {coef} coefficients via the Claret 4-term law")
            return a1, a2, a3, a4
        else:
            y1, y2 = (interp_param(p) for p in ['y1', 'y2'])
            logger.info(f"Interpolated to get the {coef} coefficients via the Claret 2-term law")
            return y1, y2

    @staticmethod
    def refit_4term(I_target: np.ndarray, mu: np.ndarray) -> tuple[float, float, float, float]:
        # I(mu) = 1 - B @ a, linear in a
        B  = np.column_stack([1.0 - mu**(k/2) for k in (1, 2, 3, 4)])
        y  = 1.0 - I_target
        a0, *_ = np.linalg.lstsq(B, y, rcond=None)            # unconstrained start

        # min ||B a - y||^2  s.t.  I(mu) = 1 - B a >= 0  on the grid
        obj  = lambda a: float((B @ a - y) @ (B @ a - y))
        cons = {"type": "ineq", "fun": lambda a: 1.0 - B @ a}
        res  = minimize(obj, a0, constraints=cons, method="SLSQP")
        return tuple(float(x) for x in res.x)

    def wd_limb_darkening(self) -> tuple[float, float, float, float]:
        data = self.query_vizier(self.logger, "J/A+A/634/A93/tablea4")
        mask = (data["Filter"] == self.filt) & (data["Mod"] == self.wdtype)
        filt_data = data[mask]
        a1, a2, a3, a4 = self.itp(self.logger, filt_data, self.wd_logg, self.wd_temp, 'ldc')
        return a1, a2, a3, a4

    def comp_limb_darkening(self) -> tuple[float, float, float, float]:
        data = self.query_vizier(self.logger, "J/A+A/546/A14/limb6")
        mask = (data["Filt"] == self.filt) & (data["Mod"] == 's')
        filt_data = data[mask]
        a1, a2, a3, a4 = self.itp(self.logger, filt_data, self.comp_logg, self.comp_temp, 'ldc')

        mu = np.linspace(0.0, 1.0, 101)
        a  = (a1, a2, a3, a4)
        I  = 1.0 - sum(ak * (1.0 - mu**(k/2)) for k, ak in enumerate(a, start=1))
        if I.min() < 0.0:
            self.logger.info(f"Companion I(mu) dips to {I.min():.4f}; clamping at 0 and refitting")
            I = np.clip(I, 0.0, None)
            a1, a2, a3, a4 = self.refit_4term(I, mu)

        return a1, a2, a3, a4

    def gravity_darkening(self) -> tuple[float, float]:
        data = self.query_vizier(self.logger, "J/A+A/634/A93/tabley")
        mask = (data["Filter"] == self.filt) & (data["Mod"] == self.wdtype)
        filt_data = data[mask]
        y1, y2 = self.itp(self.logger, filt_data, self.wd_logg, self.wd_temp, 'gdc')
        """y2 = 0.08"""
        return y1, y2

class effective_wavelength:
    def __init__(self, logger: logging.Logger, band_index: int) -> None:
        self.logger = logger
        self.band_index = band_index
    
    @staticmethod
    def load_speedyfit_spectrum(path):
        wave, flux = np.loadtxt(path, skiprows=1, usecols=(0, 1), unpack=True)
        return wave, flux

    @staticmethod
    def blackbody_spectrum(temp: float, wmin: float = 2500.0, wmax: float = 12000.0, n: int = 8000) -> tuple[np.ndarray, np.ndarray]:
        """F_lambda blackbody on a fine uniform grid (Angstrom). Normalisation is arbitrary
        since beam_factor is a ratio."""
        h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
        wave = np.linspace(wmin, wmax, n)          # Angstrom
        lam  = wave * 1e-10                          # m
        flux = (2*h*c**2 / lam**5) / (np.exp(h*c/(lam*k*temp)) - 1.0)   # F_lambda
        return wave, flux

    def transmission(self) -> tuple[np.ndarray, np.ndarray]:
        if self.band_index == 1:
            filt = "SLOAN_SDSS.u.dat"
        elif self.band_index == 2:
            filt = "SLOAN_SDSS.g.dat"
        elif self.band_index == 3:
            filt = "SLOAN_SDSS.i.dat"
        else:
            self.logger.error(f"Invalid band_index: {self.band_index}")
            raise ValueError(f"Invalid band_index: {self.band_index}")
        wave, trans = np.loadtxt(f"./transmission/{filt}", usecols=(0,1), unpack=True)
        self.logger.info(f"Loaded the wavelength and transmission for filter {filt}")
        return wave, trans

    def beam_factor(self, wave_spec, flux_spec) -> float:
        """
        Photon-weighted <3 - alpha> over this band.
        wave_spec, flux_spec : model spectrum F_lambda (Angstrom, per-Angstrom).
        """
        wave_filt, trans_filt = self.transmission()
        T = np.interp(wave_spec, wave_filt, trans_filt, left=0.0, right=0.0)

        c_Apers = 2.99792458e18
        nu      = c_Apers / wave_spec
        F_nu    = flux_spec * wave_spec**2 / c_Apers

        order = np.argsort(nu)
        alpha = np.empty_like(wave_spec)
        alpha[order] = np.gradient(np.log(F_nu[order]), np.log(nu[order]))
        B = 3.0 - alpha

        w = T * wave_spec * flux_spec 
        inband = w > 0
        B_mean = np.trapz((w * B)[inband], wave_spec[inband]) / np.trapz(w[inband], wave_spec[inband])
        self.logger.info(f"Band {self.band_index}: beam factor <3-alpha> = {B_mean:.4f}")
        return B_mean

    def pivot_wave(self) -> float:
        wave, T = self.transmission()
        lambda_pivot = np.sqrt(np.trapz(wave * T, wave) / np.trapz(T / wave, wave))
        self.logger.info(f"Calculated a pivot wavelength of {lambda_pivot / 10}")
        return lambda_pivot / 10

class adjust_parameters:
    def __init__(self, logger: logging.Logger, model_file: str, new_model_file: str, param_name: list[str], value: list[str], val_pos_index: list[int], allowed: set[str]) -> None:
        self.logger = logger
        self.model_file = model_file 
        self.new_model_file = new_model_file
        self.param_name = param_name
        self.value = value
        self.val_pos_index = val_pos_index 
        self.allowed = allowed

    def change_config(self) -> str:
        with open(self.model_file, "r") as fin, open(self.new_model_file, "w") as fout:
            self.logger.info(f"Opened {self.model_file} for modifications")
            for line in fin:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == '=':
                    for param, val, idx in zip(self.param_name, self.value, self.val_pos_index):
                        if parts[0] == param:
                            parts[idx] = f"{val}"
                    line = " ".join(parts) + "\n"
                fout.write(line) 

        self.logger.info(f"Modifications complete and saved to - {self.new_model_file}")
        return self.new_model_file
    
    def load_mcmc_params(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        params = []
        names = []
        steps = []
        self.logger.info(f"Opening {self.new_model_file} to get pbest & steps for MCMC analysis")
        with open(self.new_model_file, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and parts[1] == "=":
                    name = parts[0]
                    param_value = parts[2]
                    step = parts[4]
                    fit_flag = parts[5]
                    if fit_flag == "1" and name in self.allowed:
                        names.append(name)
                        params.append(float(param_value))
                        steps.append(float(step))
        self.logger.info(f"Saved names, pbest and steps - ready for MCMC analysis")
        return names, np.array(params), np.array(steps)