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
    assume beta_1 = 1.0, then y = y1+y2 for WD

For the gravity darkening of the companion, Claret et al. 2011 gives J/A+A/529/A75/tabley for single value: 
          
"""
import logging
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from astroquery.vizier import Vizier
from astropy.table import Table
from scipy.optimize import minimize
import astropy.io.ascii as ascii
import os

Vizier.VIZIER_SERVER = "vizier.cfa.harvard.edu"

def roche_distortion(r_VA_a: float, q: float) -> float:
    """
    Correct scaled volume-averaged radius to roche distorted radius towards L1
    given the mass ratio (M2/M1)
    """
    fname = './data/roche_conversion/roche_grid_new_corr.dat'
    q_grid, r2_a_L1, r2_va_a = np.loadtxt(fname, unpack=True)
    coords_in = list(zip(q_grid, r2_va_a))
    coords_out = list(r2_a_L1)
    roche_interpolator = LinearNDInterpolator(coords_in, coords_out, rescale=True)
    return roche_interpolator(q, r_VA_a)

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
        Vizier.ROW_LIMIT = 500000
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

        y1, y2 = (interp_param(p) for p in ['y1', 'y2'])
        logger.info(f"Interpolated to get the {coef} coefficients via the Claret 2-term law")
        return y1, y2

    @staticmethod
    def refit_4term(I_target: np.ndarray, mu: np.ndarray, logger=None) -> tuple[float, float, float, float]:
        B  = np.column_stack([1.0 - mu**(k/2) for k in (1, 2, 3, 4)])
        y  = 1.0 - I_target
        a0, *_ = np.linalg.lstsq(B, y, rcond=None)

        obj  = lambda a: float((B @ a - y) @ (B @ a - y))
        cons = {"type": "ineq", "fun": lambda a: 1.0 - B @ a}
        res  = minimize(obj, a0, constraints=cons, method="SLSQP")

        if not res.success and logger is not None:
            logger.warning(f"4-term LDC refit did not converge: {res.message}. "
                        f"Using best-effort coefficients.")

        return tuple(float(x) for x in res.x)

    def wd_limb_darkening(self) -> tuple[float, float, float, float]:
        filt_query = self.filt.replace("'", "")
        fname = f"./data/ld_coeffs/DA_LDCs_ucam_{filt_query}s.dat"
        tab_wd = Table.read(fname, format='ascii')
        wd_coords_in = list(zip(tab_wd['Teff'], tab_wd['log(g)']))
        wd_coords_out = list(zip(tab_wd['a1'], tab_wd['a2'],
                                tab_wd['a3'], tab_wd['a4']))
        ld_wd_interpolators = LinearNDInterpolator(wd_coords_in,
                                                        wd_coords_out,
                                                        rescale=True)
        a1, a2, a3, a4 = ld_wd_interpolators(self.wd_temp, self.wd_logg)
        return a1, a2, a3, a4

    def comp_limb_darkening(self) -> tuple[float, float, float, float]:
        filt_query = self.filt.replace("'", "")
        fname = f"./data/ld_coeffs/MS_LDCs_ucam_{filt_query}s.dat"
        tab_ms = Table.read(fname, format='ascii')
        ms_coords_in = list(zip(tab_ms['Teff'], tab_ms['log(g)']))
        ms_coords_out = list(zip(tab_ms['a1'], tab_ms['a2'],
                                tab_ms['a3'], tab_ms['a4']))
        ld_ms_interpolators = LinearNDInterpolator(ms_coords_in,
                                                        ms_coords_out,
                                                        rescale=True)
        a1, a2, a3, a4 = ld_ms_interpolators(self.comp_temp, self.comp_logg)
        return a1, a2, a3, a4

    def wd_gravity_darkening(self) -> float:
        data = self.query_vizier(self.logger, "J/A+A/634/A93/tabley")
        mask = (data["Filter"] == self.filt) & (data["Mod"] == self.wdtype)
        filt_data = data[mask]
        y1, y2 = self.itp(self.logger, filt_data, self.wd_logg, self.wd_temp, 'gdc')
        y = y1+y2
        return y

    def comp_gravity_darkening(self) -> float:
        filt_query = self.filt.replace("'", "")
        fname = f"./data/gravity_darkening_coeffs/MS_GDCs_{filt_query}s.dat"
        gdcs = ascii.read(fname)
        coords_in = list(zip(gdcs['Teff'], gdcs['log(g)']))
        coords_out = list(gdcs['y'])
        gdark_interpolator = LinearNDInterpolator(coords_in,
                                                coords_out,
                                                rescale=True)
        y = gdark_interpolator(self.comp_temp, self.comp_logg)
        return float(y)

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
            filt = "./data/filter_profiles/ucam_us.txt"
        elif self.band_index == 2:
            filt = "./data/filter_profiles/ucam_gs.txt"
        elif self.band_index == 3:
            filt = "./data/filter_profiles/ucam_is.txt"
        else:
            self.logger.error(f"Invalid band_index: {self.band_index}")
            raise ValueError(f"Invalid band_index: {self.band_index}")
        wave, trans = np.loadtxt(f"{filt}", usecols=(0,1), unpack=True)
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
            for line in fin:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == '=':
                    for param, val, idx in zip(self.param_name, self.value, self.val_pos_index):
                        if parts[0] == param:
                            parts[idx] = f"{val}"
                    line = " ".join(parts) + "\n"
                fout.write(line) 
        return self.new_model_file
    
    def load_mcmc_params(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        params = []
        names = []
        steps = []
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
        return names, np.array(params), np.array(steps)