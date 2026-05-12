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
from scipy.interpolate import RegularGridInterpolator
from astroquery.vizier import Vizier
from astropy.table import Table

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
        filt_data = filt_data[np.lexsort((filt_data['Teff'], filt_data['logg']))]
        logg_unique = np.unique(filt_data['logg'])
        teff_unique = np.unique(filt_data['Teff'])
        n_logg = len(logg_unique)
        n_teff = len(teff_unique)
        if coef == "ldc":
            interp = {}
            for param in ['a1', 'a2', 'a3', 'a4']:
                values_grid = filt_data[param].reshape(n_logg, n_teff)
                interp[param] = RegularGridInterpolator((logg_unique, teff_unique), values_grid)
            a1 = interp['a1']([[logg, temp]])[0]
            a2 = interp['a2']([[logg, temp]])[0]
            a3 = interp['a3']([[logg, temp]])[0]
            a4 = interp['a4']([[logg, temp]])[0]
            logger.info(f"Interpolated to get the {coef} coefficients via the Claret 4-term law")
            return a1, a2, a3, a4
        else:
            interp = {}
            for param in ["y1", "y2"]:
                values_grid = filt_data[param].reshape(n_logg, n_teff)
                interp[param] = RegularGridInterpolator((logg_unique, teff_unique), values_grid)
            y1 = interp['y1']([[logg, temp]])[0]
            y2 = interp['y2']([[logg, temp]])[0]
            logger.info(f"Interpolated to get the {coef} coefficients via the Claret 2-term law")
            return y1, y2

    def wd_limb_darkening(self) -> tuple[float, float, float, float]:
        data = self.query_vizier(self.logger, "J/A+A/634/A93/tablea4")
        mask = (data["Filter"] == self.filt) & (data["Mod"] == self.wdtype)
        filt_data = data[mask]
        a1, a2, a3, a4 = self.itp(self.logger, filt_data, self.wd_logg, self.wd_temp, 'ldc')
        return a1, a2, a3, a4

    def comp_limb_darkening(self) -> tuple[float, float, float, float]:
        data = self.query_vizier(self.logger, "J/A+A/546/A14/limb6")
        mask = (data["Filt"] == self.filt) & (data["Mod"] == 'qs')
        filt_data = data[mask]
        a1, a2, a3, a4 = self.itp(self.logger, filt_data, self.comp_logg, self.comp_temp, 'ldc')
        return a1, a2, a3, a4

    def gravity_darkening(self) -> tuple[float, float]:
        data = self.query_vizier(self.logger, "J/A+A/634/A93/tabley")
        mask = (data["Filter"] == self.filt) & (data["Mod"] == self.wdtype)
        filt_data = data[mask]
        y1, y2 = self.itp(self.logger, filt_data, self.wd_logg, self.wd_temp, 'gdc')
        return y1, y2

class effective_wavelength:
    def __init__(self, logger: logging.Logger, band_index: int) -> None:
        self.logger = logger
        self.band_index = band_index

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