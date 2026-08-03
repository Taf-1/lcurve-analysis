import logging
import numpy as np
from astropy.time import Time
from astropy.io import fits

class create_data_file:
    def __init__(self, logger: logging.Logger, pathname: str, filename: str, band_index: int, t0: str | float, period: float) -> None:
        self.logger = logger
        self.pathname = pathname 
        self.filename = filename
        self.band_index = band_index
        self.t0 = t0
        self.period = period

    def extract_ultracam_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        with fits.open(str(self.filename)) as f:
            self.logger.info(f"FITS file has {len(f)} extensions")
            if self.band_index >= len(f):
                self.logger.info(f"Band index {self.band_index} exceeds available extensions")
                raise ValueError(f"Band index {self.band_index} exceeds available extensions")
            data = f[self.band_index].data
        time = np.array(data['BMJD(TDB)'], dtype=float)
        exp_time = np.array(data['Exp_time'], dtype=float)
        flux = np.array(data['Flux'], dtype=float)
        flux_err = np.array(data['Flux_err'], dtype=float)
        oot_mask = (flux > 0.85 * np.median(flux))
        norm = np.mean(flux[oot_mask])   # OOT mean in Jy — returned for mJy plotting
        flux /= norm
        flux_err /= norm
        t0_val = float(self.t0)
        t0_tdb = Time(t0_val, format='jd', scale='tdb').mjd if t0_val > 2400000 else t0_val
        period_val = float(self.period)
        t_mid = (time.min() + time.max()) / 2
        n_cycles = round((t_mid - t0_tdb) / period_val)
        t0_nearby = t0_tdb + n_cycles * period_val
        return time, exp_time, flux, flux_err, t0_nearby, norm

    def extract_ngts_data(self) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, int]:
        self.logger.info(f"Loading NGTS data from {self.filename}")
        with fits.open(str(self.filename)) as f:
            data = f[1].data

        t = Time(data['bjd_mid'], format='jd', scale='tdb')
        time = t.mjd

        cadence = 13 / 86400

        f = data['flux_3']
        f_err = data['fluxerr_3']

        oot_mask = f > 0.85 * np.nanmedian(f)
        oot_norm = np.nanmean(f[oot_mask])

        flux = f / oot_norm
        flux_err = f_err / oot_norm

        self.logger.info(f"Loaded {len(time)} data points")
        self.logger.info(f"Time range: {time.min():.6f} to {time.max():.6f} BMJD (TDB)")

        t0_val = float(self.t0)
        t0_tdb = Time(t0_val, format='jd', scale='tdb').mjd if t0_val > 2400000 else t0_val
        period_val = float(self.period)

        cycles = np.round((time - t0_tdb) / period_val).astype(int)
        unique_cycles = np.unique(cycles)
        cycle_ref = unique_cycles[len(unique_cycles) // 2]

        t0_nearby = t0_tdb + cycle_ref * period_val

        self.logger.info(f"Reference cycle = {cycle_ref}")
        self.logger.info(f"Reference eclipse = {t0_nearby:.10f}")

        return time, cadence, flux, flux_err, t0_nearby, abs(cycle_ref)

class adjust_data_file:
    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_errors: np.ndarray, binfact: int) -> None:
        self.time = time
        self.flux = flux
        self.flux_errors = flux_errors
        self.binfact = binfact

    def bin_data_on_time(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_binned = int(len(self.time) / self.binfact)
        binned_len = int(n_binned * self.binfact)
        temp = zip(self.time[:binned_len], self.flux[:binned_len], self.flux_errors[:binned_len])
        temp = sorted(temp)
        time_s, flux_s, error_s = map(np.array, zip(*temp))
        time_bin = np.average(time_s.reshape(n_binned, self.binfact), axis=1)
        flux_bin = np.average(flux_s.reshape(n_binned, self.binfact), axis=1)
        error_bin = np.std(flux_s.reshape(n_binned, self.binfact), axis=1)
        return time_bin, flux_bin, error_bin

class save_data_file:
    def __init__(self, logger: logging.Logger, tar_name: str, band_index: int, time: np.ndarray, exp_time: np.ndarray, flux_norm: np.ndarray, flux_err_norm: np.ndarray, t0_nearby: float, output: str) -> None:
        self.logger = logger 
        self.tar_name = tar_name
        self.band_index = band_index
        self.time = time 
        self.exp_time = exp_time
        self.flux_norm = flux_norm
        self.flux_err_norm = flux_err_norm
        self.t0_nearby = t0_nearby
        self.output = output

    def write_data_file(self) -> str:
        self.logger.info(f"Writing the data file - {self.output}")
        with open(self.output, "w") as f:
            f.write(f"# Band {self.band_index} data\n")
            f.write(f"# Times relative to t0_nearby = {self.t0_nearby:.10f}\n") 
            for i in range(len(self.time)):
                rel_time = self.time[i] - self.t0_nearby
                f.write(
                    f"{rel_time:.10f} "
                    f"{self.exp_time[i]:.6e} "
                    f"{self.flux_norm[i]:.6f} "
                    f"{self.flux_err_norm[i]:.6f} "
                    f"1 1\n"
                )
        self.logger.info(f"Saved!")
        return self.output