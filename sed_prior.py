"""SED + parallax + extinction prior for the WD binary MCMC.

Physics:
  F_surf_nu  = pi * I_nu(1) * (1 - a1/5 - a2/3 - 3*a3/7 - a4/2)
  F_obs_nu   = F_surf_nu * (R1/d)^2        [erg/s/Hz/cm^2]
  F_obs [Jy] = F_obs_nu / 1e-23
  with extinction: * 10^(-0.4 * A_band * ebv)

Claret+2020 tablea4: I column is mW/(Hz sr m^2) = erg/s/Hz/sr/cm^2 (numerically equal).
"""
from __future__ import annotations
import os
import logging
import numpy as np
from astropy.io import fits
from astroquery.vizier import Vizier
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator
from astropy.table import Table

Vizier.VIZIER_SERVER = "vizier.cfa.harvard.edu"

_TRANS_DIR   = os.path.join(os.path.dirname(__file__), "data/filter_profiles")
_TRANS_FILES = {1: "ucam_us.txt", 2: "ucam_gs.txt", 3: "ucam_is.txt"}
_LDC_DIR     = os.path.join(os.path.dirname(__file__), "data/ld_coeffs")
_LDC_FILES   = {1: "DA_LDCs_ucam_us.dat", 2: "DA_LDCs_ucam_gs.dat", 3: "DA_LDCs_ucam_is.dat"}
_BAND_FILTER = {1: 'u', 2: 'g', 3: 'i'}
_PC_CM       = 3.085677581e18   # 1 parsec in cm


# ---------------------------------------------------------------------------
# Extinction law — Cardelli+89 optical/UV
# ---------------------------------------------------------------------------

def _cardelli_av(x: float, Rv: float = 3.1) -> float:
    """A_lambda / A_V via Cardelli+89. x = 1/lambda [1/um]."""
    if 1.1 <= x <= 3.3:          # optical / NIR
        y = x - 1.82
        a = (1.0 + 0.17699*y  - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4
             + 0.01979*y**5  - 0.77530*y**6 + 0.32999*y**7)
        b = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4
             - 0.62251*y**5  + 5.30260*y**6 - 2.09002*y**7)
    elif 3.3 < x <= 8.0:         # UV
        a = 1.752 - 0.316*x - 0.104 / ((x - 4.67)**2 + 0.341)
        b = -3.090 + 1.825*x + 1.206 / ((x - 4.62)**2 + 0.263)
        if 5.9 <= x <= 8.0:
            a += -0.04473*(x - 5.9)**2 - 0.009779*(x - 5.9)**3
            b +=  0.2130*(x - 5.9)**2  + 0.1207*(x - 5.9)**3
    else:
        return np.nan
    return a + b / Rv


def compute_a_over_ebv(band_idx: int, T_source: float = 10500.0,
                       Rv: float = 3.1) -> float:
    """Blackbody-weighted A/E(B-V) integrated over the SuperSDSS filter response."""
    path = os.path.join(_TRANS_DIR, _TRANS_FILES[band_idx])
    wave_A, trans = np.loadtxt(path, usecols=(0, 1), unpack=True)

    lam_cm = wave_A * 1e-8
    h, k, c = 6.626e-27, 1.381e-16, 2.998e10
    src = (2*h*c**2 / lam_cm**5) / (np.exp(h*c / (lam_cm*k*T_source)) - 1)

    x_arr     = 1e4 / wave_A     # 1/um
    A_over_AV = np.array([_cardelli_av(xi, Rv) for xi in x_arr])
    valid     = np.isfinite(A_over_AV)
    # weight by F_nu ~ B_lambda * lambda^2 (photometry is in Jy = per-Hz units)
    w         = trans * src * wave_A**2
    return (np.trapz(A_over_AV[valid] * Rv * w[valid], wave_A[valid])
            / np.trapz(w[valid], wave_A[valid]))


# ---------------------------------------------------------------------------
# Claret+2020 disk-centre intensity interpolator (tablea4)
# ---------------------------------------------------------------------------

class ClaretSEDInterp:
    """Pre-built per-band interpolators for I_nu(mu=1) and LDC from tablea4."""

    def __init__(self, logger: logging.Logger, wdtype: str = "DA",
                 band_indices: list[int] | None = None) -> None:
        self.logger  = logger
        self.wdtype  = wdtype
        self._interps: dict[int, dict[str, tuple]] = {}
        self._build(band_indices or [1, 2, 3])

    def _build(self, band_indices: list[int]) -> None:

        for idx in band_indices:
            fname = os.path.join(_LDC_DIR, _LDC_FILES[idx])
            tab_wd = Table.read(fname, format='ascii')

            teff_u = np.sort(np.unique(np.asarray(tab_wd['Teff'],    dtype=float)))
            logg_u = np.sort(np.unique(np.asarray(tab_wd['log(g)'], dtype=float)))
            t_map  = {v: i for i, v in enumerate(teff_u)}
            l_map  = {v: i for i, v in enumerate(logg_u)}

            # Build regular-grid arrays (NaN for any missing (Teff, logg) cell)
            cols_grid = {col: np.full((len(teff_u), len(logg_u)), np.nan)
                         for col in ("I", "a1", "a2", "a3", "a4")}
            pts_raw  = []
            vals_raw = {col: [] for col in cols_grid}

            for row in tab_wd:
                tf, lg = float(row['Teff']), float(row['log(g)'])
                ti, li = t_map[tf], l_map[lg]
                v = {"I":  float(row['I1']) * 2.0,
                     "a1": float(row['a1']), "a2": float(row['a2']),
                     "a3": float(row['a3']), "a4": float(row['a4'])}
                for col, grid in cols_grid.items():
                    grid[ti, li] = v[col]
                pts_raw.append((tf, lg))
                for col in cols_grid:
                    vals_raw[col].append(v[col])

            self._interps[idx] = {}
            for col, grid in cols_grid.items():
                # RegularGridInterpolator: fork-safe (no Qhull); NaN for missing cells
                lin  = RegularGridInterpolator(
                           (teff_u, logg_u), grid,
                           method='linear', bounds_error=False, fill_value=np.nan)
                # NearestNDInterpolator: fallback for out-of-bounds or NaN cells
                near = NearestNDInterpolator(pts_raw, vals_raw[col], rescale=True)
                self._interps[idx][col] = (lin, near)

            self.logger.info(f"ClaretSED: built interpolator for band {idx} ({_BAND_FILTER[idx]})")

    def get(self, band_idx: int, t1: float, logg1: float) -> tuple[float, ...]:
        """Return (I0, a1, a2, a3, a4); I0 in erg/s/Hz/sr/cm^2."""
        q = np.array([[t1, logg1]]) 
        out = []
        for col in ("I", "a1", "a2", "a3", "a4"):
            lin, near = self._interps[band_idx][col]
            val = lin(q)[0]
            if np.isnan(val):
                val = near(q)[0]
            out.append(float(val))
        return tuple(out)

def extract_wd_obs_flux(
    fits_path: str, t0: float, period: float,
    oot_excl_hw: float = 0.12, ie_hw: float = 0.025,
) -> dict[int, tuple[float, float, float]]:
    """
    Extract WD eclipse-depth flux (Jy) from the calibrated FITS file.

    Phase convention: eclipse at 0.5.
    OOT window  : |phi - 0.5| > oot_excl_hw  (generous exclusion)
    In-E window : |phi - 0.5| < ie_hw         (conservative totality window)

    Returns {band_idx: (F_WD_jy, F_WD_err_jy, oot_median_jy)}.
    """
    result: dict[int, tuple[float, float, float]] = {}
    with fits.open(fits_path) as hdul:
        for band_idx, hdu_idx in {1: 1, 2: 2, 3: 3}.items():
            hdu  = hdul[hdu_idx]
            t    = np.array(hdu.data["BMJD(TDB)"], dtype=float)
            flux = np.array(hdu.data["Flux"],       dtype=float)
            FC_E = float(hdu.header["FC_E"])

            phase = ((t - t0) % period) / period
            phase = np.where(phase > 0.5, phase - 1.0, phase) + 0.5

            oot = np.abs(phase - 0.5) > oot_excl_hw
            ie  = np.abs(phase - 0.5) < ie_hw

            if oot.sum() < 10 or ie.sum() < 3:
                raise ValueError(
                    f"Band {band_idx}: too few points OOT={oot.sum()} IE={ie.sum()}"
                )

            F_oot = np.median(flux[oot])
            F_ie  = np.median(flux[ie])
            F_wd  = F_oot - F_ie

            sig_oot  = np.std(flux[oot]) / np.sqrt(oot.sum())
            sig_ie   = np.std(flux[ie])  / np.sqrt(ie.sum())
            F_err    = np.hypot(np.hypot(sig_oot, sig_ie), FC_E * F_wd)
            result[band_idx] = (float(F_wd), float(F_err), float(F_oot))

    return result


# ---------------------------------------------------------------------------
# SED log-prior contribution
# ---------------------------------------------------------------------------

def _ld_corr(a1: float, a2: float, a3: float, a4: float) -> float:
    return 1.0 - a1/5.0 - a2/3.0 - 3.0*a3/7.0 - a4/2.0


def sed_log_prior_terms(
    logger: logging.Logger, p: dict, a_m: float, logg_obs: float, sed_data: dict
) -> float:
    """
    SED chi-squared contribution to log_prior.

    p        : parameter dict (needs 'parallax', 'ebv', 'r1', 't1')
    a_m      : semi-major axis in metres
    logg_obs : WD log g [cgs] computed from sampled m1, r1
    sed_data : dict — keys: claret_interp, F_WD_OBS, F_WD_OBS_ERR, A_OVER_EBV,
               band_order, gaia_plx, gaia_plx_err, ebv_map, ebv_err
    """
    plx = p["parallax"]
    ebv = p["ebv"]

    # box guards against divergence at extreme parallax / negative ebv
    plx_lo = max(0.1, sed_data["gaia_plx"] - 5.0*sed_data["gaia_plx_err"])
    plx_hi =          sed_data["gaia_plx"] + 5.0*sed_data["gaia_plx_err"]
    if not (plx_lo < plx < plx_hi):
        return -np.inf
    if not (0.0 <= ebv < sed_data["ebv_map"] + 5.0*sed_data["ebv_err"]):
        return -np.inf

    lp  = -0.5 * ((plx - sed_data["gaia_plx"]) / sed_data["gaia_plx_err"])**2
    lp += -0.5 * ((ebv - sed_data["ebv_map"])  / sed_data["ebv_err"])**2

    d_cm  = (1000.0 / plx) * _PC_CM
    R1_cm = p["r1"] * a_m * 100.0   # a_m in metres -> cm

    claret = sed_data["claret_interp"]
    ldc_per_band = sed_data.get("ldc_per_band", {})
    for band_idx in sed_data["band_order"]:
        I0, a1_c, a2_c, a3_c, a4_c = claret.get(band_idx, p["t1"], logg_obs)
        if not np.isfinite(I0):
            return -np.inf
        if band_idx in ldc_per_band:
            a1, a2, a3, a4 = ldc_per_band[band_idx]
        else:
            a1, a2, a3, a4 = a1_c, a2_c, a3_c, a4_c
        if not np.all(np.isfinite([a1, a2, a3, a4])):
            return -np.inf
        ldc = _ld_corr(a1, a2, a3, a4)
        if ldc <= 0.0:
            return -np.inf

        A_band     = sed_data["A_OVER_EBV"][band_idx] * ebv
        F_surf_nu  = np.pi * I0 * ldc                              # erg/s/Hz/cm^2
        F_pred_cgs = F_surf_nu * (R1_cm / d_cm)**2 * 10.0**(-0.4*A_band)
        F_pred_jy  = F_pred_cgs / 1e-23                            # 1 Jy = 1e-23 cgs

        F_obs = sed_data["F_WD_OBS"][band_idx]
        F_err = sed_data["F_WD_OBS_ERR"][band_idx]

        lp   += -0.5 * ((F_pred_jy - F_obs) / F_err)**2
    return lp
