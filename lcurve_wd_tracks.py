from __future__ import annotations
import logging
import numpy as np
import glob
import os
import re
from scipy.interpolate import RegularGridInterpolator, interp1d

# --- constants (cgs) ---
G     = 6.67430e-8
SIGMA = 5.670374419e-5
MSUN  = 1.98892e33
LSUN  = 3.828e33
RSUN  = 6.957e10

MASS_THRESHOLD = 0.45  # Msun; <= -> He (Panei), > -> C/O (Bedard)

_MODULE_DIR       = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PANEI_DIR  = os.path.join(_MODULE_DIR, "PaneiTracks2007")
_DEFAULT_BEDARD_DIR = os.path.join(_MODULE_DIR, "BedardTracks")


def _R_from_logg(logg: np.ndarray, mass_msun: float) -> np.ndarray:
    g = 10.0 ** logg
    return np.sqrt(G * mass_msun * MSUN / g) / RSUN


def _R_from_logL(logL: np.ndarray, teff: float) -> np.ndarray:
    L = (10.0 ** logL) * LSUN
    return np.sqrt(L / (4 * np.pi * SIGMA * teff**4)) / RSUN


def _interp_mass(masses: np.ndarray, R_primary: np.ndarray, R_check: np.ndarray,
                 mass_t: float, model_label: str,
                 grid_step: float | None = None, gap_tol: float = 1.5) -> tuple[float, float, bool, float, float, int]:

    ok = np.isfinite(R_primary)
    if ok.sum() < 2:
        raise ValueError(
            f"Teff covered by <2 sequences in {model_label}; "
            "check Teff is inside the grid.")
    m  = masses[ok]; Rp = R_primary[ok]; Rc = R_check[ok]
    o  = np.argsort(m)
    m, Rp, Rc = m[o], Rp[o], Rc[o]

    if grid_step is None:
        diffs = np.diff(m)
        grid_step = np.median(diffs) if len(diffs) else 0.05

    logm  = np.log10(m)
    logmt = np.log10(mass_t)

    extrap = mass_t > m.max() or mass_t < m.min()

    if not extrap:
        j = np.searchsorted(m, mass_t)
        mlo, mhi = m[j-1], m[j]
        if (mhi - mlo) > gap_tol * grid_step:
            need_lo = mlo + grid_step
            need_hi = mhi - grid_step
            raise ValueError(
                f"{model_label}: requested mass {mass_t:.4f} Msun falls in a "
                f"GAP between available sequences {mlo:.4f} and {mhi:.4f} "
                f"Msun (grid step ~{grid_step:.3f}). Interpolating across "
                f"this gap would be unreliable. Upload the missing "
                f"sequence(s) roughly {need_lo:.2f}-{need_hi:.2f} Msun.")
        def _ll(R: np.ndarray) -> float:
            lr = np.log10([R[j-1], R[j]])
            return 10.0 ** np.interp(logmt, [logm[j-1], logm[j]], lr)
    else:
        def _ll(R: np.ndarray) -> float:
            lr = np.log10(R)
            if mass_t > m.max():
                x0, x1, y0, y1 = logm[-2], logm[-1], lr[-2], lr[-1]
                return 10.0 ** (y1 + (y1 - y0)/(x1 - x0) * (logmt - x1))
            else:
                x0, x1, y0, y1 = logm[0], logm[1], lr[0], lr[1]
                return 10.0 ** (y0 + (y1 - y0)/(x1 - x0) * (logmt - x0))

    return _ll(Rp), _ll(Rc), extrap, m.min(), m.max(), int(ok.sum())


class panei_tracks:
    def __init__(self, logger: logging.Logger, data_dir: str, teff: float) -> None:
        self.logger = logger
        self.data_dir = data_dir
        self.teff = teff

    def _mass_from_name(self, path: str) -> float:
        m = re.search(r"m0\.(\d+)He", os.path.basename(path))
        return float("0." + m.group(1))

    def _load(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = np.loadtxt(path, comments="#")
        teff, logg, logL = d[:, 0], d[:, 1], d[:, 2]
        o = np.argsort(teff)
        return teff[o], logg[o], logL[o]

    def _seq_radius(self, path: str) -> tuple[float, float, float]:
        mass = self._mass_from_name(path)
        teff, logg, logL = self._load(path)
        if self.teff < teff.min() or self.teff > teff.max():
            return mass, np.nan, np.nan
        lg = np.interp(self.teff, teff, logg)
        lL = np.interp(self.teff, teff, logL)
        return mass, _R_from_logg(lg, mass), _R_from_logL(lL, self.teff)

    def get_radii(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        files = sorted(glob.glob(os.path.join(self.data_dir, "m0.*He.SDSS")))
        if not files:
            raise FileNotFoundError(f"No Panei m0.*He.SDSS files in {self.data_dir}")
        self.logger.info(f"Loading {len(files)} Panei He-core sequences")
        out = [self._seq_radius(f) for f in files]
        masses = np.array([o[0] for o in out])
        Rg     = np.array([o[1] for o in out])
        RL     = np.array([o[2] for o in out])
        return masses, Rg, RL, "Panei 2007 He-core"


class bedard_tracks:
    def __init__(self, logger: logging.Logger, data_dir: str, teff: float, layer: str="thick") -> None:
        self.logger = logger
        self.data_dir = data_dir
        self.teff = teff
        self.layer = layer

    def _mass_from_name(self, path: str) -> float:
        m = re.search(r"seq_(\d+)_", os.path.basename(path))
        return float(m.group(1)) / 100.0

    def _load(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        teff, logg, R = [], [], []
        with open(path) as fh:
            lines = [ln for ln in fh if not ln.strip().startswith("#")
                     and "===" not in ln and ln.strip()]
        i = 0
        while i < len(lines):
            parts = lines[i].split()
            if len(parts) >= 6 and parts[0].lstrip("-").isdigit():
                try:
                    teff.append(float(parts[1]))
                    logg.append(float(parts[2]))
                    R.append(float(parts[3]) / RSUN)
                except ValueError:
                    pass
                i += 3
            else:
                i += 1
        teff = np.array(teff); logg = np.array(logg); R = np.array(R)
        o = np.argsort(teff)
        return teff[o], logg[o], R[o]

    def _seq_radius(self, path: str) -> tuple[float, float, float]:
        mass = self._mass_from_name(path)
        teff, logg, R = self._load(path)
        if self.teff < teff.min() or self.teff > teff.max():
            return mass, np.nan, np.nan
        R_direct = np.interp(self.teff, teff, R)
        lg = np.interp(self.teff, teff, logg)
        R_logg = _R_from_logg(lg, mass)
        return mass, R_direct, R_logg

    def get_radii(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        patt = os.path.join(self.data_dir, f"seq_*_{self.layer}.txt")
        files = sorted(glob.glob(patt))
        if not files:
            raise FileNotFoundError(
                f"No Bedard seq_*_{self.layer}.txt files in {self.data_dir}")
        self.logger.info(f"Loading {len(files)} Bedard C/O-core ({self.layer}) sequences")
        out = [self._seq_radius(f) for f in files]
        masses = np.array([o[0] for o in out])
        Rd     = np.array([o[1] for o in out])
        Rg     = np.array([o[2] for o in out])
        label  = f"Bedard 2020 C/O-core ({self.layer})"
        return masses, Rd, Rg, label


class wd_radius:
    def __init__(self, logger: logging.Logger, teff: float, mass: float,
                 spec_type: str = "DA", data_dir: str = "/mnt/user-data/uploads") -> None:
        self.logger = logger
        self.teff = teff
        self.mass = mass
        self.spec_type = spec_type.upper()
        self.data_dir = data_dir

    def get_radius(self) -> dict:
        if self.mass <= MASS_THRESHOLD:
            layer_note = "He core (Panei 2007)"
            masses, Rprim, Rchk, label = panei_tracks(
                self.logger, self.data_dir, self.teff
            ).get_radii()
        else:
            if self.spec_type == "DA":
                layer = "thick"
            elif self.spec_type == "DB":
                layer = "thin"
            else:
                raise ValueError("spec_type must be 'DA' or 'DB' for mass>0.5")
            layer_note = f"C/O core, {layer} layer (Bedard 2020), spec={self.spec_type}"
            masses, Rprim, Rchk, label = bedard_tracks(
                self.logger, self.data_dir, self.teff, layer
            ).get_radii()

        Rp, Rc, extrap, mmin, mmax, nseq = _interp_mass(
            masses, Rprim, Rchk, self.mass, label)

        res = {
            "teff": self.teff, "mass": self.mass, "model": label,
            "R_primary_Rsun": Rp, "R_crosscheck_Rsun": Rc,
            "frac_diff": abs(Rp - Rc) / Rp,
            "extrapolated": extrap,
            "grid_mass_min": mmin, "grid_mass_max": mmax,
            "n_sequences_used": nseq,
        }

        self.logger.info(f"Teff: {self.teff:.0f} K | Mass: {self.mass:.4f} Msun")
        self.logger.info(f"Model: {label} | Composition: {layer_note}")
        self.logger.info(
            f"Grid mass span: {mmin:.4f}-{mmax:.4f} Msun ({nseq} sequences cover this Teff)")
        self.logger.info(
            f"R (primary): {Rp:.5f} Rsun | R (cross-check): {Rc:.5f} Rsun "
            f"(diff {res['frac_diff']*100:.2f}%)")
        if extrap:
            side = "ABOVE" if self.mass > mmax else "BELOW"
            self.logger.warning(
                f"Mass is {side} the available grid ({mmin:.4f}-{mmax:.4f}). "
                "Radius EXTRAPOLATED in log R-log M; treat as indicative only.")
        return res


class WDTrackInterpolator:
    """Pre-loads all WD cooling-track sequences and provides fast R(Teff, mass) lookups.

    Intended to be built once before an MCMC run and passed as an argument so that no
    file I/O occurs inside the sampler hot-loop.  Uses separate RegularGridInterpolators
    for the He-core (Panei 2007, mass ≤ MASS_THRESHOLD) and C/O-core (Bedard 2020,
    mass > MASS_THRESHOLD) grids.

    sigma_R : float
        Gaussian width [R_sun] used by the MCMC prior (default 0.001 R_sun ≈ 7% for a
        typical 0.6 Msun WD).  Set to a larger value to widen the prior.
    """

    def __init__(self, logger: logging.Logger,
                 panei_dir:  str | None = None,
                 bedard_dir: str | None = None,
                 spec_type:  str = "DA",
                 sigma_R:    float = 0.003,
                 logg1_ref:  float | None = None,
                 logg1_err:  float | None = None) -> None:
        self.logger    = logger
        self.sigma_R   = sigma_R
        self.logg1_ref = logg1_ref
        self.logg1_err = logg1_err
        spec_type      = spec_type.upper()
        if spec_type not in ("DA", "DB"):
            raise ValueError(f"spec_type must be 'DA' or 'DB', got '{spec_type}'")
        layer = "thick" if spec_type == "DA" else "thin"

        panei_dir  = panei_dir  or _DEFAULT_PANEI_DIR
        bedard_dir = bedard_dir or _DEFAULT_BEDARD_DIR

        panei_seqs  = self._load_panei(panei_dir)
        bedard_seqs = self._load_bedard(bedard_dir, layer)

        self._panei_R_interp,  self._panei_lg_interp  = self._build_interp(panei_seqs,  "Panei")
        self._bedard_R_interp, self._bedard_lg_interp = self._build_interp(bedard_seqs, "Bedard")

    def _load_panei(self, panei_dir: str) -> list:
        """Returns list of (mass, teff_arr, R_arr, logg_arr)."""
        files = sorted(glob.glob(os.path.join(panei_dir, "m0.*He.SDSS")))
        _p    = panei_tracks(self.logger, panei_dir, teff=5000.0)
        seqs  = []
        for f in files:
            mass = _p._mass_from_name(f)
            teff_arr, logg_arr, _ = _p._load(f)
            R_arr = _R_from_logg(logg_arr, mass)
            seqs.append((mass, teff_arr, R_arr, logg_arr))
        self.logger.info(f"WDTrackInterpolator: loaded {len(seqs)} Panei He-core sequences from {panei_dir}")
        return seqs

    def _load_bedard(self, bedard_dir: str, layer: str) -> list:
        """Returns list of (mass, teff_arr, R_arr, logg_arr)."""
        files = sorted(glob.glob(os.path.join(bedard_dir, f"seq_*_{layer}.txt")))
        _b    = bedard_tracks(self.logger, bedard_dir, teff=5000.0, layer=layer)
        seqs  = []
        for f in files:
            mass = _b._mass_from_name(f)
            teff_arr, logg_arr, R_arr = _b._load(f)
            seqs.append((mass, teff_arr, R_arr, logg_arr))
        self.logger.info(f"WDTrackInterpolator: loaded {len(seqs)} Bedard C/O-core ({layer}) sequences from {bedard_dir}")
        return seqs

    def _build_interp(self, seqs: list, label: str) -> tuple:
        """Build RegularGridInterpolators for R and logg on a (mass, Teff) grid.

        Returns (R_interp, logg_interp); either may be None if seqs is empty.
        """
        if not seqs:
            self.logger.warning(f"WDTrackInterpolator: no {label} sequences — prior disabled for this mass regime")
            return None, None

        seqs.sort(key=lambda x: x[0])
        masses = np.array([s[0] for s in seqs])

        teff_min = max(s[1].min() for s in seqs)
        teff_max = min(s[1].max() for s in seqs)
        if teff_min >= teff_max:
            teff_min = min(s[1].min() for s in seqs)
            teff_max = max(s[1].max() for s in seqs)
            self.logger.warning(
                f"WDTrackInterpolator: {label} sequences have no common Teff range; "
                f"using [{teff_min:.0f}, {teff_max:.0f}] K with boundary clipping")

        teff_grid  = np.linspace(teff_min, teff_max, 300)
        R_grid     = np.zeros((len(masses), 300))
        logg_grid  = np.zeros((len(masses), 300))
        for i, (_, teff_arr, R_arr, logg_arr) in enumerate(seqs):
            R_grid[i]    = np.interp(teff_grid, teff_arr, R_arr)
            logg_grid[i] = np.interp(teff_grid, teff_arr, logg_arr)

        self.logger.info(
            f"WDTrackInterpolator: {label} grid built — "
            f"mass=[{masses.min():.3f}, {masses.max():.3f}] Msun, "
            f"Teff=[{teff_min:.0f}, {teff_max:.0f}] K")

        kw = dict(method="linear", bounds_error=False, fill_value=None)
        axes = (masses, teff_grid)
        return (RegularGridInterpolator(axes, R_grid,    **kw),
                RegularGridInterpolator(axes, logg_grid, **kw))

    def _interps(self, mass: float) -> tuple:
        if mass <= MASS_THRESHOLD:
            return self._panei_R_interp, self._panei_lg_interp
        return self._bedard_R_interp, self._bedard_lg_interp

    def predict_radius(self, teff: float, mass: float) -> float:
        """Return predicted WD radius [R_sun] for the given Teff [K] and mass [M_sun]."""
        R_interp, _ = self._interps(mass)
        if R_interp is None:
            return np.nan
        return float(R_interp([[mass, teff]]))

    def predict_logg(self, teff: float, mass: float) -> float:
        """Return predicted WD log g (cgs) for the given Teff [K] and mass [M_sun]."""
        _, lg_interp = self._interps(mass)
        if lg_interp is None:
            return np.nan
        return float(lg_interp([[mass, teff]]))

class CompTrackInterpolator:
    """Companion mass–radius prior from BHAC15 tracks.

    Radius at these masses (~0.07–0.11 Msun) is age-converged above ~1 Gyr,
    so we take the oldest tabulated age per mass and interpolate in mass only.
    """

    def __init__(self, logger: logging.Logger, track_file: str,
                 sigma_R_frac: float = 0.06,
                 m_min: float = 0.070, m_max: float = 0.110) -> None:
        self.logger = logger
        self.sigma_R_frac = sigma_R_frac

        masses, radii = {}, {}
        with open(track_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6 or parts[0].startswith(("!", "#")):
                    continue
                try:
                    m    = float(parts[0])
                    logt = float(parts[1])
                    r    = float(parts[5])
                except ValueError:
                    continue   
                if m_min <= m <= m_max:
                    if m not in masses or logt > masses[m]:
                        masses[m] = logt
                        radii[m]  = r

        if len(radii) < 3:
            raise ValueError(f"Too few tracks in [{m_min}, {m_max}] Msun")

        m_grid = np.array(sorted(radii))
        r_grid = np.array([radii[m] for m in m_grid])
        self.m_min, self.m_max = m_grid[0], m_grid[-1]
        self._interp = interp1d(m_grid, r_grid, kind="linear",
                                bounds_error=False, fill_value=np.nan)
        self.logger.info(f"Companion M–R prior: masses {m_grid}, radii {r_grid}, "
                         f"sigma_R = {sigma_R_frac:.0%} of R_pred")

    def predict_radius(self, m2_msun: float) -> float:
        return float(self._interp(m2_msun))
