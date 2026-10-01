# lcurve_analysis

A Python pipeline for modelling binary star light curves using [LCURVE](https://github.com/trmrsh/cpp-lcurve). The pipeline refines the orbital ephemeris from timing measurements, fits ULTRACAM multi-band light curves, and derives physical parameters via MCMC.

## Pipeline overview

```
DESC  →  LINEAR_EPHEM_APPROACH  →  MODEL
```

The legacy `ephemeris` stage (NGTS MCMC) is still available but superseded by `linear_ephem`.

| Stage | Script | Purpose |
|---|---|---|
| **DESC** | `desc.py` | Initial ULTRACAM light curve fit across u, g, i bands; sets starting geometry |
| **LINEAR_EPHEM** | `linear_ephem_approach.py` | BLS period search → individual eclipse timing → joint NGTS+ULTRACAM linear ephemeris |
| **MODEL** | `model.py` | Full MCMC light curve modelling; derives masses, radii, temperatures |
| **EPHEMERIS** *(legacy)* | `ephemeris.py` | Period and t0 refinement from NGTS via MCMC (superseded by `linear_ephem`) |

## Requirements

- Python ≥ 3.10
- [LCURVE](https://github.com/trmrsh/cpp-lcurve) binaries (`simplex`, `levmarq`, `lroche`) on `$PATH`
- Python packages:

```
numpy==1.26.4
scipy==1.17.1
astropy==7.2.0
astropy-iers-data==0.2026.5.11.1.8.52
astroquery==0.4.11
matplotlib==3.10.9
emcee==3.1.6
corner==2.2.3
tqdm==4.67.3
```

Install with:

```bash
pip install numpy==1.26.4 scipy==1.17.1 astropy==7.2.0 astroquery==0.4.11 emcee==3.1.6 corner==2.2.3 matplotlib==3.10.9 astropy-iers-data==0.2026.5.11.1.8.52 tqdm==4.67.3
```

## Usage

All stages are driven by a single INI configuration file:

```bash
python main.py example_config.ini
```

Run a subset of stages:

```bash
python main.py example_config.ini --stages desc
python main.py example_config.ini --stages linear_ephem
python main.py example_config.ini --stages model
python main.py example_config.ini --stages desc linear_ephem model
```

### Configuration

Copy and edit `example_config.ini`:

```ini
[paths]
data_root     = /path/to/data/directory
ultracam_dat  = /path/to/target_ultracam.fits
ngts_dat      = /path/to/target_ngts.fits
example_model = /path/to/lcurve/example/model

# SPEEDYFIT model spectra used by DESC to compute relativistic beam factors
wd_model_path   = /path/to/koester_wd_model.txt
comp_model_path = /path/to/nextgen_comp_model.txt

[target]
name    = MY_TARGET
gaia_id = MY_TARGET

[ephemeris]
period = 0.065432        # days — approximate; refined by linear_ephem stage
t0     = 2458765.432100  # BJD TDB — used as literature reference

[star]
teff1     = 10000   # White dwarf Teff (K)
logg1     = 7.50    # White dwarf log g (cgs)
logg1_err = 0.10    # White dwarf log g uncertainty
wdtype    = DA      # DA or DB
teff2     = 3000    # Companion Teff (K)
logg2     = 5.00    # Companion log g (cgs)

[model]
binfact  = 100   # Raw data points per phase bin
a_r_sun  = 0.85  # Semi-major axis (R_sun) — for reference; masses now derived from velocity_scale

# Optional: spectroscopic Teff priors (split-normal; lo/up are asymmetric 1-sigma errors)
# t1_spec = 10000
# t1_low  = 200
# t1_up   = 300
# t2_spec = 2800
# t2_low  = 100
# t2_up   = 100

# Optional: SED + Gaia parallax + extinction prior
# gaia_parallax     = 5.0    # mas
# gaia_parallax_err = 0.1    # mas
# ebv               = 0.02   # E(B-V) dust map value
# ebv_err           = 0.01

# Optional: M-R track uncertainties (fractional)
# sigma_r_wd   = 0.05  # WD radius fractional tolerance (default 0.05)
# sigma_r2_frac = 0.06  # Companion radius fractional tolerance (default 0.06)

# Optional: custom Baraffe companion tracks
# comp_tracks = /path/to/baraffe.dat
```

## Stage descriptions

### DESC

Fits each ULTRACAM band (g first, then u, then i) with a simplex optimisation to establish the initial system geometry and temperatures.

Key behaviours:
- **g-band** is fitted first with geometry (iangle, r1, r2, t1, t2, absorb, period, t0) free; u and i subsequently use the g-band geometry with only temperatures free.
- **Beam factors** (`beam_factor1`, `beam_factor2`) are computed per band from the WD KOESTER model spectrum and a companion blackbody using the relativistic beaming formula.
- **Roche distortion**: r2 is scaled by `roche_distortion()` for the free-geometry fit.
- **Grid jitter search**: after the initial g-band simplex, a randomised grid search (`jitter_grid_search`) finds the optimal `nlat` resolution before re-running the g-band fit; u and i inherit the best grid.
- **Phase windows** (`phase1`, `phase2`) are computed analytically from the fitted geometry and injected into the model file.
- Limb and gravity darkening coefficients are set from Claret tables; LDC are free in the g-band simplex and fixed in u/i.

### LINEAR_EPHEM

Derives a precise linear ephemeris by combining NGTS and ULTRACAM eclipse timings.

Steps:
1. **BLS period search** on NGTS data (period grid 0.005–5 d, 40 000 points), excluding integer-day aliases and lunar/synodic aliases.
2. **Individual eclipse timing**: each detected NGTS eclipse is fitted with a trapezoid model (`fit_single_eclipse`); the centroid time and uncertainty are retained. Requires ≥15 points per eclipse window, ≥5 in-eclipse, ≥5 baseline points on each side.
3. **Initial linear ephemeris** fitted with `fit_linear_ephemeris` (weighted least squares, σ-clipping at 5σ, up to 5 iterations). NGTS timing errors are rescaled so the NGTS-only reduced χ² = 1.
4. **ULTRACAM eclipse timing**: each predicted ULTRACAM eclipse is fitted with the same trapezoid model; accepted timings (err < 43 s, offset < 0.02 d) are appended to the timing set.
5. **Final linear ephemeris** refitted to the combined NGTS + ULTRACAM timing dataset.
6. **Quadratic ephemeris test**: reports dP/dt and its significance; flags if a period derivative is detected at > 3σ.

Outputs: `ephemeris.txt` (period, T0, quality metrics, period-change test), `oc_table.csv`, `eph_phased_lc.png`.

### MODEL

Phase-folds and bins the ULTRACAM data, runs a multi-band simplex warm-start, then runs a parallel MCMC.

Key behaviours:
- **Ephemeris** is read from `{data_dir}/ephemeris.txt` (output of the `linear_ephem` stage).
- **Phase folding**: data are phase-folded using the refined T0 and period; exposure widths are set from bin-centre differences.
- **Adaptive ndiv**: `nsub` (the LCURVE finite-exposure subdivision count) is set by geometry — higher in ingress/egress, lower in-eclipse.
- **Simplex warm-start**: g-band geometry is optimised first; u and i inherit the g-band geometry with temperatures re-optimised.
- **Jacobian covariance**: a 3-band finite-difference Jacobian is computed post-simplex and used to initialise GaussianMove proposals.
- **Geometry reparametrisation**: MCMC samples `(cos_i, w, z)` instead of `(iangle, r1, r2)` directly, where `w = r_sum² − b²` and `z = r_diff² − b²` (`b = cos i`). This decorrelates the inclination banana. Samples are back-transformed to physical parameters before storage.
- **velocity_scale** is a free MCMC parameter. Stellar masses are derived from the posterior on `velocity_scale` via Kepler's third law (`a_m = velocity_scale × 1e3 × P / 2π`).
- **Walker moves**: `GaussianMove(C_Jacobian)` (50%) + `DEMove` (30%) + `DESnookerMove` (20%) when a Jacobian is available; falls back to `DEMove` + `DESnookerMove` otherwise.
- **72 walkers**, 20 000 steps. Burn-in and thinning are set adaptively from the autocorrelation time.
- **Walker sigma-clipping**: walkers with mean log-probability more than 5 MAD below the median are excluded before computing posteriors.

MCMC free parameters: `q`, `cos_i`, `w`, `z` (= geometry), `t1`, `t2`, `velocity_scale`; optionally `parallax`, `ebv` when SED prior is enabled.

#### Priors

| Prior | Source |
|---|---|
| Flat bounds on geometry, temperatures | `log_prior` hard limits |
| Roche-lobe constraint | `r2 < roche_fill_max × r_L(q)` (Eggleton 1983) |
| Spectroscopic Teff | Split-normal on t1, t2 from `t1_spec`/`t2_spec` config keys |
| RV semi-amplitudes K1, K2 | Gaussian from `*_rv_config` (X-Shooter) |
| WD surface gravity log g | Gaussian from `logg1 ± logg1_err` |
| WD mass–radius relation | Bedard + Panei cooling tracks (`WDTrackInterpolator`) |
| Companion mass–radius relation | Baraffe 2015 tracks (`CompTrackInterpolator`) |
| SED + parallax + extinction | Claret intensity + Cardelli extinction (`sed_prior.py`) |

## Module descriptions

| Module | Description |
|---|---|
| `lcurve_commands.py` | Wrappers for `simplex`, `levmarq`, and `lroche` |
| `lcurve_data_files.py` | FITS ingestion, normalisation, phase-folding, and data file writing |
| `lcurve_model_file.py` | Claret limb/gravity darkening interpolation; model file parameter editing; Roche distortion |
| `lcurve_rv_calc.py` | Mass ratio and velocity scale from X-Shooter RV measurements |
| `lcurve_stats.py` | Jacobian covariance estimation; q–i degeneracy χ² grid |
| `plotting.py` | Correlation matrices, χ² maps, marginalised profiles, light curve plots |
| `lcurve_wd_tracks.py` | Bedard + Panei WD M–R tracks; Baraffe 2015 companion tracks (0.01–0.1 M☉) |
| `sed_prior.py` | SED log-prior: Claret surface intensity × (R/d)² vs observed OOT flux, with Cardelli+89 extinction and Gaia parallax |
| `logger.py` | Rotating file + console logger |

## Output files (per target, written to `<data_root>/<name>/`)

| File | Stage | Description |
|---|---|---|
| `*_ultracam_data_file_[1-3]` | DESC | ULTRACAM time-series data files (one per band) |
| `*_ultracam_model_file_[1-3]` | DESC | Best-fit DESC model files (one per band) |
| `ultracam_model.png/.pdf` | DESC | Three-band time-series light curve with DESC model |
| `ephemeris.txt` | LINEAR_EPHEM | Refined period, T0, quality metrics, period-change test result |
| `oc_table.csv` | LINEAR_EPHEM | Eclipse timing O–C table (cycle, tmin, error, O-C in seconds) |
| `eph_phased_lc.png` | LINEAR_EPHEM | NGTS data phase-folded on the new ephemeris |
| `phased_lc.png` | MODEL | Three-band phase-folded binned data (preview before MCMC) |
| `*_phase_data_file_[1-3]` | MODEL | Phase-folded binned data files used in MCMC |
| `*_model_simplex_model_[1-3]` | MODEL | Post-simplex model files (MCMC starting point) |
| `model_vary_q_[1-3]` | MODEL | Simplex model with q unlocked (MCMC input) |
| `{N}_{param}_walker_traj.png` | MODEL | Walker trajectory plots (one per MCMC parameter) |
| `model_corner_plot.png/.pdf` | MODEL | Corner plot of full MCMC posterior |
| `model_corner_plot_transparent.png/.pdf` | MODEL | Corner plot with transparent background (publication style) |
| `*_model_params.txt` | MODEL | Masses, radii, temperatures with 1σ uncertainties |
| `lc_with_model.png/.pdf` | MODEL | Best-fit light curves (u, g, i) with residuals |
| `*_ellipsoidal_bestfit.png/.pdf` | MODEL | Zoomed i-band OOT plot showing ellipsoidal modulation |
| `*_ellipsoidal_simplex.png/.pdf` | MODEL | Same plot at simplex solution (pre-MCMC check) |
| `*_chi2_2d_map.png/.pdf` | MODEL | q–i degeneracy map |
| `*_1d_marginalised_profile.png/.pdf` | MODEL | Marginalised χ² profile for q |

## Physical parameter derivation

Stellar masses are computed from Kepler's third law using the MCMC posterior on `velocity_scale` (the full projected orbital velocity `K1 + K2` in km/s) and mass ratio `q`:

```
a_m     = velocity_scale × 1e3 × P / (2π)          [metres]
M_total = 4π² a³ / (G P²)
M_WD    = M_total / (1 + q)
M_comp  = q × M_WD
```

This replaces the earlier fixed-`a_r_sun` approach: masses now have a full MCMC posterior that is jointly constrained by the light curve geometry, the RV semi-amplitudes, and the WD surface gravity prior.
