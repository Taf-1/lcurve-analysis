# lcurve_analysis

A Python pipeline for modelling binary star light curves using [LCURVE](https://github.com/trmrsh/cpp-lcurve). The pipeline refines the orbital ephemeris from NGTS photometry, fits ULTRACAM multi-band light curves, and derives physical parameters via MCMC.

## Pipeline overview

```
DESC  →  EPHEMERIS  →  MODEL
```

| Stage | Script | Purpose |
|---|---|---|
| **DESC** | `desc.py` | Initial ULTRACAM light curve fit across u, g, i bands |
| **EPHEMERIS** | `ephemeris.py` | Period and t0 refinement from NGTS photometry via MCMC |
| **MODEL** | `model.py` | Full MCMC light curve modelling; derives masses, radii, temperatures |

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
```

Install with:

```bash
pip install numpy==1.26.4 scipy==1.17.1 astropy==7.2.0 astroquery==0.4.11 emcee==3.1.6 corner==2.2.3 matplotlib==3.10.9 astropy-iers-data==0.2026.5.11.1.8.52
```

## Usage

All three stages are driven by a single INI configuration file:

```bash
python main.py example_config.ini
```

Run a subset of stages:

```bash
python main.py example_config.ini --stages desc
python main.py example_config.ini --stages ephemeris model
```

### Configuration

Copy and edit `example_config.ini`:

```ini
[paths]
data_root     = /path/to/data/directory
ultracam_dat  = /path/to/target_ultracam.fits
ngts_dat      = /path/to/target_ngts.fits
example_model = /path/to/lcurve/example/model

[target]
name    = MY_TARGET
gaia_id = MY_TARGET

[ephemeris]
period = 0.065432        # days
t0     = 2458765.432100  # BJD TDB

[star]
teff1  = 10000   # White dwarf Teff (K)
logg1  = 7.50    # White dwarf log g
wdtype = DA      # DA or DB
teff2  = 3000    # Companion Teff (K)
logg2  = 5.00    # Companion log g

[model]
binfact  = 100   # Phase bins per data point
a_r_sun  = 0.85  # Semi-major axis (R_sun)
```

## Module descriptions

| Module | Description |
|---|---|
| `lcurve_commands.py` | Wrappers for `simplex`, `levmarq`, and `lroche` |
| `lcurve_data_files.py` | FITS ingestion, normalisation, phase-folding, and data file writing |
| `lcurve_model_file.py` | Claret limb/gravity darkening interpolation; model file parameter editing |
| `lcurve_rv_calc.py` | Mass ratio and velocity scale from X-Shooter RV measurements |
| `lcurve_stats.py` | Jacobian covariance estimation; q–i degeneracy χ² grid |
| `plotting.py` | Correlation matrices, χ² maps, marginalised profiles, light curve plots |
| `logger.py` | Rotating file + console logger |

## Output files (per target)

| File | Description |
|---|---|
| `*_ultracam_model_file_[1-3]` | Best-fit DESC model per band |
| `best_fit_ephemeris_model` | Best-fit ephemeris model from NGTS MCMC |
| `fix_mcmc_vals.txt` | Period and t0 with uncertainties |
| `model_corner_plot.png` | Corner plot of full MCMC posterior |
| `*_model_params.txt` | Masses, radii, temperatures with 1σ uncertainties |
| `lc_with_model.png/.pdf` | Best-fit light curves (u, g, i) with residuals |
| `*_ellipsoidal_bestfit.png/.pdf` | Zoomed i-band plot showing ellipsoidal modulation |
| `*_chi2_2d_map.png/.pdf` | q–i degeneracy map |
| `*_1d_marginalised_profile.png/.pdf` | Marginalised χ² profile for q |

## Physical parameter derivation

Stellar masses are computed from Kepler's third law given the orbital semi-major axis `a_r_sun` (in R☉) and the MCMC posterior on mass ratio q:

```
M_total = 4π² a³ / (G P²)
M_WD    = M_total / (1 + q)
M_comp  = q × M_WD
```
