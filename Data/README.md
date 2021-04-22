# Data
Here we describe the simulated MeerKAT-like single-dish 21cm intensity mapping data we use for our analysis, and where to access it.

## Assumed Survey Parameters:

 - Redshift = 0.2 < z < 0.58
 - Effective Central Redshift (z_eff) = 0.39 [Multi-Dark cosmology sim will be a box evolved to this redshift]
 - Frequency Channels = (899 - 1184)MHz - 1MHz channel width (285 channels)
 - Nx,Ny,Nz (cartesian box size in voxels) = 256, 256, 285
 - Cartesian Box physical size Lx,Ly,Lz = 1000,1000, 924.78 Mpc/h
 - Sky_area (determined by cartesian 1Gpc^2 perpendicular box-size at z_eff) = 54.1 x 54.1 = 2927 deg^2
 - T_obs = 1000hrs
 - Beam (Dish diameter = 15m) = smooth by constant beam, taken to be the largest one in our redshift range
 - N_dishes = 64, Tsys = 25 K

## Data folder contains:

### `data.pkl`

Master data file, contains all smoothed and unsmoothed data, including different combinations, e.g. HI signal + noise smoothed together. This is in the form of a `pandas` Series, split into the following headers and sub-headers:

- `nobeam`: Unsmoothed data, split into:
  - `foregrounds`: Our different foreground sources
    - `sync`: Galactic synchrotron emission
    - `free`: Free-free emission
    - `psource`: Point sources
    - `pleak`: Polarisation leakage
  - `HI`: HI cosmological signal
  - `noise`: Noise data

- `beam`: Smoothed data, split into: 
  - `foregrounds`: Same as above, our different foreground sources:
    - `sync`: Galactic synchrotron emission
    - `free`: Free-free emission
    - `psource`: Point sources
    - `pleak`: Polarisation leakage
    - `all_nopol`: All the foreground sources (except `pleak`)
    - `all_wpol`: All the foreground sources (including `pleak`)
  - `HI`: HI cosmological signal
  - `noise`: Noise data
  - `HI_noise`: HI + noise data
  - `FGwpol_HI_noise`: All foregrounds (including `pleak`) + HI + noise
  - `FGnopol_HI_noise`: All foregrounds (except `pleak`) + HI + noise
  - `FGwpol_HI_noise_lognorm`: All foregrounds (including `pleak`) + HI + noise + lognormal HI realisation
  - `lognorm`: lognormal HI realisation

[all temperature units are in mK]

See Section 2 of https://arxiv.org/pdf/2010.02907v2.pdf and Section 3 of https://arxiv.org/pdf/2002.05626.pdf for details on how the above data are generated.

> We acknowledge use of publicly available MultiDark simulations for generating our HI cosmological signal http://www.multidark.org/

>These maps were run with the Queen Mary’s Apocrita HPC facility, supported by QMUL Research-IT http://doi.org/10.5281/zenodo.438045 using Python 3.6.5.

### `multinest_results.pkl`

Summary of results for our kernel models, obtained using `pymultinest`'s Nested Sampling routine. This is also in the form of a `pandas` series, and is split into each different case considered. Each case includes the median and 1sigma errors of the hyperparameters, obtained from their posterior distributions. They also include the log marginal likelihood (evidence) and its uncertainty. Finally, you may also find the posterior distribution samples and their associated weights for each hyperparameter in these files.

## Access data folder here:

> https://www.dropbox.com/sh/9zftczeypu7xgt3/AABiiBw_0SBPrLgSHsjiISz8a?dl=0
