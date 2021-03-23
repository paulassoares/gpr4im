# GPRlowzIM
Investigating Gaussian Process Regression on 21cm foregrounds for low-z, single-dish intensity mapping

## Assumed Survey Parameters:

 - Redshift = 0.2 < z < 0.58
 - Effective Central Redshift (z_eff) = 0.39 [Multi-Dark cosmology sim will be a box evolved to this redshift]
 - Frequency Channels = (899 - 1184)MHz - 1MHz channel width (285 channels)
 - Nx,Ny,Nz (cartesian box size in voxels) = 256, 256, 285
 - Cartesian Box physical size Lx,Ly,Lz = 1000,1000, 924.78 Mpc/h
 - Sky_area (determined by cartesian 1Gpc^2 perpendicular box-size at z_eff) = 54.1 x 54.1 = 2927 deg^2
 - T_obs = 1000hrs
 - Beam = (Dish diameter = 15m) - for freq-independent beam and noise
 - N_dishes = 64, Tsys = 25 K

## Data Folder currently contains:
- `data.h5` (master data file, contains all smoothed and unsmoothed data, including different combinations, e.g. HI signal + noise smoothed together) - see notebook `Create_datafile.ipynb` to see exactly how we generate this file using the data described below.
 - `T_HI-MDSAGE_z_0.39.npy` (HI signal)
 - `dT_free_Stripe82_noBeam.npy` (free-free smooth foreground component)
 - `dT_psource_Stripe82_noBeam.npy` (point source smooth foreground component)
 - `dT_sync_Stripe82_noBeam.npy` (synchrotron smooth foreground component)
 - `dT_pleak_Stripe82_noBeam.npy` (polarisation leakage foreground component)
 - `dT_noise.npy` (instrumental noise)
 
 [all data is not smoothed by telescope beam, temps are in mK]
 
 See Section 3 of https://arxiv.org/pdf/2002.05626.pdf for similar process for producing the above data

## MultiDark Data:

> http://www.multidark.org/

>These maps were run with the Queen Mary’s
Apocrita HPC facility, supported by QMUL Research-IT
http://doi.org/10.5281/zenodo.438045 using Python 3.6.5.

## Access data here:

> https://www.dropbox.com/sh/9zftczeypu7xgt3/AABiiBw_0SBPrLgSHsjiISz8a?dl=0
