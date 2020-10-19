# GPRlowzIM
Investigating Gaussian Process Regression on 21cm foregrounds for low-z, single-dish intensity mapping

## Assumed Survey Parameters:

 - Redshift = 0.2 < z < 0.58
 - Effective Central Redshift (z_eff) = 0.39 [Multi-Dark cosmology sim will be a box evolved to this redshift]
 - Frequency Channels = (899 - 1184)MHz - 1MHz channel width (285 channels)
 - Nx,Ny,Nz (cartesian box size in voxels) = 256, 256, 285
 - Cartesian Box physical size Lx,L,Lz = 1000,1000, 924.78 Mpc/h
 - Sky_area (determined by cartesian 1Gpc^2 perpendicular box-size at z_eff) = 54.1 x 54.1 = 2927 deg^2
 - T_obs = 1000hrs
 - Beam = (Dish diameter = 15m) - for freq-independent beam and noise
 - N_dishes = 64, Tsys = 25 K

## Data Folder currently contains:
 - `dT_HI+noise_15mBeam.npy` (HI+noise only [no FGs])
 - `dT_HI+noise+FGnopol_Stripe82_15mBeam.npy` (as above with FG emission from Stripe82 region)
 - `dT_HI+noise+FGwpol_Stripe82_15mBeam.npy` (as above with effects from polarization leakage applied to FG)
[all temps are in mK]

See Section 3 of https://arxiv.org/pdf/2002.05626.pdf for similar process for producing the above data

## MultiDark Data:

> http://www.multidark.org/

>These maps were run with the Queen Mary’s
Apocrita HPC facility, supported by QMUL Research-IT
http://doi.org/10.5281/zenodo.438045 using Python 3.6.5.

## Access data here:

> https://www.dropbox.com/sh/9zftczeypu7xgt3/AABiiBw_0SBPrLgSHsjiISz8a?dl=0
