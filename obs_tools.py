from scipy.ndimage import gaussian_filter
import numpy as np


'''
Code for modelling observational effects in HI IM simulations, including the telescope beam smoothing effect (based on Steve's beam smoothing code)
'''


def ConvolveCube(dT,zeff,lx,ly,sigma_beam,cosmo):
    '''
    Function to smooth entire data cube one slice at a time
    
    INPUTS:
    dT: data to be smoothed, in format [Nx, Ny, Nz] where Nz is frequency direction
    zeff: effective redshift of data cube
    lx, ly: dimension of cube in x,y direction (Mpc)
    sigma_beam: FWHM of beam (deg)
    cosmo: Astropy cosmology to use in beam smoothing calculations
    '''
    nx,ny,nz = np.shape(dT)
    dpix = np.mean([lx/nx,ly/ny]) # pixel size
    r = cosmo.comoving_distance(zeff)
    R_beam = (np.radians(sigma_beam)/(2*np.sqrt(2*np.log(2))))*r.value*cosmo.h
    for j in range(nz):
        dT[:,:,j] = gaussian_filter(dT[:,:,j], sigma=R_beam/dpix, mode='reflect')
    return dT
