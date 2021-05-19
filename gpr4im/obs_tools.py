from scipy.ndimage import gaussian_filter
import numpy as np


'''
This script contains code for modelling observational effects in HI IM simulations, specifically the telescope beam smoothing effect (based on Steve Cunnington's beam smoothing code). 
It also contains code for how to convert a data cube in [Nx, Ny, Nz] format to [Npix, Nz] format.

'''


def ConvolveCube(dT, zeff, lx, ly, sigma_beam, cosmo):
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


def LoSpixels(image_cube, mean_center=True):
    '''
    Convert image data cube from [Nx, Ny, Nz] to [Nz, Npix] format.

    INPUTS:
    image_cube: input data cube in image form, with shape [Nx, Ny, Nz] where Nz
    mean_center: if True, do mean centering for each frequency slice after 
    '''

    # swap axes to [Nz, Nx, Ny] for converting to visibilities:
    image_cube = np.swapaxes(image_cube,0,1)
    image_cube = np.swapaxes(image_cube,0,2)

    # conver to LoS pixels format [Nz, Npix]
    axes = np.shape(image_cube)
    image_LoSpixels = np.reshape(image_cube,(axes[0], axes[1]*axes[2]))
    
    # mean center the data:
    if mean_center==True:
        nz = np.shape(image_LoSpixels)[0]
        for i in range(nz):
            image_LoSpixels[i] = image_LoSpixels[i] - np.mean(image_LoSpixels[i])
    
    return image_LoSpixels
