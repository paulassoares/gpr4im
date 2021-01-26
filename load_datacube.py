import numpy as np


'''
This script contains code for how to convert a data cube in [Nx, Ny, Nz] format to [Npix, Nz] format.
'''


def LoSpixels(image_cube, mean_center=True):

    '''
    Convert image data cube from [Nx, Ny, Nz] to [Nz, Npix] format.

    INPUTS:
    image_cube: input data cube in image form, with shape [Nx, Ny, Nz] where Nz
    mean_center: if True, do mean centering for each frequency slice after 
    
    OUTPUT:
    a CartDataCube object (see datacube.py in ps_eor repo)
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
