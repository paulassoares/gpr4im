import numpy as np

#from ps_eor import psutil, datacube, pspec


'''
This script contains code for how to convert a data cube in image format to a data cube in visibility format,
specifically the CartDataCube object from ps_eor required to run GPR.
'''


def generateCartDataCube(image_cube, vmin, vmax, df, nx, ny, ra_width, dec_width, mean_center=True, invert=False):

    '''
    Convert image data cube to CartDataCube visibility cube in order to run GPR.

    INPUTS:
    image_cube: input data cube in image form, with shape [Nx, Ny, Nz] where Nz
    is the radial (frequency) direction
    vmin: minimum frequency (MHz)
    vmax: maximum frequency (MHz)
    df: frequency resolution (MHz)
    nx, ny: angular pixels of data (how many pixels in the x and y directions)
    ra_width, dec_width: width of sky in RA and Dec directions (deg)
    mean_center: if True, do mean centering for each frequency slice after 
    Fourier transforming, if False, don't
    invert: whether or not to invert the frequency axis

    OUTPUT:
    a CartDataCube object (see datacube.py in ps_eor repo)
    '''

    # swap axes to [Nz, Nx, Ny] for converting to visibilities:
    image_cube = np.swapaxes(image_cube,0,1)
    image_cube = np.swapaxes(image_cube,0,2)
    
    # convert from image to visibility, input data must be in [Nz, Nx, Ny] format (frequency first):
    ft_fct = lambda data, axes: psutil.img_to_vis(data, axes=axes)
    vis_cube = ft_fct(image_cube, (1, 2))

    # convert to correct shape for running GPR [Nz, Npix]:
    axes = np.shape(vis_cube)
    vis_cube = np.reshape(vis_cube,(axes[0], axes[1]*axes[2]))
    if invert == True: vis_cube = vis_cube[::-1]
    
    # mean center the data:
    if mean_center==True:
        nz = np.shape(vis_cube)[0]
        for i in range(nz):
            vis_cube[i] = vis_cube[i] - np.mean(vis_cube[i])

    # create array of frequencies in (MHz), should be same length as Nz:
    fmhz = np.arange(vmin, vmax, df)
    freqs = fmhz * 1e6 # convert array of frequencies to Hz

    # obtain meta data object:
    res = np.radians(ra_width)/nx # angular pixel resolution in radians
    meta = datacube.ImageMetaData.from_res(res, (nx, ny))

    # obtain uu and vv:
    uu, vv, _ = psutil.get_ungrid_vis_idx((nx, ny), res, umin=0.0, umax=1000000000000000)

    # generate CartDataCube object:
    vis_cartdatacube = datacube.CartDataCube(vis_cube, uu, vv, freqs, meta)

    return vis_cartdatacube


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
