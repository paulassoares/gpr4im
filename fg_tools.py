import numpy as np
import GPy

import sys
sys.path.append('../')
import load_datacube as load


'''
This script contains Steve's code for foreground removal using PCA, and our code for GPR FG removal.
'''


def GPRclean(Input, freqs, kern, num_restarts=10, NprePCA=0, heteroscedastic=False, invert=False):
    '''
    Runs foreground clean on IM data using GPR.
    
    Input: data cube to be cleaned, in [Nx,Ny,Nz] where Nz is the frequency direction
    freqs: array of frequency range of data
    kern: kernel combination to be used in FG fit. Should be FG kernel + 21cm kernel (important that
        FG kernel is the first one)
    num_restarts: how many times to optimise the GPR regression model
    NprePCA: 0 if no pre-PCA is desired, otherwise this number is the N_FG number of components used
        in a pre-PCA clean of the data
    heteroscedastic: if True, runs Heteroscedastic regression model
    invert: if True, inverts data in the frequency direction
    '''
    # Check shape of input array. If flat data cube, collapse (ra,dec) structure
    #    to (Npix,Nz) structure:
    axes = np.shape(Input)
    # If required, do a pre-PCA with N_FG=NprePCA removed components
    if NprePCA > 0: Input = PCAclean(Input, N_FG=NprePCA)[0]
    Input = load.LoSpixels(Input)
    if invert==True: Input = Input[::-1]
    # build your model, input the freq range, the data, and the kernels
    if heteroscedastic==True: model = GPy.models.GPHeteroscedasticRegression(freqs[:, np.newaxis], Input, kern)
    else: model = GPy.models.GPRegression(freqs[:, np.newaxis], Input, kern)
    # optimise model, find best fitting hyperparameters for kernels
    model.optimize_restarts(num_restarts=num_restarts)
    # extract optimised kernels for FG and 21cm
    k_FG, k_21cm = model.kern.parts
    # make prediction of what FGs would ,ook like using this optimised FG kernel
    y_fit, y_cov = model.predict(freqs[:, np.newaxis], full_cov=True, kern=k_FG,
        include_likelihood=False)
    # subtract FG fit from data, to obtain HI residuals:
    y_sub = Input - y_fit
    # reshape residuals
    y_sub = np.swapaxes(y_sub,0,1)
    y_sub = np.reshape(y_sub,(axes[0], axes[1], axes[2]))
    return y_sub, model


def PCAclean(Input, N_FG=4):
    '''
    Takes input in either [Nx,Ny,Nz] data cube form or HEALpix [Npix,Nz]
    form where Nz is number of redshift (frequency) bins. N_FG is number of
    eigenmodes for PCA to remove
    '''
    # Check shape of input array. If flat data cube, collapse (ra,dec) structure
    #    to (Npix,Nz) structure:
    axes = np.shape(Input)
    Input = load.LoSpixels(Input)
    Nz,Npix = np.shape(Input)
    # Obtain frequenc covariance matrix for input data
    C = np.cov(Input)
    eigenval, eigenvec = np.linalg.eigh(C)
    eignumb = np.linspace(1,len(eigenval),len(eigenval))
    eigenval = eigenval[::-1] #Put largest eigenvals first
    A = eigenvec[:,Nz-N_FG:] # Mixing matrix
    S = np.dot(A.T,Input)
    #norm = np.linalg.inv(np.dot(A.T,A))
    #A = np.dot(A,norm)
    # PCA Component Maps
    Recon_FG = np.dot(A,S)
    Residual = Input - Recon_FG
    Residual = np.swapaxes(Residual,0,1) #[Nz,Npix]->[Npix,Nz]
    if len(axes)==3: Residual = np.reshape(Residual,(axes[0],axes[1],axes[2])) #Rebuild if Input was 3D datacube
    return Residual,eignumb,eigenval
