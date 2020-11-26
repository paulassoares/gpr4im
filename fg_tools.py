import numpy as np
import GPy

import sys
sys.path.append('../')
import load_datacube as load


'''
This script contains Steve's code for foreground removal using PCA, and our code for GPR FG removal.
'''


def GPRclean(Input, freqs, k_FG, k_21cm, prior_FG=None, prior_21cm=None, num_restarts=10, NprePCA=0, noise=True, heteroscedastic=False, invert=False):
    '''
    Runs foreground clean on IM data using GPR.
    
    Input: data cube to be cleaned, in [Nx,Ny,Nz] where Nz is the frequency direction
    freqs: array of frequency range of data
    k_FG, k_21cm: choice of foreground and 21cm kernels (can be a sum or product of different kernels)
    prior_FG, prior_21cm: prior bounds for lengthscale of kernels, should be a list [lower bound, upper bound].
        If None, no prior will be set for the lengthscale
    num_restarts: how many times to optimise the GPR regression model
    NprePCA: 0 if no pre-PCA is desired, otherwise this number is the N_FG number of components used
        in a pre-PCA clean of the data
    noise: if False, sets noise in model to zero (and fixed)
    heteroscedastic: if True, runs Heteroscedastic regression model (variable noise)
    invert: if True, inverts data in the frequency direction
    '''
    axes = np.shape(Input)
    
    # if desired, do a pre-PCA with N_FG=NprePCA removed components
    if NprePCA > 0: Input = PCAclean(Input, N_FG=NprePCA)[0]
    Input = load.LoSpixels(Input) # converting [Nx,Ny,Nz] -> [Npix,Nz]
    if invert==True: Input = Input[::-1] # invert frequency axis
    
    # setting priors:
    if prior_FG is not None:
        k_FG.lengthscale.constrain_bounded(prior_FG[0],prior_FG[1])
    if prior_21cm is not None:
        k_21cm.lengthscale.constrain_bounded(prior_21cm[0],prior_21cm[1])
    
    # build your model, input the freq range, the data, and the kernels
    kern = k_FG + k_21cm
    if heteroscedastic==True: model = GPy.models.GPHeteroscedasticRegression(freqs[:, np.newaxis], Input, kern)
    else: model = GPy.models.GPRegression(freqs[:, np.newaxis], Input, kern)
    
    # if data has no noise, set it to zero (and fixed):
    if noise==False: 
        model['.*Gaussian_noise'] = 0.0
        model['.*noise'].fix()
    
    # optimise model, find best fitting hyperparameters for kernels
    model.optimize_restarts(num_restarts=num_restarts)
    
    # extract optimised kernels
    k_FG, k_21cm = model.kern.parts
    
    # make prediction of what FGs would look like using this optimised FG kernel
    y_fit, y_cov = model.predict(freqs[:, np.newaxis], full_cov=True, kern=k_FG,
        include_likelihood=False)
    
    # subtract FG fit from data, to obtain HI residuals:
    y_sub = Input - y_fit
    
    # reshape residuals
    y_sub = np.swapaxes(y_sub,0,1)
    y_sub = np.reshape(y_sub,(axes[0], axes[1], axes[2]))
    
    return y_sub, y_cov, model


def PCAclean(Input, N_FG=4):
    '''
    Takes input in [Nx,Ny,Nz] data cube form where Nz is number of redshift 
    (frequency) bins. N_FG is number of eigenmodes for PCA to remove
    '''
    
    # Collapse data cube to [Npix,Nz] structure:
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
    
    # PCA Component Maps
    Recon_FG = np.dot(A,S)
    Residual = Input - Recon_FG
    Residual = np.swapaxes(Residual,0,1) #[Nz,Npix]->[Npix,Nz]
    Residual = np.reshape(Residual,(axes[0],axes[1],axes[2]))
    
    return Residual,eignumb,eigenval
