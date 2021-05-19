import numpy as np
import pandas as pd
import GPy
from . import obs_tools as obs


'''
This script contains Steve's code for foreground removal using PCA, and our code for GPR FG removal.
'''


def GPRclean(Input, freqs, k_FG, k_21cm, num_restarts=10, NprePCA=0, noise_data=None, zero_noise=False, heteroscedastic=False, invert=False):
    '''
    Runs foreground clean on IM data using GPR.
    ---------
    Arguments
    ---------
    
    INPUT: data cube to be cleaned, in [Nx,Ny,Nz] where Nz is the frequency direction
    
    FREQS: array of frequency range of data
    
    k_FG, k_21cm: choice of foreground and 21cm kernels (can be a sum or product of different kernels)
    
    NUM_RESTARTS: how many times to optimise the GPR regression model.
        [NOTE: If you already know your best fitting kernels (i.e., you're inputting the already
        optimized k_FG and k_21cm kernels), set num_restarts = 0]
        
    NprePCA: 0 if no pre-PCA is desired, otherwise this number is the N_FG number of components used
        in a pre-PCA clean of the data
    
    NOISE_DATA: input here your noise map in [Nx,Ny,Nz] if you have a reasonable estimate from your data, 
        otherwise set to None and use GPR to try to fit your noise
        
    ZERO_NOISE: if True, the noise in your GPR model will be set to zero and fixed. Otherwise it will try to
        fit to noise in your data, in either the heteroscedastic or non-heteroscedastic case. Set to zero if 
        you want to fit your noise with a separate kernel instead, otherwise you will fit to noise twice.
        
    HETEROSCEDASTIC: if True, runs Heteroscedastic regression model (variable noise)
        (Note: you cannot have zero_noise=False and heteroscedastic=True at the same time, set 
        heteroscedastic=False instead in this case).
        
    INVERT: if True, inverts data in the frequency direction
    '''
    axes = np.shape(Input)
    
    # if desired, do a pre-PCA with N_FG=NprePCA removed components
    if NprePCA > 0: Input = PCAclean(Input, N_FG=NprePCA)[0]
    
    # converting [Nx,Ny,Nz] -> [Npix,Nz]
        # note: we mean center the data but not the inidividual noise cube, since we want
        # to extract the real noise variance in our data.
    Input = obs.LoSpixels(np.copy(Input), mean_center=True)
    if noise_data is not None: noise_data = obs.LoSpixels(noise_data, mean_center=False)
    
    # invert frequency axis
    if invert==True: 
        Input = Input[::-1]
        if noise_data is not None: noise_data = noise_data[::-1]
    
    # build your model, input the freq range, the data, and the kernels
    kern = k_FG + k_21cm
    
    # this heteroscedastic case assumes a Gaussian noise variance that changes with frequency
    if heteroscedastic==True: 
        # this case assumes noise is known, sets noise level to your noise_data variances
            # at different frequencies (since heteroscedastic)
        if noise_data is not None:
            model = GPy.models.GPHeteroscedasticRegression(freqs[:, np.newaxis], Input, kern)
            model.het_Gauss.variance.constrain_fixed(noise_data.var(axis=1)[:, None])
        # this case assumes noise is not known, model will fit a variance at each frequency
        else:
            model = GPy.models.GPHeteroscedasticRegression(freqs[:, np.newaxis], Input, kern)
        # note: if you want the case of *no noise*, there's no need to use heteroscedastic,
            # so set heteroscedastic = False and see below
    
    # this non-heteroscedastic case assumes constant Gaussian noise variance throughout frequency
    else: 
        # this case assumes noise is know, sets the noise variance level to the variance
            # from the input noise_data
        if noise_data is not None:
            model = GPy.models.GPRegression(freqs[:, np.newaxis], Input, kern)
            model.Gaussian_noise.constrain_fixed(noise_data.var())
        else:
            # this case assumes there is no noise in your data
            if zero_noise == True:
                model = GPy.models.GPRegression(freqs[:, np.newaxis], Input, kern)
                model['.*Gaussian_noise'] = 0.0
                model['.*noise'].fix()
            # this case assumes there is noise but it is unknown, fits a constant variance
            else:
                model = GPy.models.GPRegression(freqs[:, np.newaxis], Input, kern)
    
    # optimise model, find best fitting hyperparameters for kernels
    model.optimize_restarts(num_restarts=num_restarts)
    
    # extract optimised foreground kernel (depending on how many kernels were considered)
    if k_FG.name == 'sum':
        k_FG_len = len(k_FG.parts)
        k_FG = model.kern.parts[0]
        if k_FG_len > 1:
            for i in range(1, k_FG_len):
                k_FG += model.kern.parts[i]
    else: k_FG = model.kern.parts[0]
    
    # make prediction of what FGs would look like using this optimised FG kernel
    y_fit, y_cov = model.predict(freqs[:, np.newaxis], full_cov=True, kern=k_FG,
        include_likelihood=False)
    
    # subtract FG fit from data, to obtain HI residuals:
    y_sub = Input - y_fit
    
    # un-revert frequency axis:
    if invert==True: 
        y_sub = y_sub[::-1]
        y_fit = y_fit[::-1]
    
    # reshape foreground fit
    y_fit = np.swapaxes(y_fit,0,1)
    y_fit = np.reshape(y_fit,(axes[0], axes[1], axes[2]))
    
    # reshape residuals
    y_sub = np.swapaxes(y_sub,0,1)
    y_sub = np.reshape(y_sub,(axes[0], axes[1], axes[2]))
    
    # create series as output object
    d = {'res': y_sub, 'fgcov': y_cov, 'fgfit': y_fit, 'model': model}
    result = pd.Series(d)
    
    return result


def GPRfit(Input, freqs, kern, num_restarts=10, NprePCA=0, noise_data=None, zero_noise=False, heteroscedastic=False, invert=False):
    '''
    Allows you to fit any kernel to any data, and obtain a prediction in your frequency range
    (e.g. prediction of what your foregrounds look like, but when fitting ONLY to the foreground data)
    ---------
    Arguments
    ---------
    
    INPUT: data cube to be cleaned, in [Nx,Ny,Nz] where Nz is the frequency direction
    
    FREQS: array of frequency range of data
    
    kern: choice of kernel
    
    NUM_RESTARTS: how many times to optimise the GPR regression model.
        [NOTE: If you already know your best fitting kernels (i.e., you're inputting the already
        optimized k_FG and k_21cm kernels), set num_restarts = 0]
        
    NprePCA: 0 if no pre-PCA is desired, otherwise this number is the N_FG number of components used
        in a pre-PCA clean of the data
    
    NOISE_DATA: input here your noise map in [Nx,Ny,Nz] if you have a reasonable estimate from your data, 
        otherwise set to None and use GPR to try to fit your noise
        
    ZERO_NOISE: if True, the noise in your GPR model will be set to zero and fixed. Otherwise it will try to
        fit to noise in your data, in either the heteroscedastic or non-heteroscedastic case. Set to zero if 
        you want to fit your noise with a separate kernel instead, otherwise you will fit to noise twice.
        
    HETEROSCEDASTIC: if True, runs Heteroscedastic regression model (variable noise)
        (Note: you cannot have zero_noise=False and heteroscedastic=True at the same time, set 
        heteroscedastic=False instead in this case).
        
    INVERT: if True, inverts data in the frequency direction
    '''
    axes = np.shape(Input)
    
    # if desired, do a pre-PCA with N_FG=NprePCA removed components
    if NprePCA > 0: Input = PCAclean(Input, N_FG=NprePCA)[0]
    
    # converting [Nx,Ny,Nz] -> [Npix,Nz]
        # note: we mean center the data but not the inidividual noise cube, since we want
        # to extract the real noise variance in our data.
    Input = obs.LoSpixels(np.copy(Input), mean_center=True)
    if noise_data is not None: noise_data = obs.LoSpixels(noise_data, mean_center=False)
    
    # invert frequency axis
    if invert==True: 
        Input = Input[::-1]
        if noise_data is not None: noise_data = noise_data[::-1]
    
    # this heteroscedastic case assumes a Gaussian noise variance that changes with frequency
    if heteroscedastic==True: 
        # this case assumes noise is known, sets noise level to your noise_data variances
            # at different frequencies (since heteroscedastic)
        if noise_data is not None:
            model = GPy.models.GPHeteroscedasticRegression(freqs[:, np.newaxis], Input, kern)
            model.het_Gauss.variance.constrain_fixed(noise_data.var(axis=1)[:, None])
        # this case assumes noise is not known, model will fit a variance at each frequency
        else:
            model = GPy.models.GPHeteroscedasticRegression(freqs[:, np.newaxis], Input, kern)
        # note: if you want the case of *no noise*, there's no need to use heteroscedastic,
            # so set heteroscedastic = False and see below
    
    # this non-heteroscedastic case assumes constant Gaussian noise variance throughout frequency
    else: 
        # this case assumes noise is know, sets the noise variance level to the variance
            # from the input noise_data
        if noise_data is not None:
            model = GPy.models.GPRegression(freqs[:, np.newaxis], Input, kern)
            model.Gaussian_noise.constrain_fixed(noise_data.var())
        else:
            # this case assumes there is no noise in your data
            if zero_noise == True:
                model = GPy.models.GPRegression(freqs[:, np.newaxis], Input, kern)
                model['.*Gaussian_noise'] = 0.0
                model['.*noise'].fix()
            # this case assumes there is noise but it is unknown, fits a constant variance
            else:
                model = GPy.models.GPRegression(freqs[:, np.newaxis], Input, kern)
    
    # optimise model, find best fitting hyperparameters for kernels
    model.optimize_restarts(num_restarts=num_restarts)
    
    # make prediction of what FGs would look like using this optimised FG kernel
    y_fit, y_cov = model.predict(freqs[:, np.newaxis], full_cov=True, kern=model.kern,
        include_likelihood=False)
    
    # un-revert frequency axis:
    if invert==True: 
        y_fit = y_fit[::-1]
    
    # reshape residuals
    y_fit = np.swapaxes(y_fit,0,1)
    y_fit = np.reshape(y_fit,(axes[0], axes[1], axes[2]))
    
    # create series as output object
    d = {'cov': y_cov, 'fit': y_fit, 'model': model}
    result = pd.Series(d)
    
    return result


def PCAclean(Input, N_FG=4):
    '''
    Takes input in [Nx,Ny,Nz] data cube form where Nz is number of redshift 
    (frequency) bins. N_FG is number of eigenmodes for PCA to remove
    '''
    
    # Collapse data cube to [Npix,Nz] structure:
    axes = np.shape(Input)
    Input = obs.LoSpixels(Input, mean_center=True)
    Nz,Npix = np.shape(Input)
    
    # Obtain frequency covariance matrix for input data
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
