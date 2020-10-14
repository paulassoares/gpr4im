import numpy as np
import scipy.stats as stats


'''
This script contains code for calculating and binning the frequency covariance matrix of 21cm IM data
in both image and visibility format.
Based on the script: https://gitlab.com/flomertens/ps_eor/-/blob/master/doc/GPR%20theory%20and%20practice.ipynb
'''


def get_cov_r(cov_matrix, dx):

    '''
    ps_eor code for binning the frequency covariance matrix by frequency widths.
    
    INPUTS:
    cov_matrix: frequency covariance matrix of data, in [Nz, Npix] format.
    dx: frequency resolution of data (MHz)
    '''
    n = cov_matrix.shape[0]
    a, b = np.indices((n, n))
    r = abs(a - np.arange(n)) * dx
    cov_m, bins, _ = stats.binned_statistic(r.flatten(), cov_matrix.flatten(), bins=n)
    return bins[:-1], cov_m / cov_m[0]


def binned_covariance(data, dx, vis=False):
    '''
    INPUTS:
    data: input data to obtain covariance matrix of in either [Nx,Ny,Nz] data cube form
    or HEALpix [Nz,Npix] form where Nz is number of redshift (frequency) bins.
    dx: frequency resolution of the data in MHz.
    vis: whether the data is in complex visibility form (True) or not (False).
    '''
    axes = np.shape(data)
    if len(axes)==3:
        data = np.reshape(data,(axes[0]*axes[1],axes[2]))
        data = np.swapaxes(data,0,1) # [Npix, Nz] -> [Nz, Npix]
    Nz, Npix = np.shape(data)
    C = np.cov(data) # Obtain frequency covariance matrix for input data
    if vis==True: bins, binned_cov = get_cov_r(np.real(C), dx)
    else: bins, binned_cov = get_cov_r(C, dx)
    return bins, binned_cov


def get_kernel_samples(kernel_name, X, n_samples, l=False):
    """
    Generate the multivariate random distribution of data with zero mean and covariance kernel
    specified by kernel_name, and obtain n samples.
    Based on https://gist.github.com/rikrd/05f5ca0c31adb9203823
    
    INPUTS:
    kernel_name: name of covariance kernel as in dir(GPy.kern)
    X: x data points to calculate kernel over
    n_samples: number of samples to obtain from multivariate random distribution
    l: lengthscale, always False if kernel doesn't have a lengthscale parameter
    """
    import GPy
    # Get the kernel from GPy:
    Kernel = getattr(GPy.kern, kernel_name)
    if l==False: k = Kernel(input_dim=1)
    else: k = Kernel(input_dim=1, lengthscale=l)
    
    X = X[:,None] # reshape X to make it n*p, they try to use 'design matrices' in GPy 
    mu = np.zeros((len(X)))# vector of the means, zero
    
    # Compute the covariance matrix associated with inputs X:
    C = k.K(X,X) 
    
    # Generate n separate samples paths from a Gaussian with mean mu and covariance C:
    Z = np.random.multivariate_normal(mu,C,n_samples)
    return Z, C
