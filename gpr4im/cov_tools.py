import numpy as np
import re


'''
This script contains code for getting samples from a particular kernel, and the corresponding equation. 
Taken from: https://gist.github.com/rikrd/05f5ca0c31adb9203823
'''


def get_kernel_samples(kernel_name, X, n_samples, l=False, var=1.0):
    """
    Generate the multivariate random distribution of data with zero mean and covariance kernel
    specified by kernel_name, and obtain n samples.
    
    INPUTS:
    kernel_name: name of covariance kernel as in dir(GPy.kern)
    X: x data points to calculate kernel over
    n_samples: number of samples to obtain from multivariate random distribution
    l: lengthscale, always False if kernel doesn't have a lengthscale hyperparameter
    var: variance
    """
    import GPy
    # Get the kernel from GPy:
    Kernel = getattr(GPy.kern, kernel_name)
    if l==False: k = Kernel(input_dim=1)
    else: k = Kernel(input_dim=1, lengthscale=l, variance=var)
    
    X = X[:,None] # reshape X to make it n*p, they try to use 'design matrices' in GPy 
    mu = np.zeros((len(X)))# vector of the means, zero
    
    # Compute the covariance matrix associated with inputs X:
    C = k.K(X,X) 
    
    # Generate n separate samples paths from a Gaussian with mean mu and covariance C:
    Z = np.random.multivariate_normal(mu,C,n_samples)
    return Z, C


def get_equation(kern):
    """
    Retrieves the LaTeX format functional form of covariance kernel (kern).
    """
    match = re.search(r'(math::)(\r\n|\r|\n)*(?P<equation>.*)(\r\n|\r|\n)*', kern.__doc__)
    return '' if match is None else match.group('equation').strip()
