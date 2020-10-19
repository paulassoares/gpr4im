import numpy as np


'''
This script contains Steve's code for foreground removal using PCA.
'''


def PCAclean(Input, N_FG=4):
    '''
    Takes input in either [Nx,Ny,Nz] data cube form or HEALpix [Npix,Nz]
    form where Nz is number of redshift (frequency) bins. N_FG is number of
    eigenmodes for PCA to remove
    '''
    # Check shape of input array. If flat data cube, collapse (ra,dec) structure
    #    to (Npix,Nz) structure:
    axes = np.shape(Input)
    if len(axes)==3: Input = np.reshape(Input,(axes[0]*axes[1],axes[2]))
    Input = np.swapaxes(Input,0,1) # [Npix,Nz]->[Nz,Npix]
    Nz,Npix = np.shape(Input)
    # Mean centre data:
    for i in range(Nz):
        Input[i] = Input[i] - np.mean(Input[i])
    C = np.cov(Input) #Obtain frequenc covariance matrix for input data
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
