import numpy as np
import pandas as pd

# adding path to GPR_for_IM directory in order to import relevant scripts
import sys
sys.path.append('../')
import pk_tools as pk


def test_pk_spher_run():
    
    #### Loading data ####
    
    data = pd.read_pickle('../Data/data.pkl')
    HI_data = data.beam.HI
    
    #### Calculating power spectrum ####
    
    # Dimension of data cube:
    lx, ly, lz = 1000,1000, 924.78 #Mpc/h
    nx, ny, nz = 256, 256, 285
    # weighting and window function:
    w = W = np.ones((nx,ny,nz))
    # minimum and maximum k in each direction:
    kmin = 2*np.pi/pow(lx*ly*lz, 1/3)
    kmax = 0.4
    # set width of k bins to be 2*kmin
    dk = 2*kmin
    # number of k bins:
    nkbin = int((kmax-kmin)/dk)
    # setting array of k bin edges:
    kbins = np.arange(kmin,kmax,dk)
    # calculating spherically averaged power spectrum
    HI_pk = pk.getpk(HI_data,w,W,nx,ny,nz,lx,ly,lz,kbins)[0]
    
    #### Comparing with hardcoded result ####

    hardcoded_result = np.load('hardcoded_results/pk_spher_result.npy')
    ratio = HI_pk/hardcoded_result

    # asserting their differences aren't > 1%
    # (some differences should be present because GPR is stochastic)
    assert all([a == 1 for a in ratio])
