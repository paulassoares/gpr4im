import numpy as np
import pandas as pd

# adding path to GPR_for_IM directory in order to import relevant scripts
import sys
sys.path.append('../')
import pk_tools as pk


def test_pk_par_run():
    
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
    kmin_par = 2*np.pi/lz
    kmax_par = 0.4
    # set width of k bins to be 2*kmin
    dk_par = 2*kmin_par
    # number of k bins:
    nkbin_par = int((kmax_par-kmin_par)/dk_par)
    # setting array of k bin edges:
    kbins_par = np.arange(kmin_par,kmax_par,dk_par)
    # calculating spherically averaged power spectrum
    HI_pk_par = pk.ParaPk(HI_data, nx, ny, nz, lx, ly, lz, kbins_par, w, W)[0]

    #### Comparing with hardcoded result ####

    hardcoded_result = np.load('hardcoded_results/pk_par_result.npy')
    ratio = HI_pk_par/hardcoded_result

    # asserting their differences aren't > 1%
    # (some differences should be present because GPR is stochastic)
    assert all([a == 1 for a in ratio])
