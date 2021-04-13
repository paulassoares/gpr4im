import numpy as np
import pandas as pd

# adding path to GPR_for_IM directory in order to import relevant scripts
import sys
sys.path.append('../')
import pk_tools as pk


def test_pk_perp_run():
    
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
    kmin_perp = 2*np.pi/np.sqrt(lx**2 + ly**2)
    kmax_perp = 0.4
    # set width of k bins to be 2*kmin
    dk_perp = 2*kmin_perp
    # number of k bins:
    nkbin_perp = int((kmax_perp-kmin_perp)/dk_perp)
    # setting array of k bin edges:
    kbins_perp = np.arange(kmin_perp,kmax_perp,dk_perp)
    # calculating spherically averaged power spectrum
    HI_pk_perp = pk.PerpPk(HI_data, nx, ny, nz, lx, ly, lz, kbins_perp, w, W)[0]

    #### Comparing with hardcoded result ####

    hardcoded_result = np.load('hardcoded_results/pk_perp_result.npy')
    ratio = HI_pk_perp/hardcoded_result

    # asserting their differences aren't > 1%
    # (some differences should be present because GPR is stochastic)
    assert all([a == 1 for a in ratio])
