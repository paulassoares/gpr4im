import numpy as np
import pandas as pd

# adding path to GPR_for_IM directory in order to import relevant scripts
import sys
sys.path.append('../')
import pk_tools as pk


def test_bias_corr_run():
    
    #### Loading data ####
    
    data = pd.read_pickle('../Data/data.pkl')
    FGnopol_HI_noise_data = data.beam.FGnopol_HI_noise
    fgcov = np.load('hardcoded_results/fgcov_result.npy')
    
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
    
    # minimum and maximum k in each direction:
    kmin_par = 2*np.pi/lz
    kmax_par = 0.4
    # set width of k bins to be 2*kmin
    dk_par = 2*kmin_par
    # number of k bins:
    nkbin_par = int((kmax_par-kmin_par)/dk_par)
    # setting array of k bin edges:
    kbins_par = np.arange(kmin_par,kmax_par,dk_par)
    
    pk3d_corr, pk1d_corr, samples = pk.get_biascorr(fgcov, FGnopol_HI_noise_data, 10, lx, ly, lz, 
                                                kbins_par, kbins, w, W)
    
    #### Comparing with hardcoded result ####

    hardcoded_result3d = np.load('hardcoded_results/pk3d_corr_result.npy')
    ratio3d = pk3d_corr/hardcoded_result3d
    
    hardcoded_result1d = np.load('hardcoded_results/pk1d_corr_result.npy')
    ratio1d = pk1d_corr/hardcoded_result1d

    # asserting their differences aren't > 1%
    # (some differences should be present because the bias correction is partly random)
    assert all([abs(1 - a) < 0.01 for a in ratio3d])
    assert all([abs(1 - a) < 0.01 for a in ratio1d])
    
