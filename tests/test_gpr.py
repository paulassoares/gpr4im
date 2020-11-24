import numpy as np
import GPy

# adding path to GPR_for_IM directory in order to import relevant scripts
import sys
sys.path.append('../')
import pk_tools as pk
import fg_tools as fg


def test_gpr_run():
    
    #### Setting up data ####
    
    # loading data:
    FGnopol_data = np.load('../Data/dT_HI+noise+FGnopol_Stripe82_15mBeam_smoothednoise.npy')
    # define frequency range of data:
    df = 1 # frequency resolution of data (MHz)
    vmin = 899 # min frequency (MHz)
    vmax = 1184 # max frequency (MHz)
    freqs = np.arange(vmin, vmax, df)
    
    #### Running GPR ####
    
    # choose your kernels, one for FG and one for HI signal
    kern_fg, kern_21cm = GPy.kern.Poly(1, order=4), GPy.kern.Exponential(1)
    # run GPR and obtain residuals
    gpr_res, gpr_cov, model = fg.GPRclean(FGnopol_data, freqs, kern_fg, kern_21cm,
                                      NprePCA=0, invert=True, heteroscedastic=True)
    
    #### Calculating power spectrum ####
    
    # dimension of data cube:
    lx,ly,lz = 1000,1000, 924.78 #Mpc/h
    nx,ny,nz = 256, 256, 285
    # deasure the auto-power spectrum, with noise:
    kmin = 2*np.pi/pow(lx*ly*lz, 1/3)
    kmax= 1
    dk = 2*kmin
    nkbin = int((kmax-kmin)/dk)
    kbins = np.arange(kmin,kmax,dk)
    k = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
    # weighting and window function:
    w = W = np.ones((nx,ny,nz))
    # calculate power spectrum:
    GPRpk = pk.getpk(gpr_res,w,W,nx,ny,nz,lx,ly,lz,kbins)[0]
    
    #### Comparing with hardcoded result ####

    hardcoded_result = np.load('hardcoded_results/pk_nopol_poly_invert_hetero.npy')
    ratio = GPRpk/hardcoded_result

    # asserting their differences aren't > 1%
    # (some differences should be present because GPR is stochastic)
    assert all([abs(1 - a) < 0.01 for a in ratio])
