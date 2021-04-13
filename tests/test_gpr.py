import numpy as np
import pandas as pd
import GPy

# adding path to GPR_for_IM directory in order to import relevant scripts
import sys
sys.path.append('../')
import fg_tools as fg


def test_gpr_run():
    
    data = pd.read_pickle('../Data/data.pkl')
    FGnopol_HI_noise_data = data.beam.FGnopol_HI_noise
    freqs = data.freqs
    
    #### Running GPR ####
    kern_fg = GPy.kern.RBF(1)
    kern_21 = GPy.kern.Exponential(1)
    kern_21.lengthscale.constrain_bounded(0,15)
    gpr_result = fg.GPRclean(FGnopol_HI_noise_data, freqs, kern_fg, kern_21, num_restarts=15, NprePCA=0, 
                                        noise_data=None, heteroscedastic=False, zero_noise=True, invert=False)
    
    #### Comparing to hardcoded result ####
    hardcoded_result = np.load('hardcoded_results/gpr_nopol_result.npy')
    ratio = gpr_result.model.param_array[:-1]/hardcoded_result

    # asserting their differences aren't > 5%
    # (some differences should be present because GPR is a stochastic process)
    assert all([abs(1 - a) < 0.05 for a in ratio])
