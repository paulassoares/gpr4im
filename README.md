# gpr4im

This package uses Gaussian Process Regression (GPR) as a foreground removal technique in the context single-dish 21cm intensity mapping. This user-friendly code shows you how to do this in the context of MeerKAT-like simulations, but any intensity mapping data can be used. This is the accompaying code to the paper `(paper link here)`, where we look at how GPR performs as a foreground removal technique in our simulations in comparison with Principal Component Analysis.

## Installation

To install this package, follow the following instructions on a terminal:

```
git clone https://github.com/psahds/gpr4im.git
cd gpr4im
pip install .
```

Make sure you do `pip install .` in the gpr4im folder, where the `setup.py` file is.

Installing `gpr4im` will also automatically install:

- `numpy`
- `matplotlib`
- `pandas`
- `GPy` (see https://github.com/SheffieldML/GPy)
- `scipy`
- `getdist`
- `astropy`
- `jupyter`

It will *not* install `pymultinest`, which is required for the `Nested sampling.ipynb` notebook. If you would like to run that notebook, please see http://johannesbuchner.github.io/PyMultiNest/install.html for more details.

## Quickstart

An very quick example of how to run GPR foreground removal using our code is shown below, but please see the Jupyter notebooks for further explanation:

```
import pandas as pd
import GPy
from gpr4im import fg_tools as fg

data = pd.read_pickle('example_data.pkl')
dirty_map = data.beam.FGnopol_HI_noise

kern_fg = GPy.kern.RBF(1)
kern_fg.variance.constrain_bounded(1000,100000000)
kern_fg.lengthscale.constrain_bounded(200,10000)
kern_21 = GPy.kern.Exponential(1)
kern_21.variance.constrain_bounded(0.000001,0.5)
kern_21.lengthscale.constrain_bounded(0.01,15)

gpr_result = fg.GPRclean(dirty_map, data.freqs, kern_fg, kern_21, 
                         NprePCA=0, num_restarts=10, noise_data=None, 
                         heteroscedastic=False, zero_noise=True, invert=False)

cleaned_map = gpr_result.res
```

For a quick introduction on how to run the code, please see `Running GPR.ipynb`. For a more thorough run through of how the code works, please see `Understanding GPR.ipynb`. The Jupyter Notebooks folder contains other introductory notebooks for how all the aspects of our code and data work, and are all user friendly. These use the data set `example_data.pkl`, which is described in the Data folder's README.

The Reproducible paper plots folder contains the notebooks showing how we obtained the analysis results for our companion paper (these are less introductory, but useful for those trying to understand how our analysis was done). The code here requires the `multinest_results.pkl` file, as well as the full data used in our analysis, `data.pkl`, which can be obtained from this link (but beware, it is 2.84 GB):

> https://www.dropbox.com/sh/9zftczeypu7xgt3/AABiiBw_0SBPrLgSHsjiISz8a?dl=0

The `Nested sampling.ipynb` notebook also uses this data, and requires `pymultinest` to be installed.

## Acknowledgment

If you make use of this code, please cite:

```
bibtex for our paper
```

This code is heavily based on the publicly available `ps_eor` code (https://gitlab.com/flomertens/ps_eor), so if you use our code please also acknowledge:

```
@article{Mertens2018,
   title={Statistical 21-cm Signal Separation via Gaussian Process Regression Analysis},
   ISSN={1365-2966},
   url={http://dx.doi.org/10.1093/mnras/sty1207},
   DOI={10.1093/mnras/sty1207},
   journal={Monthly Notices of the Royal Astronomical Society},
   publisher={Oxford University Press (OUP)},
   author={Mertens, F G and Ghosh, A and Koopmans, L V E},
   year={2018},
   month={May}
}
```
