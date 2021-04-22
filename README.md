# gpr4im

This package uses Gaussian Process Regression (GPR) as a foreground removal technique in the context single-dish 21cm intensity mapping. This user-friendly code shows you how to do this in the context of MeerKAT-like simulations, but any intensity mapping data can be used. This is the accompaying code to the paper (paper link here), where we look at how GPR performs as a foreground removal technique in our simulations in comparison with Principal Component Analysis.

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

It will *not* install `pymultinest`, which is required for the `Nested Sampling.ipynb` notebook. If you would like to run that notebook, please see http://johannesbuchner.github.io/PyMultiNest/install.html for more details.

## Quickstart

For a quick introduction on how to run the code, please see `Running GPR.ipynb`. For a more thorough run through of how the code works, please see `Understanding GPR.ipynb`.

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
