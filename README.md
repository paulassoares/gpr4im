# gpr4im

This package uses Gaussian Process Regression (GPR) as a foreground removal technique in the context single-dish 21cm intensity mapping. This user-friendly code shows you how to do this in the context of MeerKAT-like simulations, but any data can be used. This is the accompaying code to the paper (paper link here), where we look at how GPR performs as a foreground removal technique in our simulations compare with Principal Component Analysis.

## Installation

To install this package, follow the following instructions on a terminal:

```
git clone https://github.com/psahds/gpr4im.git
cd gpr4im
pip install .
```

Make sure you do `pip install .` in the gpr4im folder, where the `setup.py` file is.

## Dependencies

- `numpy`
- `matplotlib`
- `pandas`
- `GPy` (see https://github.com/SheffieldML/GPy)
- `scipy` (1.3.0 or later, due to GPy dependency)
- `getdist` (only for making corner plots)
- `astropy` (only for `Smoothing maps.ipynb`)
- `pymultinest` (only for `Nested Sampling.ipynb` notebook)
- `jupyter` (for running the Jupyter Notebooks)

## Acknowledgment

If you make use of this code, please cite:

```
bibtex for our paper
```

This code is heavily based on the publicly available `ps_eor` code (https://gitlab.com/flomertens/ps_eor), so if you use it please also acknowledge:

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
