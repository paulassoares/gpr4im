'''
Setup script, to make package pip installable
'''


from setuptools import setup


setup(name = 'gpr4im',
      version = '1.0.0',
      description = 'Uses GPR for foreground removal in 21cm intensity mapping',
      author = 'Paula S. Soares',
      author_email = 'p.s.soares@qmul.ac.uk',
      packages = ['gpr4im'],
      url = 'https://github.com/paulassoares/gpr4im',
      install_requires = ['numpy','scipy','matplotlib','astropy','pandas','GPy','getdist','jupyter'],
)
