'''
Setup script, to make package pip installable
'''


from setuptools import setup


# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name = 'gpr4im',
      version = '1.0.1',
      description = 'Python package for using GPR as a foreground removal technique in 21cm intensity mapping',
      author = 'Paula S. Soares',
      author_email = 'p.s.soares@qmul.ac.uk',
      packages = ['gpr4im'],
      url = 'https://github.com/paulassoares/gpr4im',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      install_requires = ['numpy','scipy','matplotlib','astropy','pandas','GPy','getdist','jupyter'],
)
