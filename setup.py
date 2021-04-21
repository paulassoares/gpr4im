'''
Setup script, to make package pip installable
'''


from distutils.core import setup


setup(name='gpr4im',
      version='1.0.0',
      author='Paula S. Soares',
      author_email='p.s.soares@qmul.ac.uk',
      packages=['gpr4im'],
      install_requires=['numpy','scipy','matplotlib'],
)
