from setuptools import setup, find_packages, Extension
import numpy
import glob

exec(open('fgcm/_version.py').read())

scripts = ['scripts/runFgcmFitCycle.py',
           'scripts/makeFgcmLUT.py',
           'scripts/makeFgcmStars.py']


setup(
    name='fgcm',
    version=__version__,
    description='Forward Global Calibration Method (FGCM)',
    author='Eli Rykoff, Dave Burke',
    author_email='erykoff@stanford.edu',
    packages=find_packages(),
    data_files=[],
    scripts=scripts
)


