from setuptools import setup, find_packages, Extension
import numpy
import glob

exec(open('fgcm/_version.py').read())

#scripts = ['scripts/combineAtmosphereFiles.py']
scripts = []

setup(
    name='fgcm',
    version=__version__,
    description='Tools for FGCM Y3A1',
    author='Eli Rykoff, Dave Burke',
    author_email='erykoff@slac.stanford.edu',
    packages=find_packages(),
    data_files=[],
    scripts=scripts
)


