import distutils
from distutils.core import setup
import glob
import os
import fnmatch

exec(open('fgcm/_version.py').read())

scripts = ['scripts/runFgcmFitCycle.py',
           'scripts/makeFgcmAtmosphereTable.py',
           'scripts/listFgcmAtmosphereTables.py']

name='fgcm'

def fileList(name, relpath, globstr):
    return [relpath + x for x in
            fnmatch.filter(os.listdir(name+'/'+relpath+'/'),globstr)]

datafiles = fileList(name,'data/templates/','*.fits')
datafiles.extend(fileList(name,'data/tables/','*.fits'))

setup(
    name='fgcm',
    version=__version__,
    description='Forward Global Calibration Method (FGCM)',
    author='Eli Rykoff, Dave Burke',
    author_email='erykoff@stanford.edu',
    packages=[name],
    package_dir = {name: name},
    package_data = {name: datafiles},
    scripts=scripts
)


