from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import yaml
import esutil

class DESExposureFormatter(object):
    """
    """
    def __init__(self,exposureFile,pmbFile,deltaAperFile):
        self.exposureFile = exposureFile
        self.pmbFile = pmbFile
        self.deltaAperFile = deltaAperFile

        if (not os.path.isfile(self.exposureFile)):
            raise ValueError("Could not find exposureFile: %s" % (self.exposureFile))
        if (not os.path.isfile(self.pmbFile)):
            raise ValueError("Could not find pmbFile: %s" % (self.pmbFile))
        if (not os.path.isfile(self.deltaAperFile)):
            raise ValueError("Could not find deltaAperFile: %s" % (self.deltaAperFile))

    def __call__(self,outFile,clobber=False):
        expInfoIn = fitsio.read(self.exposureFile,ext=1)

        expInfo = np.zeros(expInfoIn.size,dtype=[('EXPNUM','i8'),
                                                 ('BAND','a2'),
                                                 ('TELRA','f8'),
                                                 ('TELDEC','f8'),
                                                 ('TELHA','f8'),
                                                 ('MJD','f8'),
                                                 ('EXPTIME','f4'),
                                                 ('PMB','f4'),
                                                 ('DELTA_APER','f4'),
                                                 ('DEEPFLAG','i2')])

        expInfo['EXPNUM'] = expInfoIn['EXPNUM']
        expInfo['BAND'] = expInfoIn['BAND']
        expInfo['TELRA'] = expInfoIn['TRADEG']
        expInfo['TELDEC'] = expInfoIn['TDECDEG']
        expInfo['MJD'] = expInfoIn['MJD_OBS']
        expInfo['EXPTIME'] = expInfoIn['EXPTIME']
        expInfo['DEEPFLAG'] = 0

        deep,=np.where(np.core.defchararray.strip(expInfoIn['PROGRAM']) == 'supernova')
        expInfo['DEEPFLAG'][deep] = 1

        # and the hour angle
        for i in xrange(expInfo.size):
            parts=expInfoIn['HA'][i].split(':')
            sign = 1.0
            if parts[0][0] == '-':
                sign = -1.0
            expInfo['TELHA'][i] = sign * 15.0 * (np.abs(float(parts[0])) +
                                                 float(parts[1])/60. +
                                                 float(parts[2])/3600.0)

        # and read the pmb file...where did it come from?

        pmbExp = []
        pmbPressure = []
        with open(self.pmbFile,'r') as f:
            line=f.readline()
            line=f.readline().rstrip('\n').split(',')
            while line[0] != 'EOF':
                pmbExp.append(int(line[0]))
                pmbPressure.append(float(line[1]))
                line = f.readline().rstrip('\n').split(',')

        pmbExp = np.array(pmbExp,dtype=np.int32)
        pmbPressure = np.array(pmbPressure,dtype=np.float32)

        expInfo['PMB'] = expInfoIn['PRESSURE'].astype(np.float32)
        bad,=np.where((~np.isfinite(expInfo['PMB'])) |
                      (expInfo['PMB'] < 700.0) |
                      (expInfo['PMB'] > 1000.0))
        a,b=esutil.numpy_util.match(expInfo['EXPNUM'][bad],pmbExp)

        if (a.size < bad.size):
            raise ValueError("There are %d missing pressure values!" % (bad.size-a.size))

        expInfo['PMB'][bad[a]] = pmbPressure[b]

        # and read the aperfile
        deltaAper=fitsio.read(self.deltaAperFile,ext=1)

        a,b=esutil.numpy_util.match(deltaAper['EXPNUM'],expInfo['EXPNUM'])
        expInfo['DELTA_APER'][b] = deltaAper['APER7M9'][a]

        fitsio.write(outFile,expInfo,clobber=clobber)
