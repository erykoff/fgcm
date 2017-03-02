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

    def __run__(self,outFile):
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

        # and read the aperfile
        deltaAper=fitsio.read(self.deltaAperFile,ext=1)

        a,b=esutil.numpy_util.match(deltaAper['EXPNUM'],expInfo['EXPNUM'])
        expInfo['DELTA_APER'][b] = deltaAper['DMAG7M9'][a]

        fitsio.write(outfile,expInfo)
