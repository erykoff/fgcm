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
    def __init__(self,configFile):
        self.configFile = configFile

        with open(self.configFile) as f:
            configDict = yaml.load(f)

        requiredKeys = ['rawExposureFile','pmbFile','deltaAperFile','outFile',
                        'raRange','decRange','raWrap']

        for key in requiredKeys:
            if (key not in configDict):
                raise ValueError("required key %s not in configFile" % (key))

        self.rawExposureFile = configDict['rawExposureFile']
        self.pmbFile = configDict['pmbFile']
        self.deltaAperFile = configDict['deltaAperFile']
        self.outFile = configDict['outFile']
        self.raRange = np.array(configDict['raRange'])
        self.decRange = np.array(configDict['decRange'])
        self.raWrap = float(configDict['raWrap'])

        if (not os.path.isfile(self.rawExposureFile)):
            raise ValueError("Could not find rawExposureFile %s" % (self.rawExposureFile))
        if (not os.path.isfile(self.pmbFile)):
            raise ValueError("Could not find pmbFile %s" % (self.pmbFile))
        if (not os.path.isfile(self.deltaAperFile)):
            raise ValueError("Could not find deltaAperFile %s" % (self.deltaAperFile))

        if (self.raRange.size != 2) :
            raise ValueError("raRange must have 2 elements")
        if (self.decRange.size != 2):
            raise ValueError("decRange must have 2 elements")


    def run(self,clobber=False):
        """
        """

        if (os.path.isfile(self.outFile) and not clobber):
            print("Output file %s found and clobber == False." % (self.outFile))
            return

        expInfoIn = fitsio.read(self.rawExposureFile,ext=1)

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

        # Cut to ra/dec range
        telRAWrap = np.zeros_like(expInfo['TELRA'])
        telRAWrap[:] = expInfo['TELRA']
        hi,=np.where(telRAWrap > self.raWrap)
        telRAWrap[hi] -= 360.0

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


        inFootprint, = np.where((telRAWrap >= self.raRange[0]) &
                                (telRAWrap <= self.raRange[1]) &
                                (expInfo['TELDEC'] >= self.decRange[0]) &
                                (expInfo['TELDEC'] <= self.decRange[1]))

        expInfo = expInfo[inFootprint]

        fitsio.write(self.outFile,expInfo)
