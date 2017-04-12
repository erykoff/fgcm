from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import yaml

from fgcmUtilities import _pickle_method

import types
import copy_reg
import sharedmem as shm

copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmConfig(object):
    """
    """
    def __init__(self,configFile):
        self.configFile = configFile

        with open(self.configFile) as f:
            configDict = yaml.load(f)

        requiredKeys=['exposureFile','obsFile','indexFile','UTBoundary',
                      'washMJDs','epochMJDs','lutFile','expField',
                      'ccdField','latitude','seeingField','fitBands','extraBands',
                      'deepFlag','minObsPerBand','nCore','brightObsGrayMax',
                      'minStarPerCCD','minCCDPerExp','maxCCDGrayErr','aperCorrFitNBins',
                      'illegalValue']

        for key in requiredKeys:
            if (key not in configDict):
                raise ValueError("required %s not in configFile" % (key))

        self.exposureFile = configDict['exposureFile']
        self.minObsPerBand = configDict['minObsPerBand']
        self.obsFile = configDict['obsFile']
        self.indexFile = configDict['indexFile']
        self.UTBoundary = configDict['UTBoundary']
        self.washMJDs = np.array(configDict['washMJDs'])
        self.epochMJDs = np.array(configDict['epochMJDs'])
        self.lutFile = configDict['lutFile']
        self.expField = configDict['expField']
        self.ccdField = configDict['ccdField']
        self.latitude = configDict['latitude']
        self.seeingField = configDict['seeingField']
        self.deepFlag = configDict['deepFlag']
        self.cosLatitude = np.cos(np.radians(self.latitude))
        self.sinLatitude = np.sin(np.radians(self.latitude))
        self.fitBands = np.array(configDict['fitBands'])
        self.extraBands = np.array(configDict['extraBands'])
        self.nCore = configDict['nCore']
        self.brightObsGrayMax = configDict['brightObsGrayMax']
        self.minStarPerCCD = configDict['minStarPerCCD']
        self.minCCDPerExp = configDict['minCCDPerExp']
        self.maxCCDGrayErr = configDict['maxCCDGrayErr']
        self.aperCorrFitNBins = configDict['aperCorrFitNBins']
        self.illegalValue = configDict['illegalValue']

        if 'pwvFile' in configDict:
            self.pwvFile = configDict['pwvFile']
            if ('externalPWVDeltaT' not in configDict):
                raise ValueError("Must include externalPWVDeltaT with pwvFile")
            self.externalPWVDeltaT = configDict['externalPWVDeltaT']

        else:
            self.pwvFile = None

        if 'tauFile' in configDict:
            self.tauFile = configDict['tauFile']
            if ('externalTauDeltaT' not in configDict):
                raise ValueError("Must include externalTauDeltaT with tauFile")
            self.externalTauDeltaT = configDict['externalTauDeltaT']

        else:
            self.tauFile = None

        if 'stepUnitReference' in configDict:
            self.stepUnitReference = configDict['stepUnitReference']
        else:
            self.stepUnitReference = 0.001
        if 'stepGrain' in configDict:
            self.stepGrain = configDict['stepGrain']
        else:
            self.stepGrain = 10.0

        # and look at the lutFile
        lutStats=fitsio.read(self.lutFile,ext='INDEX')

        self.nCCD = lutStats['NCCD'][0]
        self.bands = lutStats['BANDS'][0]
        self.pmbRange = np.array([np.min(lutStats['PMB']),np.max(lutStats['PMB'])])
        self.pwvRange = np.array([np.min(lutStats['PWV']),np.max(lutStats['PWV'])])
        self.O3Range = np.array([np.min(lutStats['O3']),np.max(lutStats['O3'])])
        self.tauRange = np.array([np.min(lutStats['TAU']),np.max(lutStats['TAU'])])
        self.alphaRange = np.array([np.min(lutStats['ALPHA']),np.max(lutStats['ALPHA'])])
        self.zenithRange = np.array([np.min(lutStats['ZENITH']),np.max(lutStats['ZENITH'])])

        lutStd = fitsio.read(self.lutFile,ext='STD')
        self.pmbStd = lutStd['PMBSTD'][0]
        self.pwvStd = lutStd['PWVSTD'][0]
        self.o3Std = lutStd['O3STD'][0]
        self.tauStd = lutStd['TAUSTD'][0]
        self.alphaStd = lutStd['ALPHASTD'][0]
        self.zenithStd = lutStd['ZENITHSTD'][0]
        self.lambdaStd = lutStd['LAMBDASTD'][0]

        # and look at the exposure file and grab some stats
        expInfo = fitsio.read(self.exposureFile,ext=1)
        self.expRange = np.array([np.min(expInfo['EXPNUM']),np.max(expInfo['EXPNUM'])])
        self.mjdRange = np.array([np.min(expInfo['MJD']),np.max(expInfo['MJD'])])
        self.nExp = expInfo.size

        # based on mjdRange, look at epochs; also sort.
        # confirm that we cover all the exposures, and remove excess epochs
        self.epochMJDs.sort()
        test=np.searchsorted(self.epochMJDs,self.mjdRange)

        if test.min() == 0:
            raise ValueError("Exposure start MJD is out of epoch range!")
        if test.max() == self.epochMJDs.size:
            raise ValueError("Exposure end MJD is out of epoch range!")

        # crop to valid range
        self.epochMJDs = self.epochMJDs[test[0]-1:test[1]+1]

        # and look at washMJDs; also sort
        self.washMJDs.sort()
        gd,=np.where((self.washMJDs > self.mjdRange[0]) &
                     (self.washMJDs < self.mjdRange[1]))
        self.washMJDs = self.washMJDs[gd]

