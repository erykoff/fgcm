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
                      'illegalValue','sedFitBandFudgeFactors','sedExtraBandFudgeFactors',
                      'starColorCuts','cycleNumber','outfileBase','maxIter']

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
        self.sedFitBandFudgeFactors = np.array(configDict['sedFitBandFudgeFactors'])
        self.sedExtraBandFudgeFactors = np.array(configDict['sedExtraBandFudgeFactors'])
        self.starColorCuts = configDict['starColorCuts']
        self.cycleNumber = configDict['cycleNumber']
        self.outfileBase = configDict['outfileBase']
        self.maxIter = configDict['maxIter']


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

        if 'outputPath' in configDict:
            self.outputPath = os.path.abspath(configDict['outputPath'])
        else:
            self.outputPath = os.path.abspath('.')

        if (self.cycleNumber < 0):
            raise ValueError("Illegal cycleNumber: must be >= 0")

        if (self.cycleNumber > 1):
            if ('inParameterFile' not in configDict):
                raise ValueError("Must provide inParameterFile for cycleNumber > 0")
            self.inParameterFile = configDict['inParameterFile']

        if (self.sedFitBandFudgeFactors.size != self.fitBands.size) :
            raise ValueError("sedFitBandFudgeFactors must have same length as fitBands")

        if (self.sedExtraBandFudgeFactors.size != self.extraBands.size) :
            raise ValueError("sedExtraBandFudgeFactors must have same length as extraBands")

        self.plotPath = '%s/%s_plots_cycle_%02d' % (self.outputPath,self.outfileBase,
                                                    self.cycleNumber)

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

        # and deal with fit band indices and extra band indices
        self.bandRequired = np.zeros(self.bands.size,dtype=np.bool)
        self.bandExtra = np.zeros(self.bands.size,dtype=np.bool)
        for i in xrange(self.bands.size):
            if (self.bands[i] in self.fitBands):
                self.bandRequired[i] = True
            if (self.bands[i] in self.extraBands):
                self.bandExtra[i] = True
            if (self.bands[i] in self.fitBands and
                self.bands[i] in self.extraBands):
                raise ValueError("Cannot have the same band as fit and extra")

        # and check the star color cuts and replace with indices...
        for cCut in self.starColorCuts:
            if (cCut[0] not in self.bands):
                raise ValueError("starColorCut band %s not in list of bands!" % (cCut[0]))
            cCut[0] = list(self.bands).index(cCut[0])
            if (cCut[1] not in self.bands):
                raise ValueError("starColorCut band %s not in list of bands!" % (cCut[1]))
            cCut[1] = list(self.bands).index(cCut[1])

