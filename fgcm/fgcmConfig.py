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

        requiredKeys=['exposureFile','ccdOffsetFile','obsFile','indexFile','UTBoundary',
                      'washMJDs','epochMJDs','lutFile','expField',
                      'ccdField','latitude','seeingField','fitBands','extraBands',
                      'deepFlag','minObsPerBand','nCore','brightObsGrayMax',
                      'minStarPerCCD','minCCDPerExp','maxCCDGrayErr','aperCorrFitNBins',
                      'illegalValue','sedFitBandFudgeFactors','sedExtraBandFudgeFactors',
                      'starColorCuts','cycleNumber','outfileBase','maxIter',
                      'sigFgcmMaxErr','sigFgcmMaxEGray','ccdGrayMaxStarErr',
                      'mirrorArea','cameraGain','approxThroughput','ccdStartIndex',
                      'minExpPerNight','expGrayInitialCut','expVarGrayPhotometricCut',
                      'sigFgcmMaxErr','sigFgcmMaxEGray','ccdGrayMaxStarErr',
                      'expGrayPhotometricCut','expGrayRecoverCut','expGrayErrRecoverCut',
                      'sigma0Cal','logLevel']

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
        self.latitude = float(configDict['latitude'])
        self.seeingField = configDict['seeingField']
        self.deepFlag = configDict['deepFlag']
        self.cosLatitude = np.cos(np.radians(self.latitude))
        self.sinLatitude = np.sin(np.radians(self.latitude))
        self.fitBands = np.array(configDict['fitBands'])
        self.extraBands = np.array(configDict['extraBands'])
        self.nCore = int(configDict['nCore'])
        self.brightObsGrayMax = float(configDict['brightObsGrayMax'])
        self.minStarPerCCD = int(configDict['minStarPerCCD'])
        self.minStarPerExp = int(configDict['minStarPerExp'])
        self.minCCDPerExp = int(configDict['minCCDPerExp'])
        self.maxCCDGrayErr = float(configDict['maxCCDGrayErr'])
        #self.maxExpGrayRecoverErr = float(configDict['maxExpGrayRecoverErr'])
        #self.maxExpGrayRecoverVar = float(configDict['maxExpGrayRecoverVar'])
        #self.maxExpGrayRecover = float(configDict['maxExpGrayRecover'])
        self.expGrayPhotometricCut = float(configDict['expGrayPhotometricCut'])
        self.expGrayRecoverCut = float(configDict['expGrayRecoverCut'])
        self.expVarGrayPhotometricCut = float(configDict['expVarGrayPhotometricCut'])
        self.expGrayErrRecoverCut = float(configDict['expGrayErrRecoverCut'])
        self.minExpPerNight = int(configDict['minExpPerNight'])
        self.expGrayInitialCut = float(configDict['expGrayInitialCut'])
        #self.expGrayCut = float(configDict['expGrayCut'])
        #self.varGrayCut = float(configDict['varGrayCut'])
        self.aperCorrFitNBins = int(configDict['aperCorrFitNBins'])
        self.illegalValue = float(configDict['illegalValue'])
        self.sedFitBandFudgeFactors = np.array(configDict['sedFitBandFudgeFactors'])
        self.sedExtraBandFudgeFactors = np.array(configDict['sedExtraBandFudgeFactors'])
        self.starColorCuts = configDict['starColorCuts']
        self.cycleNumber = int(configDict['cycleNumber'])
        self.outfileBase = configDict['outfileBase']
        self.maxIter = int(configDict['maxIter'])
        self.mirrorArea = float(configDict['mirrorArea'])
        self.cameraGain = float(configDict['cameraGain'])
        self.approxThroughput = float(configDict['approxThroughput'])
        self.ccdStartIndex = int(configDict['ccdStartIndex'])
        self.ccdOffsetFile = configDict['ccdOffsetFile']
        self.sigFgcmMaxErr = float(configDict['sigFgcmMaxErr'])
        self.sigFgcmMaxEGray = float(configDict['sigFgcmMaxEGray'])
        self.ccdGrayMaxStarErr = float(configDict['ccdGrayMaxStarErr'])
        self.sigma0Cal = float(configDict['sigma0Cal'])
        self.logLevel = configDict['logLevel']

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

        if (self.expGrayPhotometricCut >= 0.0) :
            raise ValueError("expGrayPhotometricCut must be negative.")
        if (self.expGrayRecoverCut > self.expGrayPhotometricCut) :
            raise ValueError("expGrayRecoverCut must be less than expGrayPhotometricCut")
        if (self.expVarGrayPhotometricCut <= 0.0):
            raise ValueError("expVarGrayPhotometricCut must be > 0.0")
        if (self.expGrayErrRecoverCut <= 0.0):
            raise ValueError("expGrayErrRecoverCut must be > 0.0")

        if 'outputPath' in configDict:
            self.outputPath = os.path.abspath(configDict['outputPath'])
        else:
            self.outputPath = os.path.abspath('.')

        # create output path if necessary
        if (not os.path.isdir(self.outputPath)):
            try:
                os.makedirs(self.outputPath)
            except:
                raise IOError("Could not create output path: %s" % (self.outputPath))

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

        # check the cut values

        self.outfileBaseWithCycle = '%s_cycle%02d' % (self.outfileBase, self.cycleNumber)

        # set up logger are we get the name...
        self.fgcmLog = FgcmLogger('%s/%s.log' % (self.outputPath,
                                                 self.outfileBaseWithCycle),
                                  self.logLevel)

        #self.plotPath = '%s/%s_plots_cycle%02d' % (self.outputPath,self.outfileBase,
        #                                            self.cycleNumber)
        self.plotPath = '%s/%s_plots' % (self.outputPath,self.outfileBaseWithCycle)

        if (not os.path.isdir(self.plotPath)):
            try:
                os.makedirs(self.plotPath)
            except:
                raise IOError("Could not create plot path: %s" % (self.plotPath))

        if (self.illegalValue >= 0.0):
            raise ValueError("Must set illegalValue to a negative number")

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

        # read in the ccd offset file
        self.ccdOffsets = fitsio.read(self.ccdOffsetFile,ext=1)

        # based on mjdRange, look at epochs; also sort.
        # confirm that we cover all the exposures, and remove excess epochs

        try:
            self.epochNames = np.array(configDict['epochNames'],dtype='a21')
        except:
            self.epochNames = np.zeros(self.epochMJDs.size,dtype='a21')
            for i in xrange(self.epochMJDs.size):
                self.epochNames[i] = 'epoch%d' % (i)

        if (self.epochNames.size != self.epochMJDs.size):
            raise ValueError("number of epochNames must be same as epochMJDs")

        st=np.argsort(self.epochMJDs)
        self.epochMJDs = self.epochMJDs[st]
        self.epochNames = self.epochNames[st]
        test=np.searchsorted(self.epochMJDs,self.mjdRange)

        if test.min() == 0:
            raise ValueError("Exposure start MJD is out of epoch range!")
        if test.max() == self.epochMJDs.size:
            raise ValueError("Exposure end MJD is out of epoch range!")

        # crop to valid range
        self.epochMJDs = self.epochMJDs[test[0]-1:test[1]+1]
        self.epochNames = self.epochNames[test[0]-1:test[1]+1]

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

        # and AB zeropoint
        hPlanck = 6.6
        expPlanck = -27.0
        self.zptAB = (-48.6 - 2.5*expPlanck +
                       2.5*np.log10((self.mirrorArea * self.approxThroughput) /
                                    (hPlanck * self.cameraGain)))

        ## FIXME: add pmb scaling?

