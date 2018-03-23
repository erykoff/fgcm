from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import os
import sys
import yaml

from .fgcmLogger import FgcmLogger

class FgcmConfig(object):
    """
    Class which contains the FGCM Configuration.  Note that if you have fits files
     as input, use configWithFits(configDict) to initialize.

    parameters
    ----------
    configDict: dict
       Dictionary with configuration values
    lutIndex: numpy recarray
       All the information from the LUT index values
    lutStd: numpy recarray
       All the information from the LUT standard values
    expInfo: numpy recarray
       Info about each exposure
    ccdOffsets: numpy recarray
       Info about CCD positional offsets and sizes
    checkFiles: bool, default=False
       Check that all fits files exist
    """
    def __init__(self, configDict, lutIndex, lutStd, expInfo, ccdOffsets, checkFiles=False):

        requiredKeys=['bands','fitBands','extraBands','filterToBand',
                      'exposureFile','obsFile','indexFile',
                      'UTBoundary','washMJDs','epochMJDs','lutFile','expField',
                      'ccdField','latitude','seeingField',
                      'deepFlag','minObsPerBand','nCore','brightObsGrayMax',
                      'minStarPerCCD','minCCDPerExp','maxCCDGrayErr',
                      'aperCorrFitNBins','illegalValue','sedFitBandFudgeFactors',
                      'sedExtraBandFudgeFactors','starColorCuts','cycleNumber',
                      'outfileBase','maxIter','sigFgcmMaxErr','sigFgcmMaxEGray',
                      'ccdGrayMaxStarErr','mirrorArea','cameraGain',
                      'approxThroughput','ccdStartIndex','minExpPerNight',
                      'expGrayInitialCut','expVarGrayPhotometricCut',
                      'sigFgcmMaxErr','sigFgcmMaxEGray','ccdGrayMaxStarErr',
                      'expGrayPhotometricCut','expGrayRecoverCut',
                      'expGrayHighCut',
                      'expGrayErrRecoverCut','sigma0Cal','logLevel',
                      'sigma0Phot','mapLongitudeRef','mapNSide','nStarPerRun',
                      'nExpPerRun','varNSig','varMinBand','useSedLUT',
                      'freezeStdAtmosphere','reserveFraction',
                      'precomputeSuperStarInitialCycle',
                      'useRetrievedPWV','useNightlyRetrievedPWV',
                      'pwvRetrievalSmoothBlock','useRetrievedTauInit',
                      'tauRetrievalMinCCDPerNight','superStarSubCCD',
                      'clobber','printOnly','outputStars']

        for key in requiredKeys:
            if (key not in configDict):
                raise ValueError("required %s not in configFile" % (key))

        #self.bands = np.array(configDict['bands'])
        #self.fitBands = np.array(configDict['fitBands'])
        #self.extraBands = np.array(configDict['extraBands'])
        #self.filterToBand = configDict['filterToBand']
        self.bands = configDict['bands']
        self.fitBands = configDict['fitBands']
        self.extraBands = configDict['extraBands']
        self.filterToBand = configDict['filterToBand']
        self.exposureFile = configDict['exposureFile']
        self.minObsPerBand = configDict['minObsPerBand']
        self.obsFile = configDict['obsFile']
        self.indexFile = configDict['indexFile']
        self.UTBoundary = configDict['UTBoundary']
        self.washMJDs = np.array(configDict['washMJDs'],dtype='f8')
        self.epochMJDs = np.array(configDict['epochMJDs'],dtype='f8')
        self.lutFile = configDict['lutFile']
        self.expField = configDict['expField']
        self.ccdField = configDict['ccdField']
        self.latitude = float(configDict['latitude'])
        self.seeingField = configDict['seeingField']
        self.deepFlag = configDict['deepFlag']
        self.cosLatitude = np.cos(np.radians(self.latitude))
        self.sinLatitude = np.sin(np.radians(self.latitude))
        self.nCore = int(configDict['nCore'])
        self.brightObsGrayMax = float(configDict['brightObsGrayMax'])
        self.minStarPerCCD = int(configDict['minStarPerCCD'])
        self.minStarPerExp = int(configDict['minStarPerExp'])
        self.minCCDPerExp = int(configDict['minCCDPerExp'])
        self.maxCCDGrayErr = float(configDict['maxCCDGrayErr'])
        self.expGrayPhotometricCut = np.array(configDict['expGrayPhotometricCut'])
        self.expGrayHighCut = np.array(configDict['expGrayHighCut'])
        self.expGrayRecoverCut = float(configDict['expGrayRecoverCut'])
        self.expVarGrayPhotometricCut = float(configDict['expVarGrayPhotometricCut'])
        self.expGrayErrRecoverCut = float(configDict['expGrayErrRecoverCut'])
        self.minExpPerNight = int(configDict['minExpPerNight'])
        self.expGrayInitialCut = float(configDict['expGrayInitialCut'])
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
        self.sigFgcmMaxErr = float(configDict['sigFgcmMaxErr'])
        self.sigFgcmMaxEGray = float(configDict['sigFgcmMaxEGray'])
        self.ccdGrayMaxStarErr = float(configDict['ccdGrayMaxStarErr'])
        self.sigma0Cal = float(configDict['sigma0Cal'])
        self.logLevel = configDict['logLevel']
        self.sigma0Phot = float(configDict['sigma0Phot'])
        self.mapLongitudeRef =float( configDict['mapLongitudeRef'])
        self.mapNSide = int(configDict['mapNSide'])
        self.nStarPerRun = int(configDict['nStarPerRun'])
        self.nExpPerRun = int(configDict['nExpPerRun'])
        self.varNSig = float(configDict['varNSig'])
        self.varMinBand = int(configDict['varMinBand'])
        self.useSedLUT = configDict['useSedLUT']
        self.freezeStdAtmosphere = configDict['freezeStdAtmosphere']
        self.reserveFraction = configDict['reserveFraction']
        self.precomputeSuperStarInitialCycle = configDict['precomputeSuperStarInitialCycle']
        self.useRetrievedPWV = configDict['useRetrievedPWV']
        self.useNightlyRetrievedPWV = configDict['useNightlyRetrievedPWV']
        self.pwvRetrievalSmoothBlock = configDict['pwvRetrievalSmoothBlock']
        self.useRetrievedTauInit = configDict['useRetrievedTauInit']
        self.tauRetrievalMinCCDPerNight = configDict['tauRetrievalMinCCDPerNight']
        self.clobber = configDict['clobber']
        self.outputStars = configDict['outputStars']
        self.superStarSubCCD = configDict['superStarSubCCD']

        # Encode filterToBand for easy use with numpy in both py2/3 (sorry)
        #self.filterToBand = {}
        #for key in configDict['filterToBand']:
        #    self.filterToBand[key.encode('utf-8')] = configDict['filterToBand'][key].encode('utf-8')

        if 'pwvFile' in configDict:
            self.pwvFile = configDict['pwvFile']
        else:
            self.pwvFile = None
            self.externalPWVDeltaT = None

        if (self.pwvFile is not None):
            if ('externalPWVDeltaT' not in configDict):
                raise ValueError("Must include externalPWVDeltaT with pwvFile")
            self.externalPWVDeltaT = configDict['externalPWVDeltaT']

        if 'tauFile' in configDict:
            self.tauFile = configDict['tauFile']
        else:
            self.tauFile = None
            self.externalTauDeltaT = None

        if (self.tauFile is not None):
            if ('externalTauDeltaT' not in configDict):
                raise ValueError("Must include externalTauDeltaT with tauFile")
            self.externalTauDeltaT = configDict['externalTauDeltaT']

        if 'stepUnitReference' in configDict:
            self.stepUnitReference = configDict['stepUnitReference']
        else:
            self.stepUnitReference = 0.001
        if 'stepGrain' in configDict:
            self.stepGrain = configDict['stepGrain']
        else:
            self.stepGrain = 10.0
        if 'experimentalMode' in configDict:
            self.experimentalMode = bool(configDict['experimentalMode'])
        else:
            self.experimentalMode = False
        if 'resetParameters' in configDict:
            self.resetParameters = bool(configDict['resetParameters'])
        else:
            self.resetParameters = True

        if 'noChromaticCorrections' in configDict:
            self.noChromaticCorrections = bool(configDict['noChromaticCorrections'])
        else:
            self.noChromaticCorrections = False

        if 'colorSplitIndices' in configDict:
            self.colorSplitIndices = np.array(configDict['colorSplitIndices'], dtype=np.int32)
            if self.colorSplitIndices.size != 2:
                raise ValueError("colorSplitIndices must have 2 elements")
        else:
            self.colorSplitIndices = np.array([0,2])

        if 'expGrayCheckDeltaT' in configDict:
            self.expGrayCheckDeltaT = configDict['expGrayCheckDeltaT']
        else:
            self.expGrayCheckDeltaT = 10./(24.*60.)

        if (self.expGrayRecoverCut > self.expGrayPhotometricCut.min()) :
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

        self.inParameterFile = None
        self.inFlagStarFile = None

        if (self.cycleNumber >= 1) and checkFiles:
            if ('inParameterFile' not in configDict):
                raise ValueError("Must provide inParameterFile for cycleNumber > 0")
            self.inParameterFile = configDict['inParameterFile']
            if ('inFlagStarFile' not in configDict):
                raise ValueError("Must provide inFlagStarFile for cycleNumber > 0")
            self.inFlagStarFile = configDict['inFlagStarFile']


        #if (self.sedFitBandFudgeFactors.size != self.fitBands.size) :
        if (self.sedFitBandFudgeFactors.size != len(self.fitBands)):
            raise ValueError("sedFitBandFudgeFactors must have same length as fitBands")

        #if (self.sedExtraBandFudgeFactors.size != self.extraBands.size) :
        if (self.sedExtraBandFudgeFactors.size != len(self.extraBands)):
            raise ValueError("sedExtraBandFudgeFactors must have same length as extraBands")

        # check the cut values

        self.outfileBaseWithCycle = '%s_cycle%02d' % (self.outfileBase, self.cycleNumber)

        logFile = '%s/%s.log' % (self.outputPath, self.outfileBaseWithCycle)
        if os.path.isfile(logFile) and not self.clobber:
            raise RuntimeError("Found logFile %s, but clobber == False." % (logFile))

        self.plotPath = '%s/%s_plots' % (self.outputPath,self.outfileBaseWithCycle)
        if os.path.isdir(self.plotPath) and not self.clobber:
            # check if directory is empty
            if len(os.listdir(self.plotPath) > 0):
                raise RuntimeError("Found plots in %s, but clobber == False." % (self.plotPath))

        # set up logger are we get the name...
        if ('logger' not in configDict):
            self.fgcmLog = FgcmLogger('%s/%s.log' % (self.outputPath,
                                                     self.outfileBaseWithCycle),
                                      self.logLevel, printLogger=configDict['printOnly'])
            if configDict['printOnly']:
                self.fgcmLog.info('Logging to console')
            else:
                self.fgcmLog.info('Logging started to %s' % (self.fgcmLog.logFile))
        else:
            # Support an external logger such as LSST that has .info() and .debug() calls
            self.fgcmLog = configDict['logger']
            try:
                self.fgcmLog.info('Logging to external logger.')
            except:
                raise RuntimeError("Logging to configDict['logger'] failed.")

        if (self.experimentalMode) :
            self.fgcmLog.info('ExperimentalMode set to True')
        if (self.resetParameters) :
            self.fgcmLog.info('Will reset atmosphere parameters')
        if (self.noChromaticCorrections) :
            self.fgcmLog.info('WARNING: No chromatic corrections will be applied.  I hope this is what you wanted for a test!')

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
        #lutStats=fitsio.read(self.lutFile,ext='INDEX')

        self.nCCD = lutIndex['NCCD'][0]
        # these are np arrays and encoded as such
        #self.lutFilterNames = lutIndex['FILTERNAMES'][0]
        #self.lutStdFilterNames = lutIndex['STDFILTERNAMES'][0]
        self.lutFilterNames = [n.decode('utf-8') for n in lutIndex['FILTERNAMES'][0]]
        self.lutStdFilterNames = [n.decode('utf-8') for n in lutIndex['STDFILTERNAMES'][0]]
        self.pmbRange = np.array([np.min(lutIndex['PMB']),np.max(lutIndex['PMB'])])
        self.pwvRange = np.array([np.min(lutIndex['PWV']),np.max(lutIndex['PWV'])])
        self.O3Range = np.array([np.min(lutIndex['O3']),np.max(lutIndex['O3'])])
        self.tauRange = np.array([np.min(lutIndex['TAU']),np.max(lutIndex['TAU'])])
        self.alphaRange = np.array([np.min(lutIndex['ALPHA']),np.max(lutIndex['ALPHA'])])
        self.zenithRange = np.array([np.min(lutIndex['ZENITH']),np.max(lutIndex['ZENITH'])])

        # make sure we drop trailing spaces
        #self.bands = np.core.defchararray.strip(self.bands[:])
        #self.fitBands = np.core.defchararray.strip(self.fitBands[:])
        #if (self.extraBands.size > 0):
        #    self.extraBands = np.core.defchararray.strip(self.extraBands[:])

        # newer band checks
        #  1) check that all the filters in filterToBand are in lutFilterNames
        #  2) check that all the lutStdFilterNames are lutFilterNames (redundant)
        #  3) check that each band has ONE standard filter
        #  4) check that all the fitBands are in bands
        #  5) check that all the extraBands are in bands

        #  1) check that all the filters in filterToBand are in lutFilterNames
        for filterName in self.filterToBand:
            #test,=np.where(filterName == self.lutFilterNames)
            #if test.size == 0:
            #    raise ValueError("Filter %s in filterToBand not in LUT" % (filterName))
            if filterName not in self.lutFilterNames:
                raise ValueError("Filter %s in filterToBand not in LUT" % (filterName))
            #print(self.filterToBand[filterName])
            #print(self.bands)
            if self.filterToBand[filterName] not in self.bands:
                raise ValueError("Band %s in filterToBand not in bands" %
                                 (self.filterToBand[filterName]))
        #  2) check that all the lutStdFilterNames are lutFilterNames (redundant)
        for lutStdFilterName in self.lutStdFilterNames:
            if lutStdFilterName not in self.lutFilterNames:
                raise ValueError("lutStdFilterName %s not in list of lutFilterNames" % (lutStdFilterName))
        #  3) check that each band has ONE standard filter
        #bandStdFilterIndex = np.zeros(self.bands.size, dtype=np.int32) - 1
        bandStdFilterIndex = np.zeros(len(self.bands), dtype=np.int32) - 1
        for i, band in enumerate(self.bands):
            for j, filterName in enumerate(self.lutFilterNames):
                if self.filterToBand[filterName] == band:
                    # If we haven't found it yet, set the index
                    ind = list(self.lutFilterNames).index(self.lutStdFilterNames[j])
                    if bandStdFilterIndex[i] < 0:
                        bandStdFilterIndex[i] = ind
                    else:
                        if self.lutStdFilterNames[ind] != self.lutStdFilterNames[bandStdFilterIndex[i]]:
                            raise ValueError("Band %s has multiple standard filters (%s, %s)" %
                                             (band, self.lutStdFilterNames[ind],
                                              self.lutStdFilterNames[bandStdFilterIndex[i]]))
        #  4) check that all the fitBands are in bands
        for fitBand in self.fitBands:
            #test,=np.where(fitBand == self.bands)
            if fitBand not in self.bands:
            #if (test.size == 0):
                raise ValueError("Band %s from fitBands not in full bands" % (fitBand))
        #  5) check that all the extraBands are in bands
        for extraBand in self.extraBands:
            #test,=np.where(extraBand == self.bands)
            if extraBand not in self.bands:
            #if (test.size == 0):
                raise ValueError("Band %s from extraBands not in full bands" % (extraBand))

        bandString = " ".join(self.bands)
        self.fgcmLog.info('Found %d CCDs and %d bands (%s)' %
                         (self.nCCD,len(self.bands),bandString))

        # get LUT standard values
        self.pmbStd = lutStd['PMBSTD'][0]
        self.pwvStd = lutStd['PWVSTD'][0]
        self.o3Std = lutStd['O3STD'][0]
        self.tauStd = lutStd['TAUSTD'][0]
        self.alphaStd = lutStd['ALPHASTD'][0]
        self.zenithStd = lutStd['ZENITHSTD'][0]

        # And the lambdaStd and I10Std, for each *band*
        self.lambdaStdBand = lutStd['LAMBDASTD'][0][bandStdFilterIndex]
        self.I10StdBand = lutStd['I10STD'][0][bandStdFilterIndex]

        if (self.expGrayPhotometricCut.size != len(self.bands)):
            raise ValueError("expGrayPhotometricCut must have same number of elements as bands.")
        if (self.expGrayHighCut.size != len(self.bands)):
            raise ValueError("expGrayHighCut must have same number of elements as bands.")
        if (self.expGrayPhotometricCut.max() >= 0.0):
            raise ValueError("expGrayPhotometricCut must all be negative")
        if (self.expGrayHighCut.max() <= 0.0):
            raise ValueError("expGrayHighCut must all be positive")

        # and look at the exposure file and grab some stats
        self.expRange = np.array([np.min(expInfo[self.expField]),np.max(expInfo[self.expField])])
        self.mjdRange = np.array([np.min(expInfo['MJD']),np.max(expInfo['MJD'])])
        self.nExp = expInfo.size

        # read in the ccd offset file...also append a place to store sign/rotation
        dtype = ccdOffsets.dtype.descr
        dtype.extend([('XRA', np.bool),
                      ('RASIGN', 'i2'),
                      ('DECSIGN', 'i2')])

        self.ccdOffsets = np.zeros(ccdOffsets.size, dtype=dtype)
        for name in ccdOffsets.dtype.names:
            self.ccdOffsets[name][:] = ccdOffsets[name][:]

        # FIXME: check that ccdOffsets has the required information!

        # based on mjdRange, look at epochs; also sort.
        # confirm that we cover all the exposures, and remove excess epochs

        try:
            self.epochNames = configDict['epochNames']
        except:
            self.epochNames = []
            for i in xrange(self.epochMJDs.size):
                self.epochNames.append('epoch%d' % (i))

        if len(self.epochNames) != len(self.epochMJDs):
            raise ValueError("Number of epochNames must be same as epochMJDs")

        # are they sorted?
        if (self.epochMJDs != np.sort(self.epochMJDs)).any():
            raise ValueError("epochMJDs must be sorted in ascending order")

        test=np.searchsorted(self.epochMJDs,self.mjdRange)

        if test.min() == 0:
            raise ValueError("Exposure start MJD is out of epoch range!")
        if test.max() == self.epochMJDs.size:
            raise ValueError("Exposure end MJD is out of epoch range!")

        # crop to valid range
        self.epochMJDs = self.epochMJDs[test[0]-1:test[1]+1]
        self.epochNames = self.epochNames[test[0]-1:test[1]+1]

        # and look at washMJDs; also sort
        st=np.argsort(self.washMJDs)
        if (not np.array_equal(st,np.arange(self.washMJDs.size))):
            raise ValueError("Input washMJDs must be in sort order.")

        gd,=np.where((self.washMJDs > self.mjdRange[0]) &
                     (self.washMJDs < self.mjdRange[1]))
        self.washMJDs = self.washMJDs[gd]

        # and deal with fit band indices and extra band indices
        #self.bandRequired = np.zeros(self.bands.size,dtype=np.bool)
        #self.bandExtra = np.zeros(self.bands.size,dtype=np.bool)
        #for i in xrange(self.bands.size):
        #    if (self.bands[i] in self.fitBands):
        #        self.bandRequired[i] = True
        #    if (self.extraBands.size > 0):
        #        if (self.bands[i] in self.extraBands):
        #            self.bandExtra[i] = True
        #        if (self.bands[i] in self.fitBands and
        #            self.bands[i] in self.extraBands):
        #            raise ValueError("Cannot have the same band as fit and extra")
        self.bandRequiredFlag = np.zeros(len(self.bands), dtype=np.bool)
        self.bandExtraFlag = np.zeros(len(self.bands), dtype=np.bool)
        for i,band in enumerate(self.bands):
            if (band in self.fitBands):
                self.bandRequiredFlag[i] = True
            if (len(self.extraBands) > 0):
                if (band in self.extraBands):
                    self.bandExtraFlag[i] = True
                if (band in self.fitBands and
                    band in self.extraBands):
                    raise ValueError("Cannot have the same band as fit and extra")


        # and check the star color cuts and replace with indices...
        #  note that self.starColorCuts is a copy so that we don't overwrite.
        ## FIXME: allow index or name here

        for cCut in self.starColorCuts:
            if (not isinstance(cCut[0],int)) :
                if (cCut[0] not in self.bands):
                    raise ValueError("starColorCut band %s not in list of bands!" % (cCut[0]))
                cCut[0] = list(self.bands).index(cCut[0])
            if (not isinstance(cCut[1],int)) :
                if (cCut[1] not in self.bands):
                    raise ValueError("starColorCut band %s not in list of bands!" % (cCut[1]))
                cCut[1] = list(self.bands).index(cCut[1])

        # and AB zeropoint
        hPlanck = 6.6
        expPlanck = -27.0
        self.zptAB = (-48.6 - 2.5*expPlanck +
                       2.5*np.log10((self.mirrorArea * self.approxThroughput) /
                                    (hPlanck * self.cameraGain)))

        self.configDictSaved = configDict
        ## FIXME: add pmb scaling?

    @staticmethod
    def _readConfigDict(configFile):
        """
        Internal method to read a configuration dictionary from a yaml file.
        """

        with open(configFile) as f:
            configDict = yaml.load(f)

        ##self.fgcmLog.info('Configuration read from %s' % (configFile))
        print("Configuration read from %s" % (configFile))

        return configDict

    @classmethod
    def configWithFits(cls, configDict):
        """
        Initialize FgcmConfig object and read in fits files.

        parameters
        ----------
        configDict: dict
           Dictionary with config variables.
        """

        import fitsio

        expInfo = fitsio.read(configDict['exposureFile'], ext=1)

        try:
            lutIndex = fitsio.read(configDict['lutFile'], ext='INDEX')
            lutStd = fitsio.read(configDict['lutFile'], ext='STD')
        except:
            raise IOError("Could not read LUT info")

        ccdOffsets = fitsio.read(configDict['ccdOffsetFile'], ext=1)

        return cls(configDict, lutIndex, lutStd, expInfo, ccdOffsets, checkFiles=True)


    def saveConfigForNextCycle(self,fileName,parFile,flagStarFile):
        """
        Save a yaml configuration file for the next fit cycle (using fits files).

        Parameters
        ----------
        fileName: string
           Config file filename
        parFile: string
           File with saved parameters from previous cycle
        flagStarFile: string
           File with flagged stars from previous cycle
        """

        configDict = self.configDictSaved.copy()

        # save the outputPath
        configDict['outputPath'] = self.outputPath
        # update the cycleNumber
        configDict['cycleNumber'] = self.cycleNumber + 1

        # default to NOT freeze atmosphere
        configDict['freezeStdAtmosphere'] = False

        # do we want to increase maxIter?  Hmmm.

        configDict['inParameterFile'] = parFile

        configDict['inFlagStarFile'] = flagStarFile

        # do we want to guess as to the photometric cut?  not now.

        with open(fileName,'w') as f:
            yaml.dump(configDict, stream=f)

