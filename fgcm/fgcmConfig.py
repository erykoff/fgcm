from __future__ import division, absolute_import, print_function
from builtins import range

import numpy as np
import os
import sys
import yaml

from .fgcmLogger import FgcmLogger

class ConfigField(object):
    """
    A validatable field with a default
    """

    def __init__(self, datatype, value=None, default=None, required=False, length=None):
        self._datatype = datatype
        self._value = value
        self._required = required
        self._length = length

        _default = default

        if self._datatype == np.ndarray:
            if default is not None:
                _default = np.atleast_1d(default)
            if value is not None:
                self._value = np.atleast_1d(value)

        if datatype is not None:
            if _default is not None:
                if type(_default) != datatype:
                    raise TypeError("Default is the wrong datatype.")
            if self._value is not None:
                if type(self._value) != datatype:
                    raise TypeError("Value is the wrong datatype.")

        if self._value is None:
            self._value = _default

    def __get__(self, obj, type=None):
        return self._value

    def __set__(self, obj, value):
        # need to convert to numpy array if necessary

        if self._datatype == np.ndarray:
            self._value = np.atleast_1d(value)
        else:
            self._value = value

    def validate(self, name):
        if self._required:
            if self._value is None:
                raise ValueError("Required ConfigField %s is not set" % (name))
        elif self._value is None:
            # Okay to have None for not required
            return True

        if self._datatype is not None:
            if type(self._value) != self._datatype:
                raise ValueError("Datatype mismatch for %s (got %s, expected %s)" %
                                 (name, str(type(self._value)), str(self._datatype)))

        if self._length is not None:
            if len(self._value) != self._length:
                raise ValueError("ConfigField %s has the wrong length (%d != %d)" %
                                 (name, len(self._value), self._length))

        return True


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
    noOutput: bool, default=False
       Do not create an output directory.
    """

    bands = ConfigField(list, required=True)

    fitBands = ConfigField(list, required=True)
    notFitBands = ConfigField(list, required=True)
    requiredBands = ConfigField(list, required=True)
    filterToBand = ConfigField(dict, required=True)
    exposureFile = ConfigField(str, required=False)
    ccdOffsetFile = ConfigField(str, required=False)
    obsFile = ConfigField(str, required=False)
    indexFile = ConfigField(str, required=False)
    refstarFile = ConfigField(str, required=False)
    UTBoundary = ConfigField(float, default=0.0)
    washMJDs = ConfigField(np.ndarray, default=np.array((0.0)))
    epochMJDs = ConfigField(np.ndarray, default=np.array((0.0, 1e10)))
    coatingMJDs = ConfigField(np.ndarray, default=np.array((0.0)))
    epochNames = ConfigField(list, required=False)
    lutFile = ConfigField(str, required=False)
    expField = ConfigField(str, default='EXPNUM')
    ccdField = ConfigField(str, default='CCDNUM')
    latitude = ConfigField(float, required=True)
    seeingField = ConfigField(str, default='SEEING')
    seeingSubExposure = ConfigField(bool, default=False)
    deepFlag = ConfigField(str, default='DEEPFLAG')
    fwhmField = ConfigField(str, default='PSF_FWHM')
    skyBrightnessField = ConfigField(str, default='SKYBRIGHTNESS')
    minObsPerBand = ConfigField(int, default=2)
    minObsPerBandFill = ConfigField(int, default=1)
    nCore = ConfigField(int, default=1)
    randomSeed = ConfigField(int, required=False)
    logger = ConfigField(None, required=False)
    outputFgcmcalZpts = ConfigField(bool, default=False)

    brightObsGrayMax = ConfigField(float, default=0.15)
    minStarPerCCD = ConfigField(int, default=5)
    minStarPerExp = ConfigField(int, default=100)
    minCCDPerExp = ConfigField(int, default=5)
    maxCCDGrayErr = ConfigField(float, default=0.05)
    ccdGraySubCCD = ConfigField(bool, default=False)
    ccdGraySubCCDChebyshevOrder = ConfigField(int, default=1)
    ccdGraySubCCDTriangular = ConfigField(bool, default=True)
    aperCorrFitNBins = ConfigField(int, default=5)
    aperCorrInputSlopes = ConfigField(np.ndarray, default=np.array([]))
    illegalValue = ConfigField(float, default=-9999.0)
    sedBoundaryTermDict = ConfigField(dict, required=True)
    sedTermDict = ConfigField(dict, required=True)
    starColorCuts = ConfigField(list, required=True)
    quantityCuts = ConfigField(list, default=[])
    cycleNumber = ConfigField(int, default=0)
    outfileBase = ConfigField(str, required=True)
    maxIter = ConfigField(int, default=50)
    sigFgcmMaxErr = ConfigField(float, default=0.01)
    sigFgcmMaxEGray = ConfigField(list, default=[0.05])
    ccdGrayMaxStarErr = ConfigField(float, default=0.10)
    mirrorArea = ConfigField(float, required=True) # cm^2
    cameraGain = ConfigField(float, required=True)
    approxThroughput = ConfigField(list, default=[1.0])
    ccdStartIndex = ConfigField(int, default=0)
    minExpPerNight = ConfigField(int, default=10)
    expGrayInitialCut = ConfigField(float, default=-0.25)
    expVarGrayPhotometricCut = ConfigField(float, default=0.0005)
    expGrayPhotometricCut = ConfigField(np.ndarray, required=True)
    expGrayRecoverCut = ConfigField(float, default=-1.0)
    expGrayHighCut = ConfigField(np.ndarray, required=True)
    expGrayErrRecoverCut = ConfigField(float, default=0.05)
    sigmaCalRange = ConfigField(list, default=[0.001, 0.003])
    sigmaCalFitPercentile = ConfigField(list, default=[0.05, 0.15])
    sigmaCalPlotPercentile = ConfigField(list, default=[0.05, 0.95])
    sigma0Phot = ConfigField(float, default=0.003)
    logLevel = ConfigField(str, default='INFO')
    quietMode = ConfigField(bool, default=False)
    useRepeatabilityForExpGrayCuts = ConfigField(list, default=[False])

    mapLongitudeRef = ConfigField(float, default=0.0)

    autoPhotometricCutNSig = ConfigField(float, default=3.0)
    autoPhotometricCutStep = ConfigField(float, default=0.0025)
    autoHighCutNSig = ConfigField(float, default=4.0)

    instrumentParsPerBand = ConfigField(bool, default=False)
    instrumentSlopeMinDeltaT = ConfigField(float, default=5.0)

    refStarSnMin = ConfigField(float, default=20.0)
    refStarOutlierNSig = ConfigField(float, default=4.0)
    applyRefStarColorCuts = ConfigField(bool, default=True)
    useRefStarsWithInstrument = ConfigField(bool, default=True)

    mapNSide = ConfigField(int, default=256)
    nStarPerRun = ConfigField(int, default=200000)
    nExpPerRun = ConfigField(int, default=1000)
    varNSig = ConfigField(float, default=100.0)
    varMinBand = ConfigField(int, default=2)
    useSedLUT = ConfigField(bool, default=False)
    modelMagErrors = ConfigField(bool, default=False)
    freezeStdAtmosphere = ConfigField(bool, default=False)
    reserveFraction = ConfigField(float, default=0.1)
    precomputeSuperStarInitialCycle = ConfigField(bool, default=False)
    useRetrievedPwv = ConfigField(bool, default=False)
    useNightlyRetrievedPwv = ConfigField(bool, default=False)
    useQuadraticPwv = ConfigField(bool, default=False)
    pwvRetrievalSmoothBlock = ConfigField(int, default=25)
    fitMirrorChromaticity = ConfigField(bool, default=False)
    useRetrievedTauInit = ConfigField(bool, default=False)
    tauRetrievalMinCCDPerNight = ConfigField(int, default=100)
    superStarSubCCD = ConfigField(bool, default=False)
    superStarSubCCDChebyshevOrder = ConfigField(int, default=1)
    superStarSubCCDTriangular = ConfigField(bool, default=False)
    superStarSigmaClip = ConfigField(float, default=5.0)
    clobber = ConfigField(bool, default=False)
    printOnly = ConfigField(bool, default=False)
    outputStars = ConfigField(bool, default=False)
    fillStars = ConfigField(bool, default=False)
    outputZeropoints = ConfigField(bool, default=False)
    outputPath = ConfigField(str, required=False)
    saveParsForDebugging = ConfigField(bool, default=False)
    doPlots = ConfigField(bool, default=True)

    pwvFile = ConfigField(str, required=False)
    externalPwvDeltaT = ConfigField(float, default=0.1)
    tauFile = ConfigField(str, required=False)
    externalTauDeltaT = ConfigField(float, default=0.1)
    fitGradientTolerance = ConfigField(float, default=1e-5)
    stepUnitReference = ConfigField(float, default=0.0001)
    experimentalMode = ConfigField(bool, default=False)
    resetParameters = ConfigField(bool, default=True)
    noChromaticCorrections = ConfigField(bool, default=False)
    colorSplitIndices = ConfigField(np.ndarray, default=np.array((0,2)), length=2)
    expGrayCheckDeltaT = ConfigField(float, default=10. / (24. * 60.))
    modelMagErrorNObs = ConfigField(int, default=100000)

    inParameterFile = ConfigField(str, required=False)
    inFlagStarFile = ConfigField(str, required=False)

    zpsToApplyFile = ConfigField(str, required=False)
    maxFlagZpsToApply = ConfigField(int, default=2)

    def __init__(self, configDict, lutIndex, lutStd, expInfo, ccdOffsets, checkFiles=False, noOutput=False):

        self._setVarsFromDict(configDict)

        self._setDefaultLengths()

        self.validate()

        # First thing: set the random seed if desired
        if self.randomSeed is not None:
            np.random.seed(seed=self.randomSeed)

        if self.outputPath is None:
            self.outputPath = os.path.abspath('.')
        else:
            self.outputPath = os.path.abspath(self.outputPath)

        # create output path if necessary
        if not noOutput:
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

        # check the cut values

        self.outfileBaseWithCycle = '%s_cycle%02d' % (self.outfileBase, self.cycleNumber)

        logFile = '%s/%s.log' % (self.outputPath, self.outfileBaseWithCycle)
        if os.path.isfile(logFile) and not self.clobber:
            raise RuntimeError("Found logFile %s, but clobber == False." % (logFile))

        self.plotPath = None
        if self.doPlots:
            self.plotPath = '%s/%s_plots' % (self.outputPath,self.outfileBaseWithCycle)
            if os.path.isdir(self.plotPath) and not self.clobber:
                # check if directory is empty
                if len(os.listdir(self.plotPath)) > 0:
                    raise RuntimeError("Found plots in %s, but clobber == False." % (self.plotPath))

        # set up logger are we get the name...
        if ('logger' not in configDict):
            self.externalLogger = False
            self.fgcmLog = FgcmLogger('%s/%s.log' % (self.outputPath,
                                                     self.outfileBaseWithCycle),
                                      self.logLevel, printLogger=configDict['printOnly'])
            if configDict['printOnly']:
                self.fgcmLog.info('Logging to console')
            else:
                self.fgcmLog.info('Logging started to %s' % (self.fgcmLog.logFile))
        else:
            # Support an external logger such as LSST that has .info() and .debug() calls
            self.externalLogger = True
            self.fgcmLog = configDict['logger']
            try:
                if not self.quietMode:
                    self.fgcmLog.info('Logging to external logger.')
            except:
                raise RuntimeError("Logging to configDict['logger'] failed.")

        if (self.experimentalMode) :
            self.fgcmLog.info('ExperimentalMode set to True')
        if (self.resetParameters) :
            self.fgcmLog.info('Will reset atmosphere parameters')
        if (self.noChromaticCorrections) :
            self.fgcmLog.info('WARNING: No chromatic corrections will be applied.  I hope this is what you wanted for a test!')

        if (self.plotPath is not None and not os.path.isdir(self.plotPath)):
            try:
                os.makedirs(self.plotPath)
            except:
                raise IOError("Could not create plot path: %s" % (self.plotPath))

        if (self.illegalValue >= 0.0):
            raise ValueError("Must set illegalValue to a negative number")

        # and look at the lutFile
        self.nCCD = lutIndex['NCCD'][0]
        # these are np arrays and encoded as such
        self.lutFilterNames = [n.decode('utf-8') for n in lutIndex['FILTERNAMES'][0]]
        self.lutStdFilterNames = [n.decode('utf-8') for n in lutIndex['STDFILTERNAMES'][0]]

        self.pmbRange = np.array([np.min(lutIndex['PMB']),np.max(lutIndex['PMB'])])
        self.pwvRange = np.array([np.min(lutIndex['PWV']),np.max(lutIndex['PWV'])])
        self.O3Range = np.array([np.min(lutIndex['O3']),np.max(lutIndex['O3'])])
        self.tauRange = np.array([np.min(lutIndex['TAU']),np.max(lutIndex['TAU'])])
        self.alphaRange = np.array([np.min(lutIndex['ALPHA']),np.max(lutIndex['ALPHA'])])
        self.zenithRange = np.array([np.min(lutIndex['ZENITH']),np.max(lutIndex['ZENITH'])])

        # newer band checks
        #  1) check that all the filters in filterToBand are in lutFilterNames
        #  2) check that all the lutStdFilterNames are lutFilterNames (redundant)
        #  3) check that each band has ONE standard filter
        #  4) check that all the fitBands are in bands
        #  5) check that all the notFitBands are in bands
        #  6) check that all the requiredBands are in bands

        #  1) check that all the filters in filterToBand are in lutFilterNames
        for filterName in self.filterToBand:
            if filterName not in self.lutFilterNames:
                raise ValueError("Filter %s in filterToBand not in LUT" % (filterName))
        #  2) check that all the lutStdFilterNames are lutFilterNames (redundant)
        for lutStdFilterName in self.lutStdFilterNames:
            if lutStdFilterName not in self.lutFilterNames:
                raise ValueError("lutStdFilterName %s not in list of lutFilterNames" % (lutStdFilterName))
        #  3) check that each band has ONE standard filter
        bandStdFilterIndex = np.zeros(len(self.bands), dtype=np.int32) - 1
        for i, band in enumerate(self.bands):
            for j, filterName in enumerate(self.lutFilterNames):
                # Not every LUT filter must be in the filterToBand mapping.
                # If it is not there, it will not be used.
                if filterName in self.filterToBand:
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
            if fitBand not in self.bands:
                raise ValueError("Band %s from fitBands not in full bands" % (fitBand))
        #  5) check that all the notFitBands are in bands
        for notFitBand in self.notFitBands:
            if notFitBand not in self.bands:
                raise ValueError("Band %s from notFitBands not in full bands" % (notFitBand))
        #  6) check that all the requiredBands are in bands
        for requiredBand in self.requiredBands:
            if requiredBand not in self.bands:
                raise ValueError("Band %s from requiredBands not in full bands" % (requiredBand))

        bandString = " ".join(self.bands)
        self.fgcmLog.info('Found %d CCDs and %d bands (%s)' %
                         (self.nCCD,len(self.bands),bandString))

        # get LUT standard values
        self.pmbStd = lutStd['PMBSTD'][0]
        self.pwvStd = lutStd['PWVSTD'][0]
        self.lnPwvStd = np.log(lutStd['PWVSTD'][0])
        self.o3Std = lutStd['O3STD'][0]
        self.tauStd = lutStd['TAUSTD'][0]
        self.lnTauStd = np.log(lutStd['TAUSTD'][0])
        self.alphaStd = lutStd['ALPHASTD'][0]
        self.zenithStd = lutStd['ZENITHSTD'][0]

        # Cut the LUT filter names to those that are actually used
        usedFilterNames = self.filterToBand.keys()
        usedLutFilterMark = np.zeros(len(self.lutFilterNames), dtype=np.bool)
        for i, f in enumerate(self.lutFilterNames):
            if f in usedFilterNames:
                usedLutFilterMark[i] = True

        self.lutFilterNames = [f for i, f in enumerate(self.lutFilterNames) if usedLutFilterMark[i]]
        self.lutStdFilterNames = [f for i, f in enumerate(self.lutStdFilterNames) if usedLutFilterMark[i]]

        # And the lambdaStd and I10Std, for each *band*
        self.lambdaStdBand = lutStd['LAMBDASTD'][0][bandStdFilterIndex]
        self.I10StdBand = lutStd['I10STD'][0][bandStdFilterIndex]
        self.I0StdBand = lutStd['I0STD'][0][bandStdFilterIndex]
        self.I1StdBand = lutStd['I1STD'][0][bandStdFilterIndex]
        self.I2StdBand = lutStd['I2STD'][0][bandStdFilterIndex]
        self.lambdaStdFilter = lutStd['LAMBDASTDFILTER'][0][usedLutFilterMark]

        if (self.expGrayPhotometricCut.size != len(self.bands)):
            raise ValueError("expGrayPhotometricCut must have same number of elements as bands.")
        if (self.expGrayHighCut.size != len(self.bands)):
            raise ValueError("expGrayHighCut must have same number of elements as bands.")
        if (self.expGrayPhotometricCut.max() >= 0.0):
            raise ValueError("expGrayPhotometricCut must all be negative")
        if (self.expGrayHighCut.max() <= 0.0):
            raise ValueError("expGrayHighCut must all be positive")

        if len(self.sigmaCalRange) != 2:
            raise ValueError("sigmaCalRange must have 2 elements")
        if len(self.sigmaCalFitPercentile) != 2:
            raise ValueError("sigmaCalFitPercentile must have 2 elements")
        if len(self.sigmaCalPlotPercentile) != 2:
            raise ValueError("sigmaCalPlotPercentile must have 2 elements")

        if self.sigmaCalRange[1] < self.sigmaCalRange[0]:
            raise ValueError("sigmaCalRange[1] must me equal to or larger than sigmaCalRange[0]")

        if len(self.useRepeatabilityForExpGrayCuts) != 1 and \
                len(self.useRepeatabilityForExpGrayCuts) != len(self.bands):
            raise ValueError("useRepeatabilityForExpGrayCuts must be of length 1 or number of bands")
        # Expand into the full list if necessary
        if len(self.useRepeatabilityForExpGrayCuts) == 1:
            self.useRepeatabilityForExpGrayCuts = self.useRepeatabilityForExpGrayCuts * len(self.bands)

        if len(self.sigFgcmMaxEGray) != 1 and \
                len(self.sigFgcmMaxEGray) != len(self.bands):
            raise ValueError("sigFgcmMaxEGray must be of length 1 or number of bands")
        # Expand into the full list if necessary
        if len(self.sigFgcmMaxEGray) == 1:
            self.sigFgcmMaxEGray = self.sigFgcmMaxEGray * len(self.bands)

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

        if self.epochNames is None:
            self.epochNames = []
            for i in range(self.epochMJDs.size):
                self.epochNames.append('epoch%d' % (i))

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

        # and the coating MJDs
        st = np.argsort(self.coatingMJDs)
        if (not np.array_equal(st, np.arange(self.coatingMJDs.size))):
            raise ValueError("Input coatingMJDs must be in sort order.")

        gd, = np.where((self.coatingMJDs > self.mjdRange[0]) &
                       (self.coatingMJDs < self.mjdRange[1]))
        self.coatingMJDs = self.coatingMJDs[gd]

        # Deal with fit band, notfit band, required, and notrequired indices
        bandFitFlag = np.zeros(len(self.bands), dtype=np.bool)
        bandNotFitFlag = np.zeros_like(bandFitFlag)
        bandRequiredFlag = np.zeros_like(bandFitFlag)

        for i, band in enumerate(self.bands):
            if band in self.fitBands:
                bandFitFlag[i] = True
            if band in self.requiredBands:
                bandRequiredFlag[i] = True
            if len(self.notFitBands) > 0:
                if band in self.notFitBands:
                    bandNotFitFlag[i] = True
                if band in self.fitBands and band in self.notFitBands:
                    raise ValueError("Cannot have the same band in fitBands and notFitBands")

        self.bandFitIndex = np.where(bandFitFlag)[0]
        self.bandNotFitIndex = np.where(bandNotFitFlag)[0]
        self.bandRequiredIndex = np.where(bandRequiredFlag)[0]
        self.bandNotRequiredIndex = np.where(~bandRequiredFlag)[0]

        if np.array_equal(self.bandFitIndex, self.bandRequiredIndex):
            self.allFitBandsAreRequired = True
        else:
            self.allFitBandsAreRequired = False

        # and check the star color cuts and replace with indices...
        #  note that self.starColorCuts is a copy so that we don't overwrite.

        for cCut in self.starColorCuts:
            if (not isinstance(cCut[0],int)) :
                if (cCut[0] not in self.bands):
                    raise ValueError("starColorCut band %s not in list of bands!" % (cCut[0]))
                cCut[0] = list(self.bands).index(cCut[0])
            if (not isinstance(cCut[1],int)) :
                if (cCut[1] not in self.bands):
                    raise ValueError("starColorCut band %s not in list of bands!" % (cCut[1]))
                cCut[1] = list(self.bands).index(cCut[1])

        # Check for input aperture corrections.
        if self.aperCorrFitNBins == 0:
            if len(self.aperCorrInputSlopes) == 0:
                self.fgcmLog.warn("Aperture corrections will not be fit; strongly recommend setting aperCorrInputSlopes")
            else:
                if len(self.aperCorrInputSlopes) != len(self.bands):
                    raise RuntimeError("Length of aperCorrInputSlopes does not equal number of bands!")

        # Check the sed mapping dictionaries
        # First, make sure every band is listed in the sedTermDict
        for band in self.bands:
            if band not in self.sedTermDict:
                raise RuntimeError("Band %s not listed in sedTermDict." % (band))

        # Second, make sure sedBoundaryTermDict is correct format
        for boundaryTermName, boundaryTerm in self.sedBoundaryTermDict.items():
            if 'primary' not in boundaryTerm or 'secondary' not in boundaryTerm:
                raise RuntimeError("sedBoundaryTerm %s must have primary and secondary keys." % (boundaryTerm))
            if boundaryTerm['primary'] not in self.bands:
                raise RuntimeError("sedBoundaryTerm %s band %s not in list of bands." %
                                   (boundaryTermName, boundaryTerm['primary']))
            if boundaryTerm['secondary'] not in self.bands:
                raise RuntimeError("sedBoundaryTerm %s band %s not in list of bands." %
                                   (boundaryTermName, boundaryTerm['secondary']))

        # Third, extract all the terms and bands from sedTermDict, make sure all
        # are defined.
        mapBands = []
        mapTerms = []
        for band in self.sedTermDict:
            sedTerm = self.sedTermDict[band]
            if 'extrapolated' not in sedTerm:
                raise RuntimeError("sedTermDict %s must have 'extrapolated' key." % (band))
            if 'constant' not in sedTerm:
                raise RuntimeError("sedTermDict %s must have 'constant' key." % (band))
            if 'primaryTerm' not in sedTerm:
                raise RuntimeError("sedTermDict %s must have a primaryTerm." % (band))
            if 'secondaryTerm' not in sedTerm:
                raise RuntimeError("sedTermDict %s must have a secondaryTerm." % (band))
            mapTerms.append(sedTerm['primaryTerm'])
            if sedTerm['secondaryTerm'] is not None:
                mapTerms.append(sedTerm['secondaryTerm'])
            if sedTerm['extrapolated']:
                if sedTerm['secondaryTerm'] is None:
                    raise RuntimeError("sedTermDict %s must have a secondaryTerm if extrapolated." % (band))
                if 'primaryBand' not in sedTerm:
                    raise RuntimeError("sedTermDict %s must have a primaryBand if extrapolated." % (band))
                if 'secondaryBand' not in sedTerm:
                    raise RuntimeError("sedTermDict %s must have a secondaryBand if extrapolated." % (band))
                if 'tertiaryBand' not in sedTerm:
                    raise RuntimeError("sedTermDict %s must have a tertiaryBand if extrapolated." % (band))
                mapBands.append(sedTerm['primaryBand'])
                mapBands.append(sedTerm['secondaryBand'])
                mapBands.append(sedTerm['tertiaryBand'])

        for mapTerm in mapTerms:
            if mapTerm not in self.sedBoundaryTermDict:
                raise RuntimeError("Term %s is used in sedTermDict but not in sedBoundaryTermDict" % (mapTerm))
        for mapBand in mapBands:
            if mapBand not in self.bands:
                raise RuntimeError("Band %s is used in sedTermDict but not in bands" % (mapBand))

        # and AB zeropoint
        self.hPlanck = 6.6
        self.expPlanck = -27.0
        self.zptABNoThroughput = (-48.6 - 2.5 * self.expPlanck +
                                   2.5 * np.log10(self.mirrorArea) -
                                   2.5 * np.log10(self.hPlanck * self.cameraGain))

        if len(self.approxThroughput) != 1 and len(self.approxThroughput) != len(self.bands):
            raise ValueError("approxThroughput must have 1 or nbands elements.")

        self.fgcmLog.info("AB offset (w/o throughput) estimated as %.4f" % (self.zptABNoThroughput))

        self.configDictSaved = configDict
        ## FIXME: add pmb scaling?

    def updateCycleNumber(self, newCycleNumber):
        """
        Update the cycle number for re-use of config.

        Parameters
        ----------
        newCycleNumber: `int`
        """

        self.cycleNumber = newCycleNumber

        self.outfileBaseWithCycle = '%s_cycle%02d' % (self.outfileBase, self.cycleNumber)

        logFile = '%s/%s.log' % (self.outputPath, self.outfileBaseWithCycle)
        if os.path.isfile(logFile) and not self.clobber:
            raise RuntimeError("Found logFile %s, but clobber == False." % (logFile))

        self.plotPath = None
        if self.doPlots:
            self.plotPath = '%s/%s_plots' % (self.outputPath,self.outfileBaseWithCycle)
            if os.path.isdir(self.plotPath) and not self.clobber:
                # check if directory is empty
                if len(os.listdir(self.plotPath)) > 0:
                    raise RuntimeError("Found plots in %s, but clobber == False." % (self.plotPath))

        if not self.externalLogger:
            self.fgcmLog = FgcmLogger('%s/%s.log' % (self.outputPath,
                                                     self.outfileBaseWithCycle),
                                      self.logLevel, printLogger=configDict['printOnly'])

        if (self.plotPath is not None and not os.path.isdir(self.plotPath)):
            try:
                os.makedirs(self.plotPath)
            except:
                raise IOError("Could not create plot path: %s" % (self.plotPath))

    @staticmethod
    def _readConfigDict(configFile):
        """
        Internal method to read a configuration dictionary from a yaml file.
        """

        with open(configFile) as f:
            configDict = yaml.load(f, Loader=yaml.SafeLoader)

        print("Configuration read from %s" % (configFile))

        return configDict

    @classmethod
    def configWithFits(cls, configDict, noOutput=False):
        """
        Initialize FgcmConfig object and read in fits files.

        parameters
        ----------
        configDict: dict
           Dictionary with config variables.
        noOutput: bool, default=False
           Do not create output directory.
        """

        import fitsio

        expInfo = fitsio.read(configDict['exposureFile'], ext=1)

        try:
            lutIndex = fitsio.read(configDict['lutFile'], ext='INDEX')
            lutStd = fitsio.read(configDict['lutFile'], ext='STD')
        except:
            raise IOError("Could not read LUT info")

        ccdOffsets = fitsio.read(configDict['ccdOffsetFile'], ext=1)

        return cls(configDict, lutIndex, lutStd, expInfo, ccdOffsets, checkFiles=True, noOutput=noOutput)


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

        # And update the photometric cuts...
        # These need to be converted to lists of floats
        configDict['expGrayPhotometricCut'] = [float(f) for f in self.expGrayPhotometricCut]
        configDict['expGrayHighCut'] = [float(f) for f in self.expGrayHighCut]

        with open(fileName,'w') as f:
            yaml.dump(configDict, stream=f)

    def _setVarsFromDict(self, d):
        for key in d:
            if key not in type(self).__dict__:
                raise AttributeError("Unknown config variable: %s" % (key))
            setattr(self, key, d[key])

    def validate(self):
        """
        """

        for var in type(self).__dict__:
            try:
                type(self).__dict__[var].validate(var)
            except AttributeError:
                pass

    def _setDefaultLengths(self):
        """
        """

        # Check the fudge factors...
        type(self).__dict__['sedTermDict']._length = len(self.bands)

        # And the gray cuts
        type(self).__dict__['expGrayPhotometricCut']._length = len(self.bands)
        type(self).__dict__['expGrayHighCut']._length = len(self.bands)

        # And the epoch names
        type(self).__dict__['epochNames']._length = len(self.epochMJDs)
