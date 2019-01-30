from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import os
import sys
import esutil

import matplotlib.pyplot as plt

from .fgcmUtilities import expFlagDict
from .fgcmUtilities import retrievalFlagDict

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmParameters(object):
    """
    Class to contain FGCM parameters.  Initialization should be done via:
      newParsWithFits()
      newParsWithArrays()
      loadParsWithFits()
      loadParsWithArrays()

    parameters
    ----------
    fgcmConfig: FgcmConfig
       Config object
    expInfo: numpy recarray, required if New parameters
       Exposure info table
    fgcmLUT: FgcmLUT, required if New parameters
    inParInfo: numpy recarray, required if loading parameters
    inParams: numpy recarray, required if loading parameters
    inSuperStar: numpy array, required if loading parameters

    Config variables
    ----------------
    minExpPerNight: int
       Minumum number of exposures in a night for plotting
    freezeStdAtmosphere: bool
       Fit atmosphere parameters or freeze at standard values? (good for 0th cycle)
    epochMJDs: double array
       MJDs which divide observing epochs
    washMJDs: double array
       MJDs which denote mirror washing dates
    useRetrievedPwv: bool
       Use Pwv retrieved from colors from previous cycle?
    useNightlyRetrievedPwv: bool
       Re-fit offsets for each night Pwv variation (if useRetrievedPwv==True)?
    useRetrievedTauInit: bool
       Use nightly retrieved tau from previous cycle as initial guess? (experimental)
    """

    def __init__(self, fgcmConfig, expInfo=None, fgcmLUT=None,
                 inParInfo=None, inParams=None, inSuperStar=None):

        initNew = False
        loadOld = False
        if (expInfo is not None and fgcmLUT is not None):
            initNew = True
        if (inParInfo is not None and inParams is not None and inSuperStar is not None):
            loadOld = True

        if (initNew and loadOld):
            raise ValueError("Too many parameters specified: either expInfo/fgcmLUT or inParInof/inParams/inSuperStar")
        if (not initNew and not loadOld):
            raise ValueError("Too few parameters specificed: either expInfo/fgcmLUT or inParInof/inParams/inSuperStar")

        self.hasExternalPwv = False
        self.hasExternalTau = False

        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.plotPath = fgcmConfig.plotPath

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.info('Initializing FgcmParameters...')

        # for plotting
        self.minExpPerNight = fgcmConfig.minExpPerNight

        # get stuff from config file
        self.nCCD = fgcmConfig.nCCD
        self.bands = fgcmConfig.bands
        self.nBands = len(self.bands)
        self.fitBands = fgcmConfig.fitBands
        self.nFitBands = len(self.fitBands)
        self.notFitBands = fgcmConfig.notFitBands
        self.nNotFitBands = len(self.notFitBands)

        self.bandFitIndex = fgcmConfig.bandFitIndex
        self.bandNotFitIndex = fgcmConfig.bandNotFitIndex

        self.lutFilterNames = fgcmConfig.lutFilterNames
        self.lutStdFilterNames = fgcmConfig.lutStdFilterNames
        self.nLUTFilter = len(self.lutFilterNames)
        self.filterToBand = fgcmConfig.filterToBand
        self.lambdaStdFilter = fgcmConfig.lambdaStdFilter
        self.lambdaStdBand = fgcmConfig.lambdaStdBand

        self.freezeStdAtmosphere = fgcmConfig.freezeStdAtmosphere
        self.alphaStd = fgcmConfig.alphaStd
        self.o3Std = fgcmConfig.o3Std
        self.tauStd = fgcmConfig.tauStd
        self.lnTauStd = fgcmConfig.lnTauStd
        self.pwvStd = fgcmConfig.pwvStd
        self.lnPwvStd = fgcmConfig.lnPwvStd
        self.pmbStd = fgcmConfig.pmbStd
        self.zenithStd = fgcmConfig.zenithStd
        self.secZenithStd = 1./np.cos(self.zenithStd*np.pi/180.)

        self.pmbRange = fgcmConfig.pmbRange
        self.pwvRange = fgcmConfig.pwvRange
        self.lnPwvRange = np.log(self.pwvRange)
        self.O3Range = fgcmConfig.O3Range
        self.tauRange = fgcmConfig.tauRange
        self.lnTauRange = np.log(self.tauRange)
        self.alphaRange = fgcmConfig.alphaRange
        self.zenithRange = fgcmConfig.zenithRange

        self.nExp = fgcmConfig.nExp
        self.seeingField = fgcmConfig.seeingField
        self.seeingSubExposure = fgcmConfig.seeingSubExposure
        self.deepFlag = fgcmConfig.deepFlag
        self.fwhmField = fgcmConfig.fwhmField
        self.skyBrightnessField = fgcmConfig.skyBrightnessField
        self.expField = fgcmConfig.expField
        self.UTBoundary = fgcmConfig.UTBoundary
        self.latitude = fgcmConfig.latitude
        self.sinLatitude = np.sin(np.radians(self.latitude))
        self.cosLatitude = np.cos(np.radians(self.latitude))

        self.epochMJDs = fgcmConfig.epochMJDs
        self.washMJDs = fgcmConfig.washMJDs

        self.stepUnitReference = fgcmConfig.stepUnitReference
        self.stepGrain = fgcmConfig.stepGrain

        self.pwvFile = fgcmConfig.pwvFile
        self.tauFile = fgcmConfig.tauFile
        self.externalPwvDeltaT = fgcmConfig.externalPwvDeltaT
        self.externalTauDeltaT = fgcmConfig.externalTauDeltaT
        self.useRetrievedPwv = fgcmConfig.useRetrievedPwv
        self.useNightlyRetrievedPwv = fgcmConfig.useNightlyRetrievedPwv
        self.useQuadraticPwv = fgcmConfig.useQuadraticPwv
        self.useRetrievedTauInit = fgcmConfig.useRetrievedTauInit
        self.modelMagErrors = fgcmConfig.modelMagErrors

        self.superStarNPar = ((fgcmConfig.superStarSubCCDChebyshevOrder + 1) *
                              (fgcmConfig.superStarSubCCDChebyshevOrder + 1))
        self.ccdOffsets = fgcmConfig.ccdOffsets
        self.superStarSubCCD = fgcmConfig.superStarSubCCD
        self.superStarSubCCDTriangular = fgcmConfig.superStarSubCCDTriangular

        # and the default unit dict
        self.unitDictOnes = {'lnPwvUnit':1.0,
                             'lnPwvSlopeUnit':1.0,
                             'lnPwvQuadraticUnit':1.0,
                             'lnPwvGlobalUnit':1.0,
                             'o3Unit':1.0,
                             'lnTauUnit':1.0,
                             'lnTauSlopeUnit':1.0,
                             'alphaUnit':1.0,
                             'qeSysUnit':1.0,
                             'qeSysSlopeUnit':1.0,
                             'filterOffsetUnit':1.0,
                             'absOffsetUnit':1.0}

        if (initNew):
            self._initializeNewParameters(expInfo, fgcmLUT)
        else:
            self._loadOldParameters(expInfo, inParInfo, inParams, inSuperStar)

    @classmethod
    def newParsWithFits(cls, fgcmConfig, fgcmLUT):
        """
        Make a new FgcmParameters object, loading from fits.

        parameters
        ----------
        fgcmConfig: FgcmConfig
        fgcmLUT: fgcmLUT

        Config variables
        ----------------
        exposureFile: string
           File with exposure information
        """

        import fitsio

        expInfoFile = fgcmConfig.exposureFile

        expInfo = fitsio.read(expInfoFile, ext=1)
        return cls(fgcmConfig, expInfo=expInfo, fgcmLUT=fgcmLUT)

    @classmethod
    def newParsWithArrays(cls, fgcmConfig, fgcmLUT, expInfo):
        """
        Make a new FgcmParameters object, with input arrays

        parameters
        ----------
        fgcmConfig: FgcmConfig
        fgcmLUT: FgcmLUT
        expInfo: numpy recarray
           Exposure info
        """

        return cls(fgcmConfig, expInfo=expInfo, fgcmLUT=fgcmLUT)

    @classmethod
    def loadParsWithFits(cls, fgcmConfig):
        """
        Make an FgcmParameters object, loading from old parameters in fits

        parameters
        ----------
        fgcmConfig: FgcmConfig

        Config variables
        ----------------
        exposureFile: string
           File with exposure information
        inParameterFile: string
           File with input parameters (from previous cycle)
        """

        import fitsio

        expInfoFile = fgcmConfig.exposureFile
        inParFile = fgcmConfig.inParameterFile

        expInfo = fitsio.read(expInfoFile, ext=1)
        inParInfo = fitsio.read(inParFile, ext='PARINFO')
        inParams = fitsio.read(inParFile, ext='PARAMS')
        inSuperStar = fitsio.read(inParFile, ext='SUPER')

        return cls(fgcmConfig, expInfo=expInfo,
                   inParInfo=inParInfo, inParams=inParams, inSuperStar=inSuperStar)

    @classmethod
    def loadParsWithArrays(cls, fgcmConfig, expInfo, inParInfo, inParams, inSuperStar):
        """
        Make an FgcmParameters object, loading from old parameters in arrays.

        parameters
        ----------
        fgcmConfig: FgcmConfig
        expInfo: numpy recarray
           Exposure info
        inParInfo: numpy recarray
           Input parameter information array
        inParams: numpy recarray
           Input parameters
        inSuperStar: numpy array
           Input superstar
        """

        return cls(fgcmConfig, expInfo=expInfo,
                   inParInfo=inParInfo, inParams=inParams, inSuperStar=inSuperStar)

    def _initializeNewParameters(self, expInfo, fgcmLUT):
        """
        Internal method to initialize new parameters

        parameters
        ----------
        expInfo: numpy recarrat
        fgcmLUT: FgcmLUT
        """

        # link band indices
        # self._makeBandIndices()

        # load the exposure information
        self._loadExposureInfo(expInfo)

        # load observing epochs and link indices
        self._loadEpochAndWashInfo()

        # and make the new parameter arrays
        self.parAlpha = np.zeros(self.campaignNights.size,dtype=np.float32) + fgcmLUT.alphaStd
        self.parO3 = np.zeros(self.campaignNights.size,dtype=np.float32) + fgcmLUT.o3Std
        self.parLnTauIntercept = np.zeros(self.campaignNights.size,dtype=np.float32) + fgcmLUT.lnTauStd
        self.parLnTauSlope = np.zeros(self.campaignNights.size,dtype=np.float32)
        # these we will always have, won't always fit
        self.parLnPwvIntercept = np.zeros(self.campaignNights.size, dtype=np.float32) + fgcmLUT.lnPwvStd
        self.parLnPwvSlope = np.zeros(self.campaignNights.size, dtype=np.float32)
        self.parLnPwvQuadratic = np.zeros(self.campaignNights.size, dtype=np.float32)

        # parameters with per-epoch values
        self.parSuperStarFlat = np.zeros((self.nEpochs,self.nLUTFilter,self.nCCD,self.superStarNPar),dtype=np.float64)
        # The first term should be 1.0 with new flux units
        self.parSuperStarFlat[:, :, :, 0] = 1.0

        # parameters with per-wash values
        self.parQESysIntercept = np.zeros(self.nWashIntervals,dtype=np.float32)
        self.parQESysSlope = np.zeros(self.nWashIntervals,dtype=np.float32)

        # parameters for "absolute" offsets (and relative between filters)
        # Currently, this only will turn on for when there are multiple filters
        # for the same band.  In the future we can add "primary absolute calibrators"
        # to the fit, and turn these on.
        self.parFilterOffset = np.zeros(self.nLUTFilter, dtype=np.float64)
        self.parFilterOffsetFitFlag = np.zeros(self.nLUTFilter, dtype=np.bool)
        for i, f in enumerate(self.lutFilterNames):
            band = self.filterToBand[f]
            nBand = 0
            for ff in self.filterToBand:
                if self.filterToBand[ff] == band:
                    nBand += 1
            # And when there is a duplicate band and it's not the "Standard", fit the offset
            if nBand > 1 and f not in self.lutStdFilterNames:
                self.parFilterOffsetFitFlag[i] = True

        # And absolute offset parameters (used if reference mags are supplied)
        self.parAbsOffset = np.zeros(self.nBands, dtype=np.float64)

        ## FIXME: need to completely refactor
        self.externalPwvFlag = np.zeros(self.nExp,dtype=np.bool)
        if (self.pwvFile is not None):
            self.fgcmLog.info('Found external PWV file.')
            self.pwvFile = self.pwvFile
            self.hasExternalPwv = True
            self.loadExternalPwv(self.externalPwvDeltaT)

        self.externalTauFlag = np.zeros(self.nExp,dtype=np.bool)
        if (self.tauFile is not None):
            self.fgcmLog.info('Found external tau file.')
            self.tauFile = self.tauFile
            self.hasExternalTau = True
            self.loadExternalTau()

        # and the aperture corrections
        ## FIXME: should these be per band or filter??
        self.compAperCorrPivot = np.zeros(self.nBands,dtype='f8')
        self.compAperCorrSlope = np.zeros(self.nBands,dtype='f8')
        self.compAperCorrSlopeErr = np.zeros(self.nBands,dtype='f8')
        self.compAperCorrRange = np.zeros((2,self.nBands),dtype='f8')

        # The magnitude model parameters
        self.compModelErrExptimePivot = np.zeros(self.nBands, dtype='f8')
        self.compModelErrFwhmPivot = np.zeros(self.nBands, dtype='f8')
        self.compModelErrSkyPivot = np.zeros(self.nBands, dtype='f8')
        self.compModelErrPars = np.zeros((7, self.nBands), dtype='f8')

        # one of the "parameters" is expGray
        self.compExpGray = np.zeros(self.nExp,dtype='f8')
        self.compVarGray = np.zeros(self.nExp,dtype='f8')
        self.compNGoodStarPerExp = np.zeros(self.nExp,dtype='i4')

        # and sigFgcm
        self.compSigFgcm = np.zeros(self.nBands,dtype='f8')
        self.compSigmaCal = np.zeros(self.nBands, dtype='f8')

        # and the computed retrieved Pwv
        # these are set to the standard values to start
        self.compRetrievedLnPwv = np.zeros(self.nExp,dtype='f8') + self.lnPwvStd
        self.compRetrievedLnPwvInput = self.compRetrievedLnPwv.copy()
        self.compRetrievedLnPwvRaw = np.zeros(self.nExp,dtype='f8')
        self.compRetrievedLnPwvFlag = np.zeros(self.nExp,dtype='i2') + retrievalFlagDict['EXPOSURE_STANDARD']
        self.parRetrievedLnPwvScale = 1.0
        self.parRetrievedLnPwvOffset = 0.0
        self.parRetrievedLnPwvNightlyOffset = np.zeros(self.nCampaignNights,dtype='f8')

        # and retrieved tau nightly start values
        self.compRetrievedTauNight = np.zeros(self.campaignNights.size,dtype='f8') + self.tauStd
        self.compRetrievedTauNightInput = self.compRetrievedTauNight.copy()

        # compute the units
        self.unitDictSteps = fgcmLUT.computeStepUnits(self.stepUnitReference,
                                                      self.stepGrain,
                                                      self.meanNightDuration,
                                                      self.meanWashIntervalDuration,
                                                      self.fitBands,
                                                      self.bands,
                                                      self.nCampaignNights)

        # do lookups on parameter array
        self._arrangeParArray()

        # and we're done

    def _loadOldParameters(self, expInfo, inParInfo, inParams, inSuperStar):
        """
        Internal method to load old parameters

        parameters
        ----------
        expInfo: numpy recarray
        inParInfo: numpy recarray
        inParams: numpy recarray
        inSuperStar: numpy recarray
        """

        # link band indices
        # self._makeBandIndices()
        self._loadExposureInfo(expInfo)

        self._loadEpochAndWashInfo()

        # set the units from the inParInfo
        self.unitDictSteps = {'lnTauUnit': inParInfo['LNTAUUNIT'][0],
                              'lnTauSlopeUnit': inParInfo['LNTAUSLOPEUNIT'][0],
                              'alphaUnit': inParInfo['ALPHAUNIT'][0],
                              'lnPwvUnit': inParInfo['LNPWVUNIT'][0],
                              'lnPwvSlopeUnit': inParInfo['LNPWVSLOPEUNIT'][0],
                              'lnPwvQuadraticUnit': inParInfo['LNPWVQUADRATICUNIT'][0],
                              'lnPwvGlobalUnit': inParInfo['LNPWVGLOBALUNIT'][0],
                              'o3Unit': inParInfo['O3UNIT'][0],
                              'qeSysUnit': inParInfo['QESYSUNIT'][0],
                              'qeSysSlopeUnit': inParInfo['QESYSSLOPEUNIT'][0],
                              'filterOffsetUnit': inParInfo['FILTEROFFSETUNIT'][0],
                              'absOffsetUnit': inParInfo['ABSOFFSETUNIT'][0]}

        # and log
        self.fgcmLog.info('lnTau step unit set to %f' % (self.unitDictSteps['lnTauUnit']))
        self.fgcmLog.info('lnTau slope step unit set to %f' %
                         (self.unitDictSteps['lnTauSlopeUnit']))
        self.fgcmLog.info('alpha step unit set to %f' % (self.unitDictSteps['alphaUnit']))
        self.fgcmLog.info('lnPwv step unit set to %f' % (self.unitDictSteps['lnPwvUnit']))
        self.fgcmLog.info('lnPwv slope step unit set to %f' %
                         (self.unitDictSteps['lnPwvSlopeUnit']))
        self.fgcmLog.info('lnPwv quadratic step unit set to %f' %
                          (self.unitDictSteps['lnPwvQuadraticUnit']))
        self.fgcmLog.info('lnPwv global step unit set to %f' %
                          (self.unitDictSteps['lnPwvGlobalUnit']))
        self.fgcmLog.info('O3 step unit set to %f' % (self.unitDictSteps['o3Unit']))
        self.fgcmLog.info('wash step unit set to %f' % (self.unitDictSteps['qeSysUnit']))
        self.fgcmLog.info('wash slope step unit set to %f' %
                          (self.unitDictSteps['qeSysSlopeUnit']))
        self.fgcmLog.info('filter offset step unit set to %f' %
                          (self.unitDictSteps['filterOffsetUnit']))
        self.fgcmLog.info('abs offset step unit set to %f' %
                          (self.unitDictSteps['absOffsetUnit']))

        # look at external...
        self.hasExternalPwv = inParInfo['HASEXTERNALPWV'][0].astype(np.bool)
        self.hasExternalTau = inParInfo['HASEXTERNALTAU'][0].astype(np.bool)

        ## and copy the parameters
        self.parAlpha = np.atleast_1d(inParams['PARALPHA'][0])
        self.parO3 = np.atleast_1d(inParams['PARO3'][0])
        self.parLnTauIntercept = np.atleast_1d(inParams['PARLNTAUINTERCEPT'][0])
        self.parLnTauSlope = np.atleast_1d(inParams['PARLNTAUSLOPE'][0])
        self.parLnPwvIntercept = np.atleast_1d(inParams['PARLNPWVINTERCEPT'][0])
        self.parLnPwvSlope = np.atleast_1d(inParams['PARLNPWVSLOPE'][0])
        self.parLnPwvQuadratic = np.atleast_1d(inParams['PARLNPWVQUADRATIC'][0])
        self.parQESysIntercept = np.atleast_1d(inParams['PARQESYSINTERCEPT'][0])
        self.parQESysSlope = np.atleast_1d(inParams['PARQESYSSLOPE'][0])
        self.parFilterOffset = np.atleast_1d(inParams['PARFILTEROFFSET'][0])
        self.parFilterOffsetFitFlag = np.atleast_1d(inParams['PARFILTEROFFSETFITFLAG'][0]).astype(np.bool)
        self.parAbsOffset = np.atleast_1d(inParams['PARABSOFFSET'][0])

        self.externalPwvFlag = np.zeros(self.nExp,dtype=np.bool)
        if self.hasExternalPwv:
            self.pwvFile = str(inParInfo['PWVFILE'][0]).rstrip()
            self.hasExternalPwv = True
            self.loadExternalPwv(self.externalPwvDeltaT)
            self.parExternalLnPwvScale = inParams['PAREXTERNALLNPWVSCALE'][0]
            self.parExternalLnPwvOffset[:] = np.atleast_1d(inParams['PAREXTERNALLNPWVOFFSET'][0])

        self.externalTauFlag = np.zeros(self.nExp,dtype=np.bool)
        if self.hasExternalTau:
            self.tauFile = str(inParInfo['TAUFILE'][0]).rstrip()
            self.hasExternalTau = True
            self.loadExternalTau()
            self.parExternalLnTauScale = inParams['PAREXTERNALLNTAUSCALE'][0]
            self.parExternalLnTauOffset[:] = np.atleast_1d(inParams['PAREXTERNALLNTAUOFFSET'][0])

        self.compAperCorrPivot = np.atleast_1d(inParams['COMPAPERCORRPIVOT'][0])
        self.compAperCorrSlope = np.atleast_1d(inParams['COMPAPERCORRSLOPE'][0])
        self.compAperCorrSlopeErr = np.atleast_1d(inParams['COMPAPERCORRSLOPEERR'][0])
        self.compAperCorrRange = np.reshape(inParams['COMPAPERCORRRANGE'][0],(2,self.nBands))

        try:
            self.compModelErrExptimePivot = np.atleast_1d(inParams['COMPMODELERREXPTIMEPIVOT'][0])
            self.compModelErrFwhmPivot = np.atleast_1d(inParams['COMPMODELERRFWHMPIVOT'][0])
            self.compModelErrSkyPivot = np.atleast_1d(inParams['COMPMODELERRSKYPIVOT'][0])
            self.compModelErrPars = np.reshape(inParams['COMPMODELERRPARS'][0], (7, self.nBands))
        except ValueError:
            # This is okay, there will be no model error correction from old run
            pass

        self.compExpGray = np.atleast_1d(inParams['COMPEXPGRAY'][0])
        self.compVarGray = np.atleast_1d(inParams['COMPVARGRAY'][0])
        self.compNGoodStarPerExp = np.atleast_1d(inParams['COMPNGOODSTARPEREXP'][0])

        self.compSigFgcm = np.atleast_1d(inParams['COMPSIGFGCM'][0])
        self.compSigmaCal = np.atleast_1d(inParams['COMPSIGMACAL'][0])

        # These are exposure-level properties
        self.compRetrievedLnPwv = np.atleast_1d(inParams['COMPRETRIEVEDLNPWV'][0])
        self.compRetrievedLnPwvInput = self.compRetrievedLnPwv.copy()
        self.compRetrievedLnPwvRaw = np.atleast_1d(inParams['COMPRETRIEVEDLNPWVRAW'][0])
        self.compRetrievedLnPwvFlag = np.atleast_1d(inParams['COMPRETRIEVEDLNPWVFLAG'][0])
        self.parRetrievedLnPwvScale = inParams['PARRETRIEVEDLNPWVSCALE'][0]
        self.parRetrievedLnPwvOffset = inParams['PARRETRIEVEDLNPWVOFFSET'][0]
        self.parRetrievedLnPwvNightlyOffset = np.atleast_1d(inParams['PARRETRIEVEDLNPWVNIGHTLYOFFSET'][0])

        # These are nightly properties
        self.compRetrievedTauNight = np.atleast_1d(inParams['COMPRETRIEVEDTAUNIGHT'][0])
        self.compRetrievedTauNightInput = self.compRetrievedTauNight.copy()

        self._arrangeParArray()

        # need to load the superstarflats
        self.parSuperStarFlat = inSuperStar
        """
        # check if we have an "old-old-style" superstar for compatibility
        if (len(inSuperStar.shape) == 3):
            # we have an old-style superstar
            self.fgcmLog.info("Reading old-style superStarFlat")
            self.parSuperStarFlat = np.zeros((self.nEpochs,self.nLUTFilter,self.nCCD,self.superStarNPar),dtype=np.float32)
            self.parSuperStarFlat[:,:,:,0] = inSuperStar
        else:
            # Old and new style.
            # Check if we are reading in an older 2nd order polynomial file
            if inSuperStar.shape[-1] == self.superStarNParOld:
                self.fgcmLog.info("Old polynomial2d superstar flat detected")
                self.superStarPoly2d = True
                self.superStarNPar = self.superStarNParOld

            # Use the superstar as read in, raw
            self.parSuperStarFlat = inSuperStar
            """

    def resetAtmosphereParameters(self):
        self.fgcmLog.info("Resetting atmosphere parameters.")

        self.parAlpha[:] = self.alphaStd
        self.parO3[:] = self.o3Std
        self.parLnTauIntercept[:] = self.lnTauStd
        self.parLnTauSlope[:] = 0.0
        self.parLnPwvIntercept[:] = self.lnPwvStd
        self.parLnPwvSlope[:] = 0.0
        self.parLnPwvQuadratic[:] = 0.0

        # We don't reset QESysIntercept and Slope because they aren't
        #  atmosphere parameters (they are instrument parameters)
        # Similarly for filterOffset

        if self.hasExternalPwv:
            self.parExternalLnPwvScale = 1.0
            self.parExternalLnPwvOffset = 0.0

        if self.hasExternalTau:
            self.parExternalLnTauScale = 1.0
            self.parExternalLnTauOffset = 0.0

        self.parRetrievedLnPwvScale = 1.0
        self.parRetrievedLnPwvOffset = 0.0
        self.parRetrievedLnPwvNightlyOffset[:] = 0.0

        # If we are resetting parameters, and want to use retrieved tau as the initial
        #  guess, set parLnTauIntercept to that.
        if self.useRetrievedTauInit:
            self.parLnTauIntercept[:] = np.log(self.compRetrievedTauNight)

    def _loadExposureInfo(self, expInfo):
        """
        Internal method to load exposure info into variables.

        parameters
        ----------
        expInfo: numpy recarray
        """

        # ensure sorted by exposure number
        st=np.argsort(expInfo[self.expField])
        expInfo=expInfo[st]

        self.nExp = self.nExp

        self.fgcmLog.info('Loading info on %d exposures.' % (self.nExp))

        self.expArray = expInfo[self.expField]
        self.expFlag = np.zeros(self.nExp,dtype=np.int8)
        self.expExptime = expInfo['EXPTIME']

        # Load in the expSeeingVariable
        if len(expInfo[self.seeingField].shape) == 2:
            if expInfo[self.seeingField].shape[1] != self.nCCD:
                raise ValueError('ExpINfo %s field has the wrong number of ccds (%d != %d)' % (self.seeingField, self.expSeeingVariable.shape[1], self.nCCD))

            self.expSeeingVariablePerCCD = expInfo[self.seeingField]

            # And also compute the median per exposure
            # This is what will be used for calibration, and in case
            # the config.seeingSubExposure is set

            # In the future we can also compute a smooth fit to the FOV, that's
            # less necessary I think

            self.expSeeingVariable = np.zeros(expInfo.size)
            for i in xrange(expInfo.size):
                u = ((self.expSeeingVariablePerCCD[i, :] != 0.0) &
                     (self.expSeeingVariablePerCCD[i, :] > -100.0))
                if u.sum() >= 3:
                    self.expSeeingVariable[i] = np.median(self.expSeeingVariablePerCCD[i, u])
                    # Fill in the bad values with the median of the good ones
                    self.expSeeingVariablePerCCD[i, ~u] = self.expSeeingVariable[i]
                else:
                    self.expSeeingVariablePerCCD[i, :] = -1000.0
                    self.expSeeingVariable[i] = -1000.0

        else:
            # Regular per-exposure
            self.expSeeingVariable = expInfo[self.seeingField]

            if self.seeingSubExposure:
                raise RuntimeError("Config seeingSubExposure set to true, but no sub-exposure info is in the expInfo file.")

        if (len(self.expSeeingVariable.shape) == 2):
            if not self.seeingSubExposure:
                raise ValueError('ExpInfo has multi-dimensional %s field, but seeingSubExposure is False' % (self.seeingField))
            if self.expSeeingVariable.shape[1] != self.nCCD:
                raise ValueError('ExpINfo %s field has the wrong number of ccds (%d != %d)' % (self.seeingField, self.expSeeingVariable.shape[1], self.nCCD))

        self.expDeepFlag = expInfo[self.deepFlag]

        try:
            self.expFwhm = expInfo[self.fwhmField]
            self.expSkyBrightness = expInfo[self.skyBrightnessField]
        except KeyError:
            if self.modelMagErrors:
                raise ValueError("Must have columns for %s and %s to use modelMagErrors option" %
                                 (self.fwhmField, self.skyBrightnessField))

        # we need the nights of the survey (integer MJD, maybe rotated)
        self.expMJD = expInfo['MJD']
        mjdForNight = np.floor(self.expMJD + self.UTBoundary).astype(np.int32)
        self.campaignNights = np.unique(mjdForNight)
        self.nCampaignNights = self.campaignNights.size

        self.fgcmLog.info('Exposures taken on %d nights.' % (self.nCampaignNights))

        self.expDeltaUT = (self.expMJD + self.UTBoundary) - mjdForNight

        # and link the exposure numbers to the nights...
        a,b=esutil.numpy_util.match(self.campaignNights,mjdForNight)
        self.expNightIndex = np.zeros(self.nExp,dtype=np.int32)
        self.expNightIndex[b] = a

        # we need the duration of each night...
        self.nightDuration = np.zeros(self.nCampaignNights)
        self.maxDeltaUTPerNight = np.zeros(self.nCampaignNights)
        self.expPerNight = np.zeros(self.nCampaignNights,dtype=np.int32)
        for i in xrange(self.nCampaignNights):
            use,=np.where(mjdForNight == self.campaignNights[i])
            self.expPerNight[i] = use.size
            # night duration in days
            self.nightDuration[i] = (np.max(self.expMJD[use]) - np.min(self.expMJD[use]))
            # And the maximum deltaUT on a given night.
            self.maxDeltaUTPerNight[i] = np.max(self.expDeltaUT[use])
        self.meanNightDuration = np.mean(self.nightDuration)  # days
        self.meanExpPerNight = np.mean(self.expPerNight)

        # convert these to radians
        self.expTelHA = np.radians(expInfo['TELHA'])
        self.expTelRA = np.radians(expInfo['TELRA'])
        self.expTelDec = np.radians(expInfo['TELDEC'])

        # and rotate?  This may need to be revisited.
        hi,=np.where(self.expTelRA > np.pi)
        if hi.size > 0:
            self.expTelRA[hi] -= 2.0*np.pi

        # and get the secant of the Zenith angle
        self.sinLatitude = np.sin(np.radians(self.latitude))
        self.cosLatitude = np.cos(np.radians(self.latitude))

        self.expPmb = expInfo['PMB']

        # link exposures to bands
        self.expBandIndex = np.zeros(self.nExp,dtype='i2') - 1
        self.expLUTFilterIndex = np.zeros(self.nExp,dtype='i2') - 1
        expFilterName = np.core.defchararray.strip(expInfo['FILTERNAME'])
        for filterIndex,filterName in enumerate(self.lutFilterNames):
            try:
                bandIndex = self.bands.index(self.filterToBand[filterName])
            except Exception as inst:
                print(inst)
                self.fgcmLog.info('WARNING: exposures with filter %s not in config' % (filterName))
                continue

            # note that for Py3 we need to encode filterName to match to the numpy array
            use,=np.where(expFilterName == filterName.encode('utf-8'))
            if use.size == 0:
                self.fgcmLog.info('WARNING: no exposures in filter %s' % (filterName))
            else:
                self.expBandIndex[use] = bandIndex
                self.expLUTFilterIndex[use] = filterIndex

        bad,=np.where(self.expLUTFilterIndex < 0)
        if (bad.size > 0):
            self.fgcmLog.info('***Warning: %d exposures with band not in LUT!' % (bad.size))
            self.expFlag[bad] = self.expFlag[bad] | expFlagDict['BAND_NOT_IN_LUT']

        # Flag exposures that are not in the fit bands
        self.expNotFitBandFlag = np.zeros(self.nExp, dtype=np.bool)
        if self.nNotFitBands > 0:
            a, b = esutil.numpy_util.match(self.bandNotFitIndex, self.expBandIndex)
            self.expNotFitBandFlag[b] = True

        # set up the observing epochs and link indices

        # the epochs should contain all the MJDs.
        self.nEpochs = self.epochMJDs.size - 1

        self.expEpochIndex = np.zeros(self.nExp,dtype='i4')
        for i in xrange(self.nEpochs):
            use,=np.where((self.expMJD > self.epochMJDs[i]) &
                          (self.expMJD < self.epochMJDs[i+1]))
            self.expEpochIndex[use] = i

    def _loadEpochAndWashInfo(self):
        """
        Internal method to reformat epoch and wash info.
        """

        # the epochs should contain all the MJDs.
        self.nEpochs = self.epochMJDs.size - 1

        self.expEpochIndex = np.zeros(self.nExp,dtype='i4')
        for i in xrange(self.nEpochs):
            use,=np.where((self.expMJD > self.epochMJDs[i]) &
                          (self.expMJD < self.epochMJDs[i+1]))
            self.expEpochIndex[use] = i

        # and set up the wash mjds and link indices
        # the first "washMJD" is set to the first exposure date.
        # the number of *intervals* is one less than the dates?

        ## FIXME: should this happen in fgcmConfig?  But I don't want that
        # to have to have all the info...hmmm.  More refactoring!
        self.nWashIntervals = self.washMJDs.size+1
        self.washMJDs = np.insert(self.washMJDs,0,np.min(self.expMJD)-1.0)

        self.expWashIndex = np.zeros(self.nExp,dtype='i4')
        tempWashMJDs = self.washMJDs
        tempWashMJDs = np.append(tempWashMJDs,1e10)

        # record the range in each to get typical length of wash epoch
        washMJDRange = np.zeros((self.nWashIntervals,2))
        self.expPerWash = np.zeros(self.nWashIntervals,dtype=np.int32)
        for i in xrange(self.nWashIntervals):
            use,=np.where((self.expMJD > tempWashMJDs[i]) &
                          (self.expMJD < tempWashMJDs[i+1]))
            self.expWashIndex[use] = i
            washMJDRange[i,0] = np.min(self.expMJD[use])
            washMJDRange[i,1] = np.max(self.expMJD[use])
            self.expPerWash[i] = use.size

        self.meanWashIntervalDuration = np.mean(washMJDRange[:,1] - washMJDRange[:,0])
        self.meanExpPerWash = np.mean(self.expPerWash)


    def _arrangeParArray(self):
        """
        Internal method to make the full fit array
        """

        # make pointers to a fit parameter array...
        #  pwv, O3, lnTau, alpha
        self.nFitPars = (self.nCampaignNights +  # O3
                         self.nCampaignNights +  # tauIntercept
                         self.nCampaignNights +  # tauPerSlope
                         self.nCampaignNights +  # alpha
                         self.nCampaignNights +  # pwv Intercept
                         self.nCampaignNights +  # pwv Slope
                         self.nCampaignNights)   # pwv Quadratic
        ctr=0
        self.parO3Loc = ctr
        ctr+=self.nCampaignNights
        self.parLnTauInterceptLoc = ctr
        ctr+=self.nCampaignNights
        self.parLnTauSlopeLoc = ctr
        ctr+=self.nCampaignNights
        self.parAlphaLoc = ctr
        ctr+=self.nCampaignNights
        if not self.useRetrievedPwv:
            self.parLnPwvInterceptLoc = ctr
            ctr+=self.nCampaignNights
            self.parLnPwvSlopeLoc = ctr
            ctr+=self.nCampaignNights
            self.parLnPwvQuadraticLoc = ctr
            ctr+=self.nCampaignNights

        if self.hasExternalPwv and not self.useRetrievedPwv:
            self.nFitPars += (1+self.nCampaignNights)
            self.parExternalLnPwvScaleLoc = ctr
            ctr+=1
            self.parExternalLnPwvOffsetLoc = ctr
            ctr+=self.nCampaignNights

        if self.hasExternalTau:
            self.nFitPars += (1+self.nCampaignNights)
            self.parExternalLnTauScaleLoc = ctr
            ctr+=1
            self.parExternalLnTauOffsetLoc = ctr
            ctr+=self.nCampaignNights

        if self.useRetrievedPwv:
            self.nFitPars += 2
            self.parRetrievedLnPwvScaleLoc = ctr
            ctr+=1
            if self.useNightlyRetrievedPwv:
                self.parRetrievedLnPwvNightlyOffsetLoc = ctr
                ctr+=self.nCampaignNights
            else:
                self.parRetrievedLnPwvOffsetLoc = ctr
                ctr+=1

        self.nFitPars += (self.nWashIntervals + # parQESysIntercept
                          self.nWashIntervals)  # parQESysSlope

        self.parQESysInterceptLoc = ctr
        ctr+=self.nWashIntervals
        self.parQESysSlopeLoc = ctr
        ctr+=self.nWashIntervals

        self.nFitPars += self.nLUTFilter # parFilterOffset
        self.parFilterOffsetLoc = ctr
        ctr += self.nLUTFilter

        self.nFitPars += self.nBands # parAbsOffset
        self.parAbsOffsetLoc = ctr
        ctr += self.nBands


    def saveParsFits(self, parFile):
        """
        Save parameters to fits file

        parameters
        ----------
        parFile: string
           Output file
        """

        import fitsio

        # save the parameter file...
        self.fgcmLog.info('Saving parameters to %s' % (parFile))

        parInfo, pars = self.parsToArrays()

        # clobber?
        # parameter info
        fitsio.write(parFile,parInfo,extname='PARINFO',clobber=True)

        # parameters
        fitsio.write(parFile,pars,extname='PARAMS')

        # and need to record the superstar flats
        fitsio.write(parFile,self.parSuperStarFlat,extname='SUPER')

    def parsToArrays(self):
        """
        Convert parameters into recarrays for export

        parameters
        ----------
        None

        returns
        -------
        parInfo, pars: tuple
           Parameter information and parameters.
        """
        # this can be run without fits

        dtype=[('NCCD','i4'),
               ('LUTFILTERNAMES','a2',len(self.lutFilterNames)),
               ('BANDS','a2',len(self.bands)),
               ('FITBANDS','a2',len(self.fitBands)),
               ('NOTFITBANDS','a2',len(self.notFitBands)),
               ('LNTAUUNIT','f8'),
               ('LNTAUSLOPEUNIT','f8'),
               ('ALPHAUNIT','f8'),
               ('LNPWVUNIT','f8'),
               ('LNPWVSLOPEUNIT','f8'),
               ('LNPWVQUADRATICUNIT','f8'),
               ('LNPWVGLOBALUNIT','f8'),
               ('O3UNIT','f8'),
               ('QESYSUNIT','f8'),
               ('QESYSSLOPEUNIT','f8'),
               ('FILTEROFFSETUNIT','f8'),
               ('ABSOFFSETUNIT','f8'),
               ('HASEXTERNALPWV','i2'),
               ('HASEXTERNALTAU','i2')]

        ## FIXME: change from these files...
        if (self.hasExternalPwv):
            dtype.extend([('PWVFILE','a%d' % (len(self.pwvFile)+1))])
        if (self.hasExternalTau):
            dtype.extend([('TAUFILE','a%d' % (len(self.tauFile)+1))])

        parInfo=np.zeros(1,dtype=dtype)
        parInfo['NCCD'] = self.nCCD
        parInfo['LUTFILTERNAMES'] = self.lutFilterNames
        parInfo['BANDS'] = self.bands
        parInfo['FITBANDS'] = self.fitBands
        parInfo['NOTFITBANDS'] = self.notFitBands

        parInfo['LNTAUUNIT'] = self.unitDictSteps['lnTauUnit']
        parInfo['LNTAUSLOPEUNIT'] = self.unitDictSteps['lnTauSlopeUnit']
        parInfo['ALPHAUNIT'] = self.unitDictSteps['alphaUnit']
        parInfo['LNPWVUNIT'] = self.unitDictSteps['lnPwvUnit']
        parInfo['LNPWVSLOPEUNIT'] = self.unitDictSteps['lnPwvSlopeUnit']
        parInfo['LNPWVQUADRATICUNIT'] = self.unitDictSteps['lnPwvQuadraticUnit']
        parInfo['LNPWVGLOBALUNIT'] = self.unitDictSteps['lnPwvGlobalUnit']
        parInfo['O3UNIT'] = self.unitDictSteps['o3Unit']
        parInfo['QESYSUNIT'] = self.unitDictSteps['qeSysUnit']
        parInfo['QESYSSLOPEUNIT'] = self.unitDictSteps['qeSysSlopeUnit']
        parInfo['FILTEROFFSETUNIT'] = self.unitDictSteps['filterOffsetUnit']
        parInfo['ABSOFFSETUNIT'] = self.unitDictSteps['absOffsetUnit']

        parInfo['HASEXTERNALPWV'] = self.hasExternalPwv
        if (self.hasExternalPwv):
            parInfo['PWVFILE'] = self.pwvFile
        parInfo['HASEXTERNALTAU'] = self.hasExternalTau
        if (self.hasExternalTau):
            parInfo['TAUFILE'] = self.tauFile

        dtype=[('PARALPHA','f8',self.parAlpha.size),
               ('PARO3','f8',self.parO3.size),
               ('PARLNPWVINTERCEPT','f8',self.parLnPwvIntercept.size),
               ('PARLNPWVSLOPE','f8',self.parLnPwvSlope.size),
               ('PARLNPWVQUADRATIC','f8',self.parLnPwvQuadratic.size),
               ('PARLNTAUINTERCEPT','f8',self.parLnTauIntercept.size),
               ('PARLNTAUSLOPE','f8',self.parLnTauSlope.size),
               ('PARQESYSINTERCEPT','f8',self.parQESysIntercept.size),
               ('PARQESYSSLOPE','f8',self.parQESysSlope.size),
               ('PARFILTEROFFSET','f8',self.parFilterOffset.size),
               ('PARFILTEROFFSETFITFLAG','i2',self.parFilterOffsetFitFlag.size),
               ('PARABSOFFSET', 'f8', self.parAbsOffset.size),
               ('COMPAPERCORRPIVOT','f8',self.compAperCorrPivot.size),
               ('COMPAPERCORRSLOPE','f8',self.compAperCorrSlope.size),
               ('COMPAPERCORRSLOPEERR','f8',self.compAperCorrSlopeErr.size),
               ('COMPAPERCORRRANGE','f8',self.compAperCorrRange.size),
               ('COMPMODELERREXPTIMEPIVOT', 'f8', self.compModelErrExptimePivot.size),
               ('COMPMODELERRFWHMPIVOT', 'f8', self.compModelErrFwhmPivot.size),
               ('COMPMODELERRSKYPIVOT', 'f8', self.compModelErrSkyPivot.size),
               ('COMPMODELERRPARS', 'f8', self.compModelErrPars.size),
               ('COMPEXPGRAY','f8',self.compExpGray.size),
               ('COMPVARGRAY','f8',self.compVarGray.size),
               ('COMPNGOODSTARPEREXP','i4',self.compNGoodStarPerExp.size),
               ('COMPSIGFGCM','f8',self.compSigFgcm.size),
               ('COMPSIGMACAL', 'f8', self.compSigmaCal.size),
               ('COMPRETRIEVEDLNPWV','f8',self.compRetrievedLnPwv.size),
               ('COMPRETRIEVEDLNPWVRAW','f8',self.compRetrievedLnPwvRaw.size),
               ('COMPRETRIEVEDLNPWVFLAG','i2',self.compRetrievedLnPwvFlag.size),
               ('PARRETRIEVEDLNPWVSCALE','f8'),
               ('PARRETRIEVEDLNPWVOFFSET','f8'),
               ('PARRETRIEVEDLNPWVNIGHTLYOFFSET','f8',self.parRetrievedLnPwvNightlyOffset.size),
               ('COMPRETRIEVEDTAUNIGHT','f8',self.compRetrievedTauNight.size)]

        if (self.hasExternalPwv):
            dtype.extend([('PAREXTERNALLNPWVSCALE','f8'),
                          ('PAREXTERNALLNPWVOFFSET','f8',self.parExternalLnPwvOffset.size),
                          ('EXTERNALLNPWV','f8',self.nExp)])
        if (self.hasExternalTau):
            dtype.extend([('PAREXTERNALLNTAUSCALE','f8'),
                          ('PAREXTERNALLNTAUOFFSET','f8',self.parExternalLnTauOffset.size),
                          ('EXTERNALTAU','f8',self.nExp)])

        pars=np.zeros(1,dtype=dtype)

        pars['PARALPHA'][:] = self.parAlpha
        pars['PARO3'][:] = self.parO3
        pars['PARLNTAUINTERCEPT'][:] = self.parLnTauIntercept
        pars['PARLNTAUSLOPE'][:] = self.parLnTauSlope
        pars['PARLNPWVINTERCEPT'][:] = self.parLnPwvIntercept
        pars['PARLNPWVSLOPE'][:] = self.parLnPwvSlope
        pars['PARLNPWVQUADRATIC'][:] = self.parLnPwvQuadratic
        pars['PARQESYSINTERCEPT'][:] = self.parQESysIntercept
        pars['PARQESYSSLOPE'][:] = self.parQESysSlope
        pars['PARFILTEROFFSET'][:] = self.parFilterOffset
        pars['PARFILTEROFFSETFITFLAG'][:] = self.parFilterOffsetFitFlag
        pars['PARABSOFFSET'][:] = self.parAbsOffset

        if (self.hasExternalPwv):
            pars['PAREXTERNALLNPWVSCALE'] = self.parExternalLnPwvScale
            pars['PAREXTERNALLNPWVOFFSET'][:] = self.parExternalLnPwvOffset
            pars['EXTERNALLNPWV'][:] = self.externalLnPwv
        if (self.hasExternalTau):
            pars['PAREXTERNALLNTAUSCALE'] = self.parExternalLnTauScale
            pars['PAREXTERNALLNTAUOFFSET'][:] = self.parExternalLnTauOffset
            pars['EXTERNALTAU'][:] = self.externalTau

        pars['COMPAPERCORRPIVOT'][:] = self.compAperCorrPivot
        pars['COMPAPERCORRSLOPE'][:] = self.compAperCorrSlope
        pars['COMPAPERCORRSLOPEERR'][:] = self.compAperCorrSlopeErr
        pars['COMPAPERCORRRANGE'][:] = self.compAperCorrRange.flatten()

        pars['COMPMODELERREXPTIMEPIVOT'][:] = self.compModelErrExptimePivot
        pars['COMPMODELERRFWHMPIVOT'][:] = self.compModelErrFwhmPivot
        pars['COMPMODELERRSKYPIVOT'][:] = self.compModelErrSkyPivot
        pars['COMPMODELERRPARS'][:] = self.compModelErrPars.flatten()

        pars['COMPEXPGRAY'][:] = self.compExpGray
        pars['COMPVARGRAY'][:] = self.compVarGray
        pars['COMPNGOODSTARPEREXP'][:] = self.compNGoodStarPerExp

        pars['COMPSIGFGCM'][:] = self.compSigFgcm
        pars['COMPSIGMACAL'][:] = self.compSigmaCal

        pars['COMPRETRIEVEDLNPWV'][:] = self.compRetrievedLnPwv
        pars['COMPRETRIEVEDLNPWVRAW'][:] = self.compRetrievedLnPwvRaw
        pars['COMPRETRIEVEDLNPWVFLAG'][:] = self.compRetrievedLnPwvFlag
        pars['PARRETRIEVEDLNPWVSCALE'][:] = self.parRetrievedLnPwvScale
        pars['PARRETRIEVEDLNPWVOFFSET'][:] = self.parRetrievedLnPwvOffset
        pars['PARRETRIEVEDLNPWVNIGHTLYOFFSET'][:] = self.parRetrievedLnPwvNightlyOffset

        pars['COMPRETRIEVEDTAUNIGHT'][:] = self.compRetrievedTauNight

        return parInfo, pars

    def loadExternalPwv(self, externalPwvDeltaT):
        """
        Load external Pwv measurements, from fits table.

        parameters
        ----------
        externalPwvDeltaT: float
           Maximum delta-T (days) for external Pwv measurement to match exposure MJD.
        """

        import fitsio

        # loads a file with Pwv, matches to exposures/times
        # flags which ones need the nightly fit

        #self.hasExternalPwv = True

        pwvTable = fitsio.read(self.pwvFile,ext=1)

        # make sure it's sorted
        st=np.argsort(pwvTable['MJD'])
        pwvTable = pwvTable[st]

        pwvIndex = np.clip(np.searchsorted(pwvTable['MJD'],self.expMJD),0,pwvTable.size-1)
        # this will be True or False...
        self.externalPwvFlag[:] = (np.abs(pwvTable['MJD'][pwvIndex] - self.expMJD) < externalPwvDeltaT)
        self.externalLnPwv = np.zeros(self.nExp,dtype=np.float32)
        self.externalLnPwv[self.externalPwvFlag] = np.log(np.clip(pwvTable['PWV'][pwvIndex[self.externalPwvFlag]], self.pwvRange[0], self.pwvRange[1]))

        # and new PWV scaling pars!
        self.parExternalLnPwvOffset = np.zeros(self.nCampaignNights,dtype=np.float32)
        self.parExternalLnPwvScale = 1.0

        match, = np.where(self.externalPwvFlag)
        self.fgcmLog.info('%d exposures of %d have external pwv values' % (match.size,self.nExp))


    def loadExternalTau(self, withAlpha=False):
        """
        Not Supported.
        """

        # load a file with Tau values
        ## not supported yet
        raise NotImplementedError("externalTau Not supported yet")

        if (withAlpha):
            self.hasExternalAlpha = True

    def reloadParArray(self, parArray, fitterUnits=False):
        """
        Take parameter array and stuff into individual parameter attributes.

        parameters
        ----------
        parArray: float array
           Array with all fit parameters
        fitterUnits: bool, default=False
           Is the parArray in normalized fitter units?
        """

        # takes in a parameter array and loads the local split copies?
        self.fgcmLog.debug('Reloading parameter array')

        if (parArray.size != self.nFitPars):
            raise ValueError("parArray must have %d elements." % (self.nFitPars))

        unitDict = self.getUnitDict(fitterUnits=fitterUnits)

        if not self.useRetrievedPwv:
            self.parLnPwvIntercept[:] = parArray[self.parLnPwvInterceptLoc:
                                                     self.parLnPwvInterceptLoc+self.nCampaignNights] / unitDict['lnPwvUnit']
            self.parLnPwvSlope[:] = parArray[self.parLnPwvSlopeLoc:
                                                 self.parLnPwvSlopeLoc + self.nCampaignNights] / unitDict['lnPwvSlopeUnit']
            self.parLnPwvQuadratic[:] = parArray[self.parLnPwvQuadraticLoc:
                                                     self.parLnPwvQuadraticLoc + self.nCampaignNights] / unitDict['lnPwvQuadraticUnit']

        self.parO3[:] = parArray[self.parO3Loc:
                                     self.parO3Loc+self.nCampaignNights] / unitDict['o3Unit']
        self.parLnTauIntercept[:] = parArray[self.parLnTauInterceptLoc:
                                                 self.parLnTauInterceptLoc+self.nCampaignNights] / unitDict['lnTauUnit']
        self.parLnTauSlope[:] = (parArray[self.parLnTauSlopeLoc:
                                               self.parLnTauSlopeLoc + self.nCampaignNights] /
                                 unitDict['lnTauSlopeUnit'])

        self.parAlpha[:] = parArray[self.parAlphaLoc:
                                        self.parAlphaLoc+self.nCampaignNights] / unitDict['alphaUnit']
        if self.hasExternalPwv and not self.useRetrievedPwv:
            self.parExternalLnPwvScale = parArray[self.parExternalLnPwvScaleLoc] / unitDict['lnPwvGlobalUnit']
            self.parExternalLnPwvOffset = parArray[self.parExternalLnPwvOffsetLoc:
                                                       self.parExternalLnPwvOffsetLoc+self.nCampaignNights] / unitDict['lnPwvUnit']

        if (self.hasExternalTau):
            raise NotImplementedError("Not implemented")
            #self.parExternalLnTauScale = parArray[self.parExternalLnTauScaleLoc] / unitDict['tauUnit']
            #self.parExternalLnTauOffset = parArray[self.parExternalLnTauOffsetLoc:
            #                                         self.parExternalLnTauOffsetLoc+self.nCampaignNights] / unitDict['tauUnit']

        if self.useRetrievedPwv:
            self.parRetrievedLnPwvScale = parArray[self.parRetrievedLnPwvScaleLoc] / unitDict['lnPwvGlobalUnit']
            if self.useNightlyRetrievedPwv:
                self.parRetrievedLnPwvNightlyOffset = parArray[self.parRetrievedLnPwvNightlyOffsetLoc:
                                                                   self.parRetrievedLnPwvNightlyOffsetLoc + self.nCampaignNights] / unitDict['lnPwvUnit']
            else:
                self.parRetrievedLnPwvOffset = parArray[self.parRetrievedLnPwvOffsetLoc] / unitDict['lnPwvGlobalUnit']

        self.parQESysIntercept[:] = parArray[self.parQESysInterceptLoc:
                                                 self.parQESysInterceptLoc+self.nWashIntervals] / unitDict['qeSysUnit']


        self.parQESysSlope[:] = parArray[self.parQESysSlopeLoc:
                                             self.parQESysSlopeLoc+self.nWashIntervals] / unitDict['qeSysSlopeUnit']

        self.parFilterOffset[:] = parArray[self.parFilterOffsetLoc:
                                               self.parFilterOffsetLoc + self.nLUTFilter] / unitDict['filterOffsetUnit']

        self.parAbsOffset[:] = parArray[self.parAbsOffsetLoc:
                                            self.parAbsOffsetLoc + self.nBands] / unitDict['absOffsetUnit']

        # done

    def parsToExposures(self, retrievedInput=False):
        """
        Associate parameters with exposures.

        parameters
        ----------
        retrievedInput: bool, default=False
           When useRetrievedPwv, do we use the input or final values as the basis for the conversion?

        Output attributes
        -----------------
        expO3: float array
        expAlpha: float array
        expLnPwv: float array
        expLnTau: float array
        expQESys: float array
        expFilterOffset: float array
        """

        self.fgcmLog.debug('Computing exposure values from parameters')

        # I'm guessing that these don't need to be wrapped in shms but I could be wrong
        #  about the full class, which would suck.

        # first, the nightly parameters without selection...
        self.expO3 = self.parO3[self.expNightIndex]
        self.expAlpha = self.parAlpha[self.expNightIndex]

        if self.useRetrievedPwv:
            # FIXME
            if retrievedInput:
                retrievedLnPwv = self.compRetrievedLnPwvInput
            else:
                retrievedLnPwv = self.compRetrievedLnPwv

            if self.useNightlyRetrievedPwv:
                self.expLnPwv = (self.parRetrievedLnPwvNightlyOffset[self.expNightIndex] +
                                 self.parRetrievedLnPwvScale * retrievedLnPwv)
            else:
                self.expLnPwv = (self.parRetrievedLnPwvOffset +
                                 self.parRetrievedLnPwvScale * retrievedLnPwv)
        else:
            # default to the nightly slope/intercept
            self.expLnPwv = (self.parLnPwvIntercept[self.expNightIndex] +
                             self.parLnPwvSlope[self.expNightIndex] * self.expDeltaUT +
                             self.parLnPwvQuadratic[self.expNightIndex] * self.expDeltaUT**2.)
            if (self.hasExternalPwv):
                # replace where we have these
                self.expLnPwv[self.externalPwvFlag] = (self.parExternalLnPwvOffset[self.expNightIndex[self.externalPwvFlag]] +
                                                         self.parExternalLnPwvScale *
                                                         self.externalLnPwv[self.externalPwvFlag])
        # and clip to make sure it doesn't go out of bounds
        self.expLnPwv = np.clip(self.expLnPwv, self.lnPwvRange[0], self.lnPwvRange[1])

        # default to nightly slope/intercept
        self.expLnTau = (self.parLnTauIntercept[self.expNightIndex] +
                         self.parLnTauSlope[self.expNightIndex] * self.expDeltaUT)

        if (self.hasExternalTau):
            raise NotImplementedError("not implemented")
            # replace where we have these
            #self.expLnTau[self.externalTauFlag] = (self.parExternalLnTauOffset[self.expNightIndex[self.externalTauFlag]] +
            #                                     self.parExternalLnTauScale *
            #                                     self.externalTau[self.externalTauFlag])

        # and clip to make sure it doesn't go negative
        self.expLnTau = np.clip(self.expLnTau, self.lnTauRange[0], self.lnTauRange[1])

        # and QESys
        self.expQESys = (self.parQESysIntercept[self.expWashIndex] +
                         self.parQESysSlope[self.expWashIndex] *
                         (self.expMJD - self.washMJDs[self.expWashIndex]))

        # and FilterOffset + abs offset
        self.expFilterOffset = (self.parFilterOffset[self.expLUTFilterIndex] +
                                self.parAbsOffset[self.expBandIndex])

    # cannot be a property because of the keywords
    def getParArray(self,fitterUnits=False):
        """
        Take individual parameter attributes and build a parameter array.

        parameters
        ----------
        fitterUnits: bool, default=False
           Will the parArray be in normalized fitter units?

        returns
        -------
        parArray: float array
           Array with all fit parameters
        """

        self.fgcmLog.debug('Retrieving parameter array')

        # extracts parameters into a linearized array
        parArray = np.zeros(self.nFitPars,dtype=np.float64)

        unitDict = self.getUnitDict(fitterUnits=fitterUnits)

        if not self.useRetrievedPwv:
            parArray[self.parLnPwvInterceptLoc:
                         self.parLnPwvInterceptLoc+self.nCampaignNights] = self.parLnPwvIntercept[:] * unitDict['lnPwvUnit']
            parArray[self.parLnPwvSlopeLoc:
                         self.parLnPwvSlopeLoc + self.nCampaignNights] = self.parLnPwvSlope[:] * unitDict['lnPwvSlopeUnit']
            parArray[self.parLnPwvQuadraticLoc:
                         self.parLnPwvQuadraticLoc + self.nCampaignNights] = self.parLnPwvQuadratic[:] * unitDict['lnPwvQuadraticUnit']

        parArray[self.parO3Loc:
                     self.parO3Loc+self.nCampaignNights] = self.parO3[:] * unitDict['o3Unit']
        parArray[self.parLnTauInterceptLoc:
                     self.parLnTauInterceptLoc+self.nCampaignNights] = self.parLnTauIntercept[:] * unitDict['lnTauUnit']
        parArray[self.parLnTauSlopeLoc:
                     self.parLnTauSlopeLoc + self.nCampaignNights] = self.parLnTauSlope[:] * unitDict['lnTauSlopeUnit']
        parArray[self.parAlphaLoc:
                     self.parAlphaLoc+self.nCampaignNights] = self.parAlpha[:] * unitDict['alphaUnit']
        if self.hasExternalPwv and not self.useRetrievedPwv:
            parArray[self.parExternalLnPwvScaleLoc] = self.parExternalLnPwvScale * unitDict['lnPwvGlobalUnit']
            parArray[self.parExternalLnPwvOffsetLoc:
                         self.parExternalLnPwvOffsetLoc+self.nCampaignNights] = self.parExternalLnPwvOffset * unitDict['lnPwvUnit']
        if (self.hasExternalTau):
            raise NotImplementedError("not implemented")
            #parArray[self.parExternalTauScaleLoc] = self.parExternalTauScale * unitDict['lnTauUnit']
            #parArray[self.parExternalTauOffsetLoc:
            #             self.parExternalTauOffsetLoc+self.nCampaignNights] = self.parExternalTauOffset * unitDict['tauUnit']
        if self.useRetrievedPwv:
            parArray[self.parRetrievedLnPwvScaleLoc] = self.parRetrievedLnPwvScale * unitDict['lnPwvGlobalUnit']
            if self.useNightlyRetrievedPwv:
                parArray[self.parRetrievedLnPwvNightlyOffsetLoc:
                             self.parRetrievedLnPwvNightlyOffsetLoc+self.nCampaignNights] = self.parRetrievedLnPwvNightlyOffset * unitDict['lnPwvUnit']
            else:
                parArray[self.parRetrievedLnPwvOffsetLoc] = self.parRetrievedLnPwvOffset * unitDict['lnPwvGlobalUnit']

        parArray[self.parQESysInterceptLoc:
                     self.parQESysInterceptLoc+self.nWashIntervals] = self.parQESysIntercept * unitDict['qeSysUnit']
        parArray[self.parQESysSlopeLoc:
                     self.parQESysSlopeLoc+self.nWashIntervals] = self.parQESysSlope * unitDict['qeSysSlopeUnit']

        parArray[self.parFilterOffsetLoc:
                     self.parFilterOffsetLoc + self.nLUTFilter] = self.parFilterOffset * unitDict['filterOffsetUnit']

        parArray[self.parAbsOffsetLoc:
                     self.parAbsOffsetLoc + self.nBands] = self.parAbsOffset * unitDict['absOffsetUnit']

        return parArray

    # this cannot be a property because it takes units
    def getParBounds(self,fitterUnits=False):
        """
        Create parameter fit bounds

        parameters
        ----------
        fitterUnits: bool, default=False
           Will the parArray be in normalized fitter units?

        returns
        -------
        parBounds: list(zip(parLow, parHigh))

        """

        self.fgcmLog.debug('Retrieving parameter bounds')

        unitDict = self.getUnitDict(fitterUnits=fitterUnits)

        parLow = np.zeros(self.nFitPars,dtype=np.float32)
        parHigh = np.zeros(self.nFitPars,dtype=np.float32)

        if not self.useRetrievedPwv:
            parLow[self.parLnPwvInterceptLoc: \
                       self.parLnPwvInterceptLoc + \
                       self.nCampaignNights] = ( \
                self.lnPwvRange[0] * unitDict['lnPwvUnit'])
            parHigh[self.parLnPwvInterceptLoc: \
                        self.parLnPwvInterceptLoc + \
                        self.nCampaignNights] = ( \
                self.lnPwvRange[1] * unitDict['lnPwvUnit'])
            parLow[self.parLnPwvSlopeLoc: \
                       self.parLnPwvSlopeLoc + \
                       self.nCampaignNights] = ( \
                -4.0 * unitDict['lnPwvSlopeUnit'])
            parHigh[self.parLnPwvSlopeLoc: \
                       self.parLnPwvSlopeLoc + \
                       self.nCampaignNights] = ( \
                4.0 * unitDict['lnPwvSlopeUnit'])
            if self.useQuadraticPwv:
                qlo = -4.0 * unitDict['lnPwvQuadraticUnit']
                qhi = 4.0 * unitDict['lnPwvQuadraticUnit']
            else:
                qlo = 0.0
                qhi = 0.0
            parLow[self.parLnPwvQuadraticLoc: \
                       self.parLnPwvQuadraticLoc + \
                       self.nCampaignNights] = qlo
            parHigh[self.parLnPwvQuadraticLoc: \
                       self.parLnPwvQuadraticLoc + \
                       self.nCampaignNights] = qhi
        else:
            parLow[self.parRetrievedLnPwvScaleLoc] = 0.5 * unitDict['lnPwvGlobalUnit']
            parHigh[self.parRetrievedLnPwvScaleLoc] = 1.5 * unitDict['lnPwvGlobalUnit']

            if self.useNightlyRetrievedPwv:
                parLow[self.parRetrievedLnPwvNightlyOffsetLoc: \
                           self.parRetrievedLnPwvNightlyOffsetLoc + \
                           self.nCampaignNights] = ( \
                    -0.5 * unitDict['lnPwvUnit'])
                parHigh[self.parRetrievedLnPwvNightlyOffsetLoc: \
                           self.parRetrievedLnPwvNightlyOffsetLoc + \
                           self.nCampaignNights] = ( \
                    0.5 * unitDict['lnPwvUnit'])
            else:
                parLow[self.parRetrievedLnPwvOffsetLoc] = -0.5 * unitDict['lnPwvGlobalUnit']
                parHigh[self.parRetrievedLnPwvOffsetLoc] = 0.5 * unitDict['lnPwvGlobalUnit']

        parLow[self.parO3Loc: \
                   self.parO3Loc + \
                   self.nCampaignNights] = ( \
            self.O3Range[0] * unitDict['o3Unit'])
        parHigh[self.parO3Loc: \
                    self.parO3Loc + \
                    self.nCampaignNights] = ( \
            self.O3Range[1] * unitDict['o3Unit'])
        parLow[self.parLnTauInterceptLoc: \
                   self.parLnTauInterceptLoc + \
                   self.nCampaignNights] = ( \
            self.lnTauRange[0] * unitDict['lnTauUnit'])
        parHigh[self.parLnTauInterceptLoc: \
                    self.parLnTauInterceptLoc + \
                self.nCampaignNights] = ( \
            self.lnTauRange[1] * unitDict['lnTauUnit'])
        parLow[self.parLnTauSlopeLoc: \
                   self.parLnTauSlopeLoc + \
                   self.nCampaignNights] = ( \
            -4.0 * unitDict['lnTauSlopeUnit'])
        parHigh[self.parLnTauSlopeLoc: \
                    self.parLnTauSlopeLoc + \
                    self.nCampaignNights] = ( \
            4.0 * unitDict['lnTauSlopeUnit'])
        parLow[self.parAlphaLoc: \
                   self.parAlphaLoc + \
                   self.nCampaignNights] = ( \
            0.25 * unitDict['alphaUnit'])
        parHigh[self.parAlphaLoc: \
                    self.parAlphaLoc + \
                    self.nCampaignNights] = ( \
            1.75 * unitDict['alphaUnit'])
        if self.hasExternalPwv and not self.useRetrievedPwv:
            parLow[self.parExternalLnPwvScaleLoc] = 0.5 * unitDict['lnPwvGlobalUnit']
            parHigh[self.parExternalLnPwvScaleLoc] = 1.5 * unitDict['lnPwvGlobalUnit']
            parLow[self.parExternalLnPwvOffsetLoc: \
                       self.parExternalLnPwvOffsetLoc + \
                       self.nCampaignNights] = ( \
                -0.5 * unitDict['lnPwvUnit'])
            parHigh[self.parExternalLnPwvOffsetLoc: \
                       self.parExternalLnPwvOffsetLoc + \
                        self.nCampaignNights] = ( \
                0.5 * unitDict['lnPwvUnit'])
        if (self.hasExternalTau):
            raise NotImplementedError("not implemented")
            #parLow[self.parExternalTauScaleLoc] = 0.7 * unitDict['tauUnit']
            #parHigh[self.parExternalTauScaleLoc] = 1.2 * unitDict['tauUnit']
            #parLow[self.parExternalTauOffsetLoc: \
            #           self.parExternalTauOffsetLoc + \
            #       self.nCampaignNights] = ( \
            #    0.0 * unitDict['tauUnit'])
            #parHigh[self.parExternalTauOffsetLoc: \
            #            self.parExternalTauOffsetLoc + \
            #            self.nCampaignNights] = ( \
            #    0.03 * unitDict['tauUnit'])
            ## FIXME: set bounds per night?  Or clip?

        parLow[self.parQESysInterceptLoc: \
                   self.parQESysInterceptLoc + \
                   self.nWashIntervals] = ( \
            -0.2 * unitDict['qeSysUnit'])
        parHigh[self.parQESysInterceptLoc: \
                    self.parQESysInterceptLoc + \
                    self.nWashIntervals] = ( \
            0.05 * unitDict['qeSysUnit'])

        # and for the first interval, the intercept will set to zero
        parLow[self.parQESysInterceptLoc] = 0.0
        parHigh[self.parQESysInterceptLoc] = 0.0

        parLow[self.parQESysSlopeLoc: \
                   self.parQESysSlopeLoc + \
                   self.nWashIntervals] = ( \
            -0.001 * unitDict['qeSysSlopeUnit'])
        parHigh[self.parQESysSlopeLoc: \
                    self.parQESysSlopeLoc + \
                    self.nWashIntervals] = ( \
            0.001 * unitDict['qeSysSlopeUnit'])

        parLow[self.parFilterOffsetLoc: \
                   self.parFilterOffsetLoc + self.nLUTFilter] = 0.0
        parHigh[self.parFilterOffsetLoc: \
                    self.parFilterOffsetLoc + self.nLUTFilter] = 0.0
        parLow[self.parFilterOffsetLoc: \
                   self.parFilterOffsetLoc + self.nLUTFilter][self.parFilterOffsetFitFlag] = \
                   -100.0 * unitDict['filterOffsetUnit']
        parHigh[self.parFilterOffsetLoc: \
                    self.parFilterOffsetLoc + self.nLUTFilter][self.parFilterOffsetFitFlag] = \
                    100.0 * unitDict['filterOffsetUnit']

        parLow[self.parAbsOffsetLoc: \
                   self.parAbsOffsetLoc + self.nBands] = -100.0 * unitDict['absOffsetUnit']
        parHigh[self.parAbsOffsetLoc: \
                    self.parAbsOffsetLoc + self.nBands] = 100.0 * unitDict['absOffsetUnit']

        if self.freezeStdAtmosphere:
            # atmosphere parameters set to std values
            if not self.useRetrievedPwv:
                parLow[self.parLnPwvInterceptLoc: \
                           self.parLnPwvInterceptLoc + \
                           self.nCampaignNights] = self.lnPwvStd * unitDict['lnPwvUnit']
                parHigh[self.parLnPwvInterceptLoc: \
                            self.parLnPwvInterceptLoc + \
                            self.nCampaignNights] = self.lnPwvStd * unitDict['lnPwvUnit']
                parLow[self.parLnPwvSlopeLoc: \
                           self.parLnPwvSlopeLoc + \
                           self.nCampaignNights] = 0.0
                parHigh[self.parLnPwvSlopeLoc: \
                            self.parLnPwvSlopeLoc + \
                            self.nCampaignNights] = 0.0
                parLow[self.parLnPwvQuadraticLoc: \
                           self.parLnPwvQuadraticLoc + \
                           self.nCampaignNights] = 0.0
                parHigh[self.parLnPwvQuadraticLoc: \
                            self.parLnPwvQuadraticLoc + \
                            self.nCampaignNights] = 0.0
            else:
                parLow[self.parRetrievedLnPwvScaleLoc] = 1.0 * unitDict['lnPwvGlobalUnit']
                parHigh[self.parRetrievedLnPwvScaleLoc] = 1.0 * unitDict['lnPwvGlobalUnit']
                if self.useNightlyRetrievedPwv:
                    parLow[self.parRetrievedLnPwvNightlyOffsetLoc: \
                               self.parRetrievedLnPwvNightlyOffsetLoc + \
                               self.nCampaignNights] = 0.0
                    parHigh[self.parRetrievedLnPwvNightlyOffsetLoc: \
                                self.parRetrievedLnPwvNightlyOffsetLoc + \
                                self.nCampaignNights] = 0.0
                else:
                    parLow[self.parRetrievedLnPwvOffsetLoc] = 0.0
                    parHigh[self.parRetrievedLnPwvOffsetLoc] = 0.0

            if self.hasExternalPwv and not self.useRetrievedPwv:
                parLow[self.parExternalLnPwvScaleLoc] = 1.0 * unitDict['lnPwvGlobalUnit']
                parHigh[self.parExternalLnPwvScaleLoc] = 1.0 * unitDict['lnPwvGlobalUnit']
                parLow[self.parExternalLnPwvOffsetLoc: \
                           self.parExternalLnPwvOffsetLoc + \
                           self.nCampaignNights] = 0.0
                parHigh[self.parExternalLnPwvOffsetLoc: \
                            self.parExternalLnPwvOffsetLoc + \
                            self.nCampaignNights] = 0.0

            parLow[self.parO3Loc: \
                   self.parO3Loc + \
                   self.nCampaignNights] = self.o3Std * unitDict['o3Unit']
            parHigh[self.parO3Loc: \
                    self.parO3Loc + \
                    self.nCampaignNights] = self.o3Std * unitDict['o3Unit']
            parLow[self.parLnTauInterceptLoc: \
                   self.parLnTauInterceptLoc + \
                   self.nCampaignNights] = self.lnTauStd * unitDict['lnTauUnit']
            parHigh[self.parLnTauInterceptLoc: \
                    self.parLnTauInterceptLoc + \
                self.nCampaignNights] = self.lnTauStd * unitDict['lnTauUnit']
            parLow[self.parLnTauSlopeLoc: \
                   self.parLnTauSlopeLoc + \
                   self.nCampaignNights] = 0.0
            parHigh[self.parLnTauSlopeLoc: \
                    self.parLnTauSlopeLoc + \
                    self.nCampaignNights] = 0.0
            parLow[self.parAlphaLoc: \
                   self.parAlphaLoc + \
                   self.nCampaignNights] = self.alphaStd * unitDict['alphaUnit']
            parHigh[self.parAlphaLoc: \
                    self.parAlphaLoc + \
                    self.nCampaignNights] = self.alphaStd * unitDict['alphaUnit']


        # zip these into a list of tuples
        parBounds = list(zip(parLow, parHigh))

        return parBounds

    def getUnitDict(self,fitterUnits=False):
        """
        Get dictionary of unit conversions.

        parameters
        ----------
        fitterUnits: bool, default=False
           Return with normalized fitter units or just 1.0s?
        """

        if (fitterUnits):
            return self.unitDictSteps
        else:
            return self.unitDictOnes

    @property
    def superStarFlatCenter(self):
        """
        SuperStarFlat at the center of each CCD

        returns
        -------
        superStarFlatCenter: float array (nEpochs, nLUTFilter, nCCD)
        """

        # This bit of code simply returns the superStarFlat computed at the center
        # of each CCD

        from .fgcmUtilities import Cheb2dField

        # this is the version that does the center of the CCD
        # because it is operating on the whole CCD!

        superStarFlatCenter = np.zeros((self.nEpochs,
                                        self.nLUTFilter,
                                        self.nCCD))
        for e in xrange(self.nEpochs):
            for f in xrange(self.nLUTFilter):
                for c in xrange(self.nCCD):
                    field = Cheb2dField(self.ccdOffsets['X_SIZE'][c],
                                        self.ccdOffsets['Y_SIZE'][c],
                                        self.parSuperStarFlat[e, f, c, :])
                    superStarFlatCenter[e, f, c] = -2.5 * np.log10(field.evaluateCenter())

        return superStarFlatCenter

    @property
    def expCCDSuperStar(self):
        """
        SuperStarFlat for each Exposure/CCD

        returns
        -------
        expCCDSuperStar: float array (nExp, nCCD)
        """

        expCCDSuperStar = np.zeros((self.nExp, self.nCCD), dtype='f8')

        expCCDSuperStar[:, :] = self.superStarFlatCenter[self.expEpochIndex,
                                                         self.expLUTFilterIndex,
                                                         :]

        return expCCDSuperStar

    @property
    def expApertureCorrection(self):
        """
        Exposure aperture correction

        returns
        -------
        expApertureCorrection: float array (nExp)
        """

        expApertureCorrection = np.zeros(self.nExp,dtype='f8')

        ## FIXME if changing aperture correction
        # FIXME should filter not band
        expSeeingVariableClipped = np.clip(self.expSeeingVariable,
                                           self.compAperCorrRange[0,self.expBandIndex],
                                           self.compAperCorrRange[1,self.expBandIndex])

        expApertureCorrection[:] = (self.compAperCorrSlope[self.expBandIndex] *
                                    (expSeeingVariableClipped -
                                     self.compAperCorrPivot[self.expBandIndex]))

        return expApertureCorrection

    @property
    def ccdApertureCorrection(self):
        """
        CCD aperture correction

        returns
        -------
        ccdApertureCorrection: nccd x nexp float array
        """

        ccdApertureCorrection = np.zeros((self.nExp, self.nCCD), dtype='f8')


        # Run per ccd (assuming it's the shorter loop)
        for i in xrange(self.nCCD):
            ccdSeeingVariableClipped = np.clip(self.expSeeingVariablePerCCD[:, i],
                                               self.compAperCorrRange[0, self.expBandIndex],
                                               self.compAperCorrRange[1, self.expBandIndex])
            ccdApertureCorrection[:, i] = (self.compAperCorrSlope[self.expBandIndex] *
                                           (ccdSeeingVariableClipped -
                                            self.compAperCorrPivot[self.expBandIndex]))

        return ccdApertureCorrection

    def plotParameters(self):
        """
        Plot nightly average parameters
        """

        # want nightly averages, on calibratable nights (duh)

        # this is fixed here

        # make sure we have this...probably redundant
        self.parsToExposures()

        # only with photometric exposures
        expUse,=np.where(self.expFlag == 0)

        nExpPerBandPerNight = np.zeros((self.nCampaignNights,self.nBands),dtype='i4')
        nExpPerNight = np.zeros(self.nCampaignNights,dtype='i4')
        mjdNight = np.zeros(self.nCampaignNights,dtype='f8')
        alphaNight = np.zeros(self.nCampaignNights,dtype='f8')
        O3Night = np.zeros(self.nCampaignNights,dtype='f8')
        tauNight = np.zeros(self.nCampaignNights,dtype='f8')
        pwvNight = np.zeros(self.nCampaignNights,dtype='f8')

        np.add.at(nExpPerBandPerNight,
                  (self.expNightIndex[expUse],
                   self.expBandIndex[expUse]),
                  1)
        np.add.at(nExpPerNight,
                  self.expNightIndex[expUse],
                  1)
        np.add.at(mjdNight,
                  self.expNightIndex[expUse],
                  self.expMJD[expUse])
        np.add.at(alphaNight,
                  self.expNightIndex[expUse],
                  self.expAlpha[expUse])
        np.add.at(tauNight,
                  self.expNightIndex[expUse],
                  np.exp(self.expLnTau[expUse]))
        np.add.at(pwvNight,
                  self.expNightIndex[expUse],
                  np.exp(self.expLnPwv[expUse]))
        np.add.at(O3Night,
                  self.expNightIndex[expUse],
                  self.expO3[expUse])

        # hard code this for now
        gd,=np.where(nExpPerNight > self.minExpPerNight)
        mjdNight[gd] /= nExpPerNight[gd].astype(np.float64)
        alphaNight[gd] /= nExpPerNight[gd].astype(np.float64)
        tauNight[gd] /= nExpPerNight[gd].astype(np.float64)
        pwvNight[gd] /= nExpPerNight[gd].astype(np.float64)
        O3Night[gd] /= nExpPerNight[gd].astype(np.float64)

        firstMJD = np.floor(np.min(self.expMJD))

        # now alpha
        fig=plt.figure(1,figsize=(8,6))
        fig.clf()
        ax=fig.add_subplot(111)

        # alpha is good
        alphaGd, = np.where(nExpPerNight > self.minExpPerNight)

        ax.plot(mjdNight[alphaGd] - firstMJD,alphaNight[alphaGd],'r.')
        ax.set_xlabel(r'$\mathrm{MJD}\ -\ %.0f$' % (firstMJD),fontsize=16)
        ax.set_ylabel(r'$\alpha$',fontsize=16)

        fig.savefig('%s/%s_nightly_alpha.png' % (self.plotPath,
                                                 self.outfileBaseWithCycle))
        plt.close(fig)

        # Tau
        try:
            gBandIndex = self.bands.index('g')
        except ValueError:
            gBandIndex = -1
        try:
            rBandIndex = self.bands.index('r')
        except ValueError:
            rBandIndex = -1

        if gBandIndex >=0 or rBandIndex >= 0:

            fig=plt.figure(1,figsize=(8,6))
            fig.clf()
            ax=fig.add_subplot(111)

            # I'm sure there's a more elegant way of doing this
            if gBandIndex >= 0 and rBandIndex >= 0:
                tauGd, = np.where((nExpPerNight > self.minExpPerNight) &
                                  ((nExpPerBandPerNight[:,gBandIndex] +
                                    nExpPerBandPerNight[:,rBandIndex]) >
                                   self.minExpPerNight))
            elif gBandIndex >= 0:
                tauGd, = np.where((nExpPerNight > self.minExpPerNight) &
                                  ((nExpPerBandPerNight[:,gBandIndex]) >
                                   self.minExpPerNight))
            elif rBandIndex >= 0:
                tauGd, = np.where((nExpPerNight > self.minExpPerNight) &
                                  ((nExpPerBandPerNight[:,rBandIndex]) >
                                   self.minExpPerNight))

            ax.plot(mjdNight[tauGd] - firstMJD, tauNight[tauGd],'r.')
            ax.set_xlabel(r'$\mathrm{MJD}\ -\ %.0f$' % (firstMJD),fontsize=16)
            ax.set_ylabel(r'$\tau_{7750}$',fontsize=16)

            fig.savefig('%s/%s_nightly_tau.png' % (self.plotPath,
                                                   self.outfileBaseWithCycle))
            plt.close(fig)

        try:
            zBandIndex = self.bands.index('z')
        except ValueError:
            zBandIndex = -1

        if zBandIndex >= 0:
            # pwv
            fig=plt.figure(1,figsize=(8,6))
            fig.clf()
            ax=fig.add_subplot(111)

            pwvGd, = np.where((nExpPerNight > self.minExpPerNight) &
                              (nExpPerBandPerNight[:,zBandIndex] > self.minExpPerNight))

            ax.plot(mjdNight[pwvGd] - firstMJD, pwvNight[pwvGd],'r.')
            ax.set_xlabel(r'$\mathrm{MJD}\ -\ %.0f$' % (firstMJD),fontsize=16)
            ax.set_ylabel(r'$\mathrm{PWV}$',fontsize=16)

            fig.savefig('%s/%s_nightly_pwv.png' % (self.plotPath,
                                                   self.outfileBaseWithCycle))
            plt.close(fig)

        # O3
        try:
            rBandIndex = self.bands.index('r')
        except ValueError:
            rBandIndex = -1

        if rBandIndex >= 0:
            fig=plt.figure(1,figsize=(8,6))
            fig.clf()
            ax=fig.add_subplot(111)

            O3Gd, = np.where((nExpPerNight > self.minExpPerNight) &
                             (nExpPerBandPerNight[:,rBandIndex] > self.minExpPerNight))

            ax.plot(mjdNight[O3Gd] - firstMJD, O3Night[O3Gd],'r.')
            ax.set_xlabel(r'$\mathrm{MJD}\ -\ %.0f$' % (firstMJD),fontsize=16)
            ax.set_ylabel(r'$O_3$',fontsize=16)

            fig.savefig('%s/%s_nightly_o3.png' % (self.plotPath,
                                                  self.outfileBaseWithCycle))
            plt.close(fig)


        # mirror gray
        fig=plt.figure(1,figsize=(8,6))
        fig.clf()
        ax=fig.add_subplot(111)

        for i in xrange(self.nWashIntervals):
            use,=np.where(self.expWashIndex == i)
            washMJDRange = [np.min(self.expMJD[use]),np.max(self.expMJD[use])]

            ax.plot(washMJDRange - firstMJD,
                    (washMJDRange - self.washMJDs[i])*self.parQESysSlope[i] +
                    self.parQESysIntercept[i],'r--',linewidth=3)

        ax.set_xlabel(r'$\mathrm{MJD}\ -\ %.0f$' % (firstMJD),fontsize=16)
        ax.set_ylabel('$2.5 \log_{10} (S^{\mathrm{optics}})$',fontsize=16)
        ax.tick_params(axis='both',which='major',labelsize=14)

        ylim = ax.get_ylim()
        for i in xrange(self.nWashIntervals):
            ax.plot([self.washMJDs[i]-firstMJD,self.washMJDs[i]-firstMJD],
                    ylim,'k--')

        fig.savefig('%s/%s_qesys_washes.png' % (self.plotPath,
                                                self.outfileBaseWithCycle))
        plt.close(fig)

        # Filter Offset
        fig = plt.figure(1, figsize=(8, 6))
        fig.clf()
        ax = fig.add_subplot(111)

        ax.plot(self.lambdaStdFilter, self.parFilterOffset, 'r.')
        for i, f in enumerate(self.lutFilterNames):
            ax.annotate(r'$%s$' % (f), (self.lambdaStdFilter[i], self.parFilterOffset[i] - 0.01), xycoords='data', ha='center', va='top', fontsize=16)
        ax.set_xlabel('Std Wavelength (A)')
        ax.set_ylabel('Filter Offset (mag)')

        fig.savefig('%s/%s_filter_offsets.png' % (self.plotPath,
                                                  self.outfileBaseWithCycle))

        # Abs Offset
        fig = plt.figure(1, figsize=(8, 6))
        fig.clf()
        ax = fig.add_subplot(111)

        ax.plot(self.lambdaStdBand, self.parAbsOffset, 'r.')
        for i, b in enumerate(self.bands):
            ax.annotate(r'$%s$' % (b), (self.lambdaStdBand[i], self.parAbsOffset[i] - 0.01), xycoords='data', ha='center', va='top', fontsize=16)
        ax.set_xlabel('Std Wavelength (A)')
        ax.set_ylabel('Absolute offset (mag)')

        fig.savefig('%s/%s_abs_offsets.png' % (self.plotPath,
                                               self.outfileBaseWithCycle))

        ## FIXME: add pwv offset plotting routine (if external)
        ## FIXME: add tau offset plotting routing (if external)

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state
