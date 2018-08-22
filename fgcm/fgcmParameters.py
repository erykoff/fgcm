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
    useRetrievedPWV: bool
       Use PWV retrieved from colors from previous cycle?
    useNightlyRetrievedPWV: bool
       Re-fit offsets for each night PWV variation (if useRetrievedPWV==True)?
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

        self.hasExternalPWV = False
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
        self.nLUTFilter = len(self.lutFilterNames)
        self.filterToBand = fgcmConfig.filterToBand

        self.freezeStdAtmosphere = fgcmConfig.freezeStdAtmosphere
        self.alphaStd = fgcmConfig.alphaStd
        self.o3Std = fgcmConfig.o3Std
        self.tauStd = fgcmConfig.tauStd
        self.lnTauStd = np.log(self.tauStd)
        self.pwvStd = fgcmConfig.pwvStd
        self.pmbStd = fgcmConfig.pmbStd
        self.zenithStd = fgcmConfig.zenithStd
        self.secZenithStd = 1./np.cos(self.zenithStd*np.pi/180.)

        self.pmbRange = fgcmConfig.pmbRange
        self.pwvRange = fgcmConfig.pwvRange
        self.O3Range = fgcmConfig.O3Range
        self.tauRange = fgcmConfig.tauRange
        self.lnTauRange = np.log(self.tauRange)
        self.alphaRange = fgcmConfig.alphaRange
        self.zenithRange = fgcmConfig.zenithRange

        self.nExp = fgcmConfig.nExp
        self.seeingField = fgcmConfig.seeingField
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
        self.externalPWVDeltaT = fgcmConfig.externalPWVDeltaT
        self.externalTauDeltaT = fgcmConfig.externalTauDeltaT
        self.useRetrievedPWV = fgcmConfig.useRetrievedPWV
        self.useNightlyRetrievedPWV = fgcmConfig.useNightlyRetrievedPWV
        self.useRetrievedTauInit = fgcmConfig.useRetrievedTauInit
        self.modelMagErrors = fgcmConfig.modelMagErrors

        # Old style was (a,x,y,x**2,y**2,xy)
        # Retain this for reading old files
        self.superStarNParOld = 6
        self.superStarNPar = ((fgcmConfig.superStarSubCCDChebyshevOrder + 1) *
                              (fgcmConfig.superStarSubCCDChebyshevOrder + 1))
        self.superStarPoly2d = False
        self.ccdOffsets = fgcmConfig.ccdOffsets
        self.superStarSubCCD = fgcmConfig.superStarSubCCD

        # and the default unit dict
        self.unitDictOnes = {'pwvUnit':1.0,
                             'pwvPerSlopeUnit':1.0,
                             'pwvGlobalUnit':1.0,
                             'o3Unit':1.0,
                             'lnTauUnit':1.0,
                             'lnTauSlopeUnit':1.0,
                             'alphaUnit':1.0,
                             'qeSysUnit':1.0,
                             'qeSysSlopeUnit':1.0}

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
        self.parPWVIntercept = np.zeros(self.campaignNights.size,dtype=np.float32) + fgcmLUT.pwvStd
        self.parPWVPerSlope = np.zeros(self.campaignNights.size,dtype=np.float32)

        # parameters with per-epoch values
        self.parSuperStarFlat = np.zeros((self.nEpochs,self.nLUTFilter,self.nCCD,self.superStarNPar),dtype=np.float64)
        # The first term should be 1.0 with new flux units
        self.parSuperStarFlat[:, :, :, 0] = 1.0

        # parameters with per-wash values
        self.parQESysIntercept = np.zeros(self.nWashIntervals,dtype=np.float32)
        self.parQESysSlope = np.zeros(self.nWashIntervals,dtype=np.float32)

        ## FIXME: need to completely refactor
        self.externalPWVFlag = np.zeros(self.nExp,dtype=np.bool)
        if (self.pwvFile is not None):
            self.fgcmLog.info('Found external PWV file.')
            self.pwvFile = self.pwvFile
            self.hasExternalPWV = True
            self.loadExternalPWV(self.externalPWVDeltaT)
            # need to add two global parameters!
            #self.parExternalPWVScale = 1.0
            #self.parExternalPWVOffset = 0.0

        self.externalTauFlag = np.zeros(self.nExp,dtype=np.bool)
        if (self.tauFile is not None):
            self.fgcmLog.info('Found external tau file.')
            self.tauFile = self.tauFile
            self.hasExternalTau = True
            self.loadExternalTau()
            # need to add two global parameters!
            #self.parExternalTauScale = 1.0
            #self.parExternalTauOffset = 0.0

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

        # and the computed retrieved PWV
        # these are set to the standard values to start
        self.compRetrievedPWV = np.zeros(self.nExp,dtype='f8') + self.pwvStd
        self.compRetrievedPWVInput = self.compRetrievedPWV.copy()
        self.compRetrievedPWVRaw = np.zeros(self.nExp,dtype='f8')
        self.compRetrievedPWVFlag = np.zeros(self.nExp,dtype='i2') + retrievalFlagDict['EXPOSURE_STANDARD']
        self.parRetrievedPWVScale = 1.0
        self.parRetrievedPWVOffset = 0.0
        self.parRetrievedPWVNightlyOffset = np.zeros(self.nCampaignNights,dtype='f8')

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
                              'pwvUnit': inParInfo['PWVUNIT'][0],
                              'pwvPerSlopeUnit': inParInfo['PWVPERSLOPEUNIT'][0],
                              'pwvGlobalUnit': inParInfo['PWVGLOBALUNIT'][0],
                              'o3Unit': inParInfo['O3UNIT'][0],
                              'qeSysUnit': inParInfo['QESYSUNIT'][0],
                              'qeSysSlopeUnit': inParInfo['QESYSSLOPEUNIT'][0]}

        # and log
        self.fgcmLog.info('lnTau step unit set to %f' % (self.unitDictSteps['lnTauUnit']))
        self.fgcmLog.info('lnTau slope step unit set to %f' %
                         (self.unitDictSteps['lnTauSlopeUnit']))
        self.fgcmLog.info('alpha step unit set to %f' % (self.unitDictSteps['alphaUnit']))
        self.fgcmLog.info('pwv step unit set to %f' % (self.unitDictSteps['pwvUnit']))
        self.fgcmLog.info('pwv percent slope step unit set to %f' %
                         (self.unitDictSteps['pwvPerSlopeUnit']))
        self.fgcmLog.info('pwv global step unit set to %f' %
                          (self.unitDictSteps['pwvGlobalUnit']))
        self.fgcmLog.info('O3 step unit set to %f' % (self.unitDictSteps['o3Unit']))
        self.fgcmLog.info('wash step unit set to %f' % (self.unitDictSteps['qeSysUnit']))
        self.fgcmLog.info('wash slope step unit set to %f' %
                         (self.unitDictSteps['qeSysSlopeUnit']))

        # look at external...
        self.hasExternalPWV = inParInfo['HASEXTERNALPWV'][0].astype(np.bool)
        self.hasExternalTau = inParInfo['HASEXTERNALTAU'][0].astype(np.bool)

        ## and copy the parameters
        self.parAlpha = np.atleast_1d(inParams['PARALPHA'][0])
        self.parO3 = np.atleast_1d(inParams['PARO3'][0])
        self.parLnTauIntercept = np.atleast_1d(inParams['PARLNTAUINTERCEPT'][0])
        self.parLnTauSlope = np.atleast_1d(inParams['PARLNTAUSLOPE'][0])
        self.parPWVIntercept = np.atleast_1d(inParams['PARPWVINTERCEPT'][0])
        self.parPWVPerSlope = np.atleast_1d(inParams['PARPWVPERSLOPE'][0])
        self.parQESysIntercept = np.atleast_1d(inParams['PARQESYSINTERCEPT'][0])
        self.parQESysSlope = np.atleast_1d(inParams['PARQESYSSLOPE'][0])

        self.externalPWVFlag = np.zeros(self.nExp,dtype=np.bool)
        if self.hasExternalPWV:
            self.pwvFile = str(inParInfo['PWVFILE'][0]).rstrip()
            self.hasExternalPWV = True
            self.loadExternalPWV(self.externalPWVDeltaT)
            self.parExternalPWVScale = inParams['PAREXTERNALPWVSCALE'][0]
            self.parExternalPWVOffset[:] = np.atleast_1d(inParams['PAREXTERNALPWVOFFSET'][0])

        self.externalTauFlag = np.zeros(self.nExp,dtype=np.bool)
        if self.hasExternalTau:
            self.tauFile = str(inParInfo['TAUFILE'][0]).rstrip()
            self.hasExternalTau = True
            self.loadExternalTau()
            self.parExternalTauScale = inParams['PAREXTERNALTAUSCALE'][0]
            self.parExternalTauOffset[:] = np.atleast_1d(inParams['PAREXTERNALTAUOFFSET'][0])

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

        # These are exposure-level properties
        self.compRetrievedPWV = np.atleast_1d(inParams['COMPRETRIEVEDPWV'][0])
        self.compRetrievedPWVInput = self.compRetrievedPWV.copy()
        self.compRetrievedPWVRaw = np.atleast_1d(inParams['COMPRETRIEVEDPWVRAW'][0])
        self.compRetrievedPWVFlag = np.atleast_1d(inParams['COMPRETRIEVEDPWVFLAG'][0])
        self.parRetrievedPWVScale = inParams['PARRETRIEVEDPWVSCALE'][0]
        self.parRetrievedPWVOffset = inParams['PARRETRIEVEDPWVOFFSET'][0]
        self.parRetrievedPWVNightlyOffset = np.atleast_1d(inParams['PARRETRIEVEDPWVNIGHTLYOFFSET'][0])

        # These are nightly properties
        self.compRetrievedTauNight = np.atleast_1d(inParams['COMPRETRIEVEDTAUNIGHT'][0])
        self.compRetrievedTauNightInput = self.compRetrievedTauNight.copy()

        self._arrangeParArray()

        # need to load the superstarflats
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

    def resetAtmosphereParameters(self):
        self.fgcmLog.info("Resetting atmosphere parameters.")

        self.parAlpha[:] = self.alphaStd
        self.parO3[:] = self.o3Std
        self.parLnTauIntercept[:] = self.lnTauStd
        self.parLnTauSlope[:] = 0.0
        self.parPWVIntercept[:] = self.pwvStd
        self.parPWVPerSlope[:] = 0.0

        # We don't reset QESysIntercept and Slope because they aren't
        #  atmosphere parameters (they are instrument parameters)

        if self.hasExternalPWV:
            self.parExternalPWVScale = 1.0
            self.parExternalPWVOffset = 0.0

        if self.hasExternalTau:
            self.parExternalTauScale = 1.0
            self.parExternalTauOffset = 0.0

        self.parRetrievedPWVScale = 1.0
        self.parRetrievedPWVOffset = 0.0
        self.parRetrievedPWVNightlyOffset[:] = 0.0

        # If we are resetting parameters, and want to use retrieved tau as the initial
        #  guess, set parTauIntercept to that.
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

        self.expSeeingVariable = expInfo[self.seeingField]
        self.expDeepFlag = expInfo[self.deepFlag]

        try:
            self.expFwhm = expInfo[self.fwhmField]
            self.expSkyBrightness = expInfo[self.skyBrightnessField]
        except:
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
            except:
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
                         self.nCampaignNights)   # pwv Slope
        ctr=0
        self.parO3Loc = ctr
        ctr+=self.nCampaignNights
        self.parLnTauInterceptLoc = ctr
        ctr+=self.nCampaignNights
        self.parLnTauSlopeLoc = ctr
        ctr+=self.nCampaignNights
        self.parAlphaLoc = ctr
        ctr+=self.nCampaignNights
        if not self.useRetrievedPWV:
            self.parPWVInterceptLoc = ctr
            ctr+=self.nCampaignNights
            self.parPWVPerSlopeLoc = ctr
            ctr+=self.nCampaignNights

        if self.hasExternalPWV and not self.useRetrievedPWV:
            self.nFitPars += (1+self.nCampaignNights)
            self.parExternalPWVScaleLoc = ctr
            ctr+=1
            self.parExternalPWVOffsetLoc = ctr
            ctr+=self.nCampaignNights

        if self.hasExternalTau:
            self.nFitPars += (1+self.nCampaignNights)
            self.parExternalTauScaleLoc = ctr
            ctr+=1
            self.parExternalTauOffsetLoc = ctr
            ctr+=self.nCampaignNights

        if self.useRetrievedPWV:
            self.nFitPars += 2
            self.parRetrievedPWVScaleLoc = ctr
            ctr+=1
            if self.useNightlyRetrievedPWV:
                self.parRetrievedPWVNightlyOffsetLoc = ctr
                ctr+=self.nCampaignNights
            else:
                self.parRetrievedPWVOffsetLoc = ctr
                ctr+=1

        self.nFitPars += (self.nWashIntervals + # parQESysIntercept
                          self.nWashIntervals)  # parQESysSlope

        self.parQESysInterceptLoc = ctr
        ctr+=self.nWashIntervals
        self.parQESysSlopeLoc = ctr
        ctr+=self.nWashIntervals

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
               ('PWVUNIT','f8'),
               ('PWVPERSLOPEUNIT','f8'),
               ('PWVGLOBALUNIT','f8'),
               ('O3UNIT','f8'),
               ('QESYSUNIT','f8'),
               ('QESYSSLOPEUNIT','f8'),
               ('HASEXTERNALPWV','i2'),
               ('HASEXTERNALTAU','i2')]

        ## FIXME: change from these files...
        if (self.hasExternalPWV):
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
        parInfo['PWVUNIT'] = self.unitDictSteps['pwvUnit']
        parInfo['PWVPERSLOPEUNIT'] = self.unitDictSteps['pwvPerSlopeUnit']
        parInfo['PWVGLOBALUNIT'] = self.unitDictSteps['pwvGlobalUnit']
        parInfo['O3UNIT'] = self.unitDictSteps['o3Unit']
        parInfo['QESYSUNIT'] = self.unitDictSteps['qeSysUnit']
        parInfo['QESYSSLOPEUNIT'] = self.unitDictSteps['qeSysSlopeUnit']

        parInfo['HASEXTERNALPWV'] = self.hasExternalPWV
        if (self.hasExternalPWV):
            parInfo['PWVFILE'] = self.pwvFile
        parInfo['HASEXTERNALTAU'] = self.hasExternalTau
        if (self.hasExternalTau):
            parInfo['TAUFILE'] = self.tauFile

        dtype=[('PARALPHA','f8',self.parAlpha.size),
               ('PARO3','f8',self.parO3.size),
               ('PARPWVINTERCEPT','f8',self.parPWVIntercept.size),
               ('PARPWVPERSLOPE','f8',self.parPWVPerSlope.size),
               ('PARLNTAUINTERCEPT','f8',self.parLnTauIntercept.size),
               ('PARLNTAUSLOPE','f8',self.parLnTauSlope.size),
               ('PARQESYSINTERCEPT','f8',self.parQESysIntercept.size),
               ('PARQESYSSLOPE','f8',self.parQESysSlope.size),
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
               ('COMPRETRIEVEDPWV','f8',self.compRetrievedPWV.size),
               ('COMPRETRIEVEDPWVRAW','f8',self.compRetrievedPWVRaw.size),
               ('COMPRETRIEVEDPWVFLAG','i2',self.compRetrievedPWVFlag.size),
               ('PARRETRIEVEDPWVSCALE','f8'),
               ('PARRETRIEVEDPWVOFFSET','f8'),
               ('PARRETRIEVEDPWVNIGHTLYOFFSET','f8',self.parRetrievedPWVNightlyOffset.size),
               ('COMPRETRIEVEDTAUNIGHT','f8',self.compRetrievedTauNight.size)]

        if (self.hasExternalPWV):
            dtype.extend([('PAREXTERNALPWVSCALE','f8'),
                          ('PAREXTERNALPWVOFFSET','f8',self.parExternalPWVOffset.size),
                          ('EXTERNALPWV','f8',self.nExp)])
        if (self.hasExternalTau):
            dtype.extend([('PAREXTERNALTAUSCALE','f8'),
                          ('PAREXTERNALTAUOFFSET','f8',self.parExternalTauOffset.size),
                          ('EXTERNALTAU','f8',self.nExp)])

        pars=np.zeros(1,dtype=dtype)

        pars['PARALPHA'][:] = self.parAlpha
        pars['PARO3'][:] = self.parO3
        pars['PARLNTAUINTERCEPT'][:] = self.parLnTauIntercept
        pars['PARLNTAUSLOPE'][:] = self.parLnTauSlope
        pars['PARPWVINTERCEPT'][:] = self.parPWVIntercept
        pars['PARPWVPERSLOPE'][:] = self.parPWVPerSlope
        pars['PARQESYSINTERCEPT'][:] = self.parQESysIntercept
        pars['PARQESYSSLOPE'][:] = self.parQESysSlope

        if (self.hasExternalPWV):
            pars['PAREXTERNALPWVSCALE'] = self.parExternalPWVScale
            pars['PAREXTERNALPWVOFFSET'][:] = self.parExternalPWVOffset
            pars['EXTERNALPWV'][:] = self.externalPWV
        if (self.hasExternalTau):
            pars['PAREXTERNALTAUSCALE'] = self.parExternalTauScale
            pars['PAREXTERNALTAUOFFSET'][:] = self.parExternalTauOffset
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

        pars['COMPRETRIEVEDPWV'][:] = self.compRetrievedPWV
        pars['COMPRETRIEVEDPWVRAW'][:] = self.compRetrievedPWVRaw
        pars['COMPRETRIEVEDPWVFLAG'][:] = self.compRetrievedPWVFlag
        pars['PARRETRIEVEDPWVSCALE'][:] = self.parRetrievedPWVScale
        pars['PARRETRIEVEDPWVOFFSET'][:] = self.parRetrievedPWVOffset
        pars['PARRETRIEVEDPWVNIGHTLYOFFSET'][:] = self.parRetrievedPWVNightlyOffset

        pars['COMPRETRIEVEDTAUNIGHT'][:] = self.compRetrievedTauNight

        return parInfo, pars

    def loadExternalPWV(self, externalPWVDeltaT):
        """
        Load external PWV measurements, from fits table.

        parameters
        ----------
        externalPWVDeltaT: float
           Maximum delta-T (days) for external PWV measurement to match exposure MJD.
        """

        import fitsio

        # loads a file with PWV, matches to exposures/times
        # flags which ones need the nightly fit

        #self.hasExternalPWV = True

        pwvTable = fitsio.read(self.pwvFile,ext=1)

        # make sure it's sorted
        st=np.argsort(pwvTable['MJD'])
        pwvTable = pwvTable[st]

        pwvIndex = np.clip(np.searchsorted(pwvTable['MJD'],self.expMJD),0,pwvTable.size-1)
        # this will be True or False...
        self.externalPWVFlag[:] = (np.abs(pwvTable['MJD'][pwvIndex] - self.expMJD) < externalPWVDeltaT)
        self.externalPWV = np.zeros(self.nExp,dtype=np.float32)
        self.externalPWV[self.externalPWVFlag] = pwvTable['PWV'][pwvIndex[self.externalPWVFlag]]

        # and new PWV scaling pars!
        self.parExternalPWVOffset = np.zeros(self.nCampaignNights,dtype=np.float32)
        self.parExternalPWVScale = 1.0

        match, = np.where(self.externalPWVFlag)
        self.fgcmLog.info('%d exposures of %d have external pwv values' % (match.size,self.nExp))


    def loadExternalTau(self, withAlpha=False):
        """
        Not Supported.
        """

        # load a file with Tau values
        ## not supported yet
        raise ValueError("externalTau Not supported yet")

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

        if not self.useRetrievedPWV:
            self.parPWVIntercept[:] = parArray[self.parPWVInterceptLoc:
                                                   self.parPWVInterceptLoc+self.nCampaignNights] / unitDict['pwvUnit']
            self.parPWVPerSlope[:] = parArray[self.parPWVPerSlopeLoc:
                                                  self.parPWVPerSlopeLoc + self.nCampaignNights] / unitDict['pwvPerSlopeUnit']

        self.parO3[:] = parArray[self.parO3Loc:
                                     self.parO3Loc+self.nCampaignNights] / unitDict['o3Unit']
        self.parLnTauIntercept[:] = parArray[self.parLnTauInterceptLoc:
                                                 self.parLnTauInterceptLoc+self.nCampaignNights] / unitDict['lnTauUnit']
        self.parLnTauSlope[:] = (parArray[self.parLnTauSlopeLoc:
                                               self.parLnTauSlopeLoc + self.nCampaignNights] /
                                 unitDict['lnTauSlopeUnit'])

        self.parAlpha[:] = parArray[self.parAlphaLoc:
                                        self.parAlphaLoc+self.nCampaignNights] / unitDict['alphaUnit']
        if self.hasExternalPWV and not self.useRetrievedPWV:
            self.parExternalPWVScale = parArray[self.parExternalPWVScaleLoc] / unitDict['pwvGlobalUnit']
            self.parExternalPWVOffset = parArray[self.parExternalPWVOffsetLoc:
                                                     self.parExternalPWVOffsetLoc+self.nCampaignNights] / unitDict['pwvUnit']

        if (self.hasExternalTau):
            raise RuntimeError("Not implemented")
            #self.parExternalTauScale = parArray[self.parExternalTauScaleLoc] / unitDict['tauUnit']
            #self.parExternalTauOffset = parArray[self.parExternalTauOffsetLoc:
            #                                         self.parExternalTauOffsetLoc+self.nCampaignNights] / unitDict['tauUnit']

        if self.useRetrievedPWV:
            self.parRetrievedPWVScale = parArray[self.parRetrievedPWVScaleLoc] / unitDict['pwvGlobalUnit']
            if self.useNightlyRetrievedPWV:
                self.parRetrievedPWVNightlyOffset = parArray[self.parRetrievedPWVNightlyOffsetLoc:
                                                                 self.parRetrievedPWVNightlyOffsetLoc + self.nCampaignNights] / unitDict['pwvUnit']
            else:
                self.parRetrievedPWVOffset = parArray[self.parRetrievedPWVOffsetLoc] / unitDict['pwvGlobalUnit']

        self.parQESysIntercept[:] = parArray[self.parQESysInterceptLoc:
                                                 self.parQESysInterceptLoc+self.nWashIntervals] / unitDict['qeSysUnit']


        self.parQESysSlope[:] = parArray[self.parQESysSlopeLoc:
                                             self.parQESysSlopeLoc+self.nWashIntervals] / unitDict['qeSysSlopeUnit']
        # done

    def parsToExposures(self, retrievedInput=False):
        """
        Associate parameters with exposures.

        parameters
        ----------
        retrievedInput: bool, default=False
           When useRetrievedPWV, do we use the input or final values as the basis for the conversion?

        Output attributes
        -----------------
        expO3: float array
        expAlpha: float array
        expPWV: float array
        expLnTau: float array
        expQESys: float array
        """

        self.fgcmLog.debug('Computing exposure values from parameters')

        # I'm guessing that these don't need to be wrapped in shms but I could be wrong
        #  about the full class, which would suck.

        # first, the nightly parameters without selection...
        self.expO3 = self.parO3[self.expNightIndex]
        self.expAlpha = self.parAlpha[self.expNightIndex]

        if self.useRetrievedPWV:
            # FIXME
            if retrievedInput:
                retrievedPWV = self.compRetrievedPWVInput
            else:
                retrievedPWV = self.compRetrievedPWV

            if self.useNightlyRetrievedPWV:
                self.expPWV = (self.parRetrievedPWVNightlyOffset[self.expNightIndex] +
                               self.parRetrievedPWVScale * retrievedPWV)
            else:
                self.expPWV = (self.parRetrievedPWVOffset +
                               self.parRetrievedPWVScale * retrievedPWV)
        else:
            # default to the nightly slope/intercept
            self.expPWV = (self.parPWVIntercept[self.expNightIndex] +
                           (self.parPWVPerSlope[self.expNightIndex] *
                            self.parPWVIntercept[self.expNightIndex]) * self.expDeltaUT)

            if (self.hasExternalPWV):
                # replace where we have these
                self.expPWV[self.externalPWVFlag] = (self.parExternalPWVOffset[self.expNightIndex[self.externalPWVFlag]] +
                                                     self.parExternalPWVScale *
                                                     self.externalPWV[self.externalPWVFlag])

        # default to nightly slope/intercept
        #self.expTau = (self.parTauIntercept[self.expNightIndex] +
        #               (self.parTauPerSlope[self.expNightIndex] *
        #                self.parTauIntercept[self.expNightIndex]) * self.expDeltaUT)

        self.expLnTau = (self.parLnTauIntercept[self.expNightIndex] +
                         self.parLnTauSlope[self.expNightIndex] * self.expDeltaUT)

        if (self.hasExternalTau):
            raise RuntimeError("not implemented")
            # replace where we have these
            #self.expTau[self.externalTauFlag] = (self.parExternalTauOffset[self.expNightIndex[self.externalTauFlag]] +
            #                                     self.parExternalTauScale *
            #                                     self.externalTau[self.externalTauFlag])

        # and clip to make sure it doesn't go negative
        #self.expTau = np.clip(self.expTau, self.tauRange[0]+0.0001, self.tauRange[1]-0.0001)
        self.expLnTau = np.clip(self.expLnTau, self.lnTauRange[0], self.lnTauRange[1])

        # and QESys
        self.expQESys = (self.parQESysIntercept[self.expWashIndex] +
                         self.parQESysSlope[self.expWashIndex] *
                         (self.expMJD - self.washMJDs[self.expWashIndex]))

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

        if not self.useRetrievedPWV:
            parArray[self.parPWVInterceptLoc:
                         self.parPWVInterceptLoc+self.nCampaignNights] = self.parPWVIntercept[:] * unitDict['pwvUnit']
            parArray[self.parPWVPerSlopeLoc:
                         self.parPWVPerSlopeLoc + self.nCampaignNights] = self.parPWVPerSlope[:] * unitDict['pwvPerSlopeUnit']

        parArray[self.parO3Loc:
                     self.parO3Loc+self.nCampaignNights] = self.parO3[:] * unitDict['o3Unit']
        parArray[self.parLnTauInterceptLoc:
                     self.parLnTauInterceptLoc+self.nCampaignNights] = self.parLnTauIntercept[:] * unitDict['lnTauUnit']
        parArray[self.parLnTauSlopeLoc:
                     self.parLnTauSlopeLoc + self.nCampaignNights] = self.parLnTauSlope[:] * unitDict['lnTauSlopeUnit']
        parArray[self.parAlphaLoc:
                     self.parAlphaLoc+self.nCampaignNights] = self.parAlpha[:] * unitDict['alphaUnit']
        if self.hasExternalPWV and not self.useRetrievedPWV:
            parArray[self.parExternalPWVScaleLoc] = self.parExternalPWVScale * unitDict['pwvGlobalUnit']
            parArray[self.parExternalPWVOffsetLoc:
                         self.parExternalPWVOffsetLoc+self.nCampaignNights] = self.parExternalPWVOffset * unitDict['pwvUnit']
        if (self.hasExternalTau):
            raise RuntimeError("not implemented")
            #parArray[self.parExternalTauScaleLoc] = self.parExternalTauScale * unitDict['lnTauUnit']
            #parArray[self.parExternalTauOffsetLoc:
            #             self.parExternalTauOffsetLoc+self.nCampaignNights] = self.parExternalTauOffset * unitDict['tauUnit']
        if self.useRetrievedPWV:
            parArray[self.parRetrievedPWVScaleLoc] = self.parRetrievedPWVScale * unitDict['pwvGlobalUnit']
            if self.useNightlyRetrievedPWV:
                parArray[self.parRetrievedPWVNightlyOffsetLoc:
                             self.parRetrievedPWVNightlyOffsetLoc+self.nCampaignNights] = self.parRetrievedPWVNightlyOffset * unitDict['pwvUnit']
            else:
                parArray[self.parRetrievedPWVOffsetLoc] = self.parRetrievedPWVOffset * unitDict['pwvGlobalUnit']

        parArray[self.parQESysInterceptLoc:
                     self.parQESysInterceptLoc+self.nWashIntervals] = self.parQESysIntercept * unitDict['qeSysUnit']
        parArray[self.parQESysSlopeLoc:
                     self.parQESysSlopeLoc+self.nWashIntervals] = self.parQESysSlope * unitDict['qeSysSlopeUnit']

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

        if not self.useRetrievedPWV:
            parLow[self.parPWVInterceptLoc: \
                       self.parPWVInterceptLoc + \
                       self.nCampaignNights] = ( \
                self.pwvRange[0] * unitDict['pwvUnit'])
            parHigh[self.parPWVInterceptLoc: \
                        self.parPWVInterceptLoc + \
                        self.nCampaignNights] = ( \
                self.pwvRange[1] * unitDict['pwvUnit'])
            #parLow[self.parPWVPerSlopeLoc: \
            #           self.parPWVPerSlopeLoc + \
            #           self.nCampaignNights] = ( \
            #    -4.0 * unitDict['pwvPerSlopeUnit'])
            # we don't want the PWV to go below 0, so we need to set this
            # limit based on the maximum deltaUT on each night.
            parLow[self.parPWVPerSlopeLoc: \
                       self.parPWVPerSlopeLoc + \
                       self.nCampaignNights] = ( \
                (-1.0 / self.maxDeltaUTPerNight) * unitDict['pwvPerSlopeUnit'])
            parHigh[self.parPWVPerSlopeLoc: \
                        self.parPWVPerSlopeLoc + \
                        self.nCampaignNights] = ( \
                4.0 * unitDict['pwvPerSlopeUnit'])
        else:
            parLow[self.parRetrievedPWVScaleLoc] = 0.5 * unitDict['pwvGlobalUnit']
            parHigh[self.parRetrievedPWVScaleLoc] = 1.5 * unitDict['pwvGlobalUnit']
            if self.useNightlyRetrievedPWV:
                parLow[self.parRetrievedPWVNightlyOffsetLoc: \
                           self.parRetrievedPWVNightlyOffsetLoc + \
                           self.nCampaignNights] = ( \
                    -2.0 * unitDict['pwvUnit'])
                parHigh[self.parRetrievedPWVNightlyOffsetLoc: \
                           self.parRetrievedPWVNightlyOffsetLoc + \
                           self.nCampaignNights] = ( \
                    2.0 * unitDict['pwvUnit'])
            else:
                parLow[self.parRetrievedPWVOffsetLoc] = -2.0 * unitDict['pwvGlobalUnit']
                parHigh[self.parRetrievedPWVOffsetLoc] = 2.0 * unitDict['pwvGlobalUnit']

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
        if self.hasExternalPWV and not self.useRetrievedPWV:
            parLow[self.parExternalPWVScaleLoc] = 0.5 * unitDict['pwvGlobalUnit']
            parHigh[self.parExternalPWVScaleLoc] = 1.5 * unitDict['pwvGlobalUnit']
            parLow[self.parExternalPWVOffsetLoc: \
                       self.parExternalPWVOffsetLoc + \
                       self.nCampaignNights] = ( \
                -1.5 * unitDict['pwvUnit'])
            parHigh[self.parExternalPWVOffsetLoc: \
                       self.parExternalPWVOffsetLoc + \
                        self.nCampaignNights] = ( \
                3.0 * unitDict['pwvUnit'])
            ## FIXME: set bounds per night?  Or clip?
        if (self.hasExternalTau):
            raise RuntimeError("not implemented")
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

        if self.freezeStdAtmosphere:
            # atmosphere parameters set to std values
            if not self.useRetrievedPWV:
                parLow[self.parPWVInterceptLoc: \
                           self.parPWVInterceptLoc + \
                           self.nCampaignNights] = self.pwvStd * unitDict['pwvUnit']
                parHigh[self.parPWVInterceptLoc: \
                            self.parPWVInterceptLoc + \
                            self.nCampaignNights] = self.pwvStd * unitDict['pwvUnit']
                parLow[self.parPWVPerSlopeLoc: \
                           self.parPWVPerSlopeLoc + \
                           self.nCampaignNights] = 0.0
                parHigh[self.parPWVPerSlopeLoc: \
                            self.parPWVPerSlopeLoc + \
                            self.nCampaignNights] = 0.0
            else:
                parLow[self.parRetrievedPWVScaleLoc] = 1.0 * unitDict['pwvGlobalUnit']
                parHigh[self.parRetrievedPWVScaleLoc] = 1.0 * unitDict['pwvGlobalUnit']
                if self.useNightlyRetrievedPWV:
                    parLow[self.parRetrievedPWVNightlyOffsetLoc: \
                               self.parRetrievedPWVNightlyOffsetLoc + \
                               self.nCampaignNights] = 0.0
                    parHigh[self.parRetrievedPWVNightlyOffsetLoc: \
                                self.parRetrievedPWVNightlyOffsetLoc + \
                                self.nCampaignNights] = 0.0
                else:
                    parLow[self.parRetrievedPWVOffsetLoc] = 0.0
                    parHigh[self.parRetrievedPWVOffsetLoc] = 0.0

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

        from .fgcmUtilities import poly2dFunc, cheb2dFunc

        # this is the version that does the center of the CCD
        # because it is operating on the whole CCD!

        superStarFlatCenter = np.zeros((self.nEpochs,
                                        self.nLUTFilter,
                                        self.nCCD))
        for e in xrange(self.nEpochs):
            for f in xrange(self.nLUTFilter):
                for c in xrange(self.nCCD):
                    if self.superStarPoly2d:
                        xy = np.vstack((self.ccdOffsets['X_SIZE'][c]/2.,
                                        self.ccdOffsets['Y_SIZE'][c]/2.))
                        superStarFlatCenter[e, f, c] = poly2dFunc(xy,
                                                                  *self.parSuperStarFlat[e, f, c, :])
                    else:
                        superStarFlatCenter[e, f, c] = -2.5 * np.log10(cheb2dFunc(np.vstack((0.0, 0.0)),
                                                                                  *self.parSuperStarFlat[e, f, c, :]))

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
                  self.expPWV[expUse])
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

        ## FIXME: add pwv offset plotting routine (if external)
        ## FIXME: add tau offset plotting routing (if external)

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state
