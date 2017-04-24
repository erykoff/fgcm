from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil

import matplotlib.pyplot as plt

from fgcmUtilities import _pickle_method
from fgcmUtilities import expFlagDict

import types
import copy_reg

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

from fgcmLUT import FgcmLUTSHM

copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmParameters(object):
    """
    """
    #def __init__(self,parFile=None,
    #             fgcmConfig=None):
    def __init__(self,fgcmConfig,parFile=None):

        self.hasExternalPWV = False
        self.hasExternalTau = False

        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.plotPath = fgcmConfig.plotPath

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.log('INFO','Initializing FgcmParameters...')

        # for plotting
        self.minExpPerNight = fgcmConfig.minExpPerNight

        #if (fgcmConfig is not None):
        #    self._initializeParameters(fgcmConfig)
        if (parFile is not None):
            self._loadParFile(fgcmConfig,parFile)
        else:
            self._initializeParameters(fgcmConfig)


    def _initializeParameters(self, fgcmConfig):
        """
        """
        # initialize parameters from a config dictionary
        # will need to know the following:
        #   all exposure numbers (from demand list) [done]
        #   all exposure MJDs [done]
        #   UT boundary for nights (From config) [done]
        #   all exposure pressure values [done-ish]
        #   all exposure ZDs (-> secZenith and Airmass)
        #   all exposure bands (and link index) [done]
        #   all exposure exptimes [done]
        #   all exposure psf_fwhms -- or delta-aperture [done]
        #   all wash dates (will need to crop to exposure range) and link [done]
        #   all epochs (will need to crop to exposure range) and link [done]
        #   flag for special (e.g. SN) exposures? [done]
        #   lutfile here?  no.
        #   flag for exposure quality: 0 is good, and numbers for
        #      rejections of various types
        #     256 - bad band

        # default is to have a pwv_int, pwv_slope per night [done]
        #  but will change to different parameters if loadExternalPWV
        # default is to have a tau_int, tau_slope per night [done]
        #  but will change to different parameters if loadExternalTau
        # default is to have an alpha per night [done]
        #  but will change to ... something if loadExternalAlpha
        # default is to have an Ozone per night [done]
        #  but will change to global additive/multiplicative factor if loadExternalOzone

        # need an index to get quickly from exposure number to night

        #######################################################
        #######################################################

        # record necessary info here...
        self.nCCD = fgcmConfig.nCCD
        self.bands = fgcmConfig.bands
        self.nBands = self.bands.size
        self.fitBands = fgcmConfig.fitBands
        self.nFitBands = self.fitBands.size
        self.extraBands = fgcmConfig.extraBands
        self.nExtraBands = self.extraBands.size

        self._makeBandIndices()

        # first thing is to get the exposure numbers...
        self.exposureFile = fgcmConfig.exposureFile

        self._loadExposureInfo(fgcmConfig)

        # set up the observing epochs and link indices

        self._loadEpochAndWashInfo(fgcmConfig)

        # set up the parameters with nightly values
        # need to include the default stuff...

        self.parAlpha = np.zeros(self.campaignNights.size,dtype=np.float32) + fgcmConfig.alphaStd
        self.parO3 = np.zeros(self.campaignNights.size,dtype=np.float32) + fgcmConfig.o3Std
        self.parTauIntercept = np.zeros(self.campaignNights.size,dtype=np.float32) + fgcmConfig.tauStd
        self.parTauSlope = np.zeros(self.campaignNights.size,dtype=np.float32)
        self.parPWVIntercept = np.zeros(self.campaignNights.size,dtype=np.float32) + fgcmConfig.pwvStd
        self.parPWVSlope = np.zeros(self.campaignNights.size,dtype=np.float32)

        # parameters with per-epoch values
        self.parSuperStarFlat = np.zeros((self.nEpochs,self.nBands,self.nCCD),dtype=np.float32)

        # parameters with per-wash values
        self.parQESysIntercept = np.zeros(self.nWashIntervals,dtype=np.float32)
        self.parQESysSlope = np.zeros(self.nWashIntervals,dtype=np.float32)

        self.externalPWVFlag = np.zeros(self.nExp,dtype=np.bool)
        if (fgcmConfig.pwvFile is not None):
            self.fgcmLog.log('INFO','Found external PWV file.')
            self.pwvFile = fgcmConfig.pwvFile
            self.hasExternalPWV = True
            self.loadExternalPWV(fgcmConfig.externalPWVDeltaT)
            # need to add two global parameters!
            self.parExternalPWVScale = 1.0
            self.parExternalPWVOffset = 0.0

        self.externalTauFlag = np.zeros(self.nExp,dtype=np.bool)
        if (fgcmConfig.tauFile is not None):
            self.fgcmLog.log('INFO','Found external tau file.')
            self.tauFile = fgcmConfig.tauFile
            self.hasExternalTau = True
            self.loadExternalTau()
            # need to add two global parameters!
            self.parExternalTauScale = 1.0
            self.parExternalTauOffset = 0.0

        # and the aperture corrections
        self.compAperCorrPivot = np.zeros(self.nBands,dtype='f8')
        self.compAperCorrSlope = np.zeros(self.nBands,dtype='f8')
        self.compAperCorrSlopeErr = np.zeros(self.nBands,dtype='f8')
        self.compAperCorrRange = np.zeros((2,self.nBands),dtype='f8')

        # one of the "parameters" is expGray
        self.compExpGray = np.zeros(self.nExp,dtype='f8')
        self.compVarGray = np.zeros(self.nExp,dtype='f8')
        self.compNGoodStarPerExp = np.zeros(self.nExp,dtype='i4')

        # and compute the units...
        self._computeStepUnits(fgcmConfig)

        # and need to be able to pack and unpack the parameters and scalings
        #  this part is going to be the hardest

        self._arrangeParArray()
        self._setParRanges(fgcmConfig)


    def _arrangeParArray(self):
        # make pointers to a fit parameter array...
        #  pwv, O3, lnTau, alpha
        self.nFitPars = (self.campaignNights.size +  # O3
                         self.campaignNights.size +  # tauIntercept
                         self.campaignNights.size +  # tauSlope
                         self.campaignNights.size +  # alpha
                         self.campaignNights.size +  # pwv Intercept
                         self.campaignNights.size)   # pwv Slope
        ctr=0
        self.parO3Loc = ctr
        ctr+=self.campaignNights.size
        self.parTauInterceptLoc = ctr
        ctr+=self.campaignNights.size
        self.parTauSlopeLoc = ctr
        ctr+=self.campaignNights.size
        self.parAlphaLoc = ctr
        ctr+=self.campaignNights.size
        self.parPWVInterceptLoc = ctr
        ctr+=self.campaignNights.size
        self.parPWVSlopeLoc = ctr
        ctr+=self.campaignNights.size

        if (self.hasExternalPWV):
            self.nFitPars += (1+self.campaignNights.size)
            self.parExternalPWVScaleLoc = ctr
            ctr+=1
            self.parExternalPWVOffsetLoc = ctr
            ctr+=self.campaignNights.size

        if (self.hasExternalTau):
            self.nFitPars += (1+self.campaignNights.size)
            self.parExternalTauScaleLoc = ctr
            ctr+=1
            self.parExternalTauOffsetLoc = ctr
            ctr+=self.campaignNights.size

        self.nFitPars += (self.nWashIntervals + # parQESysIntercept
                          self.nWashIntervals)  # parQESysSlope

        self.parQESysInterceptLoc = ctr
        ctr+=self.nWashIntervals
        self.parQESysSlopeLoc = ctr
        ctr+=self.nWashIntervals

    def _setParRanges(self,fgcmConfig):
        """
        """

        self.pmbRange = fgcmConfig.pmbRange
        self.pwvRange = fgcmConfig.pwvRange
        self.O3Range = fgcmConfig.O3Range
        self.tauRange = fgcmConfig.tauRange
        self.alphaRange = fgcmConfig.alphaRange
        self.zenithRange = fgcmConfig.zenithRange


    def _makeBandIndices(self):
        """
        """

        self.bandIndex = np.arange(self.nBands,dtype='i2')
        self.fitBandIndex = np.zeros(self.nFitBands,dtype='i2')
        self.extraBandIndex = np.zeros(self.nExtraBands,dtype='i2')

        bandStrip = np.core.defchararray.strip(self.bands[:])
        for i in xrange(self.nFitBands):
            u,=np.where(self.fitBands[i] == self.bands)
            if (u.size == 0):
                raise ValueError("fitBand %s not in list of bands!" % (self.fitBands[i]))
            self.fitBandIndex[i] = u[0]

        for i in xrange(self.nExtraBands):
            u,=np.where(self.extraBands[i] == self.bands)
            if (u.size == 0):
                raise ValueError("extraBand %s not in list of bands!" % (self.extraBands[i]))
            self.extraBandIndex[i] = u[0]

    def _loadExposureInfo(self,fgcmConfig):
        """
        """

        expInfo = fitsio.read(self.exposureFile,ext=1)

        # ensure sorted by exposure number
        st=np.argsort(expInfo['EXPNUM'])
        expInfo=expInfo[st]

        self.nExp = fgcmConfig.nExp

        self.fgcmLog.log('INFO','Loading info on %d exposures.' % (self.nExp))

        self.expArray = expInfo['EXPNUM']
        self.expFlag = np.zeros(self.nExp,dtype=np.int8)
        self.expExptime = expInfo['EXPTIME']

        self.expSeeingVariable = expInfo[fgcmConfig.seeingField]
        self.expDeepFlag = expInfo[fgcmConfig.deepFlag]

        # we need the nights of the survey (integer MJD, maybe rotated)
        self.expMJD = expInfo['MJD']
        mjdForNight = np.floor(self.expMJD + fgcmConfig.UTBoundary).astype(np.int32)
        self.campaignNights = np.unique(mjdForNight)
        self.nCampaignNights = self.campaignNights.size

        self.fgcmLog.log('INFO','Exposures taken on %d nights.' % (self.nCampaignNights))

        self.expDeltaUT = (self.expMJD + fgcmConfig.UTBoundary) - mjdForNight

        # and link the exposure numbers to the nights...
        a,b=esutil.numpy_util.match(self.campaignNights,mjdForNight)
        self.expNightIndex = np.zeros(self.nExp,dtype=np.int32)
        self.expNightIndex[b] = a

        # we need the duration of each night...
        self.nightDuration = np.zeros(self.nCampaignNights)
        self.expPerNight = np.zeros(self.nCampaignNights,dtype=np.int32)
        for i in xrange(self.nCampaignNights):
            use,=np.where(mjdForNight == self.campaignNights[i])
            self.expPerNight[i] = use.size
            # night duration in hours
            self.nightDuration[i] = (np.max(self.expMJD[use]) - np.min(self.expMJD[use])) * 24.0
        self.meanNightDuration = np.mean(self.nightDuration)  # hours
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
        self.sinLatitude = np.sin(np.radians(fgcmConfig.latitude))
        self.cosLatitude = np.cos(np.radians(fgcmConfig.latitude))

        self.expPmb = expInfo['PMB']

        # link exposures to bands
        self.expBandIndex = np.zeros(self.nExp,dtype='i2') - 1
        for i in xrange(self.bands.size):
            use,=np.where(self.bands[i] == np.core.defchararray.strip(expInfo['BAND']))
            self.expBandIndex[use] = i

        bad,=np.where(self.expBandIndex < 0)
        if (bad.size > 0):
            self.fgcmLog.log('INFO','***Warning: %d exposures with band not in LUT!' % (bad.size))
            self.expFlag[bad] = self.expFlag[bad] | expFlagDict['BAND_NOT_IN_LUT']

        # flag those that have extra bands
        self.expExtraBandFlag = np.zeros(self.nExp,dtype=np.bool)
        if (self.nExtraBands > 0) :
            a,b=esutil.numpy_util.match(self.extraBandIndex,self.expBandIndex)
            self.expExtraBandFlag[b] = True

        # set up the observing epochs and link indices

        # the epochs should contain all the MJDs.
        self.nEpochs = fgcmConfig.epochMJDs.size - 1

        self.expEpochIndex = np.zeros(self.nExp,dtype='i4')
        for i in xrange(self.nEpochs):
            use,=np.where((self.expMJD > fgcmConfig.epochMJDs[i]) &
                          (self.expMJD < fgcmConfig.epochMJDs[i+1]))
            self.expEpochIndex[use] = i

    def _loadEpochAndWashInfo(self,fgcmConfig):
        """
        """
        # the epochs should contain all the MJDs.
        self.nEpochs = fgcmConfig.epochMJDs.size - 1

        self.expEpochIndex = np.zeros(self.nExp,dtype='i4')
        for i in xrange(self.nEpochs):
            use,=np.where((self.expMJD > fgcmConfig.epochMJDs[i]) &
                          (self.expMJD < fgcmConfig.epochMJDs[i+1]))
            self.expEpochIndex[use] = i

        # and set up the wash mjds and link indices
        # the first "washMJD" is set to the first exposure date.
        # the number of *intervals* is one less than the dates?

        self.nWashIntervals = fgcmConfig.washMJDs.size+1
        self.washMJDs = np.insert(fgcmConfig.washMJDs,0,np.min(self.expMJD)-1.0)

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



    def _computeStepUnits(self,fgcmConfig):
        """
        """
        self.stepUnitReference = fgcmConfig.stepUnitReference
        self.stepGrain = fgcmConfig.stepGrain

        LUT = FgcmLUTSHM(fgcmConfig.lutFile)

        secZenithStd = 1./np.cos(LUT.zenithStd*np.pi/180.)

        # compute tau units
        deltaMagTau = (2.5*np.log10(np.exp(-secZenithStd*LUT.tauStd)) -
                       2.5*np.log10(np.exp(-secZenithStd*(LUT.tauStd+1.0))))
        self.tauStepUnits = np.abs(deltaMagTau) / self.stepUnitReference / self.stepGrain
        self.fgcmLog.log('INFO','tau step unit set to %f' % (self.tauStepUnits))

        # and the tau slope units
        self.tauSlopeStepUnits = self.tauStepUnits * self.meanNightDuration
        self.fgcmLog.log('INFO','tau slope step unit set to %f' % (self.tauSlopeStepUnits))

        # alpha units -- reference to g, or r if not available
        ## FIXME: will need to allow band names other than g, r
        bandIndex,=np.where(self.bands == 'g')
        if bandIndex.size == 0:
            bandIndex,=np.where(self.bands == 'r')
            if bandIndex.size == 0:
                raise ValueError("Must have either g or r band...")

        deltaMagAlpha = (2.5*np.log10(np.exp(-secZenithStd*LUT.tauStd*(LUT.lambdaStd[bandIndex]/LUT.lambdaNorm)**LUT.alphaStd)) -
                         2.5*np.log10(np.exp(-secZenithStd*LUT.tauStd*(LUT.lambdaStd[bandIndex]/LUT.lambdaNorm)**(LUT.alphaStd+1.0))))
        self.alphaStepUnits = np.abs(deltaMagAlpha[0]) / self.stepUnitReference /self.stepGrain

        # scale by fraction of bands are affected...
        use,=np.where((self.fitBands == 'u') |
                      (self.fitBands == 'g') |
                      (self.fitBands == 'r'))
        self.alphaStepUnits *= float(use.size) / float(self.nFitBands)
        self.fgcmLog.log('INFO','alpha step unit set to %f' % (self.alphaStepUnits))

        # pwv units -- reference to z
        bandIndex,=np.where(self.bands == 'z')
        if bandIndex.size == 0:
            raise ValueError("Require z band for PWV ...")

        indicesStd = LUT.getIndices(bandIndex,LUT.pwvStd,LUT.o3Std,np.log(LUT.tauStd),LUT.alphaStd,secZenithStd,LUT.nCCD,LUT.pmbStd)
        i0Std = LUT.computeI0(bandIndex,LUT.pwvStd,LUT.o3Std,np.log(LUT.tauStd),LUT.alphaStd,secZenithStd,LUT.nCCD,LUT.pmbStd,indicesStd)
        indicesPlus = LUT.getIndices(bandIndex,LUT.pwvStd+1.0,LUT.o3Std,np.log(LUT.tauStd),LUT.alphaStd,secZenithStd,LUT.nCCD,LUT.pmbStd)
        i0Plus = LUT.computeI0(bandIndex,LUT.pwvStd+1.0,LUT.o3Std,np.log(LUT.tauStd),LUT.alphaStd,secZenithStd,LUT.nCCD,LUT.pmbStd,indicesPlus)
        deltaMagPWV = 2.5*np.log10(i0Std) - 2.5*np.log10(i0Plus)
        self.pwvStepUnits = np.abs(deltaMagPWV[0]) / self.stepUnitReference / self.stepGrain

        # scale by fraction of bands that are affected
        use,=np.where((self.fitBands == 'z') |
                      (self.fitBands == 'Y'))
        self.pwvStepUnits *= float(use.size) / float(self.nFitBands)
        self.fgcmLog.log('INFO','pwv step unit set to %f' % (self.pwvStepUnits))

        # PWV slope units
        self.pwvSlopeStepUnits = self.pwvStepUnits * self.meanNightDuration
        self.fgcmLog.log('INFO','pwv slope step unit set to %f' % (self.pwvSlopeStepUnits))

        # O3 units -- reference to r
        bandIndex,=np.where(self.bands == 'r')
        if bandIndex.size == 0:
            raise ValueError("Require r band for O3...")

        indicesStd = LUT.getIndices(bandIndex,LUT.pwvStd,LUT.o3Std,np.log(LUT.tauStd),LUT.alphaStd,secZenithStd,LUT.nCCD,LUT.pmbStd)
        i0Std = LUT.computeI0(bandIndex,LUT.pwvStd,LUT.o3Std,np.log(LUT.tauStd),LUT.alphaStd,secZenithStd,LUT.nCCD,LUT.pmbStd,indicesStd)
        indicesPlus = LUT.getIndices(bandIndex,LUT.pwvStd,LUT.o3Std+1.0,np.log(LUT.tauStd),LUT.alphaStd,secZenithStd,LUT.nCCD,LUT.pmbStd)
        i0Plus = LUT.computeI0(bandIndex,LUT.pwvStd,LUT.o3Std+1.0,np.log(LUT.tauStd),LUT.alphaStd,secZenithStd,LUT.nCCD,LUT.pmbStd,indicesPlus)
        deltaMagO3 = 2.5*np.log10(i0Std) - 2.5*np.log10(i0Plus)
        self.o3StepUnits = np.abs(deltaMagO3[0]) / self.stepUnitReference / self.stepGrain

        # scale by fraction of bands that are affected
        use,=np.where((self.fitBands == 'r'))
        self.o3StepUnits *= float(use.size) / float(self.nFitBands)
        self.fgcmLog.log('INFO','O3 step unit set to %f' % (self.o3StepUnits))

        # wash parameters units...
        self.washStepUnits = 1.0/self.stepUnitReference / self.stepGrain
        self.washSlopeStepUnits = self.washStepUnits / self.meanWashIntervalDuration
        self.fgcmLog.log('INFO','wash step unit set to %f' % (self.washStepUnits))
        self.fgcmLog.log('INFO','wash step unit set to %f' % (self.washSlopeStepUnits))


    def _loadParFile(self, fgcmConfig, parFile):
        """
        """
        # read in the parameter file...
        # need to decide on a format

        self.fgcmLog.log('INFO','Loading parameters from %s' % (parFile))

        parInfo=fitsio.read(parFile,ext='PARINFO')
        #self.nCCD = parInfo['NCCD'][0]
        #self.bands = parInfo['BANDS'][0]
        #self.nBands = self.bands.size
        #self.fitBands = parInfo['FITBANDS'][0]
        #self.nFitBands = self.fitBands.size
        #self.extraBands = parInfo['EXTRABANDS'][0]
        #self.nExtraBands = self.extraBands.size
        #self.exposureFile = parInfo['EXPOSUREFILE'][0]

        #self.bands = np.core.defchararray.strip(self.bands[:])
        #self.fitBands = np.core.defchararray.strip(self.fitBands[:])
        #self.extraBands = np.core.defchararray.strip(self.extraBands[:])

        self.nCCD = fgcmConfig.nCCD
        self.bands = fgcmConfig.bands
        self.nBands = self.bands.size
        self.fitBands = fgcmConfig.fitBands
        self.nFitBands = fgcmConfig.fitBands.size
        self.extraBands = fgcmConfig.extraBands
        self.nExtraBands = fgcmConfig.extraBands.size
        self.exposureFile = fgcmConfig.exposureFile

        self._makeBandIndices()
        self._loadExposureInfo(fgcmConfig)

        self._loadEpochAndWashInfo(fgcmConfig)

        self.tauStepUnits = parInfo['TAUSTEPUNITS'][0]
        self.tauSlopeStepUnits = parInfo['TAUSLOPESTEPUNITS'][0]
        self.alphaStepUnits = parInfo['ALPHASTEPUNITS'][0]
        self.pwvStepUnits = parInfo['PWVSTEPUNITS'][0]
        self.pwvSlopeStepUnits = parInfo['PWVSLOPESTEPUNITS'][0]
        self.o3StepUnits = parInfo['O3STEPUNITS'][0]
        self.washStepUnits = parInfo['WASHSTEPUNITS'][0]
        self.washSlopeStepUnits = parInfo['WASHSLOPESTEPUNITS'][0]

        self.hasExternalPWV = parInfo['HASEXTERNALPWV'][0].astype(np.bool)
        self.hasExternalTau = parInfo['HASEXTERNALTAU'][0].astype(np.bool)

        pars=fitsio.read(parFile,ext='PARAMS')
        self.parAlpha = pars['PARALPHA'][0]
        self.parO3 = pars['PARO3'][0]
        self.parTauIntercept = pars['PARTAUINTERCEPT'][0]
        self.parTauSlope = pars['PARTAUSLOPE'][0]
        self.parPWVIntercept = pars['PARPWVINTERCEPT'][0]
        self.parPWVSlope = pars['PARPWVSLOPE'][0]
        self.parQESysIntercept = pars['PARQESYSINTERCEPT'][0]
        self.parQESysSlope = pars['PARQESYSSLOPE'][0]

        self.externalPWVFlag = np.zeros(self.nExp,dtype=np.bool)
        if self.hasExternalPWV:
            self.pwvFile = str(parInfo['PWVFILE'][0]).rstrip()
            self.parExternalPWVScale = pars['PAREXTERNALPWVSCALE'][0]
            self.parExternalPWVOffset = pars['PAREXTERNALPWVOFFSET'][0]
            self.hasExternalPWV = True
            self.loadExternalPWV(fgcmConfig.externalPWVDeltaT)

        self.externalTauFlag = np.zeros(self.nExp,dtype=np.bool)
        if self.hasExternalTau:
            self.tauFile = str(parInfo['TAUFILE'][0]).rstrip()
            self.parExternalTauScale = pars['PAREXTERNALTAUSCALE'][0]
            self.parExternalTauOffset = pars['PAREXTERNALTAUOFFSET'][0]
            self.hasExternalTau = True
            self.loadExternalTau()

        self.compAperCorrPivot = pars['COMPAPERCORRPIVOT'][0]
        self.compAperCorrSlope = pars['COMPAPERCORRSLOPE'][0]
        self.compAperCorrSlopeErr = pars['COMPAPERCORRSLOPEERR'][0]
        self.compAperCorrRange = np.reshape(pars['COMPAPERCORRRANGE'][0],(2,self.nBands))

        self.compExpGray = pars['COMPEXPGRAY'][0]
        self.compVarGray = pars['COMPVARGRAY'][0]
        self.compNGoodStarPerExp = pars['COMPNGOODSTARPEREXP'][0]

        self._arrangeParArray()
        self._setParRanges(fgcmConfig)

        # should check these are all the right size...

        # need to load the superstarflats
        self.parSuperStarFlat = fitsio.read(parFile,ext='SUPER')


    def saveParFile(self, parFile):
        """
        """
        # save the parameter file...
        # need to decide on a format

        self.fgcmLog.log('INFO','Saving parameters to %s' % (parFile))

        dtype=[('NCCD','i4'),
               ('BANDS','a2',self.bands.size),
               ('FITBANDS','a2',self.fitBands.size),
               ('EXTRABANDS','a2',self.extraBands.size),
               ('EXPOSUREFILE','a%d' % (len(self.exposureFile)+1)),
               ('TAUSTEPUNITS','f8'),
               ('TAUSLOPESTEPUNITS','f8'),
               ('ALPHASTEPUNITS','f8'),
               ('PWVSTEPUNITS','f8'),
               ('PWVSLOPESTEPUNITS','f8'),
               ('O3STEPUNITS','f8'),
               ('WASHSTEPUNITS','f8'),
               ('WASHSLOPESTEPUNITS','f8'),
               ('HASEXTERNALPWV','i2'),
               ('HASEXTERNALTAU','i2')]

        if (self.hasExternalPWV):
            dtype.extend([('PWVFILE','a%d' % (len(self.pwvFile)+1))])
        if (self.hasExternalTau):
            dtype.extend([('TAUFILE','a%d' % (len(self.tauFile)+1))])

        parInfo=np.zeros(1,dtype=dtype)
        parInfo['NCCD'] = self.nCCD
        parInfo['BANDS'] = self.bands
        parInfo['FITBANDS'] = self.fitBands
        parInfo['EXTRABANDS'] = self.extraBands
        parInfo['EXPOSUREFILE'] = self.exposureFile

        parInfo['TAUSTEPUNITS'] = self.tauStepUnits
        parInfo['TAUSLOPESTEPUNITS'] = self.tauSlopeStepUnits
        parInfo['ALPHASTEPUNITS'] = self.alphaStepUnits
        parInfo['PWVSTEPUNITS'] = self.pwvStepUnits
        parInfo['PWVSLOPESTEPUNITS'] = self.pwvSlopeStepUnits
        parInfo['O3STEPUNITS'] = self.o3StepUnits
        parInfo['WASHSTEPUNITS'] = self.washStepUnits
        parInfo['WASHSLOPESTEPUNITS'] = self.washSlopeStepUnits

        parInfo['HASEXTERNALPWV'] = self.hasExternalPWV
        if (self.hasExternalPWV):
            parInfo['PWVFILE'] = self.pwvFile
        parInfo['HASEXTERNALTAU'] = self.hasExternalTau
        if (self.hasExternalTau):
            parInfo['TAUFILE'] = self.tauFile

        # clobber?
        fitsio.write(parFile,parInfo,extname='PARINFO',clobber=True)

        dtype=[('PARALPHA','f8',self.parAlpha.size),
               ('PARO3','f8',self.parO3.size),
               ('PARTAUINTERCEPT','f8',self.parTauIntercept.size),
               ('PARTAUSLOPE','f8',self.parTauSlope.size),
               ('PARPWVINTERCEPT','f8',self.parPWVIntercept.size),
               ('PARPWVSLOPE','f8',self.parPWVSlope.size),
               ('PARQESYSINTERCEPT','f8',self.parQESysIntercept.size),
               ('PARQESYSSLOPE','f8',self.parQESysSlope.size),
               ('COMPAPERCORRPIVOT','f8',self.compAperCorrPivot.size),
               ('COMPAPERCORRSLOPE','f8',self.compAperCorrSlope.size),
               ('COMPAPERCORRSLOPEERR','f8',self.compAperCorrSlopeErr.size),
               ('COMPAPERCORRRANGE','f8',self.compAperCorrRange.size),
               ('COMPEXPGRAY','f8',self.compExpGray.size),
               ('COMPVARGRAY','f8',self.compVarGray.size),
               ('COMPNGOODSTARPEREXP','i4',self.compNGoodStarPerExp.size)]

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
        pars['PARTAUINTERCEPT'][:] = self.parTauIntercept
        pars['PARTAUSLOPE'][:] = self.parTauSlope
        pars['PARPWVINTERCEPT'][:] = self.parPWVIntercept
        pars['PARPWVSLOPE'][:] = self.parPWVSlope
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

        pars['COMPEXPGRAY'][:] = self.compExpGray
        pars['COMPVARGRAY'][:] = self.compVarGray
        pars['COMPNGOODSTARPEREXP'][:] = self.compNGoodStarPerExp

        fitsio.write(parFile,pars,extname='PARAMS')

        # and need to record the superstar flats
        fitsio.write(parFile,self.parSuperStarFlat,extname='SUPER')



    def loadExternalPWV(self, externalPWVDeltaT):
        """
        """
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
        self.fgcmLog.log('INFO','%d exposures of %d have external pwv values' % (match.size,self.nExp))


    def loadExternalTau(self, withAlpha=False):
        """
        """
        # load a file with Tau values
        ## not supported yet
        raise ValueError("externalTau Not supported yet")

        if (withAlpha):
            self.hasExternalAlpha = True

    def reloadParArray(self, parArray, fitterUnits=False):
        """
        """
        # takes in a parameter array and loads the local split copies?
        self.fgcmLog.log('DEBUG','Reloading parameter array')

        if (parArray.size != self.nFitPars):
            raise ValueError("parArray must have %d elements." % (self.nFitPars))

        unitDict = self.getUnitDict(fitterUnits=fitterUnits)

        self.parPWVIntercept[:] = parArray[self.parPWVInterceptLoc:
                                               self.parPWVInterceptLoc+self.nCampaignNights] / unitDict['pwvUnit']
        self.parPWVSlope[:] = parArray[self.parPWVSlopeLoc:
                                           self.parPWVSlopeLoc+self.nCampaignNights] / unitDict['pwvSlopeUnit']
        self.parO3[:] = parArray[self.parO3Loc:
                                     self.parO3Loc+self.nCampaignNights] / unitDict['o3Unit']
        self.parTauIntercept[:] = parArray[self.parTauInterceptLoc:
                                               self.parTauInterceptLoc+self.nCampaignNights] / unitDict['tauUnit']
        self.parTauSlope[:] = parArray[self.parTauSlopeLoc:
                                           self.parTauSlopeLoc+self.nCampaignNights] / unitDict['tauSlopeUnit']
        self.parAlpha[:] = parArray[self.parAlphaLoc:
                                        self.parAlphaLoc+self.nCampaignNights] / unitDict['alphaUnit']
        if (self.hasExternalPWV):
            self.parExternalPWVScale = parArray[self.parExternalPWVScaleLoc] / unitDict['pwvUnit']
            self.parExternalPWVOffset = parArray[self.parExternalPWVOffsetLoc:
                                                     self.parExternalPWVOffsetLoc+self.nCampaignNights] / unitDict['pwvUnit']

        if (self.hasExternalTau):
            self.parExternalTauScale = parArray[self.parExternalTauScaleLoc] / unitDict['tauUnit']
            self.parExternalTauOffset = parArray[self.parExternalTauOffsetLoc:
                                                     self.parExternalTauOffsetLoc+self.nCampaignNights] / unitDict['tauUnit']

        self.parQESysIntercept[:] = parArray[self.parQESysInterceptLoc:
                                                 self.parQESysInterceptLoc+self.nWashIntervals] / unitDict['qeSysUnit']
        self.parQESysSlope[:] = parArray[self.parQESysSlopeLoc:
                                             self.parQESysSlopeLoc+self.nWashIntervals] / unitDict['qeSysSlopeUnit']
        # done


    ## MAYBE? should these be properties?
    ##  The problem is that I think I want these pre-computed, though I don't know
    ##  if that actually helps the performance.  TEST because properties would be
    ##  very convenient

    def parsToExposures(self):
        """
        """

        self.fgcmLog.log('DEBUG','Computing exposure values from parameters')

        # I'm guessing that these don't need to be wrapped in shms but I could be wrong
        #  about the full class, which would suck.

        # first, the nightly parameters without selection...
        self.expO3 = self.parO3[self.expNightIndex]
        self.expAlpha = self.parAlpha[self.expNightIndex]


        # default to the nightly slope/intercept 
        self.expPWV = (self.parPWVIntercept[self.expNightIndex] +
                       self.parPWVSlope[self.expNightIndex] * self.expDeltaUT)

        if (self.hasExternalPWV):
            # replace where we have these
            self.expPWV[self.externalPWVFlag] = (self.parExternalPWVOffset[self.expNightIndex[self.externalPWVFlag]] +
                                                 self.parExternalPWVScale *
                                                 self.externalPWV[self.externalPWVFlag])

        # default to nightly slope/intercept
        self.expTau = (self.parTauIntercept[self.expNightIndex] +
                           self.parTauSlope[self.expNightIndex] * self.expDeltaUT)

        if (self.hasExternalTau):
            # replace where we have these
            self.expTau[self.externalTauFlag] = (self.parExternalTauOffset[self.expNightIndex[self.externalTauFlag]] +
                                                 self.parExternalTauScale *
                                                 self.externalTau[self.externalTauFlag])

        # and QESys
        self.expQESys = (self.parQESysIntercept[self.expWashIndex] +
                         self.parQESysSlope[self.expWashIndex] *
                         (self.expMJD - self.washMJDs[self.expWashIndex]))

    # cannot be a property because of the keywords
    def getParArray(self,fitterUnits=False):
        """
        """

        self.fgcmLog.log('DEBUG','Retrieving parameter array')

        # extracts parameters into a linearized array
        parArray = np.zeros(self.nFitPars,dtype=np.float64)

        unitDict = self.getUnitDict(fitterUnits=fitterUnits)

        parArray[self.parPWVInterceptLoc:
                     self.parPWVInterceptLoc+self.nCampaignNights] = self.parPWVIntercept[:] * unitDict['pwvUnit']
        parArray[self.parPWVSlopeLoc:
                     self.parPWVSlopeLoc+self.nCampaignNights] = self.parPWVSlope[:] * unitDict['pwvSlopeUnit']
        parArray[self.parO3Loc:
                     self.parO3Loc+self.nCampaignNights] = self.parO3[:] * unitDict['o3Unit']
        parArray[self.parTauInterceptLoc:
                     self.parTauInterceptLoc+self.nCampaignNights] = self.parTauIntercept[:] * unitDict['tauUnit']
        parArray[self.parTauSlopeLoc:
                     self.parTauSlopeLoc+self.nCampaignNights] = self.parTauSlope[:] * unitDict['tauSlopeUnit']
        parArray[self.parAlphaLoc:
                     self.parAlphaLoc+self.nCampaignNights] = self.parAlpha[:] * unitDict['alphaUnit']
        if (self.hasExternalPWV):
            parArray[self.parExternalPWVScaleLoc] = self.parExternalPWVScale * unitDict['pwvUnit']
            parArray[self.parExternalPWVOffsetLoc:
                         self.parExternalPWVOffsetLoc+self.nCampaignNights] = self.parExternalPWVOffset * unitDict['pwvUnit']
        if (self.hasExternalTau):
            parArray[self.parExternalTauScaleLoc] = self.parExternalTauScale * unitDict['tauUnit']
            parArray[self.parExternalTauOffsetLoc:
                         self.parExternalTauOffsetLoc+self.nCampaignNights] = self.parExternalTauOffset * unitDict['tauUnit']

        parArray[self.parQESysInterceptLoc:
                     self.parQESysInterceptLoc+self.nWashIntervals] = self.parQESysIntercept * unitDict['qeSysUnit']
        parArray[self.parQESysSlopeLoc:
                     self.parQESysSlopeLoc+self.nWashIntervals] = self.parQESysSlope * unitDict['qeSysSlopeUnit']

        return parArray

    # this cannot be a property because it takes units
    def getParBounds(self,fitterUnits=False):
        """
        """

        self.fgcmLog.log('DEBUG','Retrieving parameter bounds')

        unitDict = self.getUnitDict(fitterUnits=fitterUnits)

        parLow = np.zeros(self.nFitPars,dtype=np.float32)
        parHigh = np.zeros(self.nFitPars,dtype=np.float32)

        ## MAYBE: configure slope ranges?

        parLow[self.parPWVInterceptLoc:
                   self.parPWVInterceptLoc+self.nCampaignNights] = (
            (self.pwvRange[0] + 10.0*0.2) * unitDict['pwvUnit'])
        parHigh[self.parPWVInterceptLoc:
                    self.parPWVInterceptLoc+self.nCampaignNights] = (
            (self.pwvRange[1] - 10.0*0.2) * unitDict['pwvUnit'])
        parLow[self.parPWVSlopeLoc:
                   self.parPWVSlopeLoc+self.nCampaignNights] = (
            -0.2 * unitDict['pwvSlopeUnit'])
        parHigh[self.parPWVSlopeLoc:
                    self.parPWVSlopeLoc+self.nCampaignNights] = (
            0.2 * unitDict['pwvSlopeUnit'])
        parLow[self.parO3Loc:
                   self.parO3Loc+self.nCampaignNights] = (
            self.O3Range[0] * unitDict['o3Unit'])
        parHigh[self.parO3Loc:
                    self.parO3Loc+self.nCampaignNights] = (
            self.O3Range[1] * unitDict['o3Unit'])

        parLow[self.parTauInterceptLoc:
                   self.parTauInterceptLoc+self.nCampaignNights] = (
            (self.tauRange[0] + 10.0*0.0025) * unitDict['tauUnit'])
        parHigh[self.parTauInterceptLoc:
                    self.parTauInterceptLoc+self.nCampaignNights] = (
            (self.tauRange[1] - 10.0*0.0025) * unitDict['tauUnit'])
        parLow[self.parTauSlopeLoc:
                   self.parTauSlopeLoc+self.nCampaignNights] = (
            -0.0025 * unitDict['tauSlopeUnit'])
        parHigh[self.parTauSlopeLoc:
                    self.parTauSlopeLoc+self.nCampaignNights] = (
            0.0025 * unitDict['tauSlopeUnit'])
        parLow[self.parAlphaLoc:
                   self.parAlphaLoc+self.nCampaignNights] = (
            0.25 * unitDict['alphaUnit'])
        parHigh[self.parAlphaLoc:
                    self.parAlphaLoc+self.nCampaignNights] = (
            1.75 * unitDict['alphaUnit'])
        if (self.hasExternalPWV):
            parLow[self.parExternalPWVScaleLoc] = 0.5 * unitDict['pwvUnit']
            parHigh[self.parExternalPWVScaleLoc] = 1.5 * unitDict['pwvUnit']
            parLow[self.parExternalPWVOffsetLoc:
                       self.parExternalPWVOffsetLoc+self.nCampaignNights] = (
                -1.5 * unitDict['pwvUnit'])
            parHigh[self.parExternalPWVOffsetLoc:
                       self.parExternalPWVOffsetLoc+self.nCampaignNights] = (
                3.0 * unitDict['pwvUnit'])
        if (self.hasExternalTau):
            parLow[self.parExternalTauScaleLoc] = 0.7 * unitDict['tauUnit']
            parHigh[self.parExternalTauScaleLoc] = 1.2 * unitDict['tauUnit']
            parLow[self.parExternalTauOffsetLoc:
                       self.parExternalTauOffsetLoc+self.nCampaignNights] = (
                0.0 * unitDict['tauUnit'])
            parHigh[self.parExternalTauOffsetLoc:
                        self.parExternalTauOffsetLoc+self.nCampaignNights] = (
                0.03 * unitDict['tauUnit'])

        parLow[self.parQESysInterceptLoc:
                   self.parQESysInterceptLoc+self.nWashIntervals] = (
            -0.2 * unitDict['qeSysUnit'])
        parHigh[self.parQESysInterceptLoc:
                    self.parQESysInterceptLoc+self.nWashIntervals] = (
            0.05 * unitDict['qeSysUnit'])
        parLow[self.parQESysSlopeLoc:
                   self.parQESysSlopeLoc+self.nWashIntervals] = (
            -0.001 * unitDict['qeSysSlopeUnit'])
        parHigh[self.parQESysSlopeLoc:
                    self.parQESysSlopeLoc+self.nWashIntervals] = (
            0.001 * unitDict['qeSysSlopeUnit'])

        # zip these into a list of tuples
        parBounds = zip(parLow,parHigh)

        return parBounds

    def getUnitDict(self,fitterUnits=False):
        unitDict = {'pwvUnit':1.0,
                    'pwvSlopeUnit':1.0,
                    'o3Unit':1.0,
                    'tauUnit':1.0,
                    'tauSlopeUnit':1.0,
                    'alphaUnit':1.0,
                    'qeSysUnit':1.0,
                    'qeSysSlopeUnit':1.0}
        if (fitterUnits):
            unitDict['pwvUnit'] = self.pwvStepUnits
            unitDict['pwvSlopeUnit'] = self.pwvSlopeStepUnits
            unitDict['o3Unit'] = self.o3StepUnits
            unitDict['tauUnit'] = self.tauStepUnits
            unitDict['tauSlopeUnit'] = self.tauSlopeStepUnits
            unitDict['alphaUnit'] = self.alphaStepUnits
            unitDict['qeSysUnit'] = self.washStepUnits
            unitDict['qeSysSlopeUnit'] = self.washSlopeStepUnits

        return unitDict

    @property
    def expCCDSuperStar(self):
        """
        """
        expCCDSuperStar = np.zeros((self.nExp,self.nCCD),dtype='f8')

        expCCDSuperStar[:,:] = self.parSuperStarFlat[self.expEpochIndex,
                                                     self.expBandIndex,
                                                     :]

        return expCCDSuperStar

    @property
    def expApertureCorrection(self):
        """
        """

        expApertureCorrection = np.zeros(self.nExp,dtype='f8')

        expSeeingVariableClipped = np.clip(self.expSeeingVariable,
                                           self.compAperCorrRange[0,self.expBandIndex],
                                           self.compAperCorrRange[1,self.expBandIndex])

        expApertureCorrection[:] = (self.compAperCorrSlope[self.expBandIndex] *
                                    (expSeeingVariableClipped -
                                     self.compAperCorrPivot[self.expBandIndex]))

        return expApertureCorrection


    def plotParameters(self):
        """
        """

        # want nightly averages, on calibratable nights (duh)

        # this is fixed here
        #minExpPerNight = 

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
                  self.expTau[expUse])
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

        #firstMJD = np.floor(np.min(mjdNight[gd]))
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

        # tau
        fig=plt.figure(1,figsize=(8,6))
        fig.clf()
        ax=fig.add_subplot(111)

        ## FIXME: allow other band names  (bandLike or something)
        gBandIndex,=np.where(self.bands=='g')[0]
        rBandIndex,=np.where(self.bands=='r')[0]

        tauGd, = np.where((nExpPerNight > self.minExpPerNight) &
                          ((nExpPerBandPerNight[:,gBandIndex] +
                            nExpPerBandPerNight[:,rBandIndex]) >
                           self.minExpPerNight))

        ax.plot(mjdNight[tauGd] - firstMJD, tauNight[tauGd],'r.')
        ax.set_xlabel(r'$\mathrm{MJD}\ -\ %.0f$' % (firstMJD),fontsize=16)
        ax.set_ylabel(r'$\tau_{7750}$',fontsize=16)

        fig.savefig('%s/%s_nightly_tau.png' % (self.plotPath,
                                               self.outfileBaseWithCycle))

        # pwv
        fig=plt.figure(1,figsize=(8,6))
        fig.clf()
        ax=fig.add_subplot(111)

        ## FIXME: allow other band names
        zBandIndex,=np.where(self.bands=='z')[0]

        pwvGd, = np.where((nExpPerNight > self.minExpPerNight) &
                          (nExpPerBandPerNight[:,zBandIndex] > self.minExpPerNight))

        ax.plot(mjdNight[pwvGd] - firstMJD, pwvNight[pwvGd],'r.')
        ax.set_xlabel(r'$\mathrm{MJD}\ -\ %.0f$' % (firstMJD),fontsize=16)
        ax.set_ylabel(r'$\mathrm{PWV}$',fontsize=16)

        fig.savefig('%s/%s_nightly_pwv.png' % (self.plotPath,
                                               self.outfileBaseWithCycle))

        # O3
        fig=plt.figure(1,figsize=(8,6))
        fig.clf()
        ax=fig.add_subplot(111)

        ## FIXME: allow other band names
        rBandIndex,=np.where(self.bands=='r')[0]
        O3Gd, = np.where((nExpPerNight > self.minExpPerNight) &
                         (nExpPerBandPerNight[:,rBandIndex] > self.minExpPerNight))

        ax.plot(mjdNight[O3Gd] - firstMJD, O3Night[O3Gd],'r.')
        ax.set_xlabel(r'$\mathrm{MJD}\ -\ %.0f$' % (firstMJD),fontsize=16)
        ax.set_ylabel(r'$O_3$',fontsize=16)

        fig.savefig('%s/%s_nightly_o3.png' % (self.plotPath,
                                              self.outfileBaseWithCycle))


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

        for i in xrange(self.nWashIntervals):
            ax.plot([self.washMJDs[i]-firstMJD,self.washMJDs[i]-firstMJD],
                    ax.get_ylim(),'k--')

        fig.savefig('%s/%s_qesys_washes.png' % (self.plotPath,
                                                self.outfileBaseWithCycle))

        ## FIXME: add pwv offset plotting routine (if external)
        ## FIXME: add tau offset plotting routing (if external)
