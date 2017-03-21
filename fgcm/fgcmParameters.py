from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil

from fgcmUtilities import _pickle_method

import types
import copy_reg
#import sharedmem as shm

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

from fgcmLUT import FgcmLUTSHM

copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmParameters(object):
    """
    """
    def __init__(self,parFile=None,
                 fgcmConfig=None):

        self.hasExternalPWV = False
        self.hasExternalTau = False

        if (fgcmConfig is not None):
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

        # first thing is to get the exposure numbers...
        self.exposureFile = fgcmConfig.exposureFile

        self._loadExposureInfo(fgcmConfig)

        # set up the observing epochs and link indices

        self._loadEpochAndWashInfo(fgcmConfig)

        # set up the parameters with nightly values
        # need to include the default stuff...

        self.parAlpha = np.zeros(self.expNights.size,dtype=np.float32) + fgcmConfig.alphaStd
        self.parO3 = np.zeros(self.expNights.size,dtype=np.float32) + fgcmConfig.o3Std
        self.parTauIntercept = np.zeros(self.expNights.size,dtype=np.float32) + fgcmConfig.tauStd
        self.parTauSlope = np.zeros(self.expNights.size,dtype=np.float32)
        self.parPWVIntercept = np.zeros(self.expNights.size,dtype=np.float32) + fgcmConfig.pwvStd
        self.parPWVSlope = np.zeros(self.expNights.size,dtype=np.float32)

        # parameters with per-epoch values
        self.parSuperflat = np.zeros((self.nEpochs,self.nBands,self.nCCD),dtype=np.float32)

        # parameters with per-wash values
        self.parQESysIntercept = np.zeros(self.nWashIntervals,dtype=np.float32)
        self.parQESysSlope = np.zeros(self.nWashIntervals,dtype=np.float32)

        if (fgcmConfig.pwvFile is not None):
            self.pwvFile = fgcmConfig.pwvFile
            self.hasExternalPWV = True
            self.loadExternalPWV(fgcmConfig.externalPWVDeltaT)
            # need to add two global parameters!
            self.parExternalPWVScale = 1.0
            self.parExternalPWVOffset = 0.0

        if (fgcmConfig.tauFile is not None):
            self.tauFile = fgcmConfig.tauFile
            self.hasExternalTau = True
            self.loadExternalTau()
            # need to add two global parameters!
            self.parExternalTauScale = 1.0
            self.parExternalTauOffset = 0.0

        # and compute the units...
        self._computeStepUnits(fgcmConfig)

        # and need to be able to pack and unpack the parameters and scalings
        #  this part is going to be the hardest

        self._arrangeParArray()

    def _arrangeParArray(self):
        # make pointers to a fit parameter array...
        #  pwv, O3, lnTau, alpha
        self.nFitPars = (self.expNights.size +  # O3
                         self.expNights.size +  # tauIntercept
                         self.expNights.size +  # tauSlope
                         self.expNights.size +  # alpha
                         self.expNights.size +  # pwv Intercept
                         self.expNights.size)   # pwv Slope
        ctr=0
        self.parO3Loc = ctr
        ctr+=self.expNights.size
        self.parTauInterceptLoc = ctr
        ctr+=self.expNights.size
        self.parTauSlopeLoc = ctr
        ctr+=self.expNights.size
        self.parAlphaLoc = ctr
        ctr+=self.expNights.size
        self.parPWVInterceptLoc = ctr
        ctr+=self.expNights.size
        self.parPWVSlopeLoc = ctr
        ctr+=self.expNights.size

        if (self.hasExternalPWV):
            self.nFitPars += (1+self.expNights.size)
            self.parExternalPWVScaleLoc = ctr
            ctr+=1
            self.parExternalPWVOffsetLoc = ctr
            ctr+=self.expNights.size

        if (self.hasExternalTau):
            self.nFitPars += (1+self.expNights.size)
            self.parExternalTauScaleLoc = ctr
            ctr+=1
            self.parExternalTauOffsetLoc = ctr
            ctr+=self.expNights.size

        self.nFitPars += (self.nWashIntervals + # parQESysIntercept
                          self.nWashIntervals)  # parQESysSlope

        self.parQESysInterceptLoc = ctr
        ctr+=self.nWashIntervals
        self.parQESysSlopeLoc = ctr
        ctr+=self.nWashIntervals

    def _loadExposureInfo(self,fgcmConfig):
        """
        """

        expInfo = fitsio.read(self.exposureFile,ext=1)

        # ensure sorted by exposure number
        st=np.argsort(expInfo['EXPNUM'])
        expInfo=expInfo[st]

        self.nExp = fgcmConfig.nExp

        self.expArray = expInfo['EXPNUM']
        self.expFlag = np.zeros(self.nExp,dtype=np.int8)
        self.expExptime = expInfo['EXPTIME']

        self.expSeeingVariable = expInfo[fgcmConfig.seeingField]
        self.expDeepFlag = expInfo[fgcmConfig.deepFlag]

        # we need the nights of the survey (integer MJD, maybe rotated)
        self.expMJD = expInfo['MJD']
        mjdForNight = np.floor(self.expMJD + fgcmConfig.UTBoundary).astype(np.int32)
        self.expNights = np.unique(mjdForNight)

        self.expDeltaUT = (self.expMJD + fgcmConfig.UTBoundary) - mjdForNight

        # and link the exposure numbers to the nights...
        a,b=esutil.numpy_util.match(self.expNights,mjdForNight)
        self.expNightIndex = np.zeros(self.nExp,dtype=np.int32)
        self.expNightIndex[b] = a

        # we need the duration of each night...
        self.nightDuration = np.zeros(self.expNights.size)
        self.expPerNight = np.zeros(self.expNights.size,dtype=np.int32)
        for i in xrange(self.expNights.size):
            use,=np.where(mjdForNight == self.expNights[i])
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
            print("Warning: %d exposures with band not in LUT!" % (bad.size))
            self.expFlag[bad] = self.expFlag[bad] & 256

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
        #self.washMJDs = np.concatenate([[np.min(self.expMJD)-1.0],fgcmConfig.washMJDs])
        #self.washMJDs = np.insert(fgcmConfig.washMJDs,0,np.min(self.expMJD)-1.0)
        # the number of *intervals* is one less than the dates?
        #self.nWashIntervals = self.washMJDs.size-1

        self.nWashIntervals = fgcmConfig.washMJDs.size
        self.washMJDs = np.insert(fgcmConfig.washMJDs,0,np.min(self.expMJD)-1.0)

        self.expWashIndex = np.zeros(self.nExp,dtype='i4')
        tempWashMJDs = self.washMJDs
        #tempWashMJDs = np.insert(tempWashMJDs,0,0.0)
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

        # and the tau slope units
        self.tauSlopeStepUnits = self.tauStepUnits * self.meanNightDuration

        # alpha units -- reference to g, or r if not available
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

        # PWV slope units
        self.pwvSlopeStepUnits = self.pwvStepUnits * self.meanNightDuration

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

        # wash parameters units...
        self.washStepUnits = 1.0/self.stepUnitReference / self.stepGrain
        self.washSlopeStepUnits = self.washStepUnits / self.meanWashIntervalDuration


    def loadParFile(self, fgcmConfig, parFile):
        """
        """
        # read in the parameter file...
        # need to decide on a format

        parInfo=fitsio.read(parFile,ext='PARINFO')
        self.nCCD = parInfo['NCCD'][0]
        self.bands = parInfo['BANDS'][0]
        self.fitBands = parInfo['FITBANDS'][0]
        self.exposureFile = parInfo['EXPOSUREFILE'][0]

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

        if self.hasExternalPWV:
            self.parExternalPWVScale = pars['PAREXTERNALPWVSCALE'][0]
            self.parExternalPWVOffset = pars['PAREXTERNALPWVOFFSET'][0]
        if self.hasExternalTau:
            self.parExternalTauScale = pars['PAREXTERNALTAUSCALE'][0]
            self.parExternalTauOffset = pars['PAREXTERNALTAUOFFSET'][0]


        self._arrangeParArray()
        # should check these are all the right size...

        # need to load the superstarflats


    def saveParFile(self, parFile):
        """
        """
        # save the parameter file...
        # need to decide on a format

        dtype=[('NCCD','i4'),
               ('BANDS','a2',self.bands.size),
               ('FITBANDS','a2',self.fitBands.size),
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
        fitsio.write(parfile,parInfo,extname='PARINFO',clobber=True)

        dtype=[('PARALPHA','f8',self.parAlpha.size),
               ('PARO3','f8',self.parO3.size),
               ('PARTAUINTERCEPT','f8',self.parTauIntercept.size),
               ('PARTAUSLOPE','f8',self.parTauSlope.size),
               ('PARPWVINTERCEPT','f8',self.parPWVIntercept.size),
               ('PARPWVSLOPE','f8',self.parPWVSlope.size),
               ('PARQESYSINTERCEPT','f8',self.parQESysIntercept.size),
               ('PARQESYSSLOPE','f8',self.parQESysSlope.size)]

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

        fitsio.write(parfile,pars,extname='PARAMS')

        # and need to record the superstar flats


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
        self.externalPWVFlag = (np.abs(pwvTable['MJD'][pwvIndex] - self.expMJD) < externalPWVDeltaT)
        self.externalPWV = np.zeros(self.nExp,dtype=np.float32)
        self.externalPWV[self.externalPWVFlag] = pwvTable['PWV'][pwvIndex[self.externalPWVFlag]]

        # and new PWV scaling pars!
        self.parExternalPWVOffset = np.zeros(self.expNights.size,dtype=np.float32)
        self.parExternalPWVScale = 1.0


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

        if (parArray.size != self.nFitPars):
            raise ValueError("parArray must have %d elements." % (self.nFitPars))

        if (fitterUnits):
            pwvUnit = self.pwvStepUnits
            pwvSlopeUnit = self.pwvSlopeStepUnits
            O3Unit = self.o3StepUnits
            tauUnit = self.tauStepUnits
            tauSlopeUnit = self.tauSlopeStepUnits
            alphaUnit = self.alphaUnits
            qeSysUnit = self.washStepUnits
            qeSysSlopeUnit = self.washSlopeStepUnits
        else:
            pwvUnit = 1.0
            pwvSlopeUnit = 1.0
            O3Unit = 1.0
            tauUnit = 1.0
            tauSlopeUnit = 1.0
            alphaUnit = 1.0
            qeSysUnit = 1.0
            qeSysSlopeUnit = 1.0

        self.parPWVIntercept[:] = parArray[self.parPWVInterceptLoc:
                                               self.parPWVInterceptLoc+self.expNights.size] / pwvUnit
        self.parPWVSlope[:] = parArray[self.parPWVSlopeLoc:
                                           self.parPWVSlopeLoc+self.expNights.size] / pwvSlopeUnit
        self.parO3[:] = parArray[self.parO3Loc:
                                     self.parO3Loc+self.expNights.size] / O3Unit
        self.parTauIntercept[:] = parArray[self.parTauInterceptLoc:
                                               self.parTauInterceptLoc+self.expNights.size] / tauUnit
        self.parTauSlope[:] = parArray[self.parTauSlopeLoc:
                                           self.parTauSlopeLoc+self.expNights.size] / tauSlopeUnit
        self.parAlpha[:] = parArray[self.parAlphaLoc:
                                        self.parAlphaLoc+self.expNights.size] / alphaUnit
        if (self.hasExternalPWV):
            self.parExternalPWVScale = parArray[self.parExternalPWVScaleLoc] / pwvUnit
            self.parExternalPWVOffset = parArray[self.parExternalPWVOffsetLoc:
                                                     self.parExternalPWVOffsetLoc+self.expNights.size] / pwvUnit

        if (self.hasExternalTau):
            self.parExternalTauScale = parArray[self.parExternalTauScaleLoc] / tauUnit
            self.parExternalTauOffset = parArray[self.parExternalTauOffsetLoc:
                                                     self.parExternalTauOffsetLoc+self.expNights.size] / tauUnit

        self.parQESysIntercept[:] = parArray[self.parQESysInterceptLoc:
                                                 self.parQESysInterceptLoc+self.nWashIntervals] / qeSysUnit
        self.parQESysSlope[:] = parArray[self.parQESysSlopeLoc:
                                             self.parQESysSlopeLoc+self.nWashIntervals] / qeSysSlopeUnit
        # done

    def parsToExposures(self):
        """
        """

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


    def getParArray(self,bounds=False,fitterUnits=False):
        """
        """
        # extracts parameters into a linearized array
        # also returns bounds if bounds=True
        parArray = np.zeros(self.nFitPars,dtype=np.float32)

        if (fitterUnits):
            pwvUnit = self.pwvStepUnits
            pwvSlopeUnit = self.pwvSlopeStepUnits
            O3Unit = self.o3StepUnits
            tauUnit = self.tauStepUnits
            tauSlopeUnit = self.tauSlopeStepUnits
            alphaUnit = self.alphaUnits
            qeSysUnit = self.washStepUnits
            qeSysSlopeUnit = self.washSlopeStepUnits
        else:
            pwvUnit = 1.0
            pwvSlopeUnit = 1.0
            O3Unit = 1.0
            tauUnit = 1.0
            tauSlopeUnit = 1.0
            alphaUnit = 1.0
            qeSysUnit = 1.0
            qeSysSlopeUnit = 1.0

        parArray[self.parPWVInterceptLoc:
                     self.parPWVInterceptLoc+self.expNights.size] = self.parPWVIntercept[:] * pwvUnit
        parArray[self.parPWVSlopeLoc:
                     self.parPWVSlopeLoc+self.expNights.size] = self.parPWVSlope[:] * pwvSlopeUnit
        parArray[self.parO3Loc:
                     self.parO3Loc+self.expNights.size] = self.parO3[:] * O3Unit
        parArray[self.parTauInterceptLoc:
                     self.parTauInterceptLoc+self.expNights.size] = self.parTauIntercept[:] * tauUnit
        parArray[self.parTauSlopeLoc:
                     self.parTauSlopeLoc+self.expNights.size] = self.parTauSlope[:] * tauSlopeUnit
        parArray[self.parAlphaLoc:
                     self.parAlphaLoc+self.expNights.size] = self.parAlpha[:] * alphaUnit
        if (self.hasExternalPWV):
            parArray[self.parExternalPWVScaleLoc] = self.parExternalPWVScale * pwvUnit
            parArray[self.parExternalPWVOffsetLoc:
                         self.parExternalPWVOffsetLoc+self.expNights.size] = self.parExternalPWVOffset * pwvUnit
        if (self.hasExternalTau):
            parArray[self.parExternalTauScaleLoc] = self.parExternalTauScale * tauUnit
            parArray[self.parExternalTauOffsetLoc:
                         self.parExternalTauOffsetLoc+self.expNights.size] = self.parExternalTauOffset * tauUnit

        parArray[self.parQESysInterceptLoc:
                     self.parQESysInterceptLoc+self.nWashIntervals] = self.parQESysIntercept * qeSysUnit
        parArray[self.parQESysSlopeLoc:
                     self.parQESysSlopeLoc+self.nWashIntervals] = self.parQESysSlope *  qeSysSlopeUnit

        return parArray
