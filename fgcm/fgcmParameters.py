from __future__ import print_function

import numpy as np
#import fitsio
import os
import sys
import esutil

import matplotlib.pyplot as plt

from fgcmUtilities import _pickle_method
from fgcmUtilities import expFlagDict

import types
import copy_reg

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

#from fgcmLUT import FgcmLUT

copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmParameters(object):
    """
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

        self.fgcmLog.log('INFO','Initializing FgcmParameters...')

        # for plotting
        self.minExpPerNight = fgcmConfig.minExpPerNight

        # get stuff from config file
        self.nCCD = fgcmConfig.nCCD
        self.bands = fgcmConfig.bands
        self.nBands = self.bands.size
        self.fitBands = fgcmConfig.fitBands
        self.nFitBands = self.fitBands.size
        self.extraBands = fgcmConfig.extraBands
        self.nExtraBands = self.extraBands.size

        self.alphaStd = fgcmConfig.alphaStd
        self.o3Std = fgcmConfig.o3Std
        self.tauStd = fgcmConfig.tauStd
        self.pwvStd = fgcmConfig.pwvStd
        self.pmbStd = fgcmConfig.pmbStd
        self.zenithStd = fgcmConfig.zenithStd
        self.secZenithStd = 1./np.cos(self.zenithStd*np.pi/180.)

        self.pmbRange = fgcmConfig.pmbRange
        self.pwvRange = fgcmConfig.pwvRange
        self.O3Range = fgcmConfig.O3Range
        self.tauRange = fgcmConfig.tauRange
        self.alphaRange = fgcmConfig.alphaRange
        self.zenithRange = fgcmConfig.zenithRange

        self.nExp = fgcmConfig.nExp
        self.seeingField = fgcmConfig.seeingField
        self.deepFlag = fgcmConfig.deepFlag
        self.expField = fgcmConfig.expField
        self.UTBoundary = fgcmConfig.UTBoundary
        self.latitude = fgcmConfig.latitude
        self.sinLatitude = np.sin(np.radians(self.latitude))
        self.cosLatitude = np.cos(np.radians(self.latitude))

        self.epochMJDs = fgcmConfig.epochMJDs
        self.washMJDs = fgcmConfig.washMJDs

        self.stepUnitReference = fgcmConfig.stepUnitReference
        self.stepGrain = fgcmConfig.stepGrain

        self.resetParameters = fgcmConfig.resetParameters
        self.pwvFile = fgcmConfig.pwvFile
        self.tauFile = fgcmConfig.tauFile
        self.externalPWVDeltaT = fgcmConfig.externalPWVDeltaT

        # and the default unit dict
        self.unitDictOnes = {'pwvUnit':1.0,
                             'pwvPerSlopeUnit':1.0,
                             'o3Unit':1.0,
                             'tauUnit':1.0,
                             'tauPerSlopeUnit':1.0,
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
        """

        import fitsio

        expInfoFile = fgcmConfig.exposureFile

        expInfo = fitsio.read(expInfoFile, ext=1)
        return cls(fgcmConfig, expInfo=expInfo, fgcmLUT=fgcmLUT)

    @classmethod
    def newParsWithArrays(cls, fgcmConfig, fgcmLUT, expInfo):
        """
        """

        return cls(fgcmConfig, expInfo=expInfo, fgcmLUT=fgcmLUT)

    @classmethod
    def loadParsWithFits(cls, fgcmConfig):
        """
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
        """

        return cls(fgcmConfig, expInfo=expInfo,
                   inParInfo=inParInfo, inParams=inParams, inSuperStar=inSuperStar)

    def _initializeNewParameters(self, expInfo, fgcmLUT):
        """
        """

        # link band indices
        self._makeBandIndices()

        # load the exposure information
        self._loadExposureInfo(expInfo)

        # load observing epochs and link indices
        self._loadEpochAndWashInfo()

        # and make the new parameter arrays
        self.parAlpha = np.zeros(self.campaignNights.size,dtype=np.float32) + fgcmLUT.alphaStd
        self.parO3 = np.zeros(self.campaignNights.size,dtype=np.float32) + fgcmLUT.o3Std
        self.parTauIntercept = np.zeros(self.campaignNights.size,dtype=np.float32) + fgcmLUT.tauStd
        self.parTauPerSlope = np.zeros(self.campaignNights.size,dtype=np.float32)
        self.parPWVIntercept = np.zeros(self.campaignNights.size,dtype=np.float32) + fgcmLUT.pwvStd
        self.parPWVPerSlope = np.zeros(self.campaignNights.size,dtype=np.float32)

        # parameters with per-epoch values
        self.parSuperStarFlat = np.zeros((self.nEpochs,self.nBands,self.nCCD),dtype=np.float32)

        # parameters with per-wash values
        self.parQESysIntercept = np.zeros(self.nWashIntervals,dtype=np.float32)
        self.parQESysSlope = np.zeros(self.nWashIntervals,dtype=np.float32)

        ## FIXME: need to completely refactor
        self.externalPWVFlag = np.zeros(self.nExp,dtype=np.bool)
        if (self.pwvFile is not None):
            self.fgcmLog.log('INFO','Found external PWV file.')
            self.pwvFile = self.pwvFile
            self.hasExternalPWV = True
            self.loadExternalPWV(self.externalPWVDeltaT)
            # need to add two global parameters!
            #self.parExternalPWVScale = 1.0
            #self.parExternalPWVOffset = 0.0

        self.externalTauFlag = np.zeros(self.nExp,dtype=np.bool)
        if (self.tauFile is not None):
            self.fgcmLog.log('INFO','Found external tau file.')
            self.tauFile = self.tauFile
            self.hasExternalTau = True
            self.loadExternalTau()
            # need to add two global parameters!
            #self.parExternalTauScale = 1.0
            #self.parExternalTauOffset = 0.0

        # and the aperture corrections
        self.compAperCorrPivot = np.zeros(self.nBands,dtype='f8')
        self.compAperCorrSlope = np.zeros(self.nBands,dtype='f8')
        self.compAperCorrSlopeErr = np.zeros(self.nBands,dtype='f8')
        self.compAperCorrRange = np.zeros((2,self.nBands),dtype='f8')

        # one of the "parameters" is expGray
        self.compExpGray = np.zeros(self.nExp,dtype='f8')
        self.compVarGray = np.zeros(self.nExp,dtype='f8')
        self.compNGoodStarPerExp = np.zeros(self.nExp,dtype='i4')

        # and sigFgcm
        self.compSigFgcm = np.zeros(self.nBands,dtype='f8')

        # compute the units
        self.unitDictSteps = fgcmLUT.computeStepUnits(self.stepUnitReference,
                                                      self.stepGrain,
                                                      self.meanNightDuration,
                                                      self.meanWashIntervalDuration,
                                                      self.fitBands)

        # do lookups on parameter array
        self._arrangeParArray()

        # and we're done

    def _loadOldParameters(self, expInfo, inParInfo, inParams, inSuperStar):
        """
        """

        # link band indices
        self._makeBandIndices()
        self._loadExposureInfo(expInfo)

        self._loadEpochAndWashInfo()

        # set the units from the inParInfo
        #self.tauStepUnits = inParInfo['TAUSTEPUNITS'][0]
        #self.tauPerSlopeStepUnits = inParInfo['TAUPERSLOPESTEPUNITS'][0]
        #self.alphaStepUnits = inParInfo['ALPHASTEPUNITS'][0]
        #self.pwvStepUnits = inParInfo['PWVSTEPUNITS'][0]
        #self.pwvPerSlopeStepUnits = inParInfo['PWVPERSLOPESTEPUNITS'][0]
        #self.o3StepUnits = inParInfo['O3STEPUNITS'][0]
        #self.washStepUnits = inParInfo['WASHSTEPUNITS'][0]
        #self.washSlopeStepUnits = inParInfo['WASHSLOPESTEPUNITS'][0]
        self.unitDictSteps = {'tauUnit': inParInfo['TAUUNIT'][0],
                              'tauPerSlopeUnit': inParInfo['TAUPERSLOPEUNIT'][0],
                              'alphaUnit': inParInfo['ALPHAUNIT'][0],
                              'pwvUnit': inParInfo['PWVUNIT'][0],
                              'pwvPerSlopeUnit': inParInfo['PWVPERSLOPEUNIT'][0],
                              'o3Unit': inParInfo['O3UNIT'][0],
                              'qeSysUnit': inParInfo['QESYSUNIT'][0],
                              'qeSysSlopeUnit': inParInfo['QESYSSLOPEUNIT'][0]}

        # and log
        self.fgcmLog.log('INFO','tau step unit set to %f' % (self.unitDictSteps['tauUnit']))
        self.fgcmLog.log('INFO','tau percent slope step unit set to %f' %
                         (self.unitDictSteps['tauPerSlopeUnit']))
        self.fgcmLog.log('INFO','alpha step unit set to %f' % (self.unitDictSteps['alphaUnit']))
        self.fgcmLog.log('INFO','pwv step unit set to %f' % (self.unitDictSteps['pwvUnit']))
        self.fgcmLog.log('INFO','pwv percent slope step unit set to %f' %
                         (self.unitDictSteps['pwvPerSlopeUnit']))
        self.fgcmLog.log('INFO','O3 step unit set to %f' % (self.unitDictSteps['o3Unit']))
        self.fgcmLog.log('INFO','wash step unit set to %f' % (self.unitDictSteps['qeSysUnit']))
        self.fgcmLog.log('INFO','wash slope step unit set to %f' %
                         (self.unitDictSteps['qeSysSlopeUnit']))

        # look at external...
        self.hasExternalPWV = inParInfo['HASEXTERNALPWV'][0].astype(np.bool)
        self.hasExternalTau = inParInfo['HASEXTERNALTAU'][0].astype(np.bool)

        ## and copy the parameters
        self.parAlpha = inParams['PARALPHA'][0]
        self.parO3 = inParams['PARO3'][0]
        self.parTauIntercept = inParams['PARTAUINTERCEPT'][0]
        self.parTauPerSlope = inParams['PARTAUPERSLOPE'][0]
        self.parPWVIntercept = inParams['PARPWVINTERCEPT'][0]
        self.parPWVPerSlope = inParams['PARPWVPERSLOPE'][0]
        self.parQESysIntercept = inParams['PARQESYSINTERCEPT'][0]
        self.parQESysSlope = inParams['PARQESYSSLOPE'][0]

        if (self.resetParameters):
            # reset many of the parameters
            self.parAlpha[:] = self.alphaStd
            self.parO3[:] = self.o3Std
            self.parTauIntercept[:] = self.tauStd
            self.parTauPerSlope[:] = 0.0
            self.parPWVIntercept[:] = self.pwvStd
            self.parPWVPerSlope[:] = 0.0
            # leave the QESysIntercept and Slope as previously fit
            # though we want to play with this

        self.externalPWVFlag = np.zeros(self.nExp,dtype=np.bool)
        if self.hasExternalPWV:
            self.pwvFile = str(inParInfo['PWVFILE'][0]).rstrip()
            self.hasExternalPWV = True
            self.loadExternalPWV(self.externalPWVDeltaT)
            self.parExternalPWVScale = inParams['PAREXTERNALPWVSCALE'][0]
            self.parExternalPWVOffset[:] = inParams['PAREXTERNALPWVOFFSET'][0]

            if (self.resetParameters):
                self.parExternalPWVScale = 1.0
                self.parExternalPWVOffset[:] = 0.0

        self.externalTauFlag = np.zeros(self.nExp,dtype=np.bool)
        if self.hasExternalTau:
            self.tauFile = str(inParInfo['TAUFILE'][0]).rstrip()
            self.hasExternalTau = True
            self.loadExternalTau()
            self.parExternalTauScale = inParams['PAREXTERNALTAUSCALE'][0]
            self.parExternalTauOffset[:] = inParams['PAREXTERNALTAUOFFSET'][0]

            if (self.resetParameters):
                self.parExternalTauScale = 1.0
                self.parExternalTauOffset[:] = 0.0


        self.compAperCorrPivot = inParams['COMPAPERCORRPIVOT'][0]
        self.compAperCorrSlope = inParams['COMPAPERCORRSLOPE'][0]
        self.compAperCorrSlopeErr = inParams['COMPAPERCORRSLOPEERR'][0]
        self.compAperCorrRange = np.reshape(inParams['COMPAPERCORRRANGE'][0],(2,self.nBands))

        self.compExpGray = inParams['COMPEXPGRAY'][0]
        self.compVarGray = inParams['COMPVARGRAY'][0]
        self.compNGoodStarPerExp = inParams['COMPNGOODSTARPEREXP'][0]

        self.compSigFgcm = inParams['COMPSIGFGCM'][0]

        self._arrangeParArray()

        # need to load the superstarflats
        self.parSuperStarFlat = inSuperStar

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

    def _loadExposureInfo(self, expInfo):
        """
        """

        # ensure sorted by exposure number
        st=np.argsort(expInfo[self.expField])
        expInfo=expInfo[st]

        self.nExp = self.nExp

        self.fgcmLog.log('INFO','Loading info on %d exposures.' % (self.nExp))

        self.expArray = expInfo[self.expField]
        self.expFlag = np.zeros(self.nExp,dtype=np.int8)
        self.expExptime = expInfo['EXPTIME']

        self.expSeeingVariable = expInfo[self.seeingField]
        self.expDeepFlag = expInfo[self.deepFlag]

        # we need the nights of the survey (integer MJD, maybe rotated)
        self.expMJD = expInfo['MJD']
        mjdForNight = np.floor(self.expMJD + self.UTBoundary).astype(np.int32)
        self.campaignNights = np.unique(mjdForNight)
        self.nCampaignNights = self.campaignNights.size

        self.fgcmLog.log('INFO','Exposures taken on %d nights.' % (self.nCampaignNights))

        self.expDeltaUT = (self.expMJD + self.UTBoundary) - mjdForNight

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
        self.sinLatitude = np.sin(np.radians(self.latitude))
        self.cosLatitude = np.cos(np.radians(self.latitude))

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
        self.nEpochs = self.epochMJDs.size - 1

        self.expEpochIndex = np.zeros(self.nExp,dtype='i4')
        for i in xrange(self.nEpochs):
            use,=np.where((self.expMJD > self.epochMJDs[i]) &
                          (self.expMJD < self.epochMJDs[i+1]))
            self.expEpochIndex[use] = i

    def _loadEpochAndWashInfo(self):
        """
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
        """

        # make pointers to a fit parameter array...
        #  pwv, O3, lnTau, alpha
        self.nFitPars = (self.campaignNights.size +  # O3
                         self.campaignNights.size +  # tauIntercept
                         self.campaignNights.size +  # tauPerSlope
                         self.campaignNights.size +  # alpha
                         self.campaignNights.size +  # pwv Intercept
                         self.campaignNights.size)   # pwv Slope
        ctr=0
        self.parO3Loc = ctr
        ctr+=self.campaignNights.size
        self.parTauInterceptLoc = ctr
        ctr+=self.campaignNights.size
        self.parTauPerSlopeLoc = ctr
        ctr+=self.campaignNights.size
        self.parAlphaLoc = ctr
        ctr+=self.campaignNights.size
        self.parPWVInterceptLoc = ctr
        ctr+=self.campaignNights.size
        self.parPWVPerSlopeLoc = ctr
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

    def saveParsFits(self, parFile):
        """
        """

        import fitsio

        # save the parameter file...
        self.fgcmLog.log('INFO','Saving parameters to %s' % (parFile))

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
        """
        # this can be run without fits

        dtype=[('NCCD','i4'),
               ('BANDS','a2',self.bands.size),
               ('FITBANDS','a2',self.fitBands.size),
               ('EXTRABANDS','a2',self.extraBands.size),
               ('TAUUNIT','f8'),
               ('TAUPERSLOPEUNIT','f8'),
               ('ALPHAUNIT','f8'),
               ('PWVUNIT','f8'),
               ('PWVPERSLOPEUNIT','f8'),
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
        parInfo['BANDS'] = self.bands
        parInfo['FITBANDS'] = self.fitBands
        parInfo['EXTRABANDS'] = self.extraBands

        parInfo['TAUUNIT'] = self.unitDictSteps['tauUnit']
        parInfo['TAUPERSLOPEUNIT'] = self.unitDictSteps['tauPerSlopeUnit']
        parInfo['ALPHAUNIT'] = self.unitDictSteps['alphaUnit']
        parInfo['PWVUNIT'] = self.unitDictSteps['pwvUnit']
        parInfo['PWVPERSLOPEUNIT'] = self.unitDictSteps['pwvPerSlopeUnit']
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
               ('PARTAUINTERCEPT','f8',self.parTauIntercept.size),
               ('PARTAUPERSLOPE','f8',self.parTauPerSlope.size),
               ('PARPWVINTERCEPT','f8',self.parPWVIntercept.size),
               ('PARPWVPERSLOPE','f8',self.parPWVPerSlope.size),
               ('PARQESYSINTERCEPT','f8',self.parQESysIntercept.size),
               ('PARQESYSSLOPE','f8',self.parQESysSlope.size),
               ('COMPAPERCORRPIVOT','f8',self.compAperCorrPivot.size),
               ('COMPAPERCORRSLOPE','f8',self.compAperCorrSlope.size),
               ('COMPAPERCORRSLOPEERR','f8',self.compAperCorrSlopeErr.size),
               ('COMPAPERCORRRANGE','f8',self.compAperCorrRange.size),
               ('COMPEXPGRAY','f8',self.compExpGray.size),
               ('COMPVARGRAY','f8',self.compVarGray.size),
               ('COMPNGOODSTARPEREXP','i4',self.compNGoodStarPerExp.size),
               ('COMPSIGFGCM','f8',self.compSigFgcm.size)]

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
        pars['PARTAUPERSLOPE'][:] = self.parTauPerSlope
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

        pars['COMPEXPGRAY'][:] = self.compExpGray
        pars['COMPVARGRAY'][:] = self.compVarGray
        pars['COMPNGOODSTARPEREXP'][:] = self.compNGoodStarPerExp

        pars['COMPSIGFGCM'][:] = self.compSigFgcm

        return parInfo, pars

    def loadExternalPWV(self, externalPWVDeltaT):
        """
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
        self.parPWVPerSlope[:] = parArray[self.parPWVPerSlopeLoc:
                                              self.parPWVPerSlopeLoc + self.nCampaignNights] / unitDict['pwvPerSlopeUnit']
        self.parO3[:] = parArray[self.parO3Loc:
                                     self.parO3Loc+self.nCampaignNights] / unitDict['o3Unit']
        self.parTauIntercept[:] = parArray[self.parTauInterceptLoc:
                                               self.parTauInterceptLoc+self.nCampaignNights] / unitDict['tauUnit']
        self.parTauPerSlope[:] = (parArray[self.parTauPerSlopeLoc:
                                               self.parTauPerSlopeLoc + self.nCampaignNights] /
                                  unitDict['tauPerSlopeUnit'])

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
                       (self.parPWVPerSlope[self.expNightIndex] *
                        self.parPWVIntercept[self.expNightIndex]) * self.expDeltaUT)

        if (self.hasExternalPWV):
            # replace where we have these
            self.expPWV[self.externalPWVFlag] = (self.parExternalPWVOffset[self.expNightIndex[self.externalPWVFlag]] +
                                                 self.parExternalPWVScale *
                                                 self.externalPWV[self.externalPWVFlag])

        # default to nightly slope/intercept
        self.expTau = (self.parTauIntercept[self.expNightIndex] +
                       (self.parTauPerSlope[self.expNightIndex] *
                        self.parTauIntercept[self.expNightIndex]) * self.expDeltaUT)

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
        parArray[self.parPWVPerSlopeLoc:
                     self.parPWVPerSlopeLoc + self.nCampaignNights] = self.parPWVPerSlope[:] * unitDict['pwvPerSlopeUnit']
        parArray[self.parO3Loc:
                     self.parO3Loc+self.nCampaignNights] = self.parO3[:] * unitDict['o3Unit']
        parArray[self.parTauInterceptLoc:
                     self.parTauInterceptLoc+self.nCampaignNights] = self.parTauIntercept[:] * unitDict['tauUnit']
        parArray[self.parTauPerSlopeLoc:
                     self.parTauPerSlopeLoc + self.nCampaignNights] = self.parTauPerSlope[:] * unitDict['tauPerSlopeUnit']
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

        parLow[self.parPWVInterceptLoc: \
                   self.parPWVInterceptLoc + \
               self.nCampaignNights] = ( \
            self.pwvRange[0] * unitDict['pwvUnit'])
        parHigh[self.parPWVInterceptLoc: \
                    self.parPWVInterceptLoc + \
                    self.nCampaignNights] = ( \
            self.pwvRange[1] * unitDict['pwvUnit'])
        parLow[self.parPWVPerSlopeLoc: \
                   self.parPWVPerSlopeLoc + \
                   self.nCampaignNights] = ( \
            -0.05 * unitDict['pwvPerSlopeUnit'])
        parHigh[self.parPWVPerSlopeLoc: \
                    self.parPWVPerSlopeLoc + \
                    self.nCampaignNights] = ( \
            0.05 * unitDict['pwvPerSlopeUnit'])
        parLow[self.parO3Loc: \
                   self.parO3Loc + \
                   self.nCampaignNights] = ( \
            self.O3Range[0] * unitDict['o3Unit'])
        parHigh[self.parO3Loc: \
                    self.parO3Loc + \
                    self.nCampaignNights] = ( \
            self.O3Range[1] * unitDict['o3Unit'])
        parLow[self.parTauInterceptLoc: \
                   self.parTauInterceptLoc + \
                   self.nCampaignNights] = ( \
            self.tauRange[0] * unitDict['tauUnit'])
        parHigh[self.parTauInterceptLoc: \
                    self.parTauInterceptLoc + \
                self.nCampaignNights] = ( \
            self.tauRange[1] * unitDict['tauUnit'])
        parLow[self.parTauPerSlopeLoc: \
                   self.parTauPerSlopeLoc + \
                   self.nCampaignNights] = ( \
            -0.05 * unitDict['tauPerSlopeUnit'])
        parHigh[self.parTauPerSlopeLoc: \
                    self.parTauPerSlopeLoc + \
                    self.nCampaignNights] = ( \
            0.05 * unitDict['tauPerSlopeUnit'])
        parLow[self.parAlphaLoc: \
                   self.parAlphaLoc + \
                   self.nCampaignNights] = ( \
            0.25 * unitDict['alphaUnit'])
        parHigh[self.parAlphaLoc: \
                    self.parAlphaLoc + \
                    self.nCampaignNights] = ( \
            1.75 * unitDict['alphaUnit'])
        if (self.hasExternalPWV):
            parLow[self.parExternalPWVScaleLoc] = 0.5 * unitDict['pwvUnit']
            parHigh[self.parExternalPWVScaleLoc] = 1.5 * unitDict['pwvUnit']
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
            parLow[self.parExternalTauScaleLoc] = 0.7 * unitDict['tauUnit']
            parHigh[self.parExternalTauScaleLoc] = 1.2 * unitDict['tauUnit']
            parLow[self.parExternalTauOffsetLoc: \
                       self.parExternalTauOffsetLoc + \
                   self.nCampaignNights] = ( \
                0.0 * unitDict['tauUnit'])
            parHigh[self.parExternalTauOffsetLoc: \
                        self.parExternalTauOffsetLoc + \
                        self.nCampaignNights] = ( \
                0.03 * unitDict['tauUnit'])
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

        # zip these into a list of tuples
        parBounds = zip(parLow, parHigh)

        return parBounds

    def getUnitDict(self,fitterUnits=False):
        if (fitterUnits):
            return self.unitDictSteps
        else:
            return self.unitDictOnes

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

        ylim = ax.get_ylim()
        for i in xrange(self.nWashIntervals):
            ax.plot([self.washMJDs[i]-firstMJD,self.washMJDs[i]-firstMJD],
                    ylim,'k--')

        fig.savefig('%s/%s_qesys_washes.png' % (self.plotPath,
                                                self.outfileBaseWithCycle))

        ## FIXME: add pwv offset plotting routine (if external)
        ## FIXME: add tau offset plotting routing (if external)
