from __future__ import print_function

import numpy as np
import fitsio
import os
import sys

from fgcmUtilities import _pickle_method

import types
import copy_reg
import sharedmem as shm

copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmParameters(object):
    """
    """
    def __init__(self,parFile=None,
                 fgcmConfig=None):

        self.hasExternalPWV = False

        pass

    def initializeParameters(self, fgcmConfig):
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

        # first thing is to get the exposure numbers...
        expInfo = fitsio.read(fgcmConfig.exposureFile,ext=1)

        # ensure sorted by exposure number
        st=np.argsort(expInfo['EXPNUM'])
        expInfo=expInfo[st]

        self.nExp = fgcmConfig.nExp

        self.expArray = expInfo['EXPNUM']
        self.expFlag = np.zeros(self.nExp,dtype=np.int8)
        self.expExptime = expInfo['EXPTIME']

        self.expSeeingVariable = expInfo[fgcmConfig['seeingField']]
        self.expDeepFlag = expInfo[fgcmConfig['DEEPFLAG']]

        # we need the nights of the survey (integer MJD, maybe rotated)
        self.expMJD = expInfo['MJD']
        mjdForNight = np.floor(expInfo['MJD'] + fgcmConfig.UTBoundary).astype(np.int32)
        self.expNights = np.unique(mjdForNight)

        # and link the exposure numbers to the nights...
        a,b=esutil.numpy_util.match(self.expNights,mjdForNight)
        self.expNightIndex = np.zeros(self.nExp,dtype=np.int32)
        self.expNightIndex[b] = a

        # convert these to radians
        self.expTelHA = np.radians(expInfo['TELHA'])
        self.expTelRA = np.radians(expInfo['TELRA'])
        self.expTelDec = np.radians(expInfo['TELDEC'])

        # and rotate?
        hi,=np.where(self.expTelRA > np.pi)
        if hi.size > 0:
            self.expTelRA[hi] -= 2.0*np.pi

        self.expPmb = expInfo['PMB']

        # link exposures to bands
        self.expBandIndex = np.zeros(self.nExp,dtype='i2') - 1
        for i in xrange(self.bands):
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

        # and set up the wash mjds and link indices
        self.nWashIntervals = fgcmConfig.washMJDs.size + 1

        self.expWashIndex = np.zeros(self.nExp,dtype='i4')
        tempWashMJDs = fgcmConfig.washMJDs
        tempWashMJDs = np.insert(tempWashMJDs,0,0.0)
        tempWashMJDs = np.append(tempWashMJDs,1e10)
        for i in xrange(self.nWashIntervals):
            use,=np.where((self.expMJD > tempWashMJDs[i]) &
                          (self.expMJD < tempWashMJDs[i+1]))
            self.expWashIndex[use] = i

        # set up the parameters with nightly values
        # need to include the default stuff...

        self.parAlpha = np.zeros(self.expNights.size,dtype=np.float32) + fgcmConfig.alphaStd
        self.parO3 = np.zeros(self.expNights.size,dtype=np.float32) + fgcmConfig.o3Std
        self.parTauIntercept = np.zeros(self.expNights.size,dtype=np.float32) + fgcmConfig.tauStd
        self.parTauSlope = np.zeros(self.expNights.size,dtype=np.float32)
        self.parPwvIntercept = np.zeros(self.expNights.size,dtype=np.float32) + fgcmConfig.pwvStd
        self.parPwvSlope = np.zeros(self.expNights.size,dtype=np.float32)

        # parameters with per-epoch values
        self.parSuperflat = np.zeros((self.nEpochs,self.nBands,self.nCCD),dtype=np.float32)

        # parameters with per-wash values
        self.parDustIntercept = np.zeros(self.nWashIntervals,dtype=np.float32)
        self.parDustSlope = np.zeros(self.nWashIntervals,dtype=np.float32)

        if (fgcmConfig.pwvFile is not None):
            self.loadExternalPWV(fgcmConfig.pwvFile,fgcmConfig.externalPwvDeltaT)

        if (fgcmConfig.tauFile is not None):
            self.loadExternalTau(fgcmConfig.tauFile)

        # and need to be able to pack and unpack the parameters and scalings
        #  this part is going to be the hardest


    def loadParFile(self, parFile):
        """
        """
        # read in the parameter file...
        # need to decide on a format

        pass

    def saveParFile(self, parFile):
        """
        """
        # save the parameter file...
        # need to decide on a format

        pass

    def loadExternalPWV(self, pwvFile, externalPwvDeltaT):
        """
        """
        # loads a file with PWV, matches to exposures/times
        # flags which ones need the nightly fit

        self.hasExternalPWV = True

        pwvTable = fitsio.read(pwvFile,ext=1)

        # make sure it's sorted
        st=np.argsort(pwvTable['MJD'])
        pwvTable = pwvTable[st]

        pwvIndex = np.clip(np.searchsorted(pwvTable['MJD'],self.expMJD),0,pwvTable.size-1)
        gd,=np.where(np.abs(pwvTable['MJD'][pwvIndex] - self.expMJD) < externalPwvDeltaT

    def loadExternalTau(self, tauFile, withAlpha=False):
        """
        """
        # load a file with Tau values

        self.hasExternalTau = True
        if (withAlpha):
            self.hasExternalAlpha = True

    def reloadParArray(self, parArray):
        """
        """
        # takes in a parameter array and loads the local split shm copies

        pass

    def getParArray(self,bounds=False):
        """
        """
        # extracts parameters into a linearized array
        # also returns bounds if bounds=True

        pass

