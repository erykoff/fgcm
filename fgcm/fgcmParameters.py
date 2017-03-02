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
        #   all exposure numbers (from demand list)
        #   all exposure MJDs
        #   UT boundary for nights (From config)
        #   all exposure pressure values (could be loaded separately?)
        #   all exposure ZDs (-> secZenith and Airmass)
        #   all exposure bands (and link index)
        #   all exposure exptimes
        #   all exposure psf_fwhms -- or delta-aperture
        #   all wash dates (will need to crop to exposure range)
        #   flag for special (e.g. SN) exposures?
        #   lutfile here?  no.
        #   flag for exposure quality: 0 is good, and numbers for
        #      rejections of various types

        # default is to have a pwv_int, pwv_slope per night
        #  but will change to different parameters if loadExternalPWV
        # default is to have a tau_int, tau_slope per night
        #  but will change to different parameters if loadExternalTau
        # default is to have an alpha per night
        #  but will change to ... something if loadExternalAlpha
        # default is to have an Ozone per night
        #  but will change to global additive/multiplicative factor if loadExternalOzone

        # need an index to get quickly from exposure number to night

        #######################################################
        #######################################################

        # first thing is to get the exposure numbers...
        expInfo = fitsio.read(fgcmConfig.exposureFile,ext=1)

        # ensure sorted by exposure number
        st=np.argsort(expInfo[fgcmConfig.expField])
        expInfo=expInfo[st]

        self.nExp = fgcmConfig.nExp

        self.expArray = expInfo[fgcmConfig.expField]
        self.expFlag = np.zeros(self.nExp,dtype=np.int8)

        # we need the nights of the survey (integer MJD, maybe rotated)
        mjdForNight = np.floor(expInfo[fgcmConfig.mjdField] + fgcmConfig.UTBoundary).astype(np.int32)
        self.nights = np.unique(mjdForNight)

        # and link the exposure numbers to the nights...
        a,b=esutil.numpy_util.match(self.nights,mjdForNight)
        self.nightIndex = np.zeros(self.nExp,dtype=np.int32)
        self.nightIndex[b] = a

        # set up the parameters with nightly values


        # we need to process the hour angle &c.  Blah.
        


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

    def loadExternalPWV(self, pwvFile):
        """
        """
        # loads a file with PWV, matches to exposures/times

        self.hasExternalPWV = True

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

