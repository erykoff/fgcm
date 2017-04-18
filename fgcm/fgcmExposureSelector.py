from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil

from fgcmUtilities import expFlagDict

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmExposureSelector(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars):

        self.fgcmPars = fgcmPars

        # and config variables...
        self.minStarPerExp = fgcmConfig.minStarPerExp
        self.minExpPerNight = fgcmConfig.minExpPerNight
        #self.expGrayCut = fgcmConfig.expGrayCut
        #self.varGrayCut = fgcmConfig.varGrayCut
        #self.expGrayInitialCut = fgcmConfig.expGrayInitialCut
        self.expGrayPhotometricCut = fgcmConfig.expGrayPhotometricCut
        self.expVarGrayPhotometricCut = fgcmConfig.expVarGrayPhotometricCut
        self.expGrayInitialCut = fgcmConfig.expGrayInitialCut

    def selectGoodExposures(self):
        """
        """

        # this cuts on expgray,vargray
        # based on those in the parameter file?

        self.fgcmPars.expFlag[:] = 0

        bad,=np.where(self.fgcmPars.compNGoodStarPerExp == 0)
        self.fgcmPars.expFlag[bad] |= expFlagDict['NO_STARS']

        bad,=np.where(self.fgcmPars.compNGoodStarPerExp < self.minStarPerExp)
        self.fgcmPars.expFlag[bad] |= expFlagDict['TOO_FEW_STARS']

        bad,=np.where(self.fgcmPars.compExpGray < self.expGrayPhotometricCut)
        self.fgcmPars.expFlag[bad] |= expFlagDict['EXP_GRAY_TOO_LARGE']

        bad,=np.where(self.fgcmPars.compVarGray > self.expVarGrayPhotometricCut)
        self.fgcmPars.expFlag[bad] |= expFlagDict['VAR_GRAY_TOO_LARGE']


        ## FIXME: do we want to consider minCCDPerExp?

    def selectGoodExposuresInitialSelection(self, fgcmGray):
        """
        """

        # this requires fgcmGray
        #  FIXME: ensure that fgcmGray has run initial selection
        self.fgcmPars.expFlag[:] = 0

        expGrayForInitialSelection = snmm.getArray(fgcmGray.expGrayForInitialSelectionHandle)
        expNGoodStarForInitialSelection = snmm.getArray(fgcmGray.expNGoodStarForInitialSelectionHandle)

        bad,=np.where(expNGoodStarForInitialSelection < self.minStarPerExp)
        self.fgcmPars.expFlag[bad] |= expFlagDict['TOO_FEW_STARS']

        bad,=np.where(expGrayForInitialSelection < self.expGrayInitialCut)
        self.fgcmPars.expFlag[bad] |= expFlagDict['EXP_GRAY_TOO_LARGE']


    def selectCalibratableNights(self):
        """
        """

        # this will use existing flags...

        # select good exposures,
        #  limit to those that are in the fit bands
        goodExp,=np.where((self.fgcmPars.expFlag == 0) &
                          (~self.fgcmPars.expExtraBandFlag))

        #nExpPerNight = np.zeros(self.fgcmPars.nCampaignNights)

        # we first need to look for the good nights
        nExpPerNight=esutil.stat.histogram(self.fgcmPars.expNightIndex[goodExp],min=0,
                                max=self.fgcmPars.nCampaignNights-1)

        badNights,=np.where(nExpPerNight < self.minExpPerNight)

        # and we need to use *all* the exposures to flag bad nights
        h,rev = esutil.stat.histogram(self.fgcmPars.expNightIndex,min=0,
                                      max=self.fgcmPars.nCampaignNights-1,rev=True)

        for badNight in badNights:
            i1a=rev[rev[badNight]:rev[badNight+1]]

            self.fgcmPars.expFlag[i1a] |= expFlagDict['TOO_FEW_EXP_ON_NIGHT']


