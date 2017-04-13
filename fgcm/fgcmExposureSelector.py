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
        self.expGrayCut = fgcmConfig.expGrayCut
        self.varGrayCut = fgcmConfig.varGrayCut
        self.expGrayInitialCut = fgcmConfig.expGrayInitialCut

    def selectGoodExposures(self):
        """
        """

        # this cuts on expgray,vargray
        # based on those in the parameter file?

        bad,=np.where(self.fgcmPars.compNGoodStarPerExp < self.minStarPerExp)
        self.fgcmPars.expFlag[bad] |= expFlagDict['TOO_FEW_STARS']

        bad,=np.where(self.fgcmPars.compExpGray < self.expGrayCut)
        self.fgcmPars.expFlag[bad] |= expFlagDict['EXP_GRAY_TOO_LARGE']

        bad,=np.where(self.fgcmPars.compVarGray > self.varGrayCut)
        self.fgcmPars.expFlag[bad] |= expFlagDict['VAR_GRAY_TOO_LARGE']

        # and what about number of stars?

    def selectGoodExposuresInitialSelection(self, fgcmGray):
        """
        """

        # this requires fgcmGray
        #  FIXME: ensure that fgcmGray has run initial selection

        expGrayForInitialSelection = snmm.getArray(fgcmGray.expGrayForInitialSelectionHandle)
        expGrayNGoodStarForInitialSelection = snmm.getArray(fgcmGray.expGrayNGoodStarForInitialSelectionHandle)

        bad,=np.where(expGrayNGoodStarForInitialSelection < self.minStarPerExp)
        self.fgcmPars.expFlag[bad] |= expFlagDict['TOO_FEW_STARS']

        bad,=np.where(expGrayForInitialSelection < self.expGrayInitialCut)
        self.fgcmPars.expFlag[bad] |= expFlagDict['EXP_GRAY_TOO_LARGE']


    def selectCalibratableNights(self):
        """
        """

        # this will use existing flags...

        # select good exposures
        goodExp,=np.where(self.fgcmPars.expFlag == 0)

        nExpPerNight = np.zeros(self.fgcmPars.nCampaignNights)

        h,rev=esutil.stat.histogram(self.fgcmPars.expNightIndex[goodExp],min=0,
                                    max=self.fgcmPars.nCampaignNights-1,rev=True)
        badNights,=np.where(h < self.minExpPerNight)

        for badNight in badNights:
            i1a=rev[rev[badNight]:rev[badNight+1]]

            self.fgcmPars.expFlag[i1a] |= expFlagDict['TOO_FEW_EXP_ON_NIGHT']


