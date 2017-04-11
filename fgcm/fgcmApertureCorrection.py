from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import time
import matplotlib.pyplot as plt
import scipy.optimize


from sharedNumpyMemManager import SharedNumpyMemManager as snmm
from fgcmUtilities import dataBinner

class FgcmApertureCorrection(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmGray):
        self.fgcmPars = fgcmPars

        self.fgcmGray = fgcmGray

        # and record configuration variables
        ## include plot path...
        self.aperCorrFitNBins = fgcmConfig.aperCorrFitNBins

    def computeApertureCorrections(self):
        """
        """

        # we need good exposures...
        # and then bin and fit...
        # need to remove any previous!

        # expSeeingVariable

        expGray = snmm.getArray(self.fgcmGray.expGrayHandle)

        # first, remove any previous correction if necessary...
        if (np.max(self.fgcmPars.parAperCorrRange[1,:]) >
            np.min(self.fgcmPars.parAperCorrRange[0,:])) :

            expSeeingVariableClipped = np.clip(expSeeingVariable,
                                               self.fgcmPars.parAperCorrRange[0,self.fgcmPars.expBandIndex],
                                               self.fgcmPars.parAperCorrRange[1,self.fgcmPars.expBandIndex])

            oldAperCorr = self.fgcmPars.parAperCorrSlope[self.fgcmPars.expBandIndex] * (
                expSeeingVariableClipped -
                self.fgcmPars.parAperCorrPivot[self.fgcmPars.expBandIndex])

            # Need to check sign here...
            expGray -= oldAperCorr

        expIndexUse,=np.where(self.fgcmPars.expFlag == 0)

        for i in xrange(self.fgcmPars.nBands):
            ## FIXME: marker for illegal value?
            use,=np.where((self.fgcmPars.expBandIndex[expIndexUse] == i) &
                          (self.fgcmPars.expSeeingVariable[expIndexUse] > -9999.0) &
                          (np.isfinite(self.fgcmPars.expSeeingVariable[expIndexUse])))

            # sort to set the range...
            st=np.argsort(expGray[use])
            use=use[st]

            self.fgcmPars.parAperCorrRange[0,i] = expSeeingVariable[expIndexUse[use[int(0.02*use.size)]]]
            self.fgcmPars.parAperCorrRange[1,i] = expSeeingVariable[expIndexUse[use[int(0.98*use.size)]]]

            # this will make a rounder number
            self.fgcmPars.parAperCorrPivot[i] = np.floor(np.median(expSeeingVariable[expIndexUse[use]])*1000)/1000.

            binSize = (self.fgcmPars.parAperCorrRange[1,i] -
                       self.fgcmPars.parAperCorrRange[0,i]) / self.aperCorrFitNBins

            binStruct = dataBinner(self.fgcmPars.expSeeingVariable[use],
                                   expGray[use],
                                   binSize,
                                   self.fgcmPars.parAperCorrRange[:,i])

            # remove any empty bins...
            gd,=np.where(binStruct['Y_ERR'] > 0.0)
            binStruct=binStruct[gd]

            fit,cov = np.polyfit(binStruct['X_BIN'] - self.fgcmPars.parAperCorrPivot[i],
                                 binStruct['Y'],
                                 1.0,
                                 w=(1./binStruct['Y_ERR'])**2.,
                                 cov=True)

            self.fgcmPars.parAperCorrSlope[i] = fit[0]
            self.fgcmPars.parAperCorrSlopeErr[i] = np.sqrt(cov[0,0])
