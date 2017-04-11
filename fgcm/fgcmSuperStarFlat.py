from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import time

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

class fgcmSuperStarFlat(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmGray):

        self.fgcmPars = fgcmPars

        self.fgcmGray = fgcmGray

        self.minStarPerCCD = fgcmConfig.minStarPerCCD

    def computeSuperStarFlats(self):
        """
        """

        ## FIXME: need to know which are the "Good Exposures" to use!
        ## FIXME: need to filter out SN (deep) exposures.  Hmmm.

        deltaSuperStarFlat = np.zeros_like(self.fgcmPars.parSuperStarFlat)
        deltaSuperStarFlatNCCD = np.zeros_like(self.fgcmPars.parSuperStarFlat,dtype='i4')

        ccdGray = snmm.getArray(self.fgcmGray.ccdGrayHandle)
        ccdGrayErr = snmm.getArray(self.fgcmGray.ccdGrayErrHandle)
        ccdNGoodStars = snmm.getArray(self.fgcmGray.ccdNGoodStarsHandle)

        # only select those CCDs that we have an adequate gray calculation
        expIndexUse,ccdIndexUse=np.where((ccdNGoodStars >= self.minStarPerCCD))

        # and only select exposures that should go into the SuperStarFlat
        gd,=np.where(self.fgcmPars.expFlag[expIndexUse] == 0)
        expIndexUse=expIndexUse[gd]
        ccdIndexUse=ccdIndexUse[gd]

        # sum up ccdGray values
        np.add.at(deltaSuperStarFlat,
                  (self.fgcmPars.expEpochIndex[expIndexUse],
                   self.fgcmPars.expBandIndex[expIndexUse],
                   ccdIndexUse),
                  ccdGray[expIndexUse,ccdIndexUse])
        np.add.at(deltaSuperStarFlatNCCD,
                  (self.fgcmPars.expEpochIndex[expIndexUse],
                   self.fgcmPars.expBandIndex[expIndexUse],
                   ccdIndexUse),
                  1)

        # only use exp/ccd where we have at least one observation
        gd=np.where(deltaSuperStarFlatNCCD > 0)
        deltaSuperStarFlat[gd] /= deltaSuperStarFlatNCCD[gd]

        # this accumulates onto the input parameters
        self.fgcmPars.parSuperStarFlat += deltaSuperStarFlat

        # and we're done.
