from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import time

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmGray(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmStars):

        # need fgcmPars because it tracks good exposures
        #  also this is where the gray info is stored
        self.fgcmPars = fgcmPars

        # need fgcmStars because it has the stars (duh)
        self.fgcmStars = fgcmStars

        if (not self.fgcmStars.magStdComputed):
            raise ValueError("Must run FgcmChisq to compute magStd before FgcmGray")

        # and record configuration variables...
        self.minStarPerCCD = self.fgcmConfig.minStarPerCCD
        self.minCCDPerExp = self.fgcmConfig.minCCDPerExp

        self._prepareGrayArrays()

    def _prepareGrayArrays(self):
        """
        """

        # we have expGray for Selection
        self.expGrayForSelectionHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayRMSForSelectionHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayNGoodStarForSelectionHandle = snmm.createArray(self.fgcmPars.nExp,dtype='i4')

        # and the exp/ccd gray for the zeropoints

        self.ccdGrayHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdGrayRMSHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdGrayErrHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdNGoodStarsHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='i4')

        self.expGrayHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayRMSHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayErrHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayNGoodCCDsHandle = snmm.createArray(self.fgcmPars.nExp,dtype='i2')

    def computeExpGrayForSelection(self):
        """
        """

        # useful numbers
        expGrayForSelection = snmm.getArray(self.expGrayForSelectionHandle)
        expGrayRMSForSelection = snmm.getArray(self.expGrayRMSForSelectionHandle)

        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)

        a,b=esutil.numpy_util.match(self.fgcmPars.expArray,
                                    snmm.getArray(self.fgcmStars.obsExpHandle)[:])
        obsExpIndex = np.zeros(self.fgcmStars.nStarObs,dtype='i4')
        obsExpIndex[b] = a

        # first, we need to compute E_gray == <mstd> - mstd for each observation

        # compute all the EGray values

        EGray = np.zeros(self.fgcmStars.nStarObs,dtype='f8')
        EGray[obsIndex] = (objMagStdMean[obsObjIDIndex[obsIndex],obsBandIndex[obsIndex]] -
                           obsMagStd[obsIndex])

        # only use good observations of good stars...

        # for the required bands
        req,=np.where(self.fgcmStars.bandRequired)
        minObs = objNGoodObs[:,req].min(axis=1)

        goodStars, = np.where(minObs >= self.fgcmStars.minPerBand)

        # and the extra bands
        notReq,=np.where(~self.fgcmStars.bandRequired)
        minNotObs = objNGoodObs[:,notReq].min(axis=1)

        goodExtraStars, = np.where(minNotObs >= self.fgcmStars.minPerBand)

        # select observations of these stars...
        a,b=esutil.numpy_util.match(objID[goodStars],objID[obsObjIDIndex])

        # now group per exposure and sum...

        expGrayForSelection[:] = 0.0
        expGrayRMSForSelection[:] = 0.0
        expGrayNGoodStarForSelection[:] = 0

        np.add.at(expGrayForSelection,
                  obsExpIndex[b],
                  EGray[b])
        np.add.at(expGrayRMSForSelection,
                  obsExpIndex[b],
                  EGray[b]**2.)
        np.add.at(expGrayNGoodStarForSelection,
                  obsExpIndex[b],
                  1)

        gd,=np.where(expGrayNGoodStarForSelection > 0)
        expGrayForSelection[gd] /= expGrayNGoodStarForSelection[gd]
        expGrayRMSForSelection[gd] = np.sqrt((expGrayRMSForSelection[gd]/expGrayNGoodStarForSelection[gd]) -
                                             (expGrayForSelection[gd])**2.)

    def computeCCDAndExpGray(self):
        """
        """
        pass


        
