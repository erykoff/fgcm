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
        self.minStarPerCCD = fgcmConfig.minStarPerCCD
        self.minCCDPerExp = fgcmConfig.minCCDPerExp
        self.maxCCDGrayErr = fgcmConfig.maxCCDGrayErr

        self._prepareGrayArrays()

    def _prepareGrayArrays(self):
        """
        """

        # we have expGray for Selection
        self.expGrayForInitialSelectionHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayRMSForInitialSelectionHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayNGoodStarForInitialSelectionHandle = snmm.createArray(self.fgcmPars.nExp,dtype='i4')

        # and the exp/ccd gray for the zeropoints

        self.ccdGrayHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdGrayRMSHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdGrayErrHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdNGoodStarsHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='i4')

        self.expGrayHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayRMSHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayErrHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayNGoodCCDsHandle = snmm.createArray(self.fgcmPars.nExp,dtype='i2')

    def computeExpGrayForInitialSelection(self):
        """
        """

        # Note this computes ExpGray for all exposures, good and bad


        # useful numbers
        expGrayForInitialSelection = snmm.getArray(self.expGrayForInitialSelectionHandle)
        expGrayRMSForInitialSelection = snmm.getArray(self.expGrayRMSForInitialSelectionHandle)
        expGrayNGoodStarForInitialSelection = snmm.getArray(self.expGrayNGoodStarForInitialSelectionHandle)

        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)

        #a,b=esutil.numpy_util.match(self.fgcmPars.expArray,
        #                            snmm.getArray(self.fgcmStars.obsExpHandle)[:])
        #obsExpIndex = np.zeros(self.fgcmStars.nStarObs,dtype='i4')
        #obsExpIndex[b] = a

        # first, we need to compute E_gray == <mstd> - mstd for each observation

        # compute all the EGray values

        EGray = np.zeros(self.fgcmStars.nStarObs,dtype='f8')
        EGray[obsIndex] = (objMagStdMean[obsObjIDIndex[obsIndex],obsBandIndex[obsIndex]] -
                           obsMagStd[obsIndex])

        # only use good observations of good stars...

        # for the required bands
        minObs = objNGoodObs[:,self.fgcmStars.bandRequiredIndex].min(axis=1)

        goodStars, = np.where((minObs >= self.fgcmStars.minPerBand) &
                              (objFlag == 0))

        # select observations of these stars...
        a,b=esutil.numpy_util.match(objID[goodStars],objID[obsObjIDIndex])

        # and first, we only use the required bands
        _,reqBandUse = esutil.numpy_util.match(self.fgcmStars.bandRequiredIndex,
                                               obsBandIndex[b])

        # now group per exposure and sum...

        expGrayForInitialSelection[:] = 0.0
        expGrayRMSForInitialSelection[:] = 0.0
        expGrayNGoodStarForInitialSelection[:] = 0

        np.add.at(expGrayForInitialSelection,
                  obsExpIndex[b],
                  EGray[b])
        np.add.at(expGrayRMSForInitialSelection,
                  obsExpIndex[b],
                  EGray[b]**2.)
        np.add.at(expGrayNGoodStarForInitialSelection,
                  obsExpIndex[b],
                  1)

        # loop over the extra bands...
        #  we only want to use previously determined "good" stars
        for extraBandIndex in self.fgcmStars.bandExtraIndex:
            extraBandUse, = np.where((obsBandIndex[b] == extraBandIndex) &
                                     (objNGoodObs[obsObjIDIndex[b],extraBandIndex] >=
                                      self.fgcmStars.minPerBand))

            np.add.at(expGrayForInitialSelection,
                      obsExpIndex[b[extraBandUse]],
                      EGray[b[extraBandUse]])
            np.add.at(expGrayRMSForInitialSelection,
                      obsExpIndex[b[extraBandUse]],
                      EGray[b[extraBandUse]]**2.)
            np.add.at(expGrayNGoodStarForInitialSelection,
                      obsExpIndex[b[extraBandUse]],
                      1)


        gd,=np.where(expGrayNGoodStarForInitialSelection > 0)
        expGrayForInitialSelection[gd] /= expGrayNGoodStarForInitialSelection[gd]
        expGrayRMSForInitialSelection[gd] = np.sqrt((expGrayRMSForInitialSelection[gd]/expGrayNGoodStarForInitialSelection[gd]) -
                                             (expGrayForInitialSelection[gd])**2.)

    def computeCCDAndExpGray(self):
        """
        """

        # Note: this computes the gray values for all exposures, good and bad

        # values to set
        ccdGray = snmm.getArray(self.ccdGrayHandle)
        ccdGrayRMS = snmm.getArray(self.ccdGrayRMSHandle)
        ccdGrayErr = snmm.getArray(self.ccdGrayErrHandle)
        ccdNGoodStars = snmm.getArray(self.ccdNGoodStarsHandle)

        expGray = snmm.getArray(self.expGrayHandle)
        expGrayRMS = snmm.getArray(self.expGrayRMSHandle)
        expGrayErr = snmm.getArray(self.expGrayErrHandle)
        expGrayNGoodCCDs = snmm.getArray(self.expGrayNGoodCCDsHandle)

        # input numbers
        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - 1

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)


        # first, we need to compute E_gray == <mstd> - mstd for each observation

        EGray = np.zeros(self.fgcmStars.nStarObs,dtype='f8')
        EGray[obsIndex] = (objMagStdMean[obsObjIDIndex[obsIndex],obsBandIndex[obsIndex]] -
                           obsMagStd[obsIndex])

        # and need the error for Egray: sum in quadrature of individual and avg errs
        EGrayErr2 = np.zeros(self.fgcmStars.nStarObs,dtype='f8')
        EGrayErr2[obsIndex] = (objMagStdMeanErr[obsObjIDIndex[obsIndex],obsBandIndex[obsIndex]]**2. +
                               obsMagErr[obsIndex]**2.)

        ## FIXME: only use stars that have sufficiently small EGrayErr2

        # only use good observations of good stars...
        minObs = objNGoodObs[:,self.fgcmStars.bandRequiredIndex].min(axis=1)

        goodStars, = np.where((minObs >= self.fgcmStars.minPerBand) &
                              (objFlag == 0))

        # select observations of these stars...
        a,b=esutil.numpy_util.match(objID[goodStars],objID[obsObjIDIndex])

        # first, we only use the required bands
        _,reqBandUse = esutil.numpy_util.match(self.fgcmStars.bandRequiredIndex,
                                               obsBandIndex[b])

        # group by CCD and sum

        ## ccdGray = Sum(EGray/EGrayErr^2) / Sum(1./EGrayErr^2)
        ## ccdGrayRMS = Sqrt((Sum(EGray^2/EGrayErr^2) / Sum(1./EGrayErr^2)) - ccdGray^2)
        ## ccdGrayErr = Sqrt(1./Sum(1./EGrayErr^2))

        ccdGray[:,:] = 0.0
        ccdGrayRMS[:,:] = 0.0
        ccdGrayErr[:,:] = 0.0
        ccdNGoodStars[:,:] = 0

        # temporary variable here
        ccdGrayWt = np.zeros_like(ccdGray)

        np.add.at(ccdGrayWt,
                  (obsExpIndex[b[reqBandUse]],obsCCDIndex[b[reqBandUse]]),
                  1./EGrayErr2[b[reqBandUse]])
        np.add.at(ccdGray,
                  (obsExpIndex[b[reqBandUse]],obsCCDIndex[b[reqBandUse]]),
                  EGray[b[reqBandUse]]/EGrayErr2[b[reqBandUse]])
        np.add.at(ccdGrayRMS,
                  (obsExpIndex[b[reqBandUse]],obsCCDIndex[b[reqBandUse]]),
                  EGray[b[reqBandUse]]**2./EGrayErr2[b[reqBandUse]])
        np.add.at(ccdNGoodStars,
                  (obsExpIndex[b[reqBandUse]],obsCCDIndex[b[reqBandUse]]),
                  1)

        # loop over the extra bands
        #  we only want to use previously determined "good" stars
        for extraBandIndex in self.fgcmStars.bandExtraIndex:
            extraBandUse, = np.where((obsBandIndex[b] == extraBandIndex) &
                                     (objNGoodObs[obsObjIDIndex[b],extraBandIndex] >=
                                      self.fgcmStars.minPerBand))

            np.add.at(ccdGrayWt,
                      (obsExpIndex[b[extraBandUse]],obsCCDIndex[b[extraBandUse]]),
                      1./EGrayErr2[b[extraBandUse]])
            np.add.at(ccdGray,
                      (obsExpIndex[b[extraBandUse]],obsCCDIndex[b[extraBandUse]]),
                      EGray[b[extraBandUse]]/EGrayErr2[b[extraBandUse]])
            np.add.at(ccdGrayRMS,
                      (obsExpIndex[b[extraBandUse]],obsCCDIndex[b[extraBandUse]]),
                      EGray[b[extraBandUse]]**2./EGrayErr2[b[extraBandUse]])
            np.add.at(ccdNGoodStars,
                      (obsExpIndex[b[extraBandUse]],obsCCDIndex[b[extraBandUse]]),
                      1)


        # need at least 3 or else computation can blow up
        gd = np.where(ccdNGoodStars > 2)
        ccdGray[gd] /= ccdGrayWt[gd]
        ccdGrayRMS[gd] = np.sqrt((ccdGrayRMS[gd]/ccdGrayWt[gd]) - (ccdGray[gd]**2.))
        ccdGrayErr[gd] = np.sqrt(1./ccdGrayWt[gd])

        # check for infinities
        bad=np.where(~np.isfinite(ccdGrayRMS))
        ccdGrayRMS[bad] = 0.0

        # group CCD by Exposure and Sum

        goodCCD = np.where((ccdNGoodStars >= self.minStarPerCCD) &
                           (ccdGrayErr > 0.0) &
                           (ccdGrayErr < self.maxCCDGrayErr))

        # note: goodCCD[0] refers to the expIndex, goodCCD[1] to the CCDIndex

        expGray[:] = 0.0
        expGrayRMS[:] = 0.0
        expGrayErr[:] = 0.0
        expGrayNGoodCCDs[:] = 0

        # temporary
        expGrayWt = np.zeros_like(expGray)

        np.add.at(expGrayWt,
                  goodCCD[0],
                  1./ccdGrayErr[goodCCD]**2.)
        np.add.at(expGray,
                  goodCCD[0],
                  ccdGray[goodCCD]/ccdGrayErr[goodCCD]**2.)
        np.add.at(expGrayRMS,
                  goodCCD[0],
                  ccdGray[goodCCD]**2./ccdGrayErr[goodCCD]**2.)
        np.add.at(expGrayNGoodCCDs,
                  goodCCD[0],
                  1)

        # need at least 3 or else computation can blow up
        gd, = np.where(expGrayNGoodCCDs > 2)
        expGray[gd] /= expGrayWt[gd]
        expGrayRMS[gd] = np.sqrt((expGrayRMS[gd]/expGrayWt[gd]) - (expGray[gd]**2.))
        expGrayErr[gd] = np.sqrt(1./expGrayWt[gd])

