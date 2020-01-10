from __future__ import division, absolute_import, print_function

import numpy as np
import os
import sys
import esutil
import time
import scipy.optimize

import matplotlib.pyplot as plt

from .fgcmUtilities import objFlagDict

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmFlagVariables(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmStars):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.info('Initializing fgcmFlagVariables')

        # need fgcmPars because it has the sigFgcm
        self.fgcmPars = fgcmPars

        # need fgcmStars because it has the stars (duh)
        self.fgcmStars = fgcmStars

        self.varNSig = fgcmConfig.varNSig
        self.varMinBand = fgcmConfig.varMinBand
        self.bandFitIndex = fgcmConfig.bandFitIndex

    def flagVariables(self):
        """
        """

        if (not self.fgcmStars.magStdComputed):
            raise ValueError("Must run FgcmChisq to compute magStd before computeCCDAndExpGray")

        startTime = time.time()
        self.fgcmLog.info('Flagging variables.')

        # input numbers
        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        #obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=True, checkMinObs=True)
        goodStarsSub, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=self.fgcmPars.expFlag, checkBadMag=True)

        # we need to compute E_gray == <mstd> - mstd for each observation
        # compute EGray, GO for Good Obs
        EGrayGO = (objMagStdMean[obsObjIDIndex[goodObs],obsBandIndex[goodObs]] -
                   obsMagStd[goodObs])
        # and need the error for Egray: sum in quadrature of individual and avg errs
        EGrayErr2GO = (objMagStdMeanErr[obsObjIDIndex[goodObs],obsBandIndex[goodObs]]**2. +
                       obsMagErr[goodObs]**2.)

        # set up variability counter
        varCount = np.zeros(goodStars.size,dtype='i4')

        # loop over fit bands
        for bandIndex in self.bandFitIndex:
            # which observations are considered for var checks?
            varUse, = np.where((EGrayErr2GO > 0.0) &
                             (EGrayGO != 0.0) &
                             (obsBandIndex[goodObs] == bandIndex))

            # which of these show high variability?
            isVar, = np.where(np.abs(EGrayGO[varUse]/
                                     np.sqrt(self.fgcmPars.compSigFgcm[bandIndex]**2. +
                                             EGrayErr2GO[varUse])) >
                              self.varNSig)
            # and add to the varCount.  Note that each object may be listed multiple
            #  times but this only adds 1 to each for each band
            varCount[goodStarsSub[varUse[isVar]]] += 1

        # make sure we have enough bands with variability
        varStars, = np.where(varCount >= self.varMinBand)

        # and flag
        objFlag[goodStars[varStars]] |= objFlagDict['VARIABLE']

        # log this
        self.fgcmLog.info('Found %d variable objects' % (varStars.size))


        self.fgcmLog.info('Done flagging variables in %.2f sec.' %
                         (time.time() - startTime))
