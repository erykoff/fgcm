from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import time

from fgcmUtilities import _pickle_method
from fgcmChisq import FgcmChisq

import types
import copy_reg

import multiprocessing
from multiprocessing import Pool


from sharedNumpyMemManager import SharedNumpyMemManager as snmm

copy_reg.pickle(types.MethodType, _pickle_method)


class FgcmBrightObsOld(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmStars):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.log('INFO','Initializing FgcmBrightObs')

        # need fgcmPars because it tracks good exposures
        self.fgcmPars = fgcmPars
        # need fgcmStars because it has the stars (duh)
        self.fgcmStars = fgcmStars


        self.brightObsGrayMax = fgcmConfig.brightObsGrayMax
        self.nCore = fgcmConfig.nCore

    def brightestObsMeanMag(self,debug=False,computeSEDSlopes=False):
        """
        """

        if (not self.fgcmStars.magStdComputed):
            raise ValueError("Must run FgcmChisq to compute magStd before FgcmBrightObs")

        startTime=time.time()
        self.fgcmLog.log('INFO','Selecting good stars from Bright Observations')

        self.debug = debug
        self.computeSEDSlopes = computeSEDSlopes


        # reset numbers
        snmm.getArray(self.fgcmStars.objMagStdMeanHandle)[:] = 99.0
        snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)[:] = 99.0

        # and select good stars!  This might be all stars at this point, but good to check
        goodStars,=np.where(snmm.getArray(self.fgcmStars.objFlagHandle) == 0)

        if (self.debug) :
            for goodStar in goodStars:
                self._worker(goodStar)
        else:
            self.fgcmLog.log('INFO','Running BrightObs on %d cores' % (self.nCore))
            pool = Pool(processes=self.nCore)
            pool.map(self._worker,goodStars)
            pool.close()
            pool.join()

        self.fgcmLog.log('INFO','Finished BrightObs in %.2f seconds.' %
                         (time.time() - startTime))


    def _worker(self,objIndex):
        """
        """

        # make local pointers to useful arrays
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        objNobs = snmm.getArray(self.fgcmStars.objNobsHandle)

        thisObsIndex = obsIndex[objObsIndex[objIndex]:objObsIndex[objIndex]+objNobs[objIndex]]
        thisObsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)[thisObsIndex]

        # cut to good exposures
        ## MAYBE: Check if this can be done more efficiently.
        gd,=np.where(self.fgcmPars.expFlag[thisObsExpIndex] == 0)

        thisObsIndex=thisObsIndex[gd]
        thisObsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)[thisObsIndex]

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)

        # split out the filters (instead of loop of wheres)...
        h,rev=esutil.stat.histogram(thisObsBandIndex,rev=True,
                                    min=0,max=self.fgcmPars.nBands-1)

        for j in xrange(self.fgcmPars.nBands):
            if (h[j] == 0):
                objNGoodObs[objIndex,j] = 0
                continue

            i1a=rev[rev[j]:rev[j+1]]

            # find the brightest observation
            minMag = np.amin(obsMagStd[thisObsIndex[i1a]])

            #if ((objIndex == 18781) and (j == 2)):
            #    asdljflkjlk

            # and all the observations that are comparable
            brightObs,=np.where((obsMagStd[thisObsIndex[i1a]] - minMag) <= self.brightObsGrayMax)
            # number of good observations are these bright ones
            objNGoodObs[objIndex,j] = brightObs.size

            # and compute straight, unweighted mean of bright Obs  -- no
            #objMagStdMean[objIndex,j] = np.sum(obsMagStd[thisObsIndex[i1a[brightObs]]]) / brightObs.size
            # compute weighted mean of bright observations, and also compute error
            wtSum = np.sum(1./obsMagErr[thisObsIndex[i1a[brightObs]]]**2.)
            objMagStdMean[objIndex,j] = (np.sum(obsMagStd[thisObsIndex[i1a[brightObs]]]/
                                               obsMagErr[thisObsIndex[i1a[brightObs]]]**2.) /
                                         wtSum)
            objMagStdMeanErr[objIndex,j] = np.sqrt(1./wtSum)

        if (self.computeSEDSlopes):
            self.fgcmStars.computeObjectSEDSlope(objIndex)

        # and we're done
