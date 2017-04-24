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


class FgcmBrightObs(object):
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

        self.fgcmLog.log('INFO','Found %d good stars for bright obs' % (goodStars.size))

        if (goodStars.size == 0):
            raise ValueError("No good stars to fit!")

        if (self.debug):
            self._worker(goodStars)
        else:
            self.fgcmLog.log('INFO','Running BrightObs on %d cores' % (self.nCore))

            goodStarsList = np.array_split(goodStars,self.nCore)

            # make a pool
            pool = Pool(processes=self.nCore)
            pool.map(self._worker,goodStarsList)
            pool.close()
            pool.join()


        self.fgcmLog.log('INFO','Finished BrightObs (Alt) in %.2f seconds.' %
                         (time.time() - startTime))


    def _worker(self,goodStars):
        """
        """

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsMagADUErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)

        # select the good observations that go into these stars
        _,goodObs = esutil.numpy_util.match(goodStars,obsObjIDIndex,presorted=True)

        # and cut to those exposures that are not flagged
        gd,=np.where(self.fgcmPars.expFlag[obsExpIndex[goodObs]] == 0)
        goodObs = goodObs[gd]

        obsMagErr2GO = obsMagADUErr[goodObs]**2.

        # start by saying that they're all good
        subGood = np.arange(goodObs.size)

        # we can only have the number of iterations of the max number of stars
        maxIter = objNGoodObs[goodStars,:].max()

        ctr = 0

        lastGoodSize = subGood.size+1

        # create temporary variables.  Are they too big in memory?
        nSum = np.zeros_like(objMagStdMean,dtype='i4')
        tempObjMagStdMean = np.zeros_like(objMagStdMean)

        # loop over cutting iteratively
        #  this is essentially a binary search to find the brightest mag
        #  for each object/band
        #  right now it just keeps searching until they're all done, even
        #  if that isn't the most efficient.
        while ((lastGoodSize > subGood.size) and (ctr < maxIter)) :
            # first, save lastGoodSize
            lastGoodSize = subGood.size

            # clear temp vars
            nSum[:,:] = 0
            tempObjMagStdMean[:,:] = 0

            # compute mean mags, with total and number
            np.add.at(tempObjMagStdMean,
                      (obsObjIDIndex[goodObs[subGood]],
                       obsBandIndex[goodObs[subGood]]),
                      obsMagStd[goodObs[subGood]])
            np.add.at(nSum,
                      (obsObjIDIndex[goodObs[subGood]],
                       obsBandIndex[goodObs[subGood]]),
                      1)

            # which have measurements?
            #  (note this might be redundant in the iterations)

            gd=np.where(nSum > 0)

            tempObjMagStdMean[gd] /= nSum[gd]

            # and get new subGood
            #   note that this refers to the original goodObs...not a sub of a sub
            subGood,=np.where(obsMagStd[goodObs] <=
                              tempObjMagStdMean[obsObjIDIndex[goodObs],
                                                obsBandIndex[goodObs]])

            # and increment counter
            ctr+=1


        if (ctr == maxIter):
            # this is a big problem, and shouldn't be possible.
            raise ValueError("Bright observation search failed to converge!")

        # now which observations are bright *enough* to consider?
        brightEnoughGO, = np.where((obsMagStd[goodObs] -
                                    tempObjMagStdMean[obsObjIDIndex[goodObs],
                                                      obsBandIndex[goodObs]]) <=
                                   self.brightObsGrayMax)

        # need to take the weighted mean, so a temp array here
        #  (memory issues?)
        wtSum = np.zeros_like(objMagStdMean,dtype='f8')
        obsMagErr2GOBE = obsMagADUErr[goodObs[brightEnoughGO]]**2.

        np.add.at(wtSum,
                  (obsObjIDIndex[goodObs[brightEnoughGO]],
                   obsBandIndex[goodObs[brightEnoughGO]]),
                  1./obsMagErr2GOBE)

        gd=np.where(wtSum > 0.0)
        # important: only zero accumulator for our stars
        objMagStdMean[gd] = 0.0
        objNGoodObs[gd] = 0

        # note that obsMag is already cut to goodObs
        np.add.at(objMagStdMean,
                  (obsObjIDIndex[goodObs[brightEnoughGO]],
                   obsBandIndex[goodObs[brightEnoughGO]]),
                  obsMagStd[goodObs[brightEnoughGO]]/obsMagErr2GOBE)
        np.add.at(objNGoodObs,
                  (obsObjIDIndex[goodObs[brightEnoughGO]],
                   obsBandIndex[goodObs[brightEnoughGO]]),
                  1)

        objMagStdMean[gd] /= wtSum[gd]
        objMagStdMeanErr[gd] = np.sqrt(1./wtSum[gd])

        # finally, compute SED slopes if desired
        if (self.computeSEDSlopes):
            self.fgcmStars.computeObjectSEDSlopes(goodStars)

        # and we're done
