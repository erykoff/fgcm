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
        self.nStarPerRun = fgcmConfig.nStarPerRun

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

        # do global pre-matching before giving to workers, because
        #  it is faster this way

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        preStartTime=time.time()
        self.fgcmLog.log('INFO','Pre-matching stars and observations...')
        goodStarsSub,goodObs = esutil.numpy_util.match(goodStars,
                                                       obsObjIDIndex,
                                                       presorted=True)

        if (goodStarsSub[0] != 0.0):
            raise ValueError("Very strange that the goodStarsSub first element is non-zero.")

        self.fgcmLog.log('INFO','Pre-matching done in %.1f sec.' %
                         (time.time() - preStartTime))

        if (self.debug):
            self._worker((goodStars,goodObs))
        else:
            self.fgcmLog.log('INFO','Running BrightObs on %d cores' % (self.nCore))

            # split goodStars into a list of arrays of roughly equal size

            prepStartTime = time.time()
            nSections = goodStars.size // self.nStarPerRun + 1
            goodStarsList = np.array_split(goodStars,nSections)

            # is there a better way of getting all the first elements from the list?
            #  note that we need to skip the first which should be zero (checked above)
            #  see also fgcmChisq.py
            splitValues = np.zeros(nSections-1,dtype='i4')
            for i in xrange(1,nSections):
                splitValues[i-1] = goodStarsList[i][0]

            # get the indices from the goodStarsSub matched list
            splitIndices = np.searchsorted(goodStarsSub, splitValues)

            # and split along these indices
            goodObsList = np.split(goodObs,splitIndices)

            workerList = zip(goodStarsList,goodObsList)

            # reverse sort so the longest running go first
            workerList.sort(key=lambda elt:elt[1].size, reverse=True)

            self.fgcmLog.log('INFO','Using %d sections (%.1f seconds)' %
                             (nSections,time.time() - prepStartTime))

            # make a pool
            pool = Pool(processes=self.nCore)
            pool.map(self._worker,workerList,chunksize=1)
            pool.close()
            pool.join()


        self.fgcmLog.log('INFO','Finished BrightObs in %.2f seconds.' %
                         (time.time() - startTime))


    def _worker(self,goodStarsAndObs):
        """
        """

        goodStars = goodStarsAndObs[0]
        goodObs = goodStarsAndObs[1]

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsMagADUErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        # and the arrays for locking access
        objMagStdMeanLock = snmm.getArrayBase(self.fgcmStars.objMagStdMeanHandle).get_lock()

        # select the good observations that go into these stars
        #if (self.debug) :
        #    startTime = time.time()
        #    self.fgcmLog.log('DEBUG','Matching goodStars and obsObjIDIndex')

        #_,goodObs = esutil.numpy_util.match(goodStars,obsObjIDIndex,presorted=True)

        #if (self.debug):
        #    self.fgcmLog.log('DEBUG','Matching done in %.1f seconds.' %
        #                     (time.time() - startTime))

        # and cut to those exposures that are not flagged
        if (self.debug):
            startTime = time.time()
            self.fgcmLog.log('DEBUG','Cutting to good exposures')

        gd,=np.where((self.fgcmPars.expFlag[obsExpIndex[goodObs]] == 0) &
                     (obsFlag[goodObs] == 0))
        goodObs = goodObs[gd]

        if (self.debug):
            startTime=time.time()
            self.fgcmLog.log('DEBUG','Cutting to sub-indices.')

        obsObjIDIndexGO = obsObjIDIndex[goodObs]
        obsBandIndexGO = obsBandIndex[goodObs]
        obsMagStdGO = obsMagStd[goodObs]

        if (self.debug):
            self.fgcmLog.log('DEBUG','Cut to sub-indices in %.1f seconds.' %
                             (time.time() - startTime))

        obsMagErr2GO = obsMagADUErr[goodObs]**2.

        # start by saying that they're all good
        subGood = np.arange(goodObs.size)

        # we can only have the number of iterations of the max number of stars
        maxIter = objNGoodObs[goodStars,:].max()

        ctr = 0

        lastGoodSize = subGood.size+1

        # create temporary variables.  Are they too big in memory?
        nSum = np.zeros_like(objMagStdMean,dtype='i4')
        objMagStdMeanTemp = np.zeros_like(objMagStdMean)

        # loop over cutting iteratively
        #  this is essentially a binary search to find the brightest mag
        #  for each object/band
        #  right now it just keeps searching until they're all done, even
        #  if that isn't the most efficient.
        while ((lastGoodSize > subGood.size) and (ctr < maxIter)) :
            if (self.debug):
                startTime = time.time()

            # first, save lastGoodSize
            lastGoodSize = subGood.size

            # clear temp vars
            nSum[:,:] = 0
            objMagStdMeanTemp[:,:] = 0

            # compute mean mags, with total and number
            np.add.at(objMagStdMeanTemp,
                      (obsObjIDIndexGO[subGood],
                       obsBandIndexGO[subGood]),
                      obsMagStdGO[subGood])
            np.add.at(nSum,
                      (obsObjIDIndexGO[subGood],
                       obsBandIndexGO[subGood]),
                      1)

            # which have measurements?
            #  (note this might be redundant in the iterations)

            gd=np.where(nSum > 0)

            objMagStdMeanTemp[gd] /= nSum[gd]

            # and get new subGood
            #   note that this refers to the original goodObs...not a sub of a sub
            subGood,=np.where(obsMagStdGO <=
                              objMagStdMeanTemp[obsObjIDIndexGO,
                                                obsBandIndexGO])

            if (self.debug):
                self.fgcmLog.log('DEBUG','Iteration %d done in %.1f' %
                                 (ctr, time.time() - startTime))

            # and increment counter
            ctr+=1


        if (ctr == maxIter):
            # this is a big problem, and shouldn't be possible.
            raise ValueError("Bright observation search failed to converge!")

        # now which observations are bright *enough* to consider?
        brightEnoughGO, = np.where((obsMagStdGO -
                                    objMagStdMeanTemp[obsObjIDIndexGO,
                                                      obsBandIndexGO]) <=
                                   self.brightObsGrayMax)

        # need to take the weighted mean, so a temp array here
        #  (memory issues?)
        wtSum = np.zeros_like(objMagStdMean,dtype='f8')
        objNGoodObsTemp = np.zeros_like(objNGoodObs)
        objMagStdMeanTemp[:,:] = 0

        obsMagErr2GOBE = obsMagErr2GO[brightEnoughGO]

        np.add.at(wtSum,
                  (obsObjIDIndexGO[brightEnoughGO],
                   obsBandIndexGO[brightEnoughGO]),
                  1./obsMagErr2GOBE)
        np.add.at(objMagStdMeanTemp,
                  (obsObjIDIndexGO[brightEnoughGO],
                   obsBandIndexGO[brightEnoughGO]),
                  obsMagStdGO[brightEnoughGO]/obsMagErr2GOBE)
        np.add.at(objNGoodObsTemp,
                  (obsObjIDIndexGO[brightEnoughGO],
                   obsBandIndexGO[brightEnoughGO]),
                  1)

        # these are good object/bands that were observed
        gd=np.where(wtSum > 0.0)

        # acquire lock to save values
        objMagStdMeanLock.acquire()

        objMagStdMean[gd] = objMagStdMeanTemp[gd] / wtSum[gd]
        objMagStdMeanErr[gd] = np.sqrt(1./wtSum[gd])
        objNGoodObs[gd] = objNGoodObsTemp[gd]

        # and release
        objMagStdMeanLock.release()

        # finally, compute SED slopes if desired
        if (self.computeSEDSlopes):
            self.fgcmStars.computeObjectSEDSlopes(goodStars)

        # and we're done
