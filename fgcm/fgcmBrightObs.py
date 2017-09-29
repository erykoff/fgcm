from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import time

from fgcmUtilities import _pickle_method
from fgcmChisq import FgcmChisq
from fgcmUtilities import objFlagDict


import types
import copy_reg

import multiprocessing
from multiprocessing import Pool


from sharedNumpyMemManager import SharedNumpyMemManager as snmm

copy_reg.pickle(types.MethodType, _pickle_method)


class FgcmBrightObs(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmStars,fgcmLUT):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.info('Initializing FgcmBrightObs')

        # need fgcmPars because it tracks good exposures
        self.fgcmPars = fgcmPars
        # need fgcmStars because it has the stars (duh)
        self.fgcmStars = fgcmStars
        # and fgcmLUT for the SEDs (this makes me unhappy)
        self.fgcmLUT = fgcmLUT


        self.brightObsGrayMax = fgcmConfig.brightObsGrayMax
        self.nCore = fgcmConfig.nCore
        self.nStarPerRun = fgcmConfig.nStarPerRun

        if (fgcmConfig.useSedLUT and self.fgcmLUT.hasSedLUT):
            self.useSedLUT = True
        else:
            self.useSedLUT = False

    def brightestObsMeanMag(self,debug=False,computeSEDSlopes=False):
        """
        """

        if (not self.fgcmStars.magStdComputed):
            raise ValueError("Must run FgcmChisq to compute magStd before FgcmBrightObs")

        startTime=time.time()
        self.fgcmLog.info('Selecting good stars from Bright Observations')

        self.debug = debug
        self.computeSEDSlopes = computeSEDSlopes


        # reset numbers
        snmm.getArray(self.fgcmStars.objMagStdMeanHandle)[:] = 99.0
        snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)[:] = 99.0

        # and select good stars!  This might be all stars at this point, but good to check
        #goodStars,=np.where(snmm.getArray(self.fgcmStars.objFlagHandle) == 0)
        # we want to include reserved stars for this, so we have values
        resMask = 255 & ~objFlagDict['RESERVED']
        goodStars,=np.where((snmm.getArray(self.fgcmStars.objFlagHandle) & resMask) == 0)

        self.fgcmLog.info('Found %d good stars for bright obs' % (goodStars.size))

        if (goodStars.size == 0):
            raise ValueError("No good stars to fit!")

        # do global pre-matching before giving to workers, because
        #  it is faster this way

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        preStartTime=time.time()
        self.fgcmLog.info('Pre-matching stars and observations...')
        goodStarsSub,goodObs = esutil.numpy_util.match(goodStars,
                                                       obsObjIDIndex,
                                                       presorted=True)

        if (goodStarsSub[0] != 0.0):
            raise ValueError("Very strange that the goodStarsSub first element is non-zero.")

        self.fgcmLog.info('Pre-matching done in %.1f sec.' %
                         (time.time() - preStartTime))

        if (self.debug):
            self._worker((goodStars,goodObs))
        else:
            self.fgcmLog.info('Running BrightObs on %d cores' % (self.nCore))

            # split goodStars into a list of arrays of roughly equal size

            prepStartTime = time.time()
            nSections = goodStars.size // self.nStarPerRun + 1
            goodStarsList = np.array_split(goodStars,nSections)

            # is there a better way of getting all the first elements from the list?
            #  note that we need to skip the first which should be zero (checked above)
            #  see also fgcmChisq.py
            # splitValues is the first of the goodStars in each list
            splitValues = np.zeros(nSections-1,dtype='i4')
            for i in xrange(1,nSections):
                splitValues[i-1] = goodStarsList[i][0]

            # get the indices from the goodStarsSub matched list
            splitIndices = np.searchsorted(goodStars[goodStarsSub], splitValues)

            # and split along these indices
            goodObsList = np.split(goodObs,splitIndices)

            workerList = zip(goodStarsList,goodObsList)

            # reverse sort so the longest running go first
            workerList.sort(key=lambda elt:elt[1].size, reverse=True)

            self.fgcmLog.info('Using %d sections (%.1f seconds)' %
                             (nSections,time.time() - prepStartTime))

            # make a pool
            pool = Pool(processes=self.nCore)
            pool.map(self._worker,workerList,chunksize=1)
            pool.close()
            pool.join()


        self.fgcmLog.info('Finished BrightObs in %.2f seconds.' %
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

        # and cut to those exposures that are not flagged
        if (self.debug):
            startTime = time.time()
            self.fgcmLog.debug('Cutting to good exposures')

        gd,=np.where((self.fgcmPars.expFlag[obsExpIndex[goodObs]] == 0) &
                     (obsFlag[goodObs] == 0))
        goodObs = goodObs[gd]

        if (self.debug):
            startTime=time.time()
            self.fgcmLog.debug('Cutting to sub-indices.')

        obsObjIDIndexGO = obsObjIDIndex[goodObs]
        obsBandIndexGO = obsBandIndex[goodObs]
        obsMagStdGO = obsMagStd[goodObs]

        if (self.debug):
            self.fgcmLog.debug('Cut to sub-indices in %.1f seconds.' %
                             (time.time() - startTime))

        obsMagErr2GO = obsMagADUErr[goodObs]**2.

        # new version using fmin.at()

        # start with the mean temp var, set to 99s.
        objMagStdMeanTemp = np.zeros_like(objMagStdMean)
        objMagStdMeanTemp[:,:] = 99.0

        # find the brightest (minmag) object at each index
        np.fmin.at(objMagStdMeanTemp,
                   (obsObjIDIndexGO, obsBandIndexGO),
                   obsMagStdGO)

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
            if (self.useSedLUT):
                self.fgcmStars.computeObjectSEDSlopesLUT(goodStars, self.fgcmLUT)
            else:
                self.fgcmStars.computeObjectSEDSlopes(goodStars)

        # and we're done
