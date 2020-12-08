import numpy as np
import os
import sys
import esutil
import time

from .fgcmChisq import FgcmChisq

import multiprocessing

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm


class FgcmBrightObs(object):
    """
    Class to compute approximate photometric magnitudes using a
      brightest-observation algorithm

    parameters
    ----------
    fgcmConfig: FgcmConfig
       Config object
    fgcmPars: FgcmParameters
       Parameter object
    fgcmStars: FgcmStars
       Star object
    fgcmLUT: FgcmLUT
       LUT object

    Config variables
    ----------------
    brightObsGrayMax: float
       Maximum gray compared to mean to consider averaging
    nCore: int
       Number of cores to run on (via multiprocessing)
    nStarPerRun: int
       Number of stars per run (too many uses more memory)
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmStars,fgcmLUT):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.debug('Initializing FgcmBrightObs')

        # need fgcmPars because it tracks good exposures
        self.fgcmPars = fgcmPars
        # need fgcmStars because it has the stars (duh)
        self.fgcmStars = fgcmStars
        # and fgcmLUT for the SEDs (this makes me unhappy)
        self.fgcmLUT = fgcmLUT

        self.brightObsGrayMax = fgcmConfig.brightObsGrayMax
        self.nCore = fgcmConfig.nCore
        self.nStarPerRun = fgcmConfig.nStarPerRun
        self.bandFitIndex = fgcmConfig.bandFitIndex
        self.quietMode = fgcmConfig.quietMode

        if (fgcmConfig.useSedLUT and self.fgcmLUT.hasSedLUT):
            self.useSedLUT = True
        else:
            self.useSedLUT = False

    def brightestObsMeanMag(self,debug=False,computeSEDSlopes=False):
        """
        Compute the mean magnitude of every object/band from the brightest observations.

        Parameters
        ----------
        debug: bool, default=False
           Debug mode, no multiprocessing
        computeSEDSlopes: bool, default=False
           Compute first-order SED slopes from mean magnitudes
        """

        if (not self.fgcmStars.magStdComputed):
            raise ValueError("Must run FgcmChisq to compute magStd before FgcmBrightObs")

        startTime=time.time()
        if not self.quietMode:
            self.fgcmLog.info('Selecting good stars from Bright Observations')

        self.debug = debug
        self.computeSEDSlopes = computeSEDSlopes


        # reset numbers
        snmm.getArray(self.fgcmStars.objMagStdMeanHandle)[:] = 99.0
        snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)[:] = 99.0

        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=True)

        if (goodStars.size == 0):
            raise ValueError("No good stars to fit!")

        # do global pre-matching before giving to workers, because
        #  it is faster this way

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        preStartTime=time.time()
        self.fgcmLog.debug('Pre-matching stars and observations...')

        goodStarsSub, goodObs = self.fgcmStars.getGoodObsIndices(goodStars)

        self.fgcmLog.debug('Pre-matching done in %.1f sec.' %
                           (time.time() - preStartTime))

        if (self.debug):
            self._worker((goodStars,goodObs))
        else:
            if not self.quietMode:
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
            for i in range(1,nSections):
                splitValues[i-1] = goodStarsList[i][0]

            # get the indices from the goodStarsSub matched list
            splitIndices = np.searchsorted(goodStars[goodStarsSub], splitValues)

            # and split along these indices
            goodObsList = np.split(goodObs,splitIndices)

            workerList = list(zip(goodStarsList,goodObsList))

            # reverse sort so the longest running go first
            workerList.sort(key=lambda elt:elt[1].size, reverse=True)

            self.fgcmLog.debug('Using %d sections (%.1f seconds)' %
                               (nSections,time.time() - prepStartTime))

            # make a pool
            mp_ctx = multiprocessing.get_context("fork")
            pool = mp_ctx.Pool(processes=self.nCore)
            pool.map(self._worker,workerList,chunksize=1)
            pool.close()
            pool.join()


        if not self.quietMode:
            self.fgcmLog.info('Finished BrightObs in %.2f seconds.' %
                              (time.time() - startTime))


    def _worker(self,goodStarsAndObs):
        """
        Multiprocessing worker for FgcmBrightObs.  Not to be called on its own.

        parameters
        ----------
        goodStarsAndObs: tuple[2]
           (goodStars, goodObs)
        """

        # NOTE: No logging is allowed in the _worker method

        goodStars = goodStarsAndObs[0]
        goodObs = goodStarsAndObs[1]

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        # obsMagADUErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        # Note that this will be the same when we don't fit for the error
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
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

        obsMagErr2GO = obsMagADUModelErr[goodObs]**2.

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

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state
