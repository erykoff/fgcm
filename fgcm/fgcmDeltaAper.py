from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import os
import sys
import esutil
import time

from .fgcmUtilities import _pickle_method

import types
try:
    import copy_reg as copyreg
except ImportError:
    import copyreg

import multiprocessing
from multiprocessing import Pool

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

copyreg.pickle(types.MethodType, _pickle_method)


class FgcmDeltaAper(object):
    """
    Class which computes delta aperture background offsets.

    Parameters
    ----------
    fgcmConfig: FgcmConfig
       Config object
    fgcmPars: FgcmParameters
       Parameter object
    fgcmStars: FgcmStars
       Star object
    """
    def __init__(self, fgcmConfig, fgcmPars, fgcmStars):
        self.fgcmLog = fgcmConfig.fgcmLog

        if not fgcmStars.hasDeltaAper:
            self.fgcmLog.info("Cannot compute delta aperture parameters without measurements.")
            return

        self.fgcmPars = fgcmPars
        self.fgcmStars = fgcmStars

        self.minStarPerExp = fgcmConfig.minStarPerExp
        self.illegalValue = fgcmConfig.illegalValue
        self.quietMode = fgcmConfig.quietMode
        self.nCore = fgcmConfig.nCore
        self.nStarPerRun = fgcmConfig.nStarPerRun

    def computeDeltaAperExposures(self):
        """
        Compute deltaAper per-exposure quantities
        """
        if not self.quietMode:
            self.fgcmLog.info('Computing deltaAper per exposure')

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsDeltaAper = snmm.getArray(self.fgcmStars.obsDeltaAperHandle)

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)

        # Use only good observations of good stars
        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=False, checkMinObs=True)
        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars)

        self.fgcmPars.compMedDeltaAper[:] = self.illegalValue
        self.fgcmPars.compEpsilon[:] = self.illegalValue

        h, rev = esutil.stat.histogram(obsExpIndex[goodObs], rev=True, min=0)
        expIndices, = np.where(h >= self.minStarPerExp)

        for expIndex in expIndices:
            i1a = rev[rev[expIndex]: rev[expIndex + 1]]
            mag = objMagStdMean[obsObjIDIndex[obsIndex[goodObs[i1a]]],
                                obsBandIndex[obsIndex[goodObs[i1a]]]]
            deltaAper = obsDeltaAper[goodObs[i1a]]
            err = obsMagADUModelErr[goodObs[i1a]]

            # First, we take the brightest half and compute the median
            ok, = np.where((mag < 90.0) & (np.abs(deltaAper) < 0.5))
            if ok.size < (self.minStarPerExp // 2):
                continue

            # Use 25% brightest
            st = np.argsort(mag[ok])
            cutMag = mag[ok[st[int(0.25*st.size)]]]
            bright, = np.where(mag[ok] < cutMag)
            self.fgcmPars.compMedDeltaAper[expIndex] = np.median(deltaAper[ok[bright]])

            # Next, we take the full thing and compute epsilon (nJy)
            flux = 10.**((mag[ok] - 48.6)/(-2.5)) * 1e-9
            x = (2.5/np.log(10.)) / flux
            y = deltaAper[ok]
            yerr = err[ok]

            # Will need to check for warnings here...
            fit = np.polyfit(x, y, 1, w=1./yerr)

            self.fgcmPars.compEpsilon[expIndex] = fit[0]

    def computeDeltaAperStars(self, debug=False):
        """
        Compute deltaAper per-star quantities
        """
        self.debug = debug

        startTime=time.time()
        if not self.quietMode:
            self.fgcmLog.info('Compute per-star deltaAper')

        # Reset numbers
        snmm.getArray(self.fgcmStars.objDeltaAperMeanHandle)[:] = 99.0

        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=True)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        goodStarsSub, goodObs = self.fgcmStars.getGoodObsIndices(goodStars)

        if self.debug:
            self._starWorker((goodStars, goodObs))
        else:
            if not self.quietMode:
                self.fgcmLog.info('Running DeltaAper on %d cores' % (self.nCore))

            nSections = goodStars.size // self.nStarPerRun + 1
            goodStarsList = np.array_split(goodStars, nSections)

            splitValues = np.zeros(nSections - 1,dtype='i4')
            for i in xrange(1, nSections):
                splitValues[i - 1] = goodStarsList[i][0]

            splitIndices = np.searchsorted(goodStars[goodStarsSub], splitValues)
            goodObsList = np.split(goodObs, splitIndices)

            workerList = list(zip(goodStarsList,goodObsList))

            # reverse sort so the longest running go first
            workerList.sort(key=lambda elt:elt[1].size, reverse=True)

            pool = Pool(processes=self.nCore)
            pool.map(self._starWorker, workerList, chunksize=1)
            pool.close()
            pool.join()

        if not self.quietMode:
            self.fgcmLog.info('Finished BrightObs in %.2f seconds.' %
                              (time.time() - startTime))

    def _starWorker(self, goodStarsAndObs):
        """
        Multiprocessing worker for FgcmDeltaAper.  Not to be called on its own.

        Parameters
        ----------
        goodStarsAndObs: tuple[2]
           (goodStars, goodObs)
        """
        # NOTE: No logging is allowed in the _magWorker method

        goodStars = goodStarsAndObs[0]
        goodObs = goodStarsAndObs[1]

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objDeltaAperMean = snmm.getArray(self.fgcmStars.objDeltaAperMeanHandle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsDeltaAper = snmm.getArray(self.fgcmStars.obsDeltaAperHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        # Cut to good exposures
        gd, = np.where((self.fgcmPars.expFlag[obsExpIndex[goodObs]] == 0) &
                       (obsFlag[goodObs] == 0) &
                       (np.abs(obsDeltaAper[goodObs]) < 0.5))
        goodObs = goodObs[gd]

        obsMagErr2GO = obsMagADUModelErr[goodObs]**2.

        wtSum = np.zeros_like(objMagStdMean, dtype='f8')
        objDeltaAperMeanTemp = np.zeros_like(objMagStdMean, dtype='f8')

        np.add.at(objDeltaAperMeanTemp,
                  (obsObjIDIndex[goodObs], obsBandIndex[goodObs]),
                  (obsDeltaAper[goodObs] - self.fgcmPars.compMedDeltaAper[obsExpIndex[goodObs]])/obsMagErr2GO)
        np.add.at(wtSum,
                  (obsObjIDIndex[goodObs], obsBandIndex[goodObs]),
                  1./obsMagErr2GO)

        gd = np.where(wtSum > 0.0)

        objDeltaAperMeanLock = snmm.getArrayBase(self.fgcmStars.objDeltaAperMeanHandle).get_lock()
        objDeltaAperMeanLock.acquire()

        objDeltaAperMean[gd] = objDeltaAperMeanTemp[gd] / wtSum[gd]

        objDeltaAperMeanLock.release()

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state
