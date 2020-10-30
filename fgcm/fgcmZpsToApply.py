import numpy as np
import esutil
import os

from .fgcmNumbaUtilities import numba_test, add_at_1d, add_at_2d, add_at_3d

import multiprocessing

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm


class FgcmZpsToApply(object):
    """
    Class to hold zeropoints that will be applied in FgcmApplyZeropoints.

    Parameters
    ----------
    fgcmConfig: FgcmConfig
    """

    def __init__(self, fgcmConfig, fgcmPars, fgcmStars, fgcmLUT):
        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.debug('Initializing fgcmZpsToApply.')

        self.fgcmPars = fgcmPars
        self.fgcmStars = fgcmStars
        self.fgcmLUT = fgcmLUT

        self.zpsToApplyFile = fgcmConfig.zpsToApplyFile
        self.maxFlagZpsToApply = fgcmConfig.maxFlagZpsToApply
        self.expField = fgcmConfig.expField
        self.ccdField = fgcmConfig.ccdField
        self.zptABNoThroughput = fgcmConfig.zptABNoThroughput

        self.nCore = fgcmConfig.nCore
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.nStarPerRun = fgcmConfig.nStarPerRun
        self.quietMode = fgcmConfig.quietMode
        self.hasDeltaMagBkg = fgcmStars.hasDeltaMagBkg

        self.I10StdBand = fgcmConfig.I10StdBand

        if fgcmConfig.useSedLUT and self.fgcmLUT.hasSedLUT:
            self.useSedLUT = True
        else:
            self.useSedLUT = False

        self.zeropointsLoaded = False

    def loadZeropointsFromFits(self):
        """
        Load zeropoints from fits file.

        """

        import fitsio

        if not os.path.isfile(self.zpsToApplyFile):
            raise IOError("Could not find zeropoint file %s" % (self.zpsToApplyFile))

        zps = fitsio.read(self.zpsToApplyFile, ext='ZPTS', lower=True)

        if self.hasDeltaMagBkg:
            deltaMagOffset = zps['fgcm_deltabkgmag']
        else:
            deltaMagOffset = 0.0

        self.loadZeropoints(zps[self.expField.lower()],
                            zps[self.ccdField.lower()],
                            zps['fgcm_flag'],
                            zps['fgcm_zpt'] + deltaMagOffset,
                            zps['fgcm_i10'])

        del zps

    def loadZeropoints(self, zpExpnumArray, zpCcdnumArray, zpFlagArray, zpZptArray, zpI10Array):
        """
        Load zeropoints into shared memory structures

        Parameters
        ----------
        zpExpnumArray: `int` array
           Exposure numbers of zeropoints
        zpCcdnumArray: `int` array
           Ccd numbers of zeropoints
        zpFlagArray: `int` array
           Zeropoint flags for photometricity
        zpZptArray: `float` array
           Zeropoint values
        zpI10Array: `float` array
           I10 for each zeropoint
        """

        # Rearrange into 2d arrays for easy indexing

        self.zpZptHandle = snmm.createArray((self.fgcmPars.nExp, self.fgcmPars.nCCD), dtype='f8')
        self.zpFlagHandle = snmm.createArray((self.fgcmPars.nExp, self.fgcmPars.nCCD), dtype='i4')
        self.zpI10Handle = snmm.createArray((self.fgcmPars.nExp, self.fgcmPars.nCCD), dtype='f8')

        zpZpt = snmm.getArray(self.zpZptHandle)
        zpFlag = snmm.getArray(self.zpFlagHandle)
        zpI10 = snmm.getArray(self.zpI10Handle)

        # Initially set all flags to an illegal value
        zpFlag[:, :] = 64

        a, b = esutil.numpy_util.match(self.fgcmPars.expArray, zpExpnumArray)

        zpZpt[a, zpCcdnumArray[b] - self.ccdStartIndex] = zpZptArray[b]
        zpFlag[a, zpCcdnumArray[b] - self.ccdStartIndex] = zpFlagArray[b]
        zpI10[a, zpCcdnumArray[b] - self.ccdStartIndex] = zpI10Array[b]

    def applyZeropoints(self):
        """
        Apply zeropoints to stars
        """

        if not self.quietMode:
            self.fgcmLog.info('Apply zeropoints to stars')

        # Reset mean magnitudes
        snmm.getArray(self.fgcmStars.objMagStdMeanHandle)[:] = 99.0
        snmm.getArray(self.fgcmStars.objMagStdMeanNoChromHandle)[:] = 99.0
        snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)[:] = 99.0

        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=True, checkMinObs=True)

        if not self.quietMode:
            self.fgcmLog.info('Found %d good stars to apply zeropoints' % (goodStars.size))

        if goodStars.size == 0:
            raise RuntimeError("No good stars to apply zeropoints!")

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex

        goodStarsSub, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=self.fgcmPars.expFlag)

        mp_ctx = multiprocessing.get_context('fork')
        proc = mp_ctx.Process()
        workerIndex = proc._identity[0] + 1
        proc = None

        nSections = goodStars.size // self.nStarPerRun + 1
        goodStarsList = np.array_split(goodStars, nSections)

        splitValues = np.zeros(nSections - 1, dtype='i4')
        for i in range(1, nSections):
            splitValues[i - 1] = goodStarsList[i][0]

        splitIndices = np.searchsorted(goodStars[goodStarsSub], splitValues)

        goodObsList = np.split(goodObs, splitIndices)

        workerList = list(zip(goodStarsList, goodObsList))

        workerList.sort(key=lambda elt:elt[1].size, reverse=True)

        pool = mp_ctx.Pool(processes=self.nCore)
        pool.map(self._worker, workerList, chunksize=1)

        pool.close()
        pool.join()

        self.fgcmStars.magStdComputed = True

    def _worker(self, goodStarsAndObs):
        """
        Multiprocessing worker to compute standard/mean magnitudes for FgcmZpsToApply.
        Not to be called on its own.

        Parameters
        ----------
        goodStarsAndObs: tuple[2]
           (goodStars, goodObs)
        """

        goodStars = goodStarsAndObs[0]
        goodObs = goodStarsAndObs[1]

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanNoChrom = snmm.getArray(self.fgcmStars.objMagStdMeanNoChromHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)
        objID = snmm.getArray(self.fgcmStars.objIDHandle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsLUTFilterIndex = snmm.getArray(self.fgcmStars.obsLUTFilterIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)

        zpZpt = snmm.getArray(self.zpZptHandle)
        zpI10 = snmm.getArray(self.zpI10Handle)

        # and the arrays for locking access
        objMagStdMeanLock = snmm.getArrayBase(self.fgcmStars.objMagStdMeanHandle).get_lock()
        obsMagStdLock = snmm.getArrayBase(self.fgcmStars.obsMagStdHandle).get_lock()

        # cut these down now, faster later
        obsObjIDIndexGO = esutil.numpy_util.to_native(obsObjIDIndex[goodObs])
        obsBandIndexGO = esutil.numpy_util.to_native(obsBandIndex[goodObs])
        obsLUTFilterIndexGO = esutil.numpy_util.to_native(obsLUTFilterIndex[goodObs])
        obsExpIndexGO = esutil.numpy_util.to_native(obsExpIndex[goodObs])
        obsCCDIndexGO = esutil.numpy_util.to_native(obsCCDIndex[goodObs])
        I10GO = esutil.numpy_util.to_native(zpI10[obsExpIndexGO, obsCCDIndexGO])

        obsMagErr2GO = obsMagADUModelErr[goodObs]**2.

        obsMagGO = (obsMagADU[goodObs] + zpZpt[obsExpIndexGO, obsCCDIndexGO] +
                    -2.5 * np.log10(self.fgcmPars.expExptime[obsExpIndexGO]) -
                    self.zptABNoThroughput)

        # Compute the mean

        wtSum = np.zeros_like(objMagStdMean, dtype='f8')
        objMagStdMeanTemp = np.zeros_like(objMagStdMean)

        add_at_2d(wtSum,
                  (obsObjIDIndexGO, obsBandIndexGO),
                  1./obsMagErr2GO)
        add_at_2d(objMagStdMeanTemp,
                  (obsObjIDIndexGO, obsBandIndexGO),
                  obsMagGO / obsMagErr2GO)

        gd = np.where(wtSum > 0.0)

        objMagStdMeanLock.acquire()
        objMagStdMean[gd] = objMagStdMeanTemp[gd] / wtSum[gd]
        objMagStdMeanErr[gd] = np.sqrt(1. / wtSum[gd])
        objMagStdMeanLock.release()

        # Compute the SEDs

        if self.useSedLUT:
            self.fgcmStars.computeObjectSEDSlopesLUT(goodStars, self.fgcmLUT)
        else:
            self.fgcmStars.computeObjectSEDSlopes(goodStars)

        # Compute the chromatic correction

        deltaStdGO = 2.5 * np.log10((1.0 +
                                     objSEDSlope[obsObjIDIndexGO,
                                                 obsBandIndexGO] * I10GO) /
                                    (1.0 + objSEDSlope[obsObjIDIndexGO,
                                                       obsBandIndexGO] *
                                     self.I10StdBand[obsBandIndexGO]))

        obsMagStdLock.acquire()
        obsMagStd[goodObs] = obsMagGO + deltaStdGO
        obsMagStdGO = obsMagStd[goodObs]
        obsMagStdLock.release()

        # Compute the mean (again)

        wtSum = np.zeros_like(objMagStdMean, dtype='f8')
        objMagStdMeanTemp = np.zeros_like(objMagStdMean)
        objMagStdMeanNoChromTemp = np.zeros_like(objMagStdMean)

        add_at_2d(wtSum,
                  (obsObjIDIndexGO, obsBandIndexGO),
                  1./obsMagErr2GO)
        add_at_2d(objMagStdMeanTemp,
                  (obsObjIDIndexGO, obsBandIndexGO),
                  obsMagStdGO / obsMagErr2GO)
        add_at_2d(objMagStdMeanNoChromTemp,
                  (obsObjIDIndexGO, obsBandIndexGO),
                  obsMagGO / obsMagErr2GO)

        gd = np.where(wtSum > 0.0)

        # Record the mean

        objMagStdMeanLock.acquire()
        objMagStdMean[gd] = objMagStdMeanTemp[gd] / wtSum[gd]
        objMagStdMeanNoChrom[gd] = objMagStdMeanNoChromTemp[gd] / wtSum[gd]
        objMagStdMeanErr[gd] = np.sqrt(1. / wtSum[gd])
        objMagStdMeanLock.release()

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state
