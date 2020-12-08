import numpy as np
import scipy.linalg as linalg
import os
import sys
import esutil
import time
import matplotlib.pyplot as plt

import multiprocessing

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm


class FgcmRetrieval(object):
    """
    Class to compute retrieved R0/R1 integrals from star colors.

    parameters
    ----------
    fgcmConfig: FgcmConfig
    fgcmPars: FgcmParameters
    fgcmStars: FgcmStars
    fgcmLUT: FgcmLUT

    Config variables
    ----------------
    minStarPerCCD: int
       Minimum number of stars on a CCD to try the retrieved fit
    nExpPerRun: int
       Number of exposures per multicore run (more uses more memory)
    nCore: int
       Number of cores to use.
    """

    def __init__(self,fgcmConfig,fgcmPars,fgcmStars,fgcmLUT):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.debug('Initializing FgcmRetrieval')

        self.fgcmPars = fgcmPars

        self.fgcmStars = fgcmStars

        self.I10StdBand = fgcmConfig.I10StdBand

        # and record configuration variables
        self.illegalValue = fgcmConfig.illegalValue
        self.nCore = fgcmConfig.nCore
        self.minStarPerCCD = fgcmConfig.minStarPerCCD
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.nExpPerRun = fgcmConfig.nExpPerRun
        self.outputPath = fgcmConfig.outputPath
        self.quietMode = fgcmConfig.quietMode

        self.arraysPrepared = False
        self._prepareRetrievalArrays()

    def _prepareRetrievalArrays(self):
        """
        Internal method to create shared memory arrays.
        """

        self.r0Handle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.r10Handle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')

        r0 = snmm.getArray(self.r0Handle)
        r0[:] = self.illegalValue
        r10 = snmm.getArray(self.r10Handle)
        r10[:] = self.illegalValue

        self.arraysPrepared = True

    def computeRetrievalIntegrals(self,debug=False):
        """
        Compute retrieval integrals

        parameters
        ----------
        debug: bool, default=False
           Debug mode without multicore
        """

        if (not self.fgcmStars.magStdComputed):
            raise ValueError("Must run fgcmChisq before fgcmRetrieval.")

        startTime = time.time()
        if not self.quietMode:
            self.fgcmLog.info('Computing retrieval integrals')

        # reset arrays
        r0 = snmm.getArray(self.r0Handle)
        r10 = snmm.getArray(self.r10Handle)

        r0[:] = self.illegalValue
        r10[:] = self.illegalValue


        # select good stars
        goodStars = self.fgcmStars.getGoodStarIndices()

        if not self.quietMode:
            self.fgcmLog.info('Found %d good stars for retrieval' % (goodStars.size))

        if (goodStars.size == 0):
            raise ValueError("No good stars to fit!")

        # and pre-matching stars and observations

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        preStartTime=time.time()
        self.fgcmLog.debug('Pre-matching stars and observations...')

        # Compute this for both good and bad exposures.
        goodStarsSub, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, requireSED=True, checkBadMag=True)

        self.goodObsHandle = snmm.createArray(goodObs.size,dtype='i4')
        snmm.getArray(self.goodObsHandle)[:] = goodObs

        self.obsExpIndexGOAHandle = snmm.createArray(goodObs.size,dtype='i4')
        snmm.getArray(self.obsExpIndexGOAHandle)[:] = obsExpIndex[goodObs]

        self.fgcmLog.debug('Pre-matching done in %.1f sec.' %
                           (time.time() - preStartTime))

        # which exposures have stars?
        # Note that this always returns a sorted array
        uExpIndex = np.unique(obsExpIndex[goodObs])

        self.debug=debug
        if (self.debug):
            # debug mode: single core

            #self._worker(self.fgcmPars.expArray[uExpIndex])
            self._worker(uExpIndex)

        else:
            # regular multi-core
            if not self.quietMode:
                self.fgcmLog.info('Running retrieval on %d cores' % (self.nCore))

            # split exposures into a list of arrays of roughly equal size
            nSections = uExpIndex.size // self.nExpPerRun + 1

            uExpIndexList = np.array_split(uExpIndex,nSections)

            # may want to sort by nObservations, but only if we pre-split
            mp_ctx = multiprocessing.get_context("fork")
            pool = mp_ctx.Pool(processes=self.nCore)
            pool.map(self._worker, uExpIndexList, chunksize=1)
            pool.close()
            pool.join()
            #map(self._worker, uExpIndexList)

        # free memory!
        snmm.freeArray(self.goodObsHandle)
        snmm.freeArray(self.obsExpIndexGOAHandle)

        # and we're done
        if not self.quietMode:
            self.fgcmLog.info('Computed retrieved integrals in %.2f seconds.' %
                              (time.time() - startTime))


    def _worker(self,uExpIndex):
        """
        Internal method for multicore worker

        parameters
        ----------
        uExpIndex: int array
           Unique indices of exposures in this run.
        """

        # NOTE: No logging is allowed in the _worker method

        goodObsAll = snmm.getArray(self.goodObsHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsExpIndexGOA = snmm.getArray(self.obsExpIndexGOAHandle)

        if goodObsAll.size == 0 or obsExpIndex.size == 0 or obsExpIndexGOA.size == 0:
            # there is nothing to compute because objects didn't have SED.
            return

        # Start with only those that are in range ... this saves a ton of
        # memory
        inRange, = np.where((obsExpIndexGOA >= uExpIndex.min()) &
                            (obsExpIndexGOA <= uExpIndex.max()))

        if inRange.size == 0:
            # There is nothing to do here.
            return

        _,temp = esutil.numpy_util.match(uExpIndex, obsExpIndexGOA[inRange])
        goodObs = goodObsAll[inRange[temp]]

        obsExpIndexGO = obsExpIndex[goodObs]

        temp = None

        # arrays we need...
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanNoChrom = snmm.getArray(self.fgcmStars.objMagStdMeanNoChromHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)

        obsObjIDIndexGO = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)[goodObs]
        obsBandIndexGO = snmm.getArray(self.fgcmStars.obsBandIndexHandle)[goodObs]
        obsCCDIndexGO = snmm.getArray(self.fgcmStars.obsCCDHandle)[goodObs] - self.ccdStartIndex
        obsMagADUGO = snmm.getArray(self.fgcmStars.obsMagADUHandle)[goodObs]
        obsMagErrGO = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)[goodObs]

        # compute delta mags
        # deltaMag = m_b^inst(i,j)  - <m_b^std>(j) + QE_sys
        #   we want to take out the gray term from the qe

        deltaMagGO = (obsMagADUGO -
                      objMagStdMean[obsObjIDIndexGO,
                                    obsBandIndexGO] +
                      self.fgcmPars.expQESys[obsExpIndexGO] +
                      self.fgcmPars.expFilterOffset[obsExpIndexGO])

        deltaMagErr2GO = (obsMagErrGO**2. +
                          objMagStdMeanErr[obsObjIDIndexGO,
                                           obsBandIndexGO]**2.)

        # and to flux space
        fObsGO = 10.**(-0.4*deltaMagGO)
        fObsErr2GO = deltaMagErr2GO * ((2.5/np.log(10.)) * fObsGO)**2.
        deltaStdGO = (1.0 + objSEDSlope[obsObjIDIndexGO,
                                        obsBandIndexGO] *
                      self.I10StdBand[obsBandIndexGO])

        deltaStdWeightGO = 1./(fObsErr2GO * deltaStdGO * deltaStdGO)

        # and compress obsExpIndexGO
        theseObsExpIndexGO=np.searchsorted(uExpIndex, obsExpIndexGO)

        r0 = snmm.getArray(self.r0Handle)
        r10 = snmm.getArray(self.r10Handle)

        # set up the matrices

        # IMatrix[0,0] = sum (1./sigma_f^2)
        # IMatrix[0,1] = sum (F'_nu/sigma_f^2)
        # IMatrix[1,0] = IMatrix[0,1]
        # IMatrix[1,1] = sum (F'_nu^2/sigma_f^2)

        # RHS[0] = sum (f^obs / (sigma_f^2 * deltaStd))
        # RHS[1] = sum ((F'_nu * f^obs / (sigma_f^2 * deltaStd))


        IMatrix = np.zeros((2,2,uExpIndex.size,self.fgcmPars.nCCD),dtype='f8')
        RHS = np.zeros((2,uExpIndex.size,self.fgcmPars.nCCD),dtype='f8')
        nStar = np.zeros((uExpIndex.size,self.fgcmPars.nCCD),dtype='i4')

        np.add.at(IMatrix,
                  (0,0,theseObsExpIndexGO,obsCCDIndexGO),
                  deltaStdWeightGO)
        np.add.at(IMatrix,
                  (0,1,theseObsExpIndexGO,obsCCDIndexGO),
                  objSEDSlope[obsObjIDIndexGO,
                              obsBandIndexGO] *
                  deltaStdWeightGO)
        np.add.at(IMatrix,
                  (1,0,theseObsExpIndexGO,obsCCDIndexGO),
                  objSEDSlope[obsObjIDIndexGO,
                              obsBandIndexGO] *
                  deltaStdWeightGO)
        np.add.at(IMatrix,
                  (1,1,theseObsExpIndexGO,obsCCDIndexGO),
                  objSEDSlope[obsObjIDIndexGO,
                              obsBandIndexGO]**2. *
                  deltaStdWeightGO)
        np.add.at(nStar,
                  (theseObsExpIndexGO,obsCCDIndexGO),
                  1)

        np.add.at(RHS,
                  (0,theseObsExpIndexGO,obsCCDIndexGO),
                  fObsGO / (fObsErr2GO * deltaStdGO))
        np.add.at(RHS,
                  (1,theseObsExpIndexGO,obsCCDIndexGO),
                  objSEDSlope[obsObjIDIndexGO,
                              obsBandIndexGO] *
                  fObsGO / (fObsErr2GO * deltaStdGO))

        # which can be computed?
        expIndexUse, ccdIndexUse = np.where(nStar >= self.minStarPerCCD)

        # loop, doing the linear algebra
        for i in range(expIndexUse.size):
            mat = IMatrix[:, :, expIndexUse[i], ccdIndexUse[i]]
            det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
            if not np.isfinite(det):
                continue
            inv = (1. / det) * np.array([[mat[1, 1], -mat[0, 1]],
                                         [-mat[1, 0], mat[0, 0]]])
            if np.any(~np.isfinite(inv)):
                continue

            IRetrieved = np.dot(inv, RHS[:, expIndexUse[i], ccdIndexUse[i]])

            # record these in the shared array ... should not step
            #  on each others' toes
            r0[uExpIndex[expIndexUse[i]],ccdIndexUse[i]] = IRetrieved[0]
            r10[uExpIndex[expIndexUse[i]],ccdIndexUse[i]] = IRetrieved[1]/IRetrieved[0]

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state
