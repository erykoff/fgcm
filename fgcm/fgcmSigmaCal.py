from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import os
import sys
import esutil
import time

import fitsio

from .fgcmUtilities import _pickle_method
from .fgcmUtilities import objFlagDict
from .fgcmUtilities import retrievalFlagDict

import types
try:
    import copy_reg as copyreg
except ImportError:
    import copyreg

import multiprocessing
from multiprocessing import Pool

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

copyreg.pickle(types.MethodType, _pickle_method)

class FgcmSigmaCal(object):
    """
    Class which calibrates the calibration error floor.

    parameters
    ----------
    fgcmConfig: FgcmConfig
       Config object
    fgcmPars: FgcmParameters
       Parameter object
    fgcmStars: FgcmStars
       Stars object
    """

    def __init__(self, fgcmConfig, fgcmPars, fgcmStars, fgcmGray):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.info('Initializing FgcmSigmaCal')

        self.fgcmPars = fgcmPars
        self.fgcmStars = fgcmStars
        self.fgcmGray = fgcmGray

        self.nCore = fgcmConfig.nCore
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.nStarPerRun = fgcmConfig.nStarPerRun
        self.sigma0Phot = fgcmConfig.sigma0Phot
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle

        # these are the standard *band* I10s
        self.I10StdBand = fgcmConfig.I10StdBand

        self.illegalValue = fgcmConfig.illegalValue

        self._prepareSigmaCalArrays()

    def _prepareSigmaCalArrays(self):
        """
        """

        self.objChi2Handle = snmm.createArray((self.fgcmStars.nStars, self.fgcmPars.nBands), dtype='f8')

    def run(self, applyGray=True):
        """

        """

        self.applyGray = applyGray

        # Select only reserve stars for the good stars...

        goodStars = self.fgcmStars.getGoodStarIndices(onlyReserve=True)

        self.fgcmLog.info('Found %d good reserve stars for SigmaCal' % (goodStars.size))

        if goodStars.size == 0:
            raise ValueError("No good reserve stars to fit!")

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        preStartTime=time.time()
        self.fgcmLog.info('Pre-matching stars and observations...')

        expFlag = self.fgcmPars.expFlag
        goodStarsSub, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=expFlag)

        self.fgcmLog.info('Pre-matching done in %.1f sec.' %
                          (time.time() - preStartTime))

        nSections = goodStars.size // self.nStarPerRun + 1
        goodStarsList = np.array_split(goodStars, nSections)

        splitValues = np.zeros(nSections-1,dtype='i4')
        for i in xrange(1, nSections):
            splitValues[i-1] = goodStarsList[i][0]

        # get the indices from the goodStarsSub matched list (matched to goodStars)
        splitIndices = np.searchsorted(goodStars[goodStarsSub], splitValues)

        # and split along the indices
        goodObsList = np.split(goodObs, splitIndices)

        workerList = list(zip(goodStarsList, goodObsList))

        # reverse sort so the longest running go first
        workerList.sort(key=lambda elt:elt[1].size, reverse=True)

        self.fgcmLog.info('Running SigmaCal on %d cores' % (self.nCore))

        # Debugging now just to get this to run...
        sigmaCals = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]

        objChi2 = snmm.getArray(self.objChi2Handle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)

        for i, s in enumerate(sigmaCals):
            self.sigmaCal = s

            startTime = time.time()

            pool = Pool(processes=self.nCore)
            pool.map(self._worker, workerList, chunksize=1)
            pool.close()
            pool.join()
            #map(self._worker, workerList)
            #for argh in workerList:
            #    self._worker(argh)

            self.fgcmLog.info('Computed object chisq for sigmacal = %.4f in %.2f seconds.' % (s, time.time() - startTime))

            # And save the values, make plots outside ...
            cat = np.zeros(goodStars.size, dtype=[('chi2', 'f8', self.fgcmPars.nBands),
                                                  ('mag', 'f4', self.fgcmPars.nBands),
                                                  ('ngood', 'i4', self.fgcmPars.nBands)])
            cat['mag'][:, :] = objMagStdMean[goodStars, :]
            cat['chi2'][:, :] = objChi2[goodStars, :]
            cat['ngood'][:, :] = objNGoodObs[goodStars, :]

            fitsio.write('%s_chi2_%d.fits' % (self.outfileBaseWithCycle, i), cat, clobber=True)

    def _worker(self, goodStarsAndObs):
        """
        """

        workerStartTime = time.time()

        goodStars = goodStarsAndObs[0]
        goodObs = goodStarsAndObs[1]

        # We need to make sure we don't overwrite anything we care about!!!!
        # This will be a challenge to keep the memory okay...

        # We already have the mean...
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objChi2 = snmm.getArray(self.objChi2Handle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsLUTFilterIndex = snmm.getArray(self.fgcmStars.obsLUTFilterIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)
        obsSecZenith = snmm.getArray(self.fgcmStars.obsSecZenithHandle)
        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        # We already have obsMagStd
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)

        # We apply the gray corrections here!
        # Note that obsMagStd does not have the gray corrections applied
        ccdGray = snmm.getArray(self.fgcmGray.ccdGrayHandle)
        ccdGrayErr = snmm.getArray(self.fgcmGray.ccdGrayErrHandle)
        ccdNGoodTilings = snmm.getArray(self.fgcmGray.ccdNGoodTilingsHandle)

        # Cut down goodObs to valid values
        gd, = np.where((ccdGray[obsExpIndex[goodObs], obsCCDIndex[goodObs]] > self.illegalValue) &
                       (ccdNGoodTilings[obsExpIndex[goodObs], obsCCDIndex[goodObs]] >= 2.0) &
                       (objNGoodObs[obsObjIDIndex[goodObs], obsBandIndex[goodObs]] >= 2) &
                       (objMagStdMean[obsObjIDIndex[goodObs], obsBandIndex[goodObs]] < 99.0))

        goodObs = goodObs[gd]

        # cut these down now, faster later
        obsObjIDIndexGO = obsObjIDIndex[goodObs]
        obsBandIndexGO = obsBandIndex[goodObs]
        obsLUTFilterIndexGO = obsLUTFilterIndex[goodObs]
        obsExpIndexGO = obsExpIndex[goodObs]
        obsSecZenithGO = obsSecZenith[goodObs]
        obsCCDIndexGO = obsCCDIndex[goodObs]

        # We make a sub-copy here that we can overwrite
        obsMagStdGO = obsMagStd[goodObs]

        if self.applyGray:
            obsMagStdGO += ccdGray[obsExpIndexGO, obsCCDIndexGO]

        # chi2 = 1. / (N - 1) * Sum ((m_i - mbar)**2. / (sigma_i**2.))
        # N is the number of good observations of the star objNGoodObs[obsObjIDIndexGO, obsBandIndexGO]
        # m_i is obsMagStdGO (after adjustment by ccdGray)
        # mbar is objMagStdMean[obsObjIDIndexGO, obsBandIndexGO]
        # sigma_i**2 = sigma_obs**2. + sig2fgcm / (ntile - 1) + zptvar + sigma_cal**2.
        # It needs to be computed, it is based one
        # - obsMagADUModelErr (this is sigma_obs with an additional sigma0Phot)
        # - self.sigma0Phot
        # - sig2Fgcm (self.fgcmPars.compSigFgcm[self.fgcmPars.expBandIndex[obsExpIndexGO]])
        # - Ntile (ccdNGoodTilings[obsExpIndexGO, obsCCDIndexGO])
        # - zptvar (ccdGrayErr[obsExpIndexGO, obsCCDIndexGO]**2.)
        # - sigma_cal (self.sigmaCal)

        # And recompute the errors...
        nTilingsM1 = np.clip(ccdNGoodTilings[obsExpIndexGO, obsCCDIndexGO] - 1.0, 1.0, None)

        obsMagErr2GO = ((obsMagADUModelErr[goodObs]**2. - self.sigma0Phot**2.) +
                        (self.fgcmPars.compSigFgcm[self.fgcmPars.expBandIndex[obsExpIndexGO]]**2. / nTilingsM1) +
                        (ccdGrayErr[obsExpIndexGO, obsCCDIndexGO]**2.) +
                        (self.sigmaCal**2.))

        # Now we need the per-object chi2...

        objChi2[:, :] = 0.0

        np.add.at(objChi2,
                  (obsObjIDIndexGO, obsBandIndexGO),
                  ((obsMagStdGO - objMagStdMean[obsObjIDIndexGO, obsBandIndexGO])**2. /
                   obsMagErr2GO))
        # There are duplicate indices here, but that's fine because we only want to divide once
        objChi2[obsObjIDIndexGO, obsBandIndexGO] /= (objNGoodObs[obsObjIDIndexGO, obsBandIndexGO] - 1.0)

        # And we're done

        """
        # Need to save this individual stuff just to see what's going on...
        cat = np.zeros(goodObs.size, dtype=[('obsObjIDIndexGO', 'i4'),
                                            ('obsBandIndexGO', 'i4'),
                                            ('obsMagStdGO', 'f4'),
                                            ('objMagStdMeanGO', 'f4'),
                                            ('obsMagErr2GO', 'f4'),
                                            ('obsMagADUModelErrGO', 'f4'),
                                            ('sigFgcm', 'f4'),
                                            ('ccdNGoodTilingsGO', 'f4'),
                                            ('ccdGrayErrGO', 'f4'),
                                            ('objNGoodObsGO', 'i2')])
        cat['obsObjIDIndexGO'] = obsObjIDIndexGO
        cat['obsBandIndexGO'] = obsBandIndexGO
        cat['obsMagStdGO'] = obsMagStdGO
        cat['objMagStdMeanGO'] = objMagStdMean[obsObjIDIndexGO, obsBandIndexGO]
        cat['obsMagErr2GO'] = obsMagErr2GO
        cat['obsMagADUModelErrGO'] = obsMagADUModelErr[goodObs]
        cat['sigFgcm'] = self.fgcmPars.compSigFgcm[self.fgcmPars.expBandIndex[obsExpIndexGO]]
        cat['ccdNGoodTilingsGO'] = ccdNGoodTilings[obsExpIndexGO, obsCCDIndexGO]
        cat['ccdGrayErrGO'] = ccdGrayErr[obsExpIndexGO, obsCCDIndexGO]
        cat['objNGoodObsGO'] = objNGoodObs[obsObjIDIndexGO, obsBandIndexGO]

        fitsio.write('%s_forchi2_%d.fits' % (self.outfileBaseWithCycle, int(self.sigmaCal * 10000)), cat, clobber=True)
        """

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state


