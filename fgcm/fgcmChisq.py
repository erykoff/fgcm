from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import os
import sys
import esutil
import time

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

## FIXME: derivatives should not be zero when hitting the boundary (check)

class FgcmChisq(object):
    """
    Class which computes the chi-squared for the fit.

    parameters
    ----------
    fgcmConfig: FgcmConfig
       Config object
    fgcmPars: FgcmParameters
       Parameter object
    fgcmStars: FgcmStars
       Stars object
    fgcmLUT: FgcmLUT
       LUT object

    Config variables
    ----------------
    nCore: int
       Number of cores to run in multiprocessing
    nStarPerRun: int
       Number of stars per run.  More can use more memory.
    noChromaticCorrections: bool
       If set to True, then no chromatic corrections are applied.  (bad idea).
    """

    def __init__(self,fgcmConfig,fgcmPars,fgcmStars,fgcmLUT):

        self.fgcmLog = fgcmConfig.fgcmLog

        #self.fgcmLog.log('INFO','Initializing FgcmChisq')
        self.fgcmLog.info('Initializing FgcmChisq')

        # does this need to be shm'd?
        self.fgcmPars = fgcmPars

        # this is shm'd
        self.fgcmLUT = fgcmLUT

        # also shm'd
        self.fgcmStars = fgcmStars

        # need to configure
        self.nCore = fgcmConfig.nCore
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.nStarPerRun = fgcmConfig.nStarPerRun
        self.noChromaticCorrections = fgcmConfig.noChromaticCorrections
        self.bandFitIndex = fgcmConfig.bandFitIndex
        self.useQuadraticPwv = fgcmConfig.useQuadraticPwv
        self.freezeStdAtmosphere = fgcmConfig.freezeStdAtmosphere

        # these are the standard *band* I10s
        self.I10StdBand = fgcmConfig.I10StdBand

        self.illegalValue = fgcmConfig.illegalValue

        if (fgcmConfig.useSedLUT and self.fgcmLUT.hasSedLUT):
            self.useSedLUT = True
        else:
            self.useSedLUT = False

        self.resetFitChisqList()

        # this is the default number of parameters
        self.nActualFitPars = self.fgcmPars.nFitPars
        #self.fgcmLog.log('INFO','Default: fit %d parameters.' % (self.nActualFitPars))
        self.fgcmLog.info('Default: fit %d parameters.' % (self.nActualFitPars))

        self.clearMatchCache()


    def resetFitChisqList(self):
        """
        Reset the recorded list of chi-squared values.
        """

        self.fitChisqs = []

    def clearMatchCache(self):
        """
        Clear the pre-match cache.  Note that this isn't working right.
        """
        self.matchesCached = False
        self.goodObs = None
        self.goodStarsSub = None

    def __call__(self,fitParams,fitterUnits=False,computeDerivatives=False,computeSEDSlopes=False,useMatchCache=False,debug=False,allExposures=False,includeReserve=False,fgcmGray=None):
        """
        Compute the chi-squared for a given set of parameters.

        parameters
        ----------
        fitParams: numpy array of floats
           Array with the numerical values of the parameters (properly formatted).
        fitterUnits: bool, default=False
           Are the units of fitParams normalized for the minimizer?
        computeDerivatives: bool, default=False
           Compute fit derivatives?
        computeSEDSlopes: bool, default=False
           Compute SED slopes from magnitudes?
        useMatchCache: bool, default=False
           Cache observation matches.  Do not use!
        debug: bool, default=False
           Debug mode with no multiprocessing
        allExposures: bool, default=False
           Compute using all exposures, including flagged/non-photometric
        includeReserve: bool, default=False
           Compute using all objects, including those put in reserve.
        fgcmGray: FgcmGray, default=None
           CCD Gray information for computing with "ccd crunch"
        """

        # computeDerivatives: do we want to compute the derivatives?
        # computeSEDSlope: compute SED Slope and recompute mean mags?
        # fitterUnits: units of th fitter or "true" units?

        self.computeDerivatives = computeDerivatives
        self.computeSEDSlopes = computeSEDSlopes
        self.fitterUnits = fitterUnits
        self.allExposures = allExposures
        self.useMatchCache = useMatchCache
        self.includeReserve = includeReserve
        self.fgcmGray = fgcmGray    # may be None

        self.fgcmLog.debug('FgcmChisq: computeDerivatives = %d' %
                         (int(computeDerivatives)))
        self.fgcmLog.debug('FgcmChisq: computeSEDSlopes = %d' %
                         (int(computeSEDSlopes)))
        self.fgcmLog.debug('FgcmChisq: fitterUnits = %d' %
                         (int(fitterUnits)))
        self.fgcmLog.debug('FgcmChisq: allExposures = %d' %
                         (int(allExposures)))
        self.fgcmLog.debug('FgcmChisq: includeReserve = %d' %
                         (int(includeReserve)))

        startTime = time.time()

        if (self.allExposures and (self.computeDerivatives or
                                   self.computeSEDSlopes)):
            raise ValueError("Cannot set allExposures and computeDerivatives or computeSEDSlopes")
        self.fgcmPars.reloadParArray(fitParams,fitterUnits=self.fitterUnits)
        self.fgcmPars.parsToExposures()


        # and reset numbers if necessary
        if (not self.allExposures):
            snmm.getArray(self.fgcmStars.objMagStdMeanHandle)[:] = 99.0
            snmm.getArray(self.fgcmStars.objMagStdMeanNoChromHandle)[:] = 99.0
            snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)[:] = 99.0

        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=self.includeReserve)

        self.fgcmLog.info('Found %d good stars for chisq' % (goodStars.size))

        if (goodStars.size == 0):
            raise ValueError("No good stars to fit!")

        # do global pre-matching before giving to workers, because
        #  it is faster this way

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        if (self.useMatchCache and self.matchesCached) :
            # we have already done the matching
            self.fgcmLog.info('Retrieving cached matches')
            goodObs = self.goodObs
            goodStarsSub = self.goodStarsSub
        else:
            # we need to do matching
            preStartTime=time.time()
            self.fgcmLog.info('Pre-matching stars and observations...')

            if not self.allExposures:
                expFlag = self.fgcmPars.expFlag
            else:
                expFlag = None

            goodStarsSub, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=expFlag)

            self.fgcmLog.info('Pre-matching done in %.1f sec.' %
                             (time.time() - preStartTime))

            if (self.useMatchCache) :
                self.fgcmLog.info('Caching matches for next iteration')
                self.matchesCached = True
                self.goodObs = goodObs
                self.goodStarsSub = goodStarsSub

        self.nSums = 2  # chisq, nobs
        if (self.computeDerivatives):
            # we have one for each of the derivatives
            # and a duplicate set to track which parameters were "touched"
            self.nSums += 2*self.fgcmPars.nFitPars

        self.debug = debug
        if (self.debug):
            # debug mode: single core
            self.totalHandleDict = {}
            self.totalHandleDict[0] = snmm.createArray(self.nSums,dtype='f8')

            self._worker((goodStars,goodObs))

            partialSums = snmm.getArray(self.totalHandleDict[0])[:]
        else:
            # regular multi-core


            # make a dummy process to discover starting child number
            proc = multiprocessing.Process()
            workerIndex = proc._identity[0]+1
            proc = None

            self.totalHandleDict = {}
            for thisCore in xrange(self.nCore):
                self.totalHandleDict[workerIndex + thisCore] = (
                    snmm.createArray(self.nSums,dtype='f8'))

            # split goodStars into a list of arrays of roughly equal size

            prepStartTime = time.time()
            nSections = goodStars.size // self.nStarPerRun + 1
            goodStarsList = np.array_split(goodStars,nSections)


            # is there a better way of getting all the first elements from the list?
            #  note that we need to skip the first which should be zero (checked above)
            #  see also fgcmBrightObs.py
            # splitValues is the first of the goodStars in each list
            splitValues = np.zeros(nSections-1,dtype='i4')
            for i in xrange(1,nSections):
                splitValues[i-1] = goodStarsList[i][0]

            # get the indices from the goodStarsSub matched list (matched to goodStars)
            splitIndices = np.searchsorted(goodStars[goodStarsSub], splitValues)

            # and split along the indices
            goodObsList = np.split(goodObs,splitIndices)

            workerList = list(zip(goodStarsList,goodObsList))

            # reverse sort so the longest running go first
            workerList.sort(key=lambda elt:elt[1].size, reverse=True)

            self.fgcmLog.info('Using %d sections (%.1f seconds)' %
                             (nSections,time.time()-prepStartTime))

            self.fgcmLog.info('Running chisq on %d cores' % (self.nCore))

            # make a pool
            pool = Pool(processes=self.nCore)
            pool.map(self._worker,workerList,chunksize=1)
            pool.close()
            pool.join()

            # sum up the partial sums from the different jobs
            partialSums = np.zeros(self.nSums,dtype='f8')
            for thisCore in xrange(self.nCore):
                partialSums[:] += snmm.getArray(
                    self.totalHandleDict[workerIndex + thisCore])[:]


        if (not self.allExposures):
            # we get the number of fit parameters by counting which of the parameters
            #  have been touched by the data (number of touches is irrelevant)

            if (self.computeDerivatives):
                nonZero, = np.where(partialSums[self.fgcmPars.nFitPars:
                                                    2*self.fgcmPars.nFitPars] > 0)
                self.nActualFitPars = nonZero.size
                self.fgcmLog.info('Actually fit %d parameters.' % (self.nActualFitPars))

            fitDOF = partialSums[-1] - float(self.nActualFitPars)

            if (fitDOF <= 0):
                raise ValueError("Number of parameters fitted is more than number of constraints! (%d > %d)" % (self.fgcmPars.nFitPars,partialSums[-1]))

            fitChisq = partialSums[-2] / fitDOF
            if (self.computeDerivatives):
                dChisqdP = partialSums[0:self.fgcmPars.nFitPars] / fitDOF

            # want to append this...
            self.fitChisqs.append(fitChisq)

            self.fgcmLog.info('Chisq/dof = %.2f (%d iterations)' %
                             (fitChisq, len(self.fitChisqs)))

        else:
            try:
                fitChisq = self.fitChisqs[-1]
            except IndexError:
                fitChisq = 0.0

        # free shared arrays
        for key in self.totalHandleDict.keys():
            snmm.freeArray(self.totalHandleDict[key])

        self.fgcmLog.info('Chisq computation took %.2f seconds.' %
                         (time.time() - startTime))

        self.fgcmStars.magStdComputed = True
        if (self.allExposures):
            self.fgcmStars.allMagStdComputed = True

        if (self.computeDerivatives):
            return fitChisq, dChisqdP
        else:
            return fitChisq

    def _worker(self,goodStarsAndObs):
        """
        Multiprocessing worker for FgcmChisq.  Not to be called on its own.

        parameters
        ----------
        goodStarsAndObs: tuple[2]
           (goodStars, goodObs)
        """

        # NOTE: No logging is allowed in the _worker method

        workerStartTime = time.time()

        goodStars = goodStarsAndObs[0]
        goodObs = goodStarsAndObs[1]

        if self.debug:
            thisCore = 0
        else:
            thisCore = multiprocessing.current_process()._identity[0]

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanNoChrom = snmm.getArray(self.fgcmStars.objMagStdMeanNoChromHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsLUTFilterIndex = snmm.getArray(self.fgcmStars.obsLUTFilterIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)
        obsSecZenith = snmm.getArray(self.fgcmStars.obsSecZenithHandle)
        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        # obsMagADUErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)

        # and fgcmGray stuff (if desired)
        if (self.fgcmGray is not None):
            ccdGray = snmm.getArray(self.fgcmGray.ccdGrayHandle)
            # this is ccdGray[expIndex, ccdIndex]
            # and we only apply when > self.illegalValue
            # same sign as FGCM_DUST (QESys)

        # and the arrays for locking access
        objMagStdMeanLock = snmm.getArrayBase(self.fgcmStars.objMagStdMeanHandle).get_lock()
        obsMagStdLock = snmm.getArrayBase(self.fgcmStars.obsMagStdHandle).get_lock()


        # cut these down now, faster later
        obsObjIDIndexGO = obsObjIDIndex[goodObs]
        obsBandIndexGO = obsBandIndex[goodObs]
        obsLUTFilterIndexGO = obsLUTFilterIndex[goodObs]
        obsExpIndexGO = obsExpIndex[goodObs]
        obsSecZenithGO = obsSecZenith[goodObs]
        obsCCDIndexGO = obsCCDIndex[goodObs]

        # which observations are actually used in the fit?

        _,obsFitUseGO = esutil.numpy_util.match(self.bandFitIndex,
                                                obsBandIndexGO)

        # now refer to obsBandIndex[goodObs]
        # add GO to index names that are cut to goodObs
        # add GOF to index names that are cut to goodObs[obsFitUseGO]

        lutIndicesGO = self.fgcmLUT.getIndices(obsLUTFilterIndexGO,
                                               self.fgcmPars.expLnPwv[obsExpIndexGO],
                                               self.fgcmPars.expO3[obsExpIndexGO],
                                               self.fgcmPars.expLnTau[obsExpIndexGO],
                                               self.fgcmPars.expAlpha[obsExpIndexGO],
                                               obsSecZenithGO,
                                               obsCCDIndexGO,
                                               self.fgcmPars.expPmb[obsExpIndexGO])
        I0GO = self.fgcmLUT.computeI0(self.fgcmPars.expLnPwv[obsExpIndexGO],
                                      self.fgcmPars.expO3[obsExpIndexGO],
                                      self.fgcmPars.expLnTau[obsExpIndexGO],
                                      self.fgcmPars.expAlpha[obsExpIndexGO],
                                      obsSecZenithGO,
                                      self.fgcmPars.expPmb[obsExpIndexGO],
                                      lutIndicesGO)
        I10GO = self.fgcmLUT.computeI1(self.fgcmPars.expLnPwv[obsExpIndexGO],
                                       self.fgcmPars.expO3[obsExpIndexGO],
                                       #np.log(self.fgcmPars.expTau[obsExpIndexGO]),
                                       self.fgcmPars.expLnTau[obsExpIndexGO],
                                       self.fgcmPars.expAlpha[obsExpIndexGO],
                                       obsSecZenithGO,
                                       self.fgcmPars.expPmb[obsExpIndexGO],
                                       lutIndicesGO) / I0GO


        qeSysGO = self.fgcmPars.expQESys[obsExpIndexGO]

        obsMagGO = obsMagADU[goodObs] + 2.5*np.log10(I0GO) + qeSysGO

        if (self.fgcmGray is not None):
            # We want to apply the "CCD Gray Crunch"
            # make sure we aren't adding something crazy, but this shouldn't happen
            # because we're filtering good observations (I hope!)
            ok,=np.where(ccdGray[obsExpIndexGO, obsCCDIndexGO] > self.illegalValue)
            obsMagGO[ok] += ccdGray[obsExpIndexGO[ok], obsCCDIndexGO[ok]]

        # Compute the sub-selected error-squared, using model error when available
        obsMagErr2GO = obsMagADUModelErr[goodObs]**2.

        if (self.computeSEDSlopes):
            # first, compute mean mags (code same as below.  FIXME: consolidate, but how?)

            # make temp vars.  With memory overhead

            wtSum = np.zeros_like(objMagStdMean,dtype='f8')
            objMagStdMeanTemp = np.zeros_like(objMagStdMean)

            np.add.at(wtSum,
                      (obsObjIDIndexGO,obsBandIndexGO),
                      1./obsMagErr2GO)
            np.add.at(objMagStdMeanTemp,
                  (obsObjIDIndexGO,obsBandIndexGO),
                  obsMagGO/obsMagErr2GO)

            # these are good object/bands that were observed
            gd=np.where(wtSum > 0.0)

            # and acquire lock to save the values
            objMagStdMeanLock.acquire()

            objMagStdMean[gd] = objMagStdMeanTemp[gd] / wtSum[gd]
            objMagStdMeanErr[gd] = np.sqrt(1./wtSum[gd])

            # and release the lock.
            objMagStdMeanLock.release()

            if (self.useSedLUT):
                self.fgcmStars.computeObjectSEDSlopesLUT(goodStars,self.fgcmLUT)
            else:
                self.fgcmStars.computeObjectSEDSlopes(goodStars)

        # compute linearized chromatic correction
        deltaStdGO = 2.5 * np.log10((1.0 +
                                   objSEDSlope[obsObjIDIndexGO,
                                               obsBandIndexGO] * I10GO) /
                                  (1.0 + objSEDSlope[obsObjIDIndexGO,
                                                     obsBandIndexGO] *
                                   self.I10StdBand[obsBandIndexGO]))

        if self.noChromaticCorrections:
            # NOT RECOMMENDED
            deltaStdGO *= 0.0

        # we can only do this for calibration stars.
        #  must reference the full array to save

        # acquire lock when we write to and retrieve from full array
        obsMagStdLock.acquire()

        obsMagStd[goodObs] = obsMagGO + deltaStdGO
        # this is cut here
        obsMagStdGO = obsMagStd[goodObs]

        # we now have a local cut copy, so release
        obsMagStdLock.release()

        # kick out if we're just computing magstd for all exposures
        if (self.allExposures) :
            # kick out
            return None

        # compute mean mags

        # we make temporary variables.  These are less than ideal because they
        #  take up the full memory footprint.  MAYBE look at making a smaller
        #  array just for the stars under consideration, but this would make the
        #  indexing in the np.add.at() more difficult

        wtSum = np.zeros_like(objMagStdMean,dtype='f8')
        objMagStdMeanTemp = np.zeros_like(objMagStdMean)
        objMagStdMeanNoChromTemp = np.zeros_like(objMagStdMeanNoChrom)

        np.add.at(wtSum,
                  (obsObjIDIndexGO,obsBandIndexGO),
                  1./obsMagErr2GO)

        np.add.at(objMagStdMeanTemp,
                  (obsObjIDIndexGO,obsBandIndexGO),
                  obsMagStdGO/obsMagErr2GO)

        # And the same thing with the non-chromatic corrected values
        np.add.at(objMagStdMeanNoChromTemp,
                  (obsObjIDIndexGO,obsBandIndexGO),
                  obsMagGO/obsMagErr2GO)

        # which objects/bands have observations?
        gd=np.where(wtSum > 0.0)

        # and acquire lock to save the values
        objMagStdMeanLock.acquire()

        objMagStdMean[gd] = objMagStdMeanTemp[gd] / wtSum[gd]
        objMagStdMeanNoChrom[gd] = objMagStdMeanNoChromTemp[gd] / wtSum[gd]
        objMagStdMeanErr[gd] = np.sqrt(1./wtSum[gd])

        # also make local copies for Good Observations
        objMagStdMeanGO = objMagStdMean[obsObjIDIndexGO,obsBandIndexGO]
        objMagStdMeanErr2GO = objMagStdMeanErr[obsObjIDIndexGO,obsBandIndexGO]**2.

        # and release the lock.
        objMagStdMeanLock.release()

        # compute delta-mags

        deltaMagGO = (obsMagStdGO - objMagStdMeanGO)

        # Note that this is the model error when we have it
        obsWeightGO = 1. / obsMagErr2GO

        deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

        partialChisq = np.sum(deltaMagGO[obsFitUseGO]**2. * obsWeightGO[obsFitUseGO])

        partialArray = np.zeros(self.nSums,dtype='f8')
        partialArray[-2] = partialChisq
        partialArray[-1] = obsFitUseGO.size

        if (self.computeDerivatives):
            unitDict=self.fgcmPars.getUnitDict(fitterUnits=self.fitterUnits)

            # this is going to be ugly.  wow, how many indices and sub-indices?
            #  or does it simplify since we need all the obs on a night?
            #  we shall see!  And speed up!

            (dLdLnPwvGO,dLdO3GO,dLdLnTauGO,dLdAlphaGO) = (
                self.fgcmLUT.computeLogDerivatives(lutIndicesGO,
                                                   I0GO))

            if (self.fgcmLUT.hasI1Derivatives):
                (dLdLnPwvI1GO,dLdO3I1GO,dLdLnTauI1GO,dLdAlphaI1GO) = (
                    self.fgcmLUT.computeLogDerivativesI1(lutIndicesGO,
                                                         I0GO,
                                                         I10GO,
                                                         objSEDSlope[obsObjIDIndexGO,
                                                                     obsBandIndexGO]))
                dLdLnPwvGO += dLdLnPwvI1GO
                dLdO3GO += dLdO3I1GO
                dLdLnTauGO += dLdLnTauI1GO
                dLdAlphaGO += dLdAlphaI1GO


            # we have objMagStdMeanErr[objIndex,:] = \Sum_{i"} 1/\sigma^2_{i"j}
            #   note that this is summed over all observations of an object in a band
            #   so that this is already done

            # we need magdLdp = \Sum_{i'} (1/\sigma^2_{i'j}) dL(i',j|p)
            #   note that this is summed over all observations in a filter that
            #   touch a given parameter

            # set up arrays
            magdLdLnPwvIntercept = np.zeros((self.fgcmPars.nCampaignNights,
                                           self.fgcmPars.nFitBands))
            magdLdLnPwvSlope = np.zeros_like(magdLdLnPwvIntercept)
            magdLdLnPwvQuadratic = np.zeros_like(magdLdLnPwvIntercept)
            magdLdLnPwvOffset = np.zeros_like(magdLdLnPwvIntercept)
            magdLdLnTauIntercept = np.zeros_like(magdLdLnPwvIntercept)
            magdLdLnTauSlope = np.zeros_like(magdLdLnPwvIntercept)
            magdLdLnTauOffset = np.zeros_like(magdLdLnPwvIntercept)
            magdLdAlpha = np.zeros_like(magdLdLnPwvIntercept)
            magdLdO3 = np.zeros_like(magdLdLnPwvIntercept)

            magdLdLnPwvScale = np.zeros(self.fgcmPars.nFitBands,dtype='f4')
            magdLdLnTauScale = np.zeros_like(magdLdLnPwvScale)

            magdLdLnPwvRetrievedScale = np.zeros(self.fgcmPars.nFitBands,dtype='f4')
            magdLdLnPwvRetrievedOffset = np.zeros_like(magdLdLnPwvRetrievedScale)
            magdLdLnPwvRetrievedNightlyOffset = np.zeros_like(magdLdLnPwvIntercept)

            magdLdWashIntercept = np.zeros((self.fgcmPars.nWashIntervals,
                                            self.fgcmPars.nFitBands))
            magdLdWashSlope = np.zeros_like(magdLdWashIntercept)

            # note below that objMagStdMeanErr2GO is the the square of the error,
            #  and already cut to [obsObjIDIndexGO,obsBandIndexGO]

            if not self.freezeStdAtmosphere:
                ##########
                ## O3
                ##########

                expNightIndexGOF = self.fgcmPars.expNightIndex[obsExpIndexGO[obsFitUseGO]]
                uNightIndex = np.unique(expNightIndexGOF)

                np.add.at(magdLdO3,
                          (expNightIndexGOF,obsBandIndexGO[obsFitUseGO]),
                          dLdO3GO[obsFitUseGO] / obsMagErr2GO[obsFitUseGO])
                np.multiply.at(magdLdO3,
                               (expNightIndexGOF,obsBandIndexGO[obsFitUseGO]),
                               objMagStdMeanErr2GO[obsFitUseGO])
                np.add.at(partialArray[self.fgcmPars.parO3Loc:
                                           (self.fgcmPars.parO3Loc+
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF,
                          deltaMagWeightedGOF * (
                        (dLdO3GO[obsFitUseGO] -
                         magdLdO3[expNightIndexGOF,obsBandIndexGO[obsFitUseGO]])))

                partialArray[self.fgcmPars.parO3Loc +
                             uNightIndex] *= (2.0 / unitDict['o3Unit'])
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parO3Loc +
                             uNightIndex] += 1

                ###########
                ## Alpha
                ###########

                np.add.at(magdLdAlpha,
                          (expNightIndexGOF,obsBandIndexGO[obsFitUseGO]),
                          dLdAlphaGO[obsFitUseGO] / obsMagErr2GO[obsFitUseGO])
                np.multiply.at(magdLdAlpha,
                               (expNightIndexGOF,obsBandIndexGO[obsFitUseGO]),
                               objMagStdMeanErr2GO[obsFitUseGO])
                np.add.at(partialArray[self.fgcmPars.parAlphaLoc:
                                           (self.fgcmPars.parAlphaLoc+
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF,
                          deltaMagWeightedGOF * (
                        (dLdAlphaGO[obsFitUseGO] -
                         magdLdAlpha[expNightIndexGOF,obsBandIndexGO[obsFitUseGO]])))

                partialArray[self.fgcmPars.parAlphaLoc +
                             uNightIndex] *= (2.0 / unitDict['alphaUnit'])
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parAlphaLoc +
                             uNightIndex] += 1


                ###########
                ## PWV External
                ###########

                if (self.fgcmPars.hasExternalPwv and not self.fgcmPars.useRetrievedPwv):
                    hasExtGOF,=np.where(self.fgcmPars.externalPwvFlag[obsExpIndexGO[obsFitUseGO]])
                    uNightIndexHasExt = np.unique(expNightIndexGOF[hasExtGOF])

                    # PWV Nightly Offset
                    np.add.at(magdLdLnPwvOffset,
                              (expNightIndexGOF[hasExtGOF],
                               obsBandIndexGO[obsFitUseGO[hasExtGOF]]),
                              dLdLnPwvGO[obsFitUseGO[hasExtGOF]] /
                              obsMagErr2GO[obsFitUseGO[hasExtGOF]])
                    np.multiply.at(magdLdLnPwvOffset,
                                   (expNightIndexGOF[hasExtGOF],
                                    obsBandIndexGO[obsFitUseGO[hasExtGOF]]),
                                   objMagStdMeanErr2GO[obsFitUseGO[hasExtGOF]])
                    np.add.at(partialArray[self.fgcmPars.parExternalLnPwvOffsetLoc:
                                               (self.fgcmPars.parExternalLnPwvOffsetLoc+
                                                self.fgcmPars.nCampaignNights)],
                              expNightIndexGOF[hasExtGOF],
                              deltaMagWeightedGOF[hasExtGOF] * (
                            (dLdLnPwvGO[obsFitUseGO[hasExtGOF]] -
                             magdLdLnPwvOffset[expNightIndexGOF[hasExtGOF],
                                             obsBandIndexGO[obsFitUseGO[hasExtGOF]]])))
                    partialArray[self.fgcmPars.parExternalLnPwvOffsetLoc +
                                 uNightIndexHasExt] *= (2.0 / unitDict['lnPwvUnit'])
                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parExternalLnPwvOffsetLoc +
                                 uNightIndexHasExt] += 1


                    # PWV Global Scale
                    np.add.at(magdLdLnPwvScale,
                              obsBandIndexGO[obsFitUseGO[hasExtGOF]],
                              self.fgcmPars.expLnPwv[obsExpIndexGO[obsFitUseGO[hasExtGOF]]] *
                              dLdLnPwvGO[obsFitUseGO[hasExtGOF]] /
                              obsMagErr2GO[obsFitUseGO[hasExtGOF]])
                    np.multiply.at(magdLdLnPwvScale,
                                   obsBandIndexGO[obsFitUseGO[hasExtGOF]],
                                   objMagStdMeanErr2GO[obsFitUseGO[hasExtGOF]])
                    partialArray[self.fgcmPars.parExternalLnPwvScaleLoc] = 2.0 * (
                        np.sum(deltaMagWeightedGOF[hasExtGOF] * (
                                self.fgcmPars.expLnPwv[obsExpIndexGO[obsFitUseGO[hasExtGOF]]] *
                                dLdLnPwvGO[obsFitUseGO[hasExtGOF]] -
                                magdLdLnPwvScale[obsBandIndexGO[obsFitUseGO[hasExtGOF]]])) /
                        unitDict['lnPwvGlobalUnit'])
                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parExternalLnPwvScaleLoc] += 1

                ################
                ## PWV Retrieved
                ################

                if (self.fgcmPars.useRetrievedPwv):
                    hasRetrievedPwvGOF, = np.where((self.fgcmPars.compRetrievedLnPwvFlag[obsExpIndexGO[obsFitUseGO]] &
                                                    retrievalFlagDict['EXPOSURE_RETRIEVED']) > 0)

                    if hasRetrievedPwvGOF.size > 0:
                        # note this might be zero-size on first run

                        # PWV Retrieved Global Scale
                        np.add.at(magdLdLnPwvRetrievedScale,
                                  obsBandIndexGO[obsFitUseGO[hasRetrievedPwvGOF]],
                                  self.fgcmPars.expLnPwv[obsExpIndexGO[obsFitUseGO[hasRetrievedPwvGOF]]] *
                                  dLdLnPwvGO[obsFitUseGO[hasRetrievedPwvGOF]] /
                                  obsMagErr2GO[obsFitUseGO[hasRetrievedPwvGOF]])
                        np.multiply.at(magdLdLnPwvRetrievedScale,
                                       obsBandIndexGO[obsFitUseGO[hasRetrievedPwvGOF]],
                                       objMagStdMeanErr2GO[obsFitUseGO[hasRetrievedPwvGOF]])
                        partialArray[self.fgcmPars.parRetrievedLnPwvScaleLoc] = 2.0 * (
                            np.sum(deltaMagWeightedGOF[hasRetrievedPwvGOF] * (
                                    self.fgcmPars.expLnPwv[obsExpIndexGO[obsFitUseGO[hasRetrievedPwvGOF]]] *
                                    dLdLnPwvGO[obsFitUseGO[hasRetrievedPwvGOF]] -
                                    magdLdLnPwvRetrievedScale[obsBandIndexGO[obsFitUseGO[hasRetrievedPwvGOF]]])) /
                            unitDict['lnPwvGlobalUnit'])
                        partialArray[self.fgcmPars.nFitPars +
                                     self.fgcmPars.parRetrievedLnPwvScaleLoc] += 1

                        if self.fgcmPars.useNightlyRetrievedPwv:
                            # PWV Retrieved Nightly Offset

                            uNightIndexHasRetrievedPwv = np.unique(expNightIndexGOF[hasRetrievedPwvGOF])

                            np.add.at(magdLdLnPwvRetrievedNightlyOffset,
                                      (expNightIndexGOF[hasRetrievedPwvGOF],
                                       obsBandIndexGO[obsFitUseGO[hasRetrievedPwvGOF]]),
                                      dLdLnPwvGO[obsFitUseGO[hasRetrievedPwvGOF]] /
                                      obsMagErr2GO[obsFitUseGO[hasRetrievedPwvGOF]])
                            np.multiply.at(magdLdLnPwvRetrievedNightlyOffset,
                                           (expNightIndexGOF[hasRetrievedPwvGOF],
                                            obsBandIndexGO[obsFitUseGO[hasRetrievedPwvGOF]]),
                                           objMagStdMeanErr2GO[obsFitUseGO[hasRetrievedPwvGOF]])
                            np.add.at(partialArray[self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc:
                                                       (self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc+
                                                        self.fgcmPars.nCampaignNights)],
                                      expNightIndexGOF[hasRetrievedPwvGOF],
                                      deltaMagWeightedGOF[hasRetrievedPwvGOF] * (
                                    (dLdLnPwvGO[obsFitUseGO[hasRetrievedPwvGOF]] -
                                     magdLdLnPwvRetrievedNightlyOffset[expNightIndexGOF[hasRetrievedPwvGOF],
                                                                     obsBandIndexGO[obsFitUseGO[hasRetrievedPwvGOF]]])))
                            partialArray[self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc +
                                         uNightIndexHasRetrievedPwv] *= (2.0 / unitDict['lnPwvUnit'])
                            partialArray[self.fgcmPars.nFitPars +
                                         self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc +
                                         uNightIndexHasRetrievedPwv] += 1

                        else:
                            # PWV Retrieved Global Offset
                            np.add.at(magdLdLnPwvRetrievedOffset,
                                      obsBandIndexGO[obsFitUseGO[hasRetrievedPwvGOF]],
                                      dLdLnPwvGO[obsFitUseGO[hasRetrievedPwvGOF]] /
                                      obsMagErr2GO[obsFitUseGO[hasRetrievedPwvGOF]])
                            np.multiply.at(magdLdLnPwvRetrievedOffset,
                                           obsBandIndexGO[obsFitUseGO[hasRetrievedPwvGOF]],
                                           objMagStdMeanErr2GO[obsFitUseGO[hasRetrievedPwvGOF]])
                            partialArray[self.fgcmPars.parRetrievedLnPwvOffsetLoc] = 2.0 * (
                                np.sum(deltaMagWeightedGOF[hasRetrievedPwvGOF] * (
                                        dLdLnPwvGO[obsFitUseGO[hasRetrievedPwvGOF]] -
                                        magdLdLnPwvRetrievedOffset[obsBandIndexGO[obsFitUseGO[hasRetrievedPwvGOF]]])) /
                                unitDict['pwvGlobalUnit'])
                            partialArray[self.fgcmPars.nFitPars +
                                         self.fgcmPars.parRetrievedLnPwvOffsetLoc] += 1

                else:
                    ###########
                    ## Pwv No External
                    ###########

                    noExtGOF, = np.where(~self.fgcmPars.externalPwvFlag[obsExpIndexGO[obsFitUseGO]])
                    uNightIndexNoExt = np.unique(expNightIndexGOF[noExtGOF])

                    # Pwv Nightly Intercept

                    np.add.at(magdLdLnPwvIntercept,
                              (expNightIndexGOF[noExtGOF],
                               obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                              dLdLnPwvGO[obsFitUseGO[noExtGOF]] /
                              obsMagErr2GO[obsFitUseGO[noExtGOF]])
                    np.multiply.at(magdLdLnPwvIntercept,
                                   (expNightIndexGOF[noExtGOF],
                                    obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                                   objMagStdMeanErr2GO[obsFitUseGO[noExtGOF]])
                    np.add.at(partialArray[self.fgcmPars.parLnPwvInterceptLoc:
                                               (self.fgcmPars.parLnPwvInterceptLoc+
                                                self.fgcmPars.nCampaignNights)],
                              expNightIndexGOF[noExtGOF],
                              deltaMagWeightedGOF[noExtGOF] * (
                            (dLdLnPwvGO[obsFitUseGO[noExtGOF]] -
                             magdLdLnPwvOffset[expNightIndexGOF[noExtGOF],
                                             obsBandIndexGO[obsFitUseGO[noExtGOF]]])))

                    partialArray[self.fgcmPars.parLnPwvInterceptLoc +
                                 uNightIndexNoExt] *= (2.0 / unitDict['lnPwvUnit'])
                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parLnPwvInterceptLoc +
                                 uNightIndexNoExt] += 1

                    # lnPwv Nightly Slope
                    np.add.at(magdLdLnPwvSlope,
                              (expNightIndexGOF[noExtGOF],
                               obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                              self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]] *
                              dLdLnPwvGO[obsFitUseGO[noExtGOF]] /
                              obsMagErr2GO[obsFitUseGO[noExtGOF]])
                    np.multiply.at(magdLdLnPwvSlope,
                                   (expNightIndexGOF[noExtGOF],
                                    obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                                   objMagStdMeanErr2GO[obsFitUseGO[noExtGOF]])
                    np.add.at(partialArray[self.fgcmPars.parLnPwvSlopeLoc:
                                               (self.fgcmPars.parLnPwvSlopeLoc+
                                                self.fgcmPars.nCampaignNights)],
                              expNightIndexGOF[noExtGOF],
                              deltaMagWeightedGOF[noExtGOF] * (
                            (self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]] *
                             dLdLnPwvGO[obsFitUseGO[noExtGOF]] -
                             magdLdLnPwvSlope[expNightIndexGOF[noExtGOF],
                                                 obsBandIndexGO[obsFitUseGO[noExtGOF]]])))

                    partialArray[self.fgcmPars.parLnPwvSlopeLoc +
                                 uNightIndex] *= (2.0 / unitDict['lnPwvSlopeUnit'])
                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parLnPwvSlopeLoc] += 1

                    # lnPwv Nightly Quadratic
                    if self.useQuadraticPwv:
                        np.add.at(magdLdLnPwvQuadratic,
                                  (expNightIndexGOF[noExtGOF],
                                   obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                                  self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]]**2. *
                                  dLdLnPwvGO[obsFitUseGO[noExtGOF]] /
                                  obsMagErr2GO[obsFitUseGO[noExtGOF]])
                        np.multiply.at(magdLdLnPwvQuadratic,
                                       (expNightIndexGOF[noExtGOF],
                                        obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                                       objMagStdMeanErr2GO[obsFitUseGO[noExtGOF]])
                        np.add.at(partialArray[self.fgcmPars.parLnPwvQuadraticLoc:
                                                   (self.fgcmPars.parLnPwvQuadraticLoc+
                                                    self.fgcmPars.nCampaignNights)],
                                  expNightIndexGOF[noExtGOF],
                                  deltaMagWeightedGOF[noExtGOF] * (
                                (self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]]**2. *
                                 dLdLnPwvGO[obsFitUseGO[noExtGOF]] -
                                 magdLdLnPwvQuadratic[expNightIndexGOF[noExtGOF],
                                                      obsBandIndexGO[obsFitUseGO[noExtGOF]]])))

                        partialArray[self.fgcmPars.parLnPwvQuadraticLoc +
                                     uNightIndex] *= (2.0 / unitDict['lnPwvQuadraticUnit'])
                        partialArray[self.fgcmPars.nFitPars +
                                     self.fgcmPars.parLnPwvQuadraticLoc] += 1


                #############
                ## Tau External
                #############

                if (self.fgcmPars.hasExternalTau):
                    hasExtGOF,=np.where(self.fgcmPars.externalTauFlag[obsExpIndexGO[obsFitUseGO]])
                    uNightIndexHasExt = np.unique(expNightIndexGOF[hasExtGOF])

                    # Tau Nightly Offset
                    np.add.at(magdLdLnTauOffset,
                              (expNightIndexGOF[hasExtGOF],
                               obsBandIndexGO[obsFitUseGO[hasExtGOF]]),
                              dLdLnTauGO[obsFitUseGO[hasExtGOF]] /
                              obsMagErr2GO[obsFitUseGO[hasExtGOF]])
                    np.multiply.at(magdLdLnTauOffset,
                                   (expNightIndexGOF[hasExtGOF],
                                    obsBandIndexGO[obsFitUseGO[hasExtGOF]]),
                                   objMagStdMeanErr2GO[obsFitUseGO[hasExtGOF]])
                    np.add.at(partialArray[self.fgcmPars.parExternalLnTauOffsetLoc:
                                               (self.fgcmPars.parExternalLnTauOffsetLoc+
                                                self.fgcmPars.nCampaignNights)],
                              expNightIndexGOF[hasExtGOF],
                              deltaMagWeightedGOF[hasExtGOF] * (
                            (dLdLnTauGO[obsFitUseGO[hasExtGOF]] -
                             magdLdLnTauOffset[expNightIndexGOF[hasExtGOF],
                                             obsBandIndexGO[obsFitUseGO[hasExtGOF]]])))

                    partialArray[self.fgcmPars.parExternalLnTauOffsetLoc +
                                 uNightIndexHasExt] *= (2.0 / unitDict['lnTauUnit'])
                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parExternalLnTauOffsetLoc +
                                 uNightIndexHasExt] += 1

                    # Tau Global Scale
                    ## MAYBE: is this correct with the logs?
                    np.add.at(magdLdLnTauScale,
                              obsBandIndexGO[obsFitUseGO[hasExtGOF]],
                              self.fgcmPars.expLnTau[obsExpIndexGO[obsFitUseGO[hasExtGOF]]] *
                              dLdLnTauGO[obsFitUseGO[hasExtGOF]] /
                              obsMagErr2GO[obsFitUseGO[hasExtGOF]])
                    np.multiply.at(magdLdLnTauScale,
                                   obsBandIndexGO[obsFitUseGO[hasExtGOF]],
                                   objMagStdMeanErr2GO[obsFitUseGO[hasExtGOF]])
                    partialArray[self.fgcmPars.parExternalLnTauScaleLoc] = 2.0 * (
                        np.sum(deltaMagWeightedGOF[hasExtGOF] * (
                                self.fgcmPars.expLnTau[obsExpIndexGO[obsFitUseGO[hasExtGOF]]] *
                                dLdLnTauGO[obsFitUseGO[hasExtGOF]] -
                                magdLdLnTauScale[obsBandIndexGO[obsFitUseGO[hasExtGOF]]])) /
                        unitDict['lnTauUnit'])
                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parExternalLnTauScaleLoc] += 1

                ###########
                ## Tau No External
                ###########

                noExtGOF, = np.where(~self.fgcmPars.externalTauFlag[obsExpIndexGO[obsFitUseGO]])
                uNightIndexNoExt = np.unique(expNightIndexGOF[noExtGOF])

                # lnTau Nightly Intercept
                np.add.at(magdLdLnTauIntercept,
                          (expNightIndexGOF[noExtGOF],
                           obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                          dLdLnTauGO[obsFitUseGO[noExtGOF]] /
                          obsMagErr2GO[obsFitUseGO[noExtGOF]])
                np.multiply.at(magdLdLnTauIntercept,
                               (expNightIndexGOF[noExtGOF],
                                obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                               objMagStdMeanErr2GO[obsFitUseGO[noExtGOF]])
                np.add.at(partialArray[self.fgcmPars.parLnTauInterceptLoc:
                                           (self.fgcmPars.parLnTauInterceptLoc+
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF[noExtGOF],
                          deltaMagWeightedGOF[noExtGOF] * (
                        (dLdLnTauGO[obsFitUseGO[noExtGOF]] -
                         magdLdLnTauOffset[expNightIndexGOF[noExtGOF],
                                         obsBandIndexGO[obsFitUseGO[noExtGOF]]])))

                partialArray[self.fgcmPars.parLnTauInterceptLoc +
                             uNightIndexNoExt] *= (2.0 / unitDict['lnTauUnit'])
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parLnTauInterceptLoc +
                             uNightIndexNoExt] += 1

                # lnTau nightly slope
                np.add.at(magdLdLnTauSlope,
                          (expNightIndexGOF[noExtGOF],
                           obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                          self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]] *
                          dLdLnTauGO[obsFitUseGO[noExtGOF]] /
                          obsMagErr2GO[obsFitUseGO[noExtGOF]])
                np.multiply.at(magdLdLnTauSlope,
                               (expNightIndexGOF[noExtGOF],
                                obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                               objMagStdMeanErr2GO[obsFitUseGO[noExtGOF]])
                np.add.at(partialArray[self.fgcmPars.parLnTauSlopeLoc:
                                           (self.fgcmPars.parLnTauSlopeLoc+
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF[noExtGOF],
                          deltaMagWeightedGOF[noExtGOF] * (
                        (self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]] *
                         #self.fgcmPars.expLnTau[obsExpIndexGO[obsFitUseGO[noExtGOF]]] *
                         dLdLnTauGO[obsFitUseGO[noExtGOF]] -
                         magdLdLnTauSlope[expNightIndexGOF[noExtGOF],
                                           obsBandIndexGO[obsFitUseGO[noExtGOF]]])))

                partialArray[self.fgcmPars.parLnTauSlopeLoc +
                             uNightIndexNoExt] *= (2.0 / unitDict['lnTauSlopeUnit'])
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parLnTauSlopeLoc +
                             uNightIndexNoExt] += 1

            #############
            ## Washes (QE Sys)
            #############

            expWashIndexGOF = self.fgcmPars.expWashIndex[obsExpIndexGO[obsFitUseGO]]
            uWashIndex = np.unique(expWashIndexGOF)

            # Wash Intercept
            np.add.at(magdLdWashIntercept,
                      (expWashIndexGOF,obsBandIndexGO[obsFitUseGO]),
                      1./obsMagErr2GO[obsFitUseGO])
            np.multiply.at(magdLdWashIntercept,
                           (expWashIndexGOF,obsBandIndexGO[obsFitUseGO]),
                           objMagStdMeanErr2GO[obsFitUseGO])
            np.add.at(partialArray[self.fgcmPars.parQESysInterceptLoc:
                                       (self.fgcmPars.parQESysInterceptLoc +
                                        self.fgcmPars.nWashIntervals)],
                      expWashIndexGOF,
                      deltaMagWeightedGOF * (
                    (1.0 - magdLdWashIntercept[expWashIndexGOF,
                                               obsBandIndexGO[obsFitUseGO]])))

            partialArray[self.fgcmPars.parQESysInterceptLoc +
                         uWashIndex] *= (2.0 / unitDict['qeSysUnit'])
            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parQESysInterceptLoc +
                         uWashIndex] += 1

            # Wash Slope
            np.add.at(magdLdWashSlope,
                      (expWashIndexGOF,obsBandIndexGO[obsFitUseGO]),
                      (self.fgcmPars.expMJD[obsExpIndexGO[obsFitUseGO]] -
                       self.fgcmPars.washMJDs[expWashIndexGOF]) /
                       obsMagErr2GO[obsFitUseGO])
            np.multiply.at(magdLdWashSlope,
                           (expWashIndexGOF,obsBandIndexGO[obsFitUseGO]),
                           objMagStdMeanErr2GO[obsFitUseGO])
            np.add.at(partialArray[self.fgcmPars.parQESysSlopeLoc:
                                       (self.fgcmPars.parQESysSlopeLoc +
                                        self.fgcmPars.nWashIntervals)],
                      expWashIndexGOF,
                      deltaMagWeightedGOF * (
                    (self.fgcmPars.expMJD[obsExpIndexGO[obsFitUseGO]] -
                     self.fgcmPars.washMJDs[expWashIndexGOF]) -
                    magdLdWashSlope[expWashIndexGOF,
                                    obsBandIndexGO[obsFitUseGO]]))
            partialArray[self.fgcmPars.parQESysSlopeLoc +
                         uWashIndex] *= (2.0 / unitDict['qeSysSlopeUnit'])
            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parQESysSlopeLoc +
                         uWashIndex] += 1


        # note that this store doesn't need locking because we only access
        #  a given array from a single process

        totalArr = snmm.getArray(self.totalHandleDict[thisCore])
        totalArr[:] = totalArr[:] + partialArray

        # and we're done
        return None

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state
