import numpy as np
import os
import sys
import esutil
import time

from .fgcmUtilities import retrievalFlagDict
from .fgcmUtilities import MaxFitIterations
from .fgcmUtilities import Cheb2dField
from .fgcmUtilities import objFlagDict

from .fgcmNumbaUtilities import numba_test, add_at_1d, add_at_2d, add_at_3d

import multiprocessing

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm


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

        self.fgcmLog.debug('Initializing FgcmChisq')

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
        self.nStarPerGrayRun = fgcmConfig.nStarPerGrayRun
        self.noChromaticCorrections = fgcmConfig.noChromaticCorrections
        self.bandFitIndex = fgcmConfig.bandFitIndex
        self.useQuadraticPwv = fgcmConfig.useQuadraticPwv
        self.freezeStdAtmosphere = fgcmConfig.freezeStdAtmosphere
        self.ccdGraySubCCD = fgcmConfig.ccdGraySubCCD
        self.useRefStarsWithInstrument = fgcmConfig.useRefStarsWithInstrument
        self.instrumentParsPerBand = fgcmConfig.instrumentParsPerBand
        self.useExposureReferenceOffset = fgcmConfig.useExposureReferenceOffset
        self.saveParsForDebugging = fgcmConfig.saveParsForDebugging
        self.quietMode = fgcmConfig.quietMode

        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle

        # these are the standard *band* I10s
        self.I10StdBand = fgcmConfig.I10StdBand

        self.illegalValue = fgcmConfig.illegalValue

        if (fgcmConfig.useSedLUT and self.fgcmLUT.hasSedLUT):
            self.useSedLUT = True
        else:
            self.useSedLUT = False

        self.deltaMapperDefault = None

        self.resetFitChisqList()

        # this is the default number of parameters
        self.nActualFitPars = self.fgcmPars.nFitPars
        if not self.quietMode:
            self.fgcmLog.info('Default: fit %d parameters.' % (self.nActualFitPars))

        self.clearMatchCache()

        self.maxIterations = -1

        numba_test(0)

    def resetFitChisqList(self):
        """
        Reset the recorded list of chi-squared values.
        """

        self.fitChisqs = []
        self._nIterations = 0

    @property
    def maxIterations(self):
        return self._maxIterations

    @maxIterations.setter
    def maxIterations(self, value):
        self._maxIterations = value

    def clearMatchCache(self):
        """
        Clear the pre-match cache.  Note that this isn't working right.
        """
        self.matchesCached = False
        self.goodObs = None
        self.goodStarsSub = None

    def setDeltaMapperDefault(self, deltaMapperDefault):
        """
        Set the deltaMapperDefault array.

        Parameters
        ----------
        deltaMapperDefault : `np.recarray`
        """
        self.deltaMapperDefault = deltaMapperDefault

    def __call__(self,fitParams,fitterUnits=False,computeDerivatives=False,computeSEDSlopes=False,useMatchCache=False,computeAbsThroughput=False,ignoreRef=False,debug=False,allExposures=False,includeReserve=False,fgcmGray=None):
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
        computeAbsThroughputt: `bool`, default=False
           Compute the absolute throughput after computing mean mags
        ignoreRef: `bool`, default=False
           Ignore reference stars for computation...
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
        self.computeAbsThroughput = computeAbsThroughput
        self.ignoreRef = ignoreRef

        nStarPerRun = self.nStarPerRun
        if fgcmGray is not None:
            nStarPerRun = self.nStarPerGrayRun

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

        # When we're doing the fitting, we want to fill in the missing qe sys values if needed
        self.fgcmPars.reloadParArray(fitParams, fitterUnits=self.fitterUnits)
        self.fgcmPars.parsToExposures()

        if self.saveParsForDebugging:
            # put in saving of the parameters...
            # this will be in both units
            import astropy.io.fits as pyfits
            tempCat = np.zeros(1, dtype=[('o3', 'f8', (self.fgcmPars.nCampaignNights, )),
                                         ('lnTauIntercept', 'f8', (self.fgcmPars.nCampaignNights, )),
                                         ('lnTauSlope', 'f8', (self.fgcmPars.nCampaignNights, )),
                                         ('alpha', 'f8', (self.fgcmPars.nCampaignNights, )),
                                         ('lnPwvIntercept', 'f8', (self.fgcmPars.nCampaignNights, )),
                                         ('lnPwvSlope', 'f8', (self.fgcmPars.nCampaignNights, )),
                                         ('lnPwvQuadratic', 'f8', (self.fgcmPars.nCampaignNights, )),
                                         ('qeSysIntercept', 'f8', (self.fgcmPars.parQESysIntercept.size, ))])
            tempCat['o3'][0][:] = fitParams[self.fgcmPars.parO3Loc:
                                                (self.fgcmPars.parO3Loc +
                                                 self.fgcmPars.nCampaignNights)]
            tempCat['lnTauIntercept'][0][:] = fitParams[self.fgcmPars.parLnTauInterceptLoc:
                                                            (self.fgcmPars.parLnTauInterceptLoc +
                                                             self.fgcmPars.nCampaignNights)]
            tempCat['lnTauSlope'][0][:] = fitParams[self.fgcmPars.parLnTauSlopeLoc:
                                                        (self.fgcmPars.parLnTauSlopeLoc +
                                                         self.fgcmPars.nCampaignNights)]
            tempCat['alpha'][0][:] = fitParams[self.fgcmPars.parAlphaLoc:
                                                   (self.fgcmPars.parAlphaLoc +
                                                    self.fgcmPars.nCampaignNights)]
            tempCat['lnPwvIntercept'][0][:] = fitParams[self.fgcmPars.parLnPwvInterceptLoc:
                                                            (self.fgcmPars.parLnPwvInterceptLoc +
                                                             self.fgcmPars.nCampaignNights)]
            tempCat['lnPwvSlope'][0][:] = fitParams[self.fgcmPars.parLnPwvSlopeLoc:
                                                        (self.fgcmPars.parLnPwvSlopeLoc +
                                                         self.fgcmPars.nCampaignNights)]
            tempCat['lnPwvQuadratic'][0][:] = fitParams[self.fgcmPars.parLnPwvQuadraticLoc:
                                                            (self.fgcmPars.parLnPwvQuadraticLoc +
                                                             self.fgcmPars.nCampaignNights)]
            tempCat['qeSysIntercept'][0][:] = fitParams[self.fgcmPars.parQESysInterceptLoc:
                                                            (self.fgcmPars.parQESysInterceptLoc +
                                                             self.fgcmPars.parQESysIntercept.size)]

            pyfits.writeto('%s_fitParams_%d_fitterunits.fits' % (self.outfileBaseWithCycle, len(self.fitChisqs) + 1), tempCat, overwrite=True)

            tempCat = np.zeros((1, ), dtype=[('o3', 'f8', self.fgcmPars.nCampaignNights),
                                             ('lnTauIntercept', 'f8', self.fgcmPars.nCampaignNights),
                                             ('lnTauSlope', 'f8', self.fgcmPars.nCampaignNights),
                                             ('alpha', 'f8', self.fgcmPars.nCampaignNights),
                                             ('lnPwvIntercept', 'f8', self.fgcmPars.nCampaignNights),
                                             ('lnPwvSlope', 'f8', self.fgcmPars.nCampaignNights),
                                             ('lnPwvQuadratic', 'f8', self.fgcmPars.nCampaignNights),
                                             ('qeSysIntercept', 'f8', (self.fgcmPars.nBands, self.fgcmPars.nWashIntervals))])
            tempCat['o3'][0][:] = self.fgcmPars.parO3
            tempCat['lnTauIntercept'][0][:] = self.fgcmPars.parLnTauIntercept
            tempCat['lnTauSlope'][0][:] = self.fgcmPars.parLnTauSlope
            tempCat['alpha'][0][:] = self.fgcmPars.parAlpha
            tempCat['lnPwvIntercept'][0][:] = self.fgcmPars.parLnPwvIntercept
            tempCat['lnPwvSlope'][0][:] = self.fgcmPars.parLnPwvSlope
            tempCat['lnPwvQuadratic'][0][:] = self.fgcmPars.parLnPwvQuadratic
            tempCat['qeSysIntercept'][0][:, :] = self.fgcmPars.parQESysIntercept

            pyfits.writeto('%s_fitParams_%s_parunits.fits' % (self.outfileBaseWithCycle, len(self.fitChisqs) + 1), tempCat, overwrite=True)

        #############

        # and reset numbers if necessary
        if (not self.allExposures):
            snmm.getArray(self.fgcmStars.objMagStdMeanHandle)[:] = 99.0
            snmm.getArray(self.fgcmStars.objMagStdMeanNoChromHandle)[:] = 99.0
            snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)[:] = 99.0

        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=self.includeReserve)

        if self._nIterations == 0:
            self.fgcmLog.info('Found %d good stars for chisq' % (goodStars.size))

        if (goodStars.size == 0):
            raise RuntimeError("No good stars to fit!")

        if self.fgcmStars.hasRefstars:
            objRefIDIndex = snmm.getArray(self.fgcmStars.objRefIDIndexHandle)

        # do global pre-matching before giving to workers, because
        #  it is faster this way

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        if (self.useMatchCache and self.matchesCached) :
            # we have already done the matching
            self.fgcmLog.debug('Retrieving cached matches')
            goodObs = self.goodObs
            goodStarsSub = self.goodStarsSub
        else:
            # we need to do matching
            preStartTime=time.time()
            self.fgcmLog.debug('Pre-matching stars and observations...')

            if not self.allExposures:
                expFlag = self.fgcmPars.expFlag
            else:
                expFlag = None

            goodStarsSub, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=expFlag)

            self.fgcmLog.debug('Pre-matching done in %.1f sec.' %
                               (time.time() - preStartTime))

            if (self.useMatchCache) :
                self.fgcmLog.debug('Caching matches for next iteration')
                self.matchesCached = True
                self.goodObs = goodObs
                self.goodStarsSub = goodStarsSub

        self.nSums = 4 # chisq, chisq_ref, nobs, nobs_ref
        if self.computeDerivatives:
            # 0: nFitPars -> derivative for regular chisq
            # nFitPars: 2*nFitPars -> parameters which are touched (regular chisq)
            # 2*nFitPars: 3*nFitPars -> derivative for reference stars
            # 3*nFitPars: 4*nFitPars -> parameters that are touched (reference chisq)
            self.nSums += 4 * self.fgcmPars.nFitPars

        self.applyDelta = False

        self.debug = debug
        if (self.debug):
            # debug mode: single core
            self.totalHandleDict = {}
            self.totalHandleDict[0] = snmm.createArray(self.nSums,dtype='f8')

            self._magWorker((goodStars, goodObs))

            if self.computeAbsThroughput:
                self.applyDelta = True
                self.deltaAbsOffset = self.fgcmStars.computeAbsOffset()
                self.fgcmPars.compAbsThroughput *= 10.**(-self.deltaAbsOffset / 2.5)

            if not self.allExposures:
                self._chisqWorker((goodStars, goodObs))

            partialSums = snmm.getArray(self.totalHandleDict[0])[:]
        else:
            # regular multi-core

            mp_ctx = multiprocessing.get_context('fork')
            # make a dummy process to discover starting child number
            proc = mp_ctx.Process()
            workerIndex = proc._identity[0]+1
            proc = None

            self.totalHandleDict = {}
            for thisCore in range(self.nCore):
                self.totalHandleDict[workerIndex + thisCore] = (
                    snmm.createArray(self.nSums,dtype='f8'))

            # split goodStars into a list of arrays of roughly equal size

            prepStartTime = time.time()
            nSections = goodStars.size // nStarPerRun + 1
            goodStarsList = np.array_split(goodStars,nSections)

            # is there a better way of getting all the first elements from the list?
            #  note that we need to skip the first which should be zero (checked above)
            #  see also fgcmBrightObs.py
            # splitValues is the first of the goodStars in each list
            splitValues = np.zeros(nSections-1,dtype='i4')
            for i in range(1,nSections):
                splitValues[i-1] = goodStarsList[i][0]

            # get the indices from the goodStarsSub matched list (matched to goodStars)
            splitIndices = np.searchsorted(goodStars[goodStarsSub], splitValues)

            # and split along the indices
            goodObsList = np.split(goodObs,splitIndices)

            workerList = list(zip(goodStarsList,goodObsList))

            # reverse sort so the longest running go first
            workerList.sort(key=lambda elt:elt[1].size, reverse=True)

            self.fgcmLog.debug('Using %d sections (%.1f seconds)' %
                               (nSections,time.time()-prepStartTime))

            self.fgcmLog.debug('Running chisq on %d cores' % (self.nCore))

            # make a pool
            pool = mp_ctx.Pool(processes=self.nCore)
            # Compute magnitudes
            pool.map(self._magWorker, workerList, chunksize=1)

            # And compute absolute offset if desired...
            if self.computeAbsThroughput:
                self.applyDelta = True
                self.deltaAbsOffset = self.fgcmStars.computeAbsOffset()
                self.fgcmPars.compAbsThroughput *= 10.**(-self.deltaAbsOffset / 2.5)

            # And the follow-up chisq and derivatives
            if not self.allExposures:
                pool.map(self._chisqWorker, workerList, chunksize=1)

            pool.close()
            pool.join()

            # sum up the partial sums from the different jobs
            partialSums = np.zeros(self.nSums,dtype='f8')
            for thisCore in range(self.nCore):
                partialSums[:] += snmm.getArray(
                    self.totalHandleDict[workerIndex + thisCore])[:]

        if (not self.allExposures):
            # we get the number of fit parameters by counting which of the parameters
            #  have been touched by the data (number of touches is irrelevant)

            if self.computeDerivatives:
                # Note that the extra partialSums for the reference stars will be zero
                # if there are no reference stars.
                nonZero, = np.where((partialSums[self.fgcmPars.nFitPars:
                                                     2*self.fgcmPars.nFitPars] > 0) |
                                    (partialSums[3*self.fgcmPars.nFitPars:
                                                     4*self.fgcmPars.nFitPars] > 0))
                self.nActualFitPars = nonZero.size
                if self._nIterations == 0:
                    self.fgcmLog.info('Actually fit %d parameters.' % (self.nActualFitPars))

            fitDOF = partialSums[-3] + partialSums[-1] - float(self.nActualFitPars)

            if (fitDOF <= 0):
                raise ValueError("Number of parameters fitted is more than number of constraints! (%d > %d)" % (self.fgcmPars.nFitPars,partialSums[-1]))

            fitChisq = (partialSums[-4] + partialSums[-2]) / fitDOF
            if self.computeDerivatives:
                dChisqdP = (partialSums[0:self.fgcmPars.nFitPars] +
                            partialSums[2*self.fgcmPars.nFitPars: 3*self.fgcmPars.nFitPars]) / fitDOF

                if self.saveParsForDebugging:

                    import astropy.io.fits as pyfits
                    tempCat = np.zeros((1, ), dtype=[('chisq', 'f8'),
                                                     ('o3', 'f8', self.fgcmPars.nCampaignNights),
                                                     ('lnTauIntercept', 'f8', self.fgcmPars.nCampaignNights),
                                                     ('lnTauSlope', 'f8', self.fgcmPars.nCampaignNights),
                                                     ('alpha', 'f8', self.fgcmPars.nCampaignNights),
                                                     ('lnPwvIntercept', 'f8', self.fgcmPars.nCampaignNights),
                                                     ('lnPwvSlope', 'f8', self.fgcmPars.nCampaignNights),
                                                     ('lnPwvQuadratic', 'f8', self.fgcmPars.nCampaignNights),
                                                     ('qeSysIntercept', 'f8', self.fgcmPars.nWashIntervals)])
                    tempCat['o3'][0][:] = dChisqdP[self.fgcmPars.parO3Loc:
                                                       (self.fgcmPars.parO3Loc +
                                                        self.fgcmPars.nCampaignNights)]
                    tempCat['lnTauIntercept'][0][:] = dChisqdP[self.fgcmPars.parLnTauInterceptLoc:
                                                                   (self.fgcmPars.parLnTauInterceptLoc +
                                                                    self.fgcmPars.nCampaignNights)]
                    tempCat['lnTauSlope'][0][:] = dChisqdP[self.fgcmPars.parLnTauSlopeLoc:
                                                               (self.fgcmPars.parLnTauSlopeLoc +
                                                                self.fgcmPars.nCampaignNights)]
                    tempCat['alpha'][0][:] = dChisqdP[self.fgcmPars.parAlphaLoc:
                                                          (self.fgcmPars.parAlphaLoc +
                                                           self.fgcmPars.nCampaignNights)]
                    tempCat['lnPwvIntercept'][0][:] = dChisqdP[self.fgcmPars.parLnPwvInterceptLoc:
                                                                   (self.fgcmPars.parLnPwvInterceptLoc +
                                                                    self.fgcmPars.nCampaignNights)]
                    tempCat['lnPwvSlope'][0][:] = dChisqdP[self.fgcmPars.parLnPwvSlopeLoc:
                                                               (self.fgcmPars.parLnPwvSlopeLoc +
                                                                self.fgcmPars.nCampaignNights)]
                    tempCat['lnPwvQuadratic'][0][:] = dChisqdP[self.fgcmPars.parLnPwvQuadraticLoc:
                                                                   (self.fgcmPars.parLnPwvQuadraticLoc +
                                                                    self.fgcmPars.nCampaignNights)]
                    tempCat['qeSysIntercept'][0][:] = dChisqdP[self.fgcmPars.parQESysInterceptLoc:
                                                                   (self.fgcmPars.parQESysInterceptLoc +
                                                                    self.fgcmPars.nWashIntervals)]
                    tempCat['chisq'][0] = fitChisq

                    pyfits.writeto('%s_dChisqdP_%d_fitterunits.fits' % (self.outfileBaseWithCycle, len(self.fitChisqs) + 1), tempCat, overwrite=True)

            # want to append this...
            self.fitChisqs.append(fitChisq)
            self._nIterations += 1

            self.fgcmLog.info('Chisq/dof = %.6f (%d iterations)' %
                             (fitChisq, len(self.fitChisqs)))

            # Make sure the final chisq is at or near the minimum.  Otherwise sometimes
            # we cut out at one of the cray-cray points, and that is bad.
            if (self.maxIterations > 0 and self._nIterations > self.maxIterations and
                fitChisq < (np.min(np.array(self.fitChisqs)) + 0.1)):
                self.fgcmLog.info('Ran over maximum number of iterations.')
                raise MaxFitIterations

        else:
            try:
                fitChisq = self.fitChisqs[-1]
            except IndexError:
                fitChisq = 0.0

        # free shared arrays
        for key in self.totalHandleDict.keys():
            snmm.freeArray(self.totalHandleDict[key])

        if not self.quietMode:
            self.fgcmLog.info('Chisq computation took %.2f seconds.' %
                              (time.time() - startTime))

        self.fgcmStars.magStdComputed = True
        if (self.allExposures):
            self.fgcmStars.allMagStdComputed = True

        if (self.computeDerivatives):
            return fitChisq, dChisqdP
        else:
            return fitChisq

    def _magWorker(self, goodStarsAndObs):
        """
        Multiprocessing worker to compute standard/mean magnitudes for FgcmChisq.
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

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsLUTFilterIndex = snmm.getArray(self.fgcmStars.obsLUTFilterIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)
        obsSecZenith = snmm.getArray(self.fgcmStars.obsSecZenithHandle)
        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsDeltaStd = snmm.getArray(self.fgcmStars.obsDeltaStdHandle)

        # and fgcmGray stuff (if desired)
        if (self.fgcmGray is not None):
            ccdGray = snmm.getArray(self.fgcmGray.ccdGrayHandle)
            # this is ccdGray[expIndex, ccdIndex]
            # and we only apply when > self.illegalValue
            # same sign as FGCM_DUST (QESys)
            if np.any(self.ccdGraySubCCD):
                ccdGraySubCCDPars = snmm.getArray(self.fgcmGray.ccdGraySubCCDParsHandle)

        # and the arrays for locking access
        objMagStdMeanLock = snmm.getArrayBase(self.fgcmStars.objMagStdMeanHandle).get_lock()
        obsMagStdLock = snmm.getArrayBase(self.fgcmStars.obsMagStdHandle).get_lock()

        # cut these down now, faster later
        obsObjIDIndexGO = esutil.numpy_util.to_native(obsObjIDIndex[goodObs])
        obsBandIndexGO = esutil.numpy_util.to_native(obsBandIndex[goodObs])
        obsLUTFilterIndexGO = esutil.numpy_util.to_native(obsLUTFilterIndex[goodObs])
        obsExpIndexGO = esutil.numpy_util.to_native(obsExpIndex[goodObs])
        obsCCDIndexGO = esutil.numpy_util.to_native(obsCCDIndex[goodObs])

        obsSecZenithGO = obsSecZenith[goodObs]

        # which observations are actually used in the fit?

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
                                       self.fgcmPars.expLnTau[obsExpIndexGO],
                                       self.fgcmPars.expAlpha[obsExpIndexGO],
                                       obsSecZenithGO,
                                       self.fgcmPars.expPmb[obsExpIndexGO],
                                       lutIndicesGO) / I0GO


        qeSysGO = self.fgcmPars.expQESys[obsExpIndexGO]
        filterOffsetGO = self.fgcmPars.expFilterOffset[obsExpIndexGO]

        # Explicitly update obsMagADU to float64 (internally is 32-bit)
        # I0GO, qeSysGO, filterOffsetGO are 64 bit
        obsMagGO = obsMagADU[goodObs].astype(np.float64) + \
            2.5*np.log10(I0GO) + qeSysGO + filterOffsetGO

        if (self.fgcmGray is not None):
            # We want to apply the "CCD Gray Crunch"
            # make sure we aren't adding something crazy, but this shouldn't happen
            # because we're filtering good observations (I hope!)
            ok,=np.where(ccdGray[obsExpIndexGO, obsCCDIndexGO] > self.illegalValue)

            if np.any(self.ccdGraySubCCD):
                obsXGO = snmm.getArray(self.fgcmStars.obsXHandle)[goodObs]
                obsYGO = snmm.getArray(self.fgcmStars.obsYHandle)[goodObs]

                h0, rev0 = esutil.stat.histogram(obsCCDIndexGO[ok], rev=True)
                use0, = np.where(h0 > 0)
                for i0 in use0:
                    i0a = rev0[rev0[i0]: rev0[i0 + 1]]

                    h1, rev1 = esutil.stat.histogram(obsExpIndexGO[ok][i0a], rev=True)
                    use1, = np.where(h1 > 0)
                    for i1 in use1:
                        i1a = i0a[rev1[rev1[i1]: rev1[i1 + 1]]]
                        eInd = obsExpIndexGO[ok[i1a[0]]]
                        cInd = obsCCDIndexGO[ok[i1a[0]]]
                        field = Cheb2dField(self.deltaMapperDefault['x_size'][cInd],
                                            self.deltaMapperDefault['y_size'][cInd],
                                            ccdGraySubCCDPars[eInd, cInd, :])
                        fluxScale = field.evaluate(obsXGO[ok[i1a]], obsYGO[ok[i1a]])
                        obsMagGO[ok[i1a]] += -2.5 * np.log10(np.clip(fluxScale, 0.1, None))
            else:
                # Regular non-sub-ccd
                obsMagGO[ok] += ccdGray[obsExpIndexGO[ok], obsCCDIndexGO[ok]]

            if self.useExposureReferenceOffset:
                # Apply the reference offsets as well.
                ok, = np.where(self.fgcmPars.compExpRefOffset[obsExpIndexGO] > self.illegalValue)
                obsMagGO[ok] += self.fgcmPars.compExpRefOffset[obsExpIndexGO[ok]]

        # Compute the sub-selected error-squared, using model error when available
        obsMagErr2GO = obsMagADUModelErr[goodObs].astype(np.float64)**2.

        if (self.computeSEDSlopes):
            # first, compute mean mags (code same as below.  FIXME: consolidate, but how?)

            # make temp vars.  With memory overhead

            wtSum = np.zeros_like(objMagStdMean, dtype='f8')
            objMagStdMeanTemp = np.zeros_like(objMagStdMean, dtype='f8')

            add_at_2d(wtSum,
                   (obsObjIDIndexGO,obsBandIndexGO),
                   (1./obsMagErr2GO).astype(wtSum.dtype))
            add_at_2d(objMagStdMeanTemp,
                   (obsObjIDIndexGO,obsBandIndexGO),
                   (obsMagGO/obsMagErr2GO).astype(objMagStdMeanTemp.dtype))

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
        obsDeltaStd[goodObs] = deltaStdGO

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
        objMagStdMeanTemp = np.zeros_like(objMagStdMean, dtype='f8')
        objMagStdMeanNoChromTemp = np.zeros_like(objMagStdMeanNoChrom, dtype='f8')

        add_at_2d(wtSum,
               (obsObjIDIndexGO,obsBandIndexGO),
               (1./obsMagErr2GO).astype(wtSum.dtype))

        add_at_2d(objMagStdMeanTemp,
               (obsObjIDIndexGO,obsBandIndexGO),
               (obsMagStdGO/obsMagErr2GO).astype(objMagStdMeanTemp.dtype))

        # And the same thing with the non-chromatic corrected values
        add_at_2d(objMagStdMeanNoChromTemp,
               (obsObjIDIndexGO,obsBandIndexGO),
               (obsMagGO/obsMagErr2GO).astype(objMagStdMeanNoChromTemp.dtype))

        # which objects/bands have observations?
        gd=np.where(wtSum > 0.0)

        # and acquire lock to save the values
        objMagStdMeanLock.acquire()

        objMagStdMean[gd] = objMagStdMeanTemp[gd] / wtSum[gd]
        objMagStdMeanNoChrom[gd] = objMagStdMeanNoChromTemp[gd] / wtSum[gd]
        objMagStdMeanErr[gd] = np.sqrt(1./wtSum[gd])

        # and release the lock.
        objMagStdMeanLock.release()

        # this is the end of the _magWorker

    def _chisqWorker(self, goodStarsAndObs):
        """
        Multiprocessing worker to compute chisq and derivatives for FgcmChisq.
        Not to be called on its own.

        Parameters
        ----------
        goodStarsAndObs: tuple[2]
           (goodStars, goodObs)
        """

        # kick out if we're just computing magstd for all exposures
        if (self.allExposures) :
            # kick out
            return None

        goodStars = goodStarsAndObs[0]
        goodObs = goodStarsAndObs[1]

        if self.debug:
            thisCore = 0
        else:
            thisCore = multiprocessing.current_process()._identity[0]

        # Set things up
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanNoChrom = snmm.getArray(self.fgcmStars.objMagStdMeanNoChromHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsLUTFilterIndex = snmm.getArray(self.fgcmStars.obsLUTFilterIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)
        obsSecZenith = snmm.getArray(self.fgcmStars.obsSecZenithHandle)
        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)

        # and the arrays for locking access
        objMagStdMeanLock = snmm.getArrayBase(self.fgcmStars.objMagStdMeanHandle).get_lock()
        obsMagStdLock = snmm.getArrayBase(self.fgcmStars.obsMagStdHandle).get_lock()

        # cut these down now, faster later
        obsObjIDIndexGO = esutil.numpy_util.to_native(obsObjIDIndex[goodObs])
        obsBandIndexGO = esutil.numpy_util.to_native(obsBandIndex[goodObs])
        obsLUTFilterIndexGO = esutil.numpy_util.to_native(obsLUTFilterIndex[goodObs])
        obsExpIndexGO = esutil.numpy_util.to_native(obsExpIndex[goodObs])
        obsSecZenithGO = obsSecZenith[goodObs]
        obsCCDIndexGO = esutil.numpy_util.to_native(obsCCDIndex[goodObs])


        # now refer to obsBandIndex[goodObs]
        # add GO to index names that are cut to goodObs
        # add GOF to index names that are cut to goodObs[obsFitUseGO] (see below)

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
                                       self.fgcmPars.expLnTau[obsExpIndexGO],
                                       self.fgcmPars.expAlpha[obsExpIndexGO],
                                       obsSecZenithGO,
                                       self.fgcmPars.expPmb[obsExpIndexGO],
                                       lutIndicesGO) / I0GO

        # Compute the sub-selected error-squared, using model error when available
        obsMagErr2GO = obsMagADUModelErr[goodObs].astype(np.float64)**2.

        obsMagStdLock.acquire()

        # If we want to apply the deltas, do it here
        if self.applyDelta:
            obsMagStd[goodObs] -= self.deltaAbsOffset[obsBandIndexGO]

        # Make local copy of mags
        obsMagStdGO = obsMagStd[goodObs]

        obsMagStdLock.release()

        # and acquire lock to save the values
        objMagStdMeanLock.acquire()

        if self.applyDelta:
            gdMeanStar, gdMeanBand = np.where(objMagStdMean[goodStars, :] < 90.0)
            objMagStdMean[goodStars[gdMeanStar], gdMeanBand] -= self.deltaAbsOffset[gdMeanBand]

        objMagStdMeanGO = objMagStdMean[obsObjIDIndexGO,obsBandIndexGO]
        objMagStdMeanErr2GO = objMagStdMeanErr[obsObjIDIndexGO,obsBandIndexGO]**2.

        objMagStdMeanLock.release()

        # New logic:
        #  Select out reference stars (if desired)
        #  Select out non-reference stars
        #  Compute deltas and chisq for reference stars
        #  Compute deltas and chisq for non-reference stars
        #  Compute derivatives...

        # Default mask is not to mask
        maskGO = np.ones(goodObs.size, dtype=bool)

        useRefstars = False
        if self.fgcmStars.hasRefstars and not self.ignoreRef:
            # Prepare arrays
            objRefIDIndex = snmm.getArray(self.fgcmStars.objRefIDIndexHandle)
            refMag = snmm.getArray(self.fgcmStars.refMagHandle)
            refMagErr = snmm.getArray(self.fgcmStars.refMagErrHandle)

            # Are there any reference stars in this set of stars?
            use, = np.where(objRefIDIndex[goodStars] >= 0)
            if use.size == 0:
                # There are no reference stars in this list of stars.  That's okay!
                useRefstars = False
            else:
                # Get good observations of reference stars
                # This must be two steps because we first need the indices to
                # avoid out-of-bounds
                mask = objFlagDict['REFSTAR_OUTLIER'] | objFlagDict['REFSTAR_BAD_COLOR'] | objFlagDict['REFSTAR_RESERVED']
                goodRefObsGO, = np.where((objRefIDIndex[obsObjIDIndexGO] >= 0) &
                                         ((objFlag[obsObjIDIndexGO] & mask) == 0))

                # And check that these are all quality observations
                tempUse, = np.where((objMagStdMean[obsObjIDIndexGO[goodRefObsGO],
                                                   obsBandIndexGO[goodRefObsGO]] < 90.0) &
                                    (refMag[objRefIDIndex[obsObjIDIndexGO[goodRefObsGO]],
                                            obsBandIndexGO[goodRefObsGO]] < 90.0))

                if tempUse.size > 0:
                    useRefstars = True
                    goodRefObsGO = goodRefObsGO[tempUse]

                if useRefstars:
                    # Down-select to remove reference stars
                    maskGO[goodRefObsGO] = False

        # which observations are actually used in the fit?
        useGO, = np.where(maskGO)
        _, obsFitUseGO = esutil.numpy_util.match(self.bandFitIndex,
                                                 obsBandIndexGO[useGO])
        obsFitUseGO = useGO[obsFitUseGO]

        # Now we can compute delta and chisq for non-reference stars

        deltaMagGO = obsMagStdGO - objMagStdMeanGO

        # Note that this is computed from the model error
        obsWeightGO = 1. / obsMagErr2GO

        deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

        # For correlation tracking
        gsGOF, indGOF = esutil.numpy_util.match(goodStars, obsObjIDIndexGO[obsFitUseGO])

        partialChisq = 0.0
        partialChisqRef = 0.0

        partialChisq = np.sum(deltaMagGO[obsFitUseGO]**2. * obsWeightGO[obsFitUseGO])

        # And for the reference stars (if we want)
        if useRefstars:
            # Only use the specific fit bands, for derivatives
            _, obsFitUseGRO = esutil.numpy_util.match(self.bandFitIndex,
                                                      obsBandIndexGO[goodRefObsGO])

            # useful below
            goodRefObsGOF = goodRefObsGO[obsFitUseGRO]

            deltaMagGRO = obsMagStdGO[goodRefObsGO] - refMag[objRefIDIndex[obsObjIDIndexGO[goodRefObsGO]],
                                                             obsBandIndexGO[goodRefObsGO]]

            obsWeightGRO = 1. / (obsMagErr2GO[goodRefObsGO] + refMagErr[objRefIDIndex[obsObjIDIndexGO[goodRefObsGO]],
                                                                        obsBandIndexGO[goodRefObsGO]]**2.)

            deltaRefMagWeightedGROF = deltaMagGRO[obsFitUseGRO] * obsWeightGRO[obsFitUseGRO]

            partialChisqRef += np.sum(deltaMagGRO[obsFitUseGRO]**2. * obsWeightGRO[obsFitUseGRO])

        partialArray = np.zeros(self.nSums, dtype='f8')
        partialArray[-4] = partialChisq
        partialArray[-3] = obsFitUseGO.size
        if useRefstars:
            partialArray[-2] = partialChisqRef
            partialArray[-1] = obsFitUseGRO.size

        if (self.computeDerivatives):
            if self.fitterUnits:
                units = self.fgcmPars.stepUnits
            else:
                units = np.ones(self.fgcmPars.nFitPars)

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

            # note below that objMagStdMeanErr2GO is the the square of the error,
            #  and already cut to [obsObjIDIndexGO,obsBandIndexGO]

            innerTermGOF = np.zeros(obsFitUseGO.size)

            obsExpIndexGOF = obsExpIndexGO[obsFitUseGO]
            obsBandIndexGOF = obsBandIndexGO[obsFitUseGO]
            obsBandIndexGOFI = obsBandIndexGOF[indGOF]
            obsFitUseGOI = obsFitUseGO[indGOF]
            obsMagErr2GOFI = obsMagErr2GO[obsFitUseGO[indGOF]]

            if not self.freezeStdAtmosphere:
                # And more initialization for atmosphere terms
                sumNumerator = np.zeros((goodStars.size, self.fgcmPars.nBands, self.fgcmPars.nCampaignNights))

                expNightIndexGOF = esutil.numpy_util.to_native(self.fgcmPars.expNightIndex[obsExpIndexGOF])
                expNightIndexGOFI = expNightIndexGOF[indGOF]

                ##########
                ## O3
                ##########

                uNightIndex = np.unique(expNightIndexGOF)

                sumNumerator = np.zeros((goodStars.size, self.fgcmPars.nBands, self.fgcmPars.nCampaignNights))
                sumNumerator[:, :, :] = 0.0
                add_at_3d(sumNumerator,
                          (gsGOF, obsBandIndexGOFI, expNightIndexGOFI),
                          (dLdO3GO[obsFitUseGOI] / obsMagErr2GOFI).astype(np.float64))

                innerTermGOF[:] = 0.0
                innerTermGOF[indGOF] = (dLdO3GO[obsFitUseGOI] -
                                        sumNumerator[gsGOF, obsBandIndexGOFI, expNightIndexGOFI] * objMagStdMeanErr2GO[obsFitUseGOI])

                add_at_1d(partialArray[self.fgcmPars.parO3Loc:
                                           (self.fgcmPars.parO3Loc +
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF,
                          (2.0 * deltaMagWeightedGOF * innerTermGOF).astype(np.float64))

                partialArray[self.fgcmPars.parO3Loc +
                             uNightIndex] /= units[self.fgcmPars.parO3Loc +
                                                   uNightIndex]
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parO3Loc +
                             uNightIndex] += 1

                if useRefstars:
                    # We assume that the unique nights must be a subset of those above
                    expNightIndexGROF = esutil.numpy_util.to_native(self.fgcmPars.expNightIndex[obsExpIndexGO[goodRefObsGO[obsFitUseGRO]]])
                    uRefNightIndex = np.unique(expNightIndexGROF)

                    add_at_1d(partialArray[2*self.fgcmPars.nFitPars +
                                           self.fgcmPars.parO3Loc:
                                               (2*self.fgcmPars.nFitPars +
                                                self.fgcmPars.parO3Loc +
                                                self.fgcmPars.nCampaignNights)],
                              expNightIndexGROF,
                              (2.0 * deltaRefMagWeightedGROF * dLdO3GO[goodRefObsGOF]).astype(np.float64))

                    partialArray[2*self.fgcmPars.nFitPars +
                                 self.fgcmPars.parO3Loc +
                                 uRefNightIndex] /= units[self.fgcmPars.parO3Loc +
                                                          uRefNightIndex]
                    partialArray[3*self.fgcmPars.nFitPars +
                                 self.fgcmPars.parO3Loc +
                                 uRefNightIndex] += 1

                ###########
                ## Alpha
                ###########

                sumNumerator[:, :, :] = 0.0
                add_at_3d(sumNumerator,
                          (gsGOF, obsBandIndexGOFI, expNightIndexGOFI),
                          (dLdAlphaGO[obsFitUseGOI] / obsMagErr2GOFI).astype(np.float64))

                innerTermGOF[:] = 0.0
                innerTermGOF[indGOF] = (dLdAlphaGO[obsFitUseGOI] -
                                        sumNumerator[gsGOF, obsBandIndexGOFI, expNightIndexGOFI] * objMagStdMeanErr2GO[obsFitUseGOI])

                add_at_1d(partialArray[self.fgcmPars.parAlphaLoc:
                                           (self.fgcmPars.parAlphaLoc+
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF,
                          (2.0 * deltaMagWeightedGOF * innerTermGOF).astype(np.float64))

                partialArray[self.fgcmPars.parAlphaLoc +
                             uNightIndex] /= units[self.fgcmPars.parAlphaLoc +
                                                   uNightIndex]
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parAlphaLoc +
                             uNightIndex] += 1

                if useRefstars:

                    add_at_1d(partialArray[2*self.fgcmPars.nFitPars +
                                           self.fgcmPars.parAlphaLoc:
                                               (2*self.fgcmPars.nFitPars +
                                                self.fgcmPars.parAlphaLoc+
                                                self.fgcmPars.nCampaignNights)],
                              expNightIndexGROF,
                              (2.0 * deltaRefMagWeightedGROF * dLdAlphaGO[goodRefObsGOF]).astype(np.float64))

                    partialArray[2*self.fgcmPars.nFitPars +
                                 self.fgcmPars.parAlphaLoc +
                                 uRefNightIndex] /= units[self.fgcmPars.parAlphaLoc +
                                                          uRefNightIndex]
                    partialArray[3*self.fgcmPars.nFitPars +
                                 self.fgcmPars.parAlphaLoc +
                                 uRefNightIndex] += 1

                ###########
                ## PWV External
                ###########

                if (self.fgcmPars.hasExternalPwv and not self.fgcmPars.useRetrievedPwv):
                    hasExtGOF, = np.where(self.fgcmPars.externalPwvFlag[obsExpIndexGOF])
                    uNightIndexHasExt = np.unique(expNightIndexGOF[hasExtGOF])
                    hasExtGOFG, = np.where(~self.fgcmPars.externalPwvFlag[obsExpIndexGOF[indGOF]])

                    # lnPw Nightly Offset

                    sumNumerator[:, :, :] = 0.0
                    add_at_3d(sumNumerator,
                              (gsGOF[hasExtGOFG],
                               obsBandIndexGOFI[hasExtGOFG],
                               expNightIndexGOFI[hasExtGOFG]),
                              (dLdLnPwvGO[obsFitUseGOI[hasExtGOFG]] /
                               obsMagErr2GOFI[hasExtGOFG]).astype(np.float64))

                    innerTermGOF[:] = 0.0
                    innerTermGOF[indGOF[hasExtGOFG]] = (dLdLnPwvGO[obsFitUseGOI[hasExtGOFG]] -
                                                        sumNumerator[gsGOF[hasExtGOFG],
                                                                     obsBandIndexGOFI[hasExtGOFG],
                                                                     expNightIndexGOFI[hasExtGOFG]] *
                                                        objMagStdMeanErr2GO[obsFitUseGOI[hasExtGOFG]])

                    add_at_1d(partialArray[self.fgcmPars.parExternalLnPwvOffsetLoc:
                                               (self.fgcmPars.parExternalLnPwvOffsetLoc+
                                                self.fgcmPars.nCampaignNights)],
                              expNightIndexGOF[hasExtGOF],
                              (2.0 * deltaMagWeightedGOF[hasExtGOF] * innerTermGOF[hasExtGOF]).astype(np.float64))

                    partialArray[self.fgcmPars.parExternalLnPwvOffsetLoc +
                                 uNightIndexHasExt] /= units[self.fgcmPars.parExternalLnPwvOffsetLoc +
                                                             uNightIndexHasExt]
                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parExternalLnPwvOffsetLoc +
                                 uNightIndexHasExt] += 1

                    if useRefstars:
                        hasExtGROF, = np.where(self.fgcmPars.externalPwvFlag[obsExpIndexGO[goodRefObsGOF]])
                        uRefNightIndexHasExt = np.unique(expNightIndexGROF[hasExtGROF])

                        add_at_1d(partialArray[2*self.fgcmPars.nFitPars +
                                            self.fgcmPars.parExternalLnPwvOffsetLoc:
                                                (2*self.fgcmPars.nFitPars +
                                                 self.fgcmPars.parExternalLnPwvOffsetLoc +
                                                 self.fgcmPars.nCampaignNights)],
                               expNightIndexGROF[hasExtGROF],
                               (2.0 * deltaRefMagWeightedGROF[hasExtGROF] *
                                dLdLnPwvGO[goodRefObsGOF[hasExtGROF]]).astype(np.float64))

                        partialArray[2*self.fgcmPars.nFitPars +
                                     self.fgcmPars.parExternalLnPwvOffsetLoc +
                                     uRefNightIndexHasExt] /= units[self.fgcmPars.parExternalLnPwvOffsetLoc +
                                                                    uRefNightIndexHasExt]
                        partialArray[3*self.fgcmPars.nFitPars +
                                     self.fgcmPars.parExternalLnPwvOffsetLoc +
                                     uRefNightIndexHasExt] += 1

                    # lnPwv Global Scale

                    # NOTE: this may be wrong.  Needs thought.

                    partialArray[self.fgcmPars.parExternalLnPwvScaleLoc] = 2.0 * (
                        np.sum(deltaMagWeightedGOF[hasExtGOF] * (
                                self.fgcmPars.expLnPwv[obsExpIndexGOF[hasExtGOF]] *
                                dLdLnPwvGO[obsFitUseGO[hasExtGOF]])))

                    partialArray[self.fgcmPars.parExternalLnPwvScaleLoc] /= units[self.fgcmPars.parExternalLnPwvScaleLoc]

                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parExternalLnPwvScaleLoc] += 1

                    if useRefstars:
                        temp = np.sum(2.0 * deltaRefMagWeightedGROF[hasExtGROF] *
                                      dLdLnPwvGO[goodRefObsGOF[hasExtGROF]])

                        partialArray[2*self.fgcmPars.nFitPars +
                                     self.fgcmPars.parExternalLnPwvScaleLoc] = temp / units[self.fgcmPars.parExternalLnPwvScaleLoc]
                        partialArray[3*self.fgcmPars.nFitPars +
                                     self.fgcmPars.parExternalLnPwvScaleLoc] += 1

                ################
                ## PWV Retrieved
                ################

                if (self.fgcmPars.useRetrievedPwv):
                    hasRetrievedPwvGOF, = np.where((self.fgcmPars.compRetrievedLnPwvFlag[obsExpIndexGOF] &
                                                    retrievalFlagDict['EXPOSURE_RETRIEVED']) > 0)
                    hasRetrievedPWVGOFG, = np.where((self.fgcmPars.compRetrievedLnPwvFlag[obsExpIndexGOF[indGOF]] &
                                                     retrievalFlagDict['EXPOSURE_RETRIEVED']) > 0)

                    if hasRetrievedPwvGOF.size > 0:
                        # note this might be zero-size on first run

                        # lnPwv Retrieved Global Scale

                        # This may be wrong

                        partialArray[self.fgcmPars.parRetrievedLnPwvScaleLoc] = 2.0 * (
                            np.sum(deltaMagWeightedGOF[hasRetrievedPwvGOF] * (
                                    self.fgcmPars.expLnPwv[obsExpIndexGOF[hasRetreivedPwvGOF]] *
                                    dLdLnPwvGO[obsFitUseGO[hasRetrievedPwvGOF]])))

                        partialArray[self.fgcmPars.parRetrievedLnPwvScaleLoc] /= units[self.fgcmPars.parRetrievedLnPwvScaleLoc]

                        partialArray[self.fgcmPars.nFitPars +
                                     self.fgcmPars.parRetrievedLnPwvScaleLoc] += 1

                        if useRefstars:
                            hasRetrievedPwvGROF, = np.where((self.fgcmPars.computeRetrievedLnPwvFlag[obsExpIndexGO[goodRefObsGOF]] &
                                                             (retrievalFlagDict['EXPOSURE_RETRIEVED']) > 0))

                            temp = np.sum(2.0 * deltaRefMagWeightedGROF[hasRetrievedPwvGROF] *
                                          dLdLnPwvGO[goodRefObsGROF[hasRetrievedPwvGROF]])
                            partialArray[2*self.fgcmPars.nFitPars +
                                         self.fgcmPars.parRetrievedLnPwvScaleLoc] = temp / units[self.fgcmPars.parRetrievedLnPwvScaleLoc]
                            partialArray[3*self.fgcmPars.nFitPars +
                                         self.fgcmPars.parRetrievedLnPwvScaleLoc] += 1

                        if self.fgcmPars.useNightlyRetrievedPwv:
                            # lnPwv Retrieved Nightly Offset

                            uNightIndexHasRetrievedPwv = np.unique(expNightIndexGOF[hasRetrievedPwvGOF])

                            sumNumerator[:, :, :] = 0.0
                            add_at_3d(sumNumerator,
                                      (gsGOF[hasRetrievedPwvGOFG],
                                       obsBandIndexGOFI[hasRetrievedPwvGOFG],
                                       expNightIndexGOFI[hasRetrievedPwvGOFG]),
                                      (dLdLnPwvGO[obsFitUseGOI[hasRetrievedPwvGOFG]] /
                                       obsMagErr2GOFI[hasRetrievedPwvGOFG]).astype(np.float64))

                            innerTermGOF[:] = 0.0
                            innerTermGOF[indGOF[hasRetrievedPwvGOFG]] = (dLdLnPwvGO[obsFitUseGOI[hasRetrievedPwvGOFG]] -
                                                                         sumNumerator[gsGOF[hasRetrievedPwvGOFG],
                                                                                      obsBandIndexGOFI[hasRetrievedPwvGOFG],
                                                                                      expNightIndexGOFI[hasRetrievedPwvGOFG]] *
                                                                         objMagStdMeanErr2GO[obsFitUseGOI[hasRetrievedPwvGOFG]])

                            add_at_1d(partialArray[self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc:
                                                       (self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc+
                                                        self.fgcmPars.nCampaignNights)],
                                      (2.0 * deltaMagWeightedGOF[hasRetrievedPwvGOF] * innerTermGOF[hasRetrievedPwvGOF]).astype(np.float64))

                            partialArray[self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc +
                                         uNightIndexHasRetrievedPwv] /= units[self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc +
                                                                              uNightIndexHasRetrievedPwv]
                            partialArray[self.fgcmPars.nFitPars +
                                         self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc +
                                         uNightIndexHasRetrievedPwv] += 1

                            if useRefstars:
                                uRefNightIndexHasRetrievedPWV = np.unique(expNightIndexGROF[hasRetrievedPwvGROF])

                                add_at_1d(partialArray[2*self.fgcmPars.nFitPars +
                                                    self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc:
                                                        (2*self.fgcmPars.nFitPars +
                                                         self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc+
                                                         self.fgcmPars.nCampaignNights)],
                                       expNightIndexGROF[hasRetrievedPwvGROF],
                                       (2.0 * deltaRefMagWeightedGROF *
                                       dLdLnPwvGO[goodRefObsGOF[hasRetrievedPwvGROF]]).astype(np.float64))

                                partialArray[2*self.fgcmPars.nFitPars +
                                             self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc +
                                             uRefNightIndexHasRetrievedPwv] /= units[self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc +
                                                                                     uRefNightIndexHasRetrievedPwv]
                                partialArray[3*self.fgcmPars.nFitPars +
                                             self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc +
                                             uRefNightIndexHasRetrievedPwv] += 1

                        else:
                            # lnPwv Retrieved Global Offset

                            # This may be wrong

                            partialArray[self.fgcmPars.parRetrievedLnPwvOffsetLoc] = 2.0 * (
                                np.sum(deltaMagWeightedGOF[hasRetrievedPwvGOF] * (
                                        dLdLnPwvGO[obsFitUseGO[hasRetrievedPwvGOF]])))

                            partialArray[self.fgcmPars.parRetrievedLnPwvOffsetLoc] /= units[self.fgcmPars.parRetrievedLnPwvOffsetLoc]
                            partialArray[self.fgcmPars.nFitPars +
                                         self.fgcmPars.parRetrievedLnPwvOffsetLoc] += 1

                            if useRefstars:
                                temp = np.sum(2.0 * deltaRefMagWeightedGROF *
                                              dLdLnPwvGO[goodRefObsGOF[hasRetrievedPwvGROF]])

                                partialArray[2*self.fgcmPars.nFitPars +
                                             self.fgcmPars.parRetrievedLnPwvOffsetLoc] = temp / units[self.fgcmPars.parRetrievedLnPwvOffsetLoc]
                                partialArray[3*self.fgcmPars.nFitPars +
                                             self.fgcmPars.parRetrievedLnPwvOffsetLoc] += 1


                else:
                    ###########
                    ## Pwv No External
                    ###########

                    noExtGOF, = np.where(~self.fgcmPars.externalPwvFlag[obsExpIndexGOF])
                    uNightIndexNoExt = np.unique(expNightIndexGOF[noExtGOF])
                    noExtGOFG, = np.where(~self.fgcmPars.externalPwvFlag[obsExpIndexGOF[indGOF]])

                    # lnPwv Nightly Intercept

                    sumNumerator[:, :, :] = 0.0
                    add_at_3d(sumNumerator,
                              (gsGOF[noExtGOFG],
                               obsBandIndexGOFI[noExtGOFG],
                               expNightIndexGOFI[noExtGOFG]),
                              (dLdLnPwvGO[obsFitUseGOI[noExtGOFG]] /
                              obsMagErr2GOFI[noExtGOFG]).astype(np.float64))

                    innerTermGOF[:] = 0.0
                    innerTermGOF[indGOF[noExtGOFG]] = (dLdLnPwvGO[obsFitUseGOI[noExtGOFG]] -
                                                       sumNumerator[gsGOF[noExtGOFG],
                                                                    obsBandIndexGOFI[noExtGOFG],
                                                                    expNightIndexGOFI[noExtGOFG]] *
                                                       objMagStdMeanErr2GO[obsFitUseGOI[noExtGOFG]])

                    add_at_1d(partialArray[self.fgcmPars.parLnPwvInterceptLoc:
                                               (self.fgcmPars.parLnPwvInterceptLoc+
                                                self.fgcmPars.nCampaignNights)],
                              expNightIndexGOF[noExtGOF],
                              (2.0 * deltaMagWeightedGOF[noExtGOF] * innerTermGOF[noExtGOF]).astype(np.float64))

                    partialArray[self.fgcmPars.parLnPwvInterceptLoc +
                                 uNightIndexNoExt] /= units[self.fgcmPars.parLnPwvInterceptLoc +
                                                            uNightIndexNoExt]
                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parLnPwvInterceptLoc +
                                 uNightIndexNoExt] += 1

                    if useRefstars:
                        noExtGROF, = np.where(~self.fgcmPars.externalPwvFlag[obsExpIndexGO[goodRefObsGOF]])
                        uRefNightIndexNoExt = np.unique(expNightIndexGROF[noExtGROF])

                        add_at_1d(partialArray[2*self.fgcmPars.nFitPars +
                                               self.fgcmPars.parLnPwvInterceptLoc:
                                                   (2*self.fgcmPars.nFitPars +
                                                    self.fgcmPars.parLnPwvInterceptLoc+
                                                    self.fgcmPars.nCampaignNights)],
                                  expNightIndexGROF[noExtGROF],
                                  (2.0 * deltaRefMagWeightedGROF[noExtGROF] *
                                  dLdLnPwvGO[goodRefObsGOF[noExtGROF]]).astype(np.float64))

                        partialArray[2*self.fgcmPars.nFitPars + self.fgcmPars.parLnPwvInterceptLoc + uRefNightIndexNoExt] = units[self.fgcmPars.parLnPwvInterceptLoc + uRefNightIndexNoExt]
                        partialArray[3*self.fgcmPars.nFitPars + self.fgcmPars.parLnPwvInterceptLoc + uRefNightIndexNoExt] += 1

                    # lnPwv Nightly Slope

                    dLdLnPwvSlopeGOFI = self.fgcmPars.expDeltaUT[obsExpIndexGOF[indGOF]] * dLdLnPwvGO[obsFitUseGOI]

                    sumNumerator[:, :, :] = 0.0
                    add_at_3d(sumNumerator,
                              (gsGOF[noExtGOFG],
                               obsBandIndexGOFI[noExtGOFG],
                               expNightIndexGOFI[noExtGOFG]),
                              (dLdLnPwvSlopeGOFI[noExtGOFG] /
                               obsMagErr2GOFI[noExtGOFG]).astype(np.float64))

                    innerTermGOF[:] = 0.0
                    innerTermGOF[indGOF[noExtGOFG]] = (dLdLnPwvSlopeGOFI[noExtGOFG] -
                                                       sumNumerator[gsGOF[noExtGOFG],
                                                                    obsBandIndexGOFI[noExtGOFG],
                                                                    expNightIndexGOFI[noExtGOFG]] *
                                                       objMagStdMeanErr2GO[obsFitUseGOI[noExtGOFG]])
                    add_at_1d(partialArray[self.fgcmPars.parLnPwvSlopeLoc:
                                               (self.fgcmPars.parLnPwvSlopeLoc+
                                                self.fgcmPars.nCampaignNights)],
                              expNightIndexGOF[noExtGOF],
                              (2.0 * deltaMagWeightedGOF[noExtGOF] * innerTermGOF[noExtGOF]).astype(np.float64))

                    partialArray[self.fgcmPars.parLnPwvSlopeLoc +
                                 uNightIndexNoExt] /= units[self.fgcmPars.parLnPwvSlopeLoc +
                                                            uNightIndexNoExt]
                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parLnPwvSlopeLoc +
                                 uNightIndexNoExt] += 1

                    if useRefstars:
                        add_at_1d(partialArray[2*self.fgcmPars.nFitPars +
                                               self.fgcmPars.parLnPwvSlopeLoc:
                                                   (2*self.fgcmPars.nFitPars +
                                                    self.fgcmPars.parLnPwvSlopeLoc+
                                                    self.fgcmPars.nCampaignNights)],
                                  expNightIndexGROF[noExtGROF],
                                  (2.0 * deltaRefMagWeightedGROF[noExtGROF] *
                                  self.fgcmPars.expDeltaUT[obsExpIndexGO[goodRefObsGOF[noExtGROF]]] *
                                  dLdLnPwvGO[goodRefObsGOF[noExtGROF]]).astype(np.float64))

                        partialArray[2*self.fgcmPars.nFitPars +
                                     self.fgcmPars.parLnPwvSlopeLoc +
                                     uRefNightIndexNoExt] /= units[self.fgcmPars.parLnPwvSlopeLoc +
                                                                   uRefNightIndexNoExt]
                        partialArray[3*self.fgcmPars.nFitPars +
                                     self.fgcmPars.parLnPwvSlopeLoc +
                                     uRefNightIndexNoExt] += 1

                    # lnPwv Nightly Quadratic
                    if self.useQuadraticPwv:

                        dLdLnPwvQuadraticGOFI = self.fgcmPars.expDeltaUT[obsExpIndexGOF[indGOF]]**2. * dLdLnPwvGO[obsFitUseGOI]

                        sumNumerator = np.zeros((goodStars.size, self.fgcmPars.nBands, self.fgcmPars.nCampaignNights))
                        sumNumerator[:, :, :] = 0.0
                        add_at_3d(sumNumerator,
                                  (gsGOF[noExtGOFG],
                                   obsBandIndexGOFI[noExtGOFG],
                                   expNightIndexGOFI[noExtGOFG]),
                                  (dLdLnPwvQuadraticGOFI[noExtGOFG] /
                                   obsMagErr2GOFI[noExtGOFG]).astype(np.float64))

                        innerTermGOF = np.zeros(obsFitUseGO.size)
                        innerTermGOF[:] = 0.0
                        innerTermGOF[indGOF[noExtGOFG]] = (dLdLnPwvQuadraticGOFI[noExtGOFG] -
                                                           sumNumerator[gsGOF[noExtGOFG],
                                                                        obsBandIndexGOFI[noExtGOFG],
                                                                        expNightIndexGOFI[noExtGOFG]] *
                                                           objMagStdMeanErr2GO[obsFitUseGOI[noExtGOFG]])

                        add_at_1d(partialArray[self.fgcmPars.parLnPwvQuadraticLoc:
                                                   (self.fgcmPars.parLnPwvQuadraticLoc+
                                                    self.fgcmPars.nCampaignNights)],
                                  expNightIndexGOF[noExtGOF],
                                  (2.0 * deltaMagWeightedGOF[noExtGOF] * innerTermGOF[noExtGOF]).astype(np.float64))
                        partialArray[self.fgcmPars.parLnPwvQuadraticLoc +
                                     uNightIndexNoExt] /= units[self.fgcmPars.parLnPwvQuadraticLoc +
                                                                uNightIndexNoExt]
                        partialArray[self.fgcmPars.nFitPars +
                                     self.fgcmPars.parLnPwvQuadraticLoc +
                                     uNightIndexNoExt] += 1

                        if useRefstars:
                            add_at_1d(partialArray[2*self.fgcmPars.nFitPars +
                                                   self.fgcmPars.parLnPwvQuadraticLoc:
                                                       (2*self.fgcmPars.nFitPars +
                                                        self.fgcmPars.parLnPwvQuadraticLoc+
                                                        self.fgcmPars.nCampaignNights)],
                                      expNightIndexGROF[noExtGROF],
                                      (2.0 * deltaRefMagWeightedGROF[noExtGROF] *
                                      self.fgcmPars.expDeltaUT[obsExpIndexGO[goodRefObsGOF[noExtGROF]]]**2. *
                                      dLdLnPwvGO[goodRefObsGOF[noExtGROF]]).astype(np.float64))

                            partialArray[2*self.fgcmPars.nFitPars +
                                         self.fgcmPars.parLnPwvQuadraticLoc +
                                         uRefNightIndexNoExt] /= units[self.fgcmPars.parLnPwvQuadraticLoc +
                                                                       uRefNightIndexNoExt]
                            partialArray[3*self.fgcmPars.nFitPars +
                                         self.fgcmPars.parLnPwvQuadraticLoc +
                                         uRefNightIndexNoExt] += 1

                #############
                ## Tau External
                #############

                if (self.fgcmPars.hasExternalTau):
                    # NOT IMPLEMENTED PROPERLY YET

                    raise NotImplementedError("external tau not implemented.")

                ###########
                ## Tau No External
                ###########

                noExtGOF, = np.where(~self.fgcmPars.externalTauFlag[obsExpIndexGOF])
                uNightIndexNoExt = np.unique(expNightIndexGOF[noExtGOF])
                noExtGOFG, = np.where(~self.fgcmPars.externalTauFlag[obsExpIndexGOF[indGOF]])

                # lnTau Nightly Intercept

                sumNumerator[:, :, :] = 0.0
                add_at_3d(sumNumerator,
                          (gsGOF[noExtGOFG],
                           obsBandIndexGOFI[noExtGOFG],
                           expNightIndexGOFI[noExtGOFG]),
                          (dLdLnTauGO[obsFitUseGOI[noExtGOFG]] /
                          obsMagErr2GOFI[noExtGOFG]).astype(np.float64))

                innerTermGOF = np.zeros(obsFitUseGO.size)
                innerTermGOF[:] = 0.0
                innerTermGOF[indGOF[noExtGOFG]] = (dLdLnTauGO[obsFitUseGOI[noExtGOFG]] -
                                                   sumNumerator[gsGOF[noExtGOFG],
                                                                obsBandIndexGOFI[noExtGOFG],
                                                                expNightIndexGOFI[noExtGOFG]] *
                                                   objMagStdMeanErr2GO[obsFitUseGOI[noExtGOFG]])
                add_at_1d(partialArray[self.fgcmPars.parLnTauInterceptLoc:
                                           (self.fgcmPars.parLnTauInterceptLoc+
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF[noExtGOF],
                          (2.0 * deltaMagWeightedGOF[noExtGOF] * innerTermGOF[noExtGOF]).astype(np.float64))

                partialArray[self.fgcmPars.parLnTauInterceptLoc +
                             uNightIndexNoExt] /= units[self.fgcmPars.parLnTauInterceptLoc +
                                                        uNightIndexNoExt]
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parLnTauInterceptLoc +
                             uNightIndexNoExt] += 1

                if useRefstars:
                    noExtGROF, = np.where(~self.fgcmPars.externalTauFlag[obsExpIndexGO[goodRefObsGOF]])
                    uRefNightIndexNoExt = np.unique(expNightIndexGROF[noExtGROF])

                    add_at_1d(partialArray[2*self.fgcmPars.nFitPars +
                                           self.fgcmPars.parLnTauInterceptLoc:
                                               (2*self.fgcmPars.nFitPars +
                                                self.fgcmPars.parLnTauInterceptLoc+
                                                self.fgcmPars.nCampaignNights)],
                              expNightIndexGROF[noExtGROF],
                              (2.0 * deltaRefMagWeightedGROF[noExtGROF] *
                              dLdLnTauGO[goodRefObsGOF[noExtGROF]]).astype(np.float64))

                    partialArray[2*self.fgcmPars.nFitPars +
                                 self.fgcmPars.parLnTauInterceptLoc +
                                 uRefNightIndexNoExt] /= units[self.fgcmPars.parLnTauInterceptLoc +
                                                               uRefNightIndexNoExt]
                    partialArray[3*self.fgcmPars.nFitPars +
                                 self.fgcmPars.parLnTauInterceptLoc +
                                 uRefNightIndexNoExt] += 1

                # lnTau nightly slope

                dLdLnTauSlopeGOFI = self.fgcmPars.expDeltaUT[obsExpIndexGOF[indGOF]] * dLdLnTauGO[obsFitUseGOI]
                sumNumerator[:, :, :] = 0.0
                add_at_3d(sumNumerator,
                          (gsGOF[noExtGOFG],
                           obsBandIndexGOFI[noExtGOFG],
                           expNightIndexGOFI[noExtGOFG]),
                          (dLdLnTauSlopeGOFI[noExtGOFG] /
                           obsMagErr2GOFI[noExtGOFG]).astype(np.float64))

                innerTermGOF[:] = 0.0
                innerTermGOF[indGOF[noExtGOFG]] = (dLdLnTauSlopeGOFI[noExtGOFG] -
                                                   sumNumerator[gsGOF[noExtGOFG],
                                                                obsBandIndexGOFI[noExtGOFG],
                                                                expNightIndexGOFI[noExtGOFG]] *
                                                   objMagStdMeanErr2GO[obsFitUseGOI[noExtGOFG]])
                add_at_1d(partialArray[self.fgcmPars.parLnTauSlopeLoc:
                                           (self.fgcmPars.parLnTauSlopeLoc+
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF[noExtGOF],
                          (2.0 * deltaMagWeightedGOF[noExtGOF] * innerTermGOF[noExtGOF]).astype(np.float64))

                partialArray[self.fgcmPars.parLnTauSlopeLoc +
                             uNightIndexNoExt] /= units[self.fgcmPars.parLnTauSlopeLoc +
                                                        uNightIndexNoExt]
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parLnTauSlopeLoc +
                             uNightIndexNoExt] += 1

                if useRefstars:
                    add_at_1d(partialArray[2*self.fgcmPars.nFitPars +
                                           self.fgcmPars.parLnTauSlopeLoc:
                                               (2*self.fgcmPars.nFitPars +
                                                self.fgcmPars.parLnTauSlopeLoc+
                                                self.fgcmPars.nCampaignNights)],
                              expNightIndexGROF[noExtGROF],
                              (2.0 * deltaRefMagWeightedGROF[noExtGROF] *
                              self.fgcmPars.expDeltaUT[obsExpIndexGO[goodRefObsGOF[noExtGROF]]] *
                              dLdLnTauGO[goodRefObsGOF[noExtGROF]]).astype(np.float64))

                    partialArray[2*self.fgcmPars.nFitPars +
                                 self.fgcmPars.parLnTauSlopeLoc +
                                 uRefNightIndexNoExt] /= units[self.fgcmPars.parLnTauSlopeLoc +
                                                               uRefNightIndexNoExt]
                    partialArray[3*self.fgcmPars.nFitPars +
                                 self.fgcmPars.parLnTauSlopeLoc +
                                 uRefNightIndexNoExt] += 1

            ##################
            ## Washes (QE Sys)
            ##################

            # Note that we do this derivative even if we've frozen the atmosphere.

            expWashIndexGOF = self.fgcmPars.expWashIndex[obsExpIndexGOF]

            # Wash Intercept

            if self.instrumentParsPerBand:
                # We have per-band intercepts
                # Non-fit bands will be given the mean of the others (in fgcmParameters),
                # because they aren't in the chi2.
                uWashBandIndex = np.unique(np.ravel_multi_index((obsBandIndexGOF,
                                                                 expWashIndexGOF),
                                                                self.fgcmPars.parQESysIntercept.shape))
                ravelIndexGOF = np.ravel_multi_index((obsBandIndexGOF,
                                                      expWashIndexGOF),
                                                     self.fgcmPars.parQESysIntercept.shape)

                sumNumerator = np.zeros((goodStars.size, self.fgcmPars.nBands, self.fgcmPars.parQESysIntercept.size))
                add_at_3d(sumNumerator,
                          (gsGOF, obsBandIndexGOFI, ravelIndexGOF[indGOF]),
                          (1.0 / obsMagErr2GOFI).astype(np.float64))

                innerTermGOF[:] = 0.0
                innerTermGOF[indGOF] = (1.0 - sumNumerator[gsGOF,
                                                           obsBandIndexGOFI,
                                                           ravelIndexGOF[indGOF]] *
                                        objMagStdMeanErr2GO[obsFitUseGOI])

                add_at_1d(partialArray[self.fgcmPars.parQESysInterceptLoc:
                                           (self.fgcmPars.parQESysInterceptLoc +
                                            self.fgcmPars.parQESysIntercept.size)],
                          ravelIndexGOF,
                          (2.0 * deltaMagWeightedGOF * innerTermGOF).astype(np.float64))

                partialArray[self.fgcmPars.parQESysInterceptLoc +
                             uWashBandIndex] /= units[self.fgcmPars.parQESysInterceptLoc +
                                                      uWashBandIndex]
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parQESysInterceptLoc +
                             uWashBandIndex] += 1

                # And reference stars
                if useRefstars and self.useRefStarsWithInstrument:
                    add_at_1d(partialArray[2*self.fgcmPars.nFitPars +
                                           self.fgcmPars.parQESysInterceptLoc:
                                               (2*self.fgcmPars.nFitPars +
                                                self.fgcmPars.parQESysInterceptLoc +
                                                self.fgcmPars.parQESysIntercept.size)],
                              np.ravel_multi_index((obsBandIndexGO[goodRefObsGOF],
                                                    esutil.numpy_util.to_native(self.fgcmPars.expWashIndex[obsExpIndexGO[goodRefObsGOF]])),
                                                   self.fgcmPars.parQESysIntercept.shape),
                              (2.0 * deltaRefMagWeightedGROF).astype(np.float64))

                    partialArray[2*self.fgcmPars.nFitPars +
                                 self.fgcmPars.parQESysInterceptLoc +
                                 uWashBandIndex] /= units[self.fgcmPars.parQESysInterceptLoc +
                                                          uWashBandIndex]
                    partialArray[3*self.fgcmPars.nFitPars +
                                 self.fgcmPars.parQESysInterceptLoc +
                                 uWashBandIndex] += 1

            else:
                # We have one gray mirror term for all bands
                uWashIndex = np.unique(expWashIndexGOF)

                sumNumerator = np.zeros((goodStars.size, self.fgcmPars.nBands, self.fgcmPars.nWashIntervals))
                add_at_3d(sumNumerator,
                          (gsGOF, obsBandIndexGOFI, expWashIndexGOF[indGOF]),
                          (1.0 / obsMagErr2GOFI).astype(np.float64))

                innerTermGOF[:] = 0.0
                innerTermGOF[indGOF] = (1.0 - sumNumerator[gsGOF,
                                                           obsBandIndexGOFI,
                                                           expWashIndexGOF[indGOF]] *
                                        objMagStdMeanErr2GO[obsFitUseGOI])
                add_at_1d(partialArray[self.fgcmPars.parQESysInterceptLoc:
                                           (self.fgcmPars.parQESysInterceptLoc +
                                            self.fgcmPars.nWashIntervals)],
                          expWashIndexGOF,
                          (2.0 * deltaMagWeightedGOF * innerTermGOF).astype(np.float64))

                partialArray[self.fgcmPars.parQESysInterceptLoc +
                             uWashIndex] /= units[self.fgcmPars.parQESysInterceptLoc +
                                                  uWashIndex]
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parQESysInterceptLoc +
                             uWashIndex] += 1

                # We don't want to use the reference stars for the wash intercept
                # or slope by default because if they aren't evenly sampled it can
                # cause the fitter to go CRAZY.  Though improvements in slope computation
                # means this should be revisited.
                if useRefstars and self.useRefStarsWithInstrument:
                    add_at_1d(partialArray[2*self.fgcmPars.nFitPars +
                                           self.fgcmPars.parQESysInterceptLoc:
                                               (2*self.fgcmPars.nFitPars +
                                                self.fgcmPars.parQESysInterceptLoc +
                                                self.fgcmPars.nWashIntervals)],
                              esutil.numpy_util.to_native(self.fgcmPars.expWashIndex[obsExpIndexGO[goodRefObsGOF]]),
                              (2.0 * deltaRefMagWeightedGROF).astype(np.float64))

                    partialArray[2*self.fgcmPars.nFitPars +
                                 self.fgcmPars.parQESysInterceptLoc +
                                 uWashIndex] /= units[self.fgcmPars.parQESysInterceptLoc +
                                                      uWashIndex]
                    partialArray[3*self.fgcmPars.nFitPars +
                                 self.fgcmPars.parQESysInterceptLoc +
                                 uWashIndex] += 1

            #################
            ## Filter offset
            #################

            sumNumerator = np.zeros((goodStars.size, self.fgcmPars.nBands, self.fgcmPars.nLUTFilter))
            add_at_3d(sumNumerator,
                      (gsGOF, obsBandIndexGOFI, obsLUTFilterIndexGO[obsFitUseGOI]),
                      (1.0 / obsMagErr2GOFI).astype(np.float64))

            innerTermGOF[:] = 0.0
            innerTermGOF[indGOF] = (1.0 - sumNumerator[gsGOF,
                                                       obsBandIndexGOFI,
                                                       obsLUTFilterIndexGO[obsFitUseGOI]] *
                                    objMagStdMeanErr2GO[obsFitUseGOI])

            add_at_1d(partialArray[self.fgcmPars.parFilterOffsetLoc:
                                       (self.fgcmPars.parFilterOffsetLoc +
                                        self.fgcmPars.nLUTFilter)],
                      obsLUTFilterIndexGO[obsFitUseGO],
                      (2.0 * deltaMagWeightedGOF * innerTermGOF).astype(np.float64))
            partialArray[self.fgcmPars.parFilterOffsetLoc:
                             (self.fgcmPars.parFilterOffsetLoc +
                              self.fgcmPars.nLUTFilter)] /= units[self.fgcmPars.parFilterOffsetLoc:
                                                                      (self.fgcmPars.parFilterOffsetLoc +
                                                                       self.fgcmPars.nLUTFilter)]

            # Note that using the refstars with the filter offset derivative
            # seems to make things go haywire, so don't do that.

            # Now set those to zero the derivatives we aren't using
            partialArray[self.fgcmPars.parFilterOffsetLoc:
                             (self.fgcmPars.parFilterOffsetLoc +
                              self.fgcmPars.nLUTFilter)][~self.fgcmPars.parFilterOffsetFitFlag] = 0.0
            uOffsetIndex, = np.where(self.fgcmPars.parFilterOffsetFitFlag)
            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parFilterOffsetLoc +
                         uOffsetIndex] += 1

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
