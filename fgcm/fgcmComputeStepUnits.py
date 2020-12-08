import numpy as np
import os
import sys
import esutil
import time

from .fgcmUtilities import objFlagDict

from .fgcmNumbaUtilities import numba_test, add_at_1d, add_at_2d, add_at_3d

import multiprocessing

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm


class FgcmComputeStepUnits(object):
    """
    Class which computes the step units for each parameter.

    Parameters
    ----------
    fgcmConfig: FgcmConfig
       Config object
    fgcmPars: FgcmParameters
       Parameter object
    fgcmStars: FgcmStars
       Stars object
    fgcmLUT: FgcmLUT
       LUT object
    """

    def __init__(self, fgcmConfig, fgcmPars, fgcmStars, fgcmLUT):
        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.debug('Initializing FgcmComputeStepUnits')

        # does this need to be shm'd?
        self.fgcmPars = fgcmPars

        # this is shm'd
        self.fgcmLUT = fgcmLUT

        # also shm'd
        self.fgcmStars = fgcmStars

        self.nCore = fgcmConfig.nCore
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.nStarPerRun = fgcmConfig.nStarPerRun
        self.noChromaticCorrections = fgcmConfig.noChromaticCorrections
        self.bandFitIndex = fgcmConfig.bandFitIndex
        self.useQuadraticPwv = fgcmConfig.useQuadraticPwv
        self.freezeStdAtmosphere = fgcmConfig.freezeStdAtmosphere
        self.ccdGraySubCCD = fgcmConfig.ccdGraySubCCD
        self.ccdOffsets = fgcmConfig.ccdOffsets
        self.useRefStarsWithInstrument = fgcmConfig.useRefStarsWithInstrument
        self.instrumentParsPerBand = fgcmConfig.instrumentParsPerBand
        self.stepUnitReference = fgcmConfig.stepUnitReference
        self.fitGradientTolerance = fgcmConfig.fitGradientTolerance
        self.saveParsForDebugging = fgcmConfig.saveParsForDebugging
        self.quietMode = fgcmConfig.quietMode

        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle

        # these are the standard *band* I10s
        self.I10StdBand = fgcmConfig.I10StdBand

        self.illegalValue = fgcmConfig.illegalValue

        self.maxParStepFraction = 0.1 # hard code this

        if (fgcmConfig.useSedLUT and self.fgcmLUT.hasSedLUT):
            self.useSedLUT = True
        else:
            self.useSedLUT = False

        numba_test(0)

    def run(self, fitParams):
        """
        Compute step units for all parameters

        Parameters
        ----------
        fitParams: numpy array of floats
           Array with the numerical values of the parameters (properly formatted).
        """
        startTime = time.time()

        self.fgcmPars.reloadParArray(fitParams, fitterUnits=False)
        self.fgcmPars.parsToExposures()

        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=False)
        if not self.quietMode:
            self.fgcmLog.info('Found %d good stars for step units' % (goodStars.size))

        if (goodStars.size == 0):
            raise RuntimeError("No good stars to fit!")

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        expFlag = self.fgcmPars.expFlag

        goodStarsSub, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=expFlag,
                                                                 checkBadMag=True)

        self.nSums = 1 # nobs
        # 0: nFitPars -> derivative for step calculation
        # nFitPars: 2*nFitPars -> parameters which are touched
        # Note that we only need nobs, not ref because we assume all observations are
        # going to have one or the other, and this doesn't care which is which
        self.nSums += 2 * self.fgcmPars.nFitPars

        mp_ctx = multiprocessing.get_context("fork")
        proc = mp_ctx.Process()
        workerIndex = proc._identity[0]+1
        proc = None

        self.totalHandleDict = {}
        for thisCore in range(self.nCore):
            self.totalHandleDict[workerIndex + thisCore] = (
                snmm.createArray(self.nSums,dtype='f8'))

        self._testing = workerIndex

        nSections = goodStars.size // self.nStarPerRun + 1
        goodStarsList = np.array_split(goodStars,nSections)

        splitValues = np.zeros(nSections-1,dtype='i4')
        for i in range(1,nSections):
            splitValues[i-1] = goodStarsList[i][0]

        splitIndices = np.searchsorted(goodStars[goodStarsSub], splitValues)

        # and split along the indices
        goodObsList = np.split(goodObs,splitIndices)

        workerList = list(zip(goodStarsList,goodObsList))

        # reverse sort so the longest running go first
        workerList.sort(key=lambda elt:elt[1].size, reverse=True)

        # make a pool
        pool = mp_ctx.Pool(processes=self.nCore)
        # Compute magnitudes
        pool.map(self._stepWorker, workerList, chunksize=1)

        pool.close()
        pool.join()

        # sum up the partial sums from the different jobs
        partialSums = np.zeros(self.nSums,dtype='f8')
        for thisCore in range(self.nCore):
            partialSums[:] += snmm.getArray(
                self.totalHandleDict[workerIndex + thisCore])[:]

        nonZero, = np.where((partialSums[self.fgcmPars.nFitPars: 2*self.fgcmPars.nFitPars] > 0) &
                            (partialSums[0: self.fgcmPars.nFitPars] != 0.0))
        nActualFitPars = nonZero.size

        # Get the number of degrees of freedom
        fitDOF = partialSums[-1] - float(nActualFitPars)

        dChisqdPNZ = partialSums[nonZero] / fitDOF

        # default step is 1.0
        self.fgcmPars.stepUnits[:] = 1.0

        # And the actual step size for good pars
        self.fgcmPars.stepUnits[nonZero] = np.abs(dChisqdPNZ) / self.fitGradientTolerance

        if not self.freezeStdAtmosphere:
            # This provides a bit more wiggle room when fitting lots of parameters
            self.fgcmPars.stepUnits[nonZero] /= (2.0 * nActualFitPars)

        # And reset to median value for each class of steps

        # O3
        vals = self.fgcmPars.stepUnits[self.fgcmPars.parO3Loc:
                                           (self.fgcmPars.parO3Loc +
                                            self.fgcmPars.nCampaignNights)]
        self.fgcmPars.stepUnits[self.fgcmPars.parO3Loc:
                                    (self.fgcmPars.parO3Loc +
                                     self.fgcmPars.nCampaignNights)] = np.median(vals)
        # Alpha
        vals = self.fgcmPars.stepUnits[self.fgcmPars.parAlphaLoc:
                                           (self.fgcmPars.parAlphaLoc +
                                            self.fgcmPars.nCampaignNights)]
        self.fgcmPars.stepUnits[self.fgcmPars.parAlphaLoc:
                                    (self.fgcmPars.parAlphaLoc +
                                     self.fgcmPars.nCampaignNights)] = np.median(vals)
        # lnTau Intercept
        vals = self.fgcmPars.stepUnits[self.fgcmPars.parLnTauInterceptLoc:
                                          (self.fgcmPars.parLnTauInterceptLoc +
                                           self.fgcmPars.nCampaignNights)]
        self.fgcmPars.stepUnits[self.fgcmPars.parLnTauInterceptLoc:
                                          (self.fgcmPars.parLnTauInterceptLoc +
                                           self.fgcmPars.nCampaignNights)] = np.median(vals)
        # lnTau Slope
        vals = self.fgcmPars.stepUnits[self.fgcmPars.parLnTauSlopeLoc:
                                          (self.fgcmPars.parLnTauSlopeLoc +
                                           self.fgcmPars.nCampaignNights)]
        self.fgcmPars.stepUnits[self.fgcmPars.parLnTauSlopeLoc:
                                          (self.fgcmPars.parLnTauSlopeLoc +
                                           self.fgcmPars.nCampaignNights)] = np.median(vals)
        # lnPwv Intercept
        vals = self.fgcmPars.stepUnits[self.fgcmPars.parLnPwvInterceptLoc:
                                          (self.fgcmPars.parLnPwvInterceptLoc +
                                           self.fgcmPars.nCampaignNights)]
        self.fgcmPars.stepUnits[self.fgcmPars.parLnPwvInterceptLoc:
                                          (self.fgcmPars.parLnPwvInterceptLoc +
                                           self.fgcmPars.nCampaignNights)] = np.median(vals)
        # lnPwv Slope
        vals = self.fgcmPars.stepUnits[self.fgcmPars.parLnPwvSlopeLoc:
                                          (self.fgcmPars.parLnPwvSlopeLoc +
                                           self.fgcmPars.nCampaignNights)]
        self.fgcmPars.stepUnits[self.fgcmPars.parLnPwvSlopeLoc:
                                          (self.fgcmPars.parLnPwvSlopeLoc +
                                           self.fgcmPars.nCampaignNights)] = np.median(vals)
        # lnPwv Quadratic (if necessary)
        if self.useQuadraticPwv:
            vals = self.fgcmPars.stepUnits[self.fgcmPars.parLnPwvQuadraticLoc:
                                          (self.fgcmPars.parLnPwvQuadraticLoc +
                                           self.fgcmPars.nCampaignNights)]
            self.fgcmPars.stepUnits[self.fgcmPars.parLnPwvQuadraticLoc:
                                          (self.fgcmPars.parLnPwvQuadraticLoc +
                                           self.fgcmPars.nCampaignNights)] = np.median(vals)

        # retrieved pwv nightly (if necessary)
        if self.fgcmPars.hasExternalPwv and not self.fgcmPars.useRetrievedPwv:
            vals = self.fgcmPars.stepUnits[self.fgcmPars.parExternalLnPwvOffsetLoc:
                                               (self.fgcmPars.parExternalLnPwvOffsetLoc+
                                                self.fgcmPars.nCampaignNights)]
            self.fgcmPars.stepUnits[self.fgcmPars.parExternalLnPwvOffsetLoc:
                                        (self.fgcmPars.parExternalLnPwvOffsetLoc+
                                         self.fgcmPars.nCampaignNights)] = np.median(vals)
        if self.fgcmPars.useRetrievedPwv:
            if self.fgcmPars.useNightlyRetrievedPwv:
                vals = self.fgcmPars.stepUnits[self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc:
                                                   (self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc+
                                                    self.fgcmPars.nCampaignNights)]
                self.fgcmPars.stepUnits[self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc:
                                            (self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc+
                                             self.fgcmPars.nCampaignNights)] = np.median(vals)

        """
        vals = self.fgcmPars.stepUnits[self.fgcmPars.parQESysInterceptLoc:
                                           (self.fgcmPars.parQESysInterceptLoc +
                                            self.fgcmPars.nWashIntervals)]
        self.fgcmPars.stepUnits[self.fgcmPars.parQESysInterceptLoc:
                                           (self.fgcmPars.parQESysInterceptLoc +
                                            self.fgcmPars.nWashIntervals)] = np.median(vals)
                                            """

        if self.saveParsForDebugging:
            import astropy.io.fits as pyfits
            tempCat = np.zeros(1, dtype=[('o3', 'f8', self.fgcmPars.nCampaignNights),
                                         ('lnTauIntercept', 'f8', self.fgcmPars.nCampaignNights),
                                         ('lnTauSlope', 'f8', self.fgcmPars.nCampaignNights),
                                         ('alpha', 'f8', self.fgcmPars.nCampaignNights),
                                         ('lnPwvIntercept', 'f8', self.fgcmPars.nCampaignNights),
                                         ('lnPwvSlope', 'f8', self.fgcmPars.nCampaignNights),
                                         ('lnPwvQuadratic', 'f8', self.fgcmPars.nCampaignNights),
                                         ('qeSysIntercept', 'f8', self.fgcmPars.nWashIntervals)])
            tempCat['o3'][0][:] = self.fgcmPars.stepUnits[self.fgcmPars.parO3Loc:
                                                              (self.fgcmPars.parO3Loc +
                                                               self.fgcmPars.nCampaignNights)]
            tempCat['lnTauIntercept'][0][:] = self.fgcmPars.stepUnits[self.fgcmPars.parLnTauInterceptLoc:
                                                                          (self.fgcmPars.parLnTauInterceptLoc +
                                                                           self.fgcmPars.nCampaignNights)]
            tempCat['lnTauSlope'][0][:] = self.fgcmPars.stepUnits[self.fgcmPars.parLnTauSlopeLoc:
                                                                          (self.fgcmPars.parLnTauSlopeLoc +
                                                                           self.fgcmPars.nCampaignNights)]
            tempCat['alpha'][0][:] = self.fgcmPars.stepUnits[self.fgcmPars.parAlphaLoc:
                                                              (self.fgcmPars.parAlphaLoc +
                                                               self.fgcmPars.nCampaignNights)]
            tempCat['lnPwvIntercept'][0][:] = self.fgcmPars.stepUnits[self.fgcmPars.parLnPwvInterceptLoc:
                                                                          (self.fgcmPars.parLnPwvInterceptLoc +
                                                                           self.fgcmPars.nCampaignNights)]
            tempCat['lnPwvSlope'][0][:] = self.fgcmPars.stepUnits[self.fgcmPars.parLnPwvSlopeLoc:
                                                                      (self.fgcmPars.parLnPwvSlopeLoc +
                                                                       self.fgcmPars.nCampaignNights)]
            tempCat['lnPwvQuadratic'][0][:] = self.fgcmPars.stepUnits[self.fgcmPars.parLnPwvQuadraticLoc:
                                                                      (self.fgcmPars.parLnPwvQuadraticLoc +
                                                                       self.fgcmPars.nCampaignNights)]
            tempCat['qeSysIntercept'][0][:] = self.fgcmPars.stepUnits[self.fgcmPars.parQESysInterceptLoc:
                                                                          (self.fgcmPars.parQESysInterceptLoc +
                                                                           self.fgcmPars.nWashIntervals)]

            pyfits.writeto('%s_stepUnits3.fits' % (self.outfileBaseWithCycle), tempCat, overwrite=True)

        # free shared arrays
        for key in self.totalHandleDict.keys():
            snmm.freeArray(self.totalHandleDict[key])

        if not self.quietMode:
            self.fgcmLog.info('Step size computation took %.2f seconds.' %
                              (time.time() - startTime))

    def _stepWorker(self, goodStarsAndObs):
        """
        Multiprocessing worker to compute fake derivatives for FgcmComputeStepUnits.
        Not to be called on its own.

        Parameters
        ----------
        goodStarsAndObs: tuple[2]
           (goodStars, goodObs)
        """

        goodStars = goodStarsAndObs[0]
        goodObs = goodStarsAndObs[1]

        thisCore = multiprocessing.current_process()._identity[0]

        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsLUTFilterIndex = snmm.getArray(self.fgcmStars.obsLUTFilterIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)
        obsSecZenith = snmm.getArray(self.fgcmStars.obsSecZenithHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)

        # cut these down now, faster later
        obsObjIDIndexGO = esutil.numpy_util.to_native(obsObjIDIndex[goodObs])
        obsBandIndexGO = esutil.numpy_util.to_native(obsBandIndex[goodObs])
        obsLUTFilterIndexGO = esutil.numpy_util.to_native(obsLUTFilterIndex[goodObs])
        obsExpIndexGO = esutil.numpy_util.to_native(obsExpIndex[goodObs])
        obsCCDIndexGO = esutil.numpy_util.to_native(obsCCDIndex[goodObs])

        obsSecZenithGO = obsSecZenith[goodObs]
        objMagStdMeanErr2GO = objMagStdMeanErr[obsObjIDIndexGO, obsBandIndexGO]**2.

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

        obsMagErr2GO = obsMagADUModelErr[goodObs]**2.

        # Note that we don't care if something is a refstar or not for this

        # Default mask is not to mask
        maskGO = np.ones(goodObs.size, dtype=np.bool)

        # which observations are actually used in the fit?
        useGO, = np.where(maskGO)
        _, obsFitUseGO = esutil.numpy_util.match(self.bandFitIndex,
                                                 obsBandIndexGO[useGO])
        obsFitUseGO = useGO[obsFitUseGO]

        deltaMagGO = np.zeros(goodObs.size) + self.stepUnitReference
        obsWeightGO = 1. / obsMagErr2GO
        deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

        gsGOF, indGOF = esutil.numpy_util.match(goodStars, obsObjIDIndexGO[obsFitUseGO])

        partialArray = np.zeros(self.nSums, dtype='f8')
        partialArray[-1] = obsFitUseGO.size

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

        # Set up temporary storage and sub-indexed arrays
        sumNumerator = np.zeros((goodStars.size, self.fgcmPars.nBands, self.fgcmPars.nCampaignNights))
        innerTermGOF = np.zeros(obsFitUseGO.size)

        obsExpIndexGOF = obsExpIndexGO[obsFitUseGO]
        expNightIndexGOF = esutil.numpy_util.to_native(self.fgcmPars.expNightIndex[obsExpIndexGOF])
        obsBandIndexGOF = obsBandIndexGO[obsFitUseGO]
        obsBandIndexGOFI = obsBandIndexGOF[indGOF]
        expNightIndexGOFI = expNightIndexGOF[indGOF]
        obsFitUseGOI = obsFitUseGO[indGOF]
        obsMagErr2GOFI = obsMagErr2GO[obsFitUseGO[indGOF]]

        ##########
        ## O3
        ##########

        uNightIndex = np.unique(expNightIndexGOF)

        obsBandIndexGOFI = obsBandIndexGOF[indGOF]
        expNightIndexGOFI = expNightIndexGOF[indGOF]

        sumNumerator[:, :, :] = 0.0
        add_at_3d(sumNumerator,
                  (gsGOF, obsBandIndexGOFI, expNightIndexGOFI),
                  dLdO3GO[obsFitUseGOI] / obsMagErr2GOFI)

        innerTermGOF[:] = 0.0
        innerTermGOF[indGOF] = (dLdO3GO[obsFitUseGOI] -
                                sumNumerator[gsGOF, obsBandIndexGOFI, expNightIndexGOFI] * objMagStdMeanErr2GO[obsFitUseGOI])

        add_at_1d(partialArray[self.fgcmPars.parO3Loc:
                                   (self.fgcmPars.parO3Loc +
                                    self.fgcmPars.nCampaignNights)],
                  expNightIndexGOF,
                  2.0 * deltaMagWeightedGOF * innerTermGOF)

        partialArray[self.fgcmPars.nFitPars +
                     self.fgcmPars.parO3Loc +
                     uNightIndex] += 1

        ###########
        ## Alpha
        ###########

        # Note that expNightIndexGOF, obsBandIndexGOF are the same as above

        sumNumerator[:, :, :] =  0.0
        add_at_3d(sumNumerator,
                  (gsGOF, obsBandIndexGOFI, expNightIndexGOFI),
                  dLdAlphaGO[obsFitUseGOI] / obsMagErr2GOFI)

        innerTermGOF[:] = 0.0
        innerTermGOF[indGOF] = (dLdAlphaGO[obsFitUseGOI] -
                                sumNumerator[gsGOF, obsBandIndexGOFI, expNightIndexGOFI] * objMagStdMeanErr2GO[obsFitUseGOI])

        add_at_1d(partialArray[self.fgcmPars.parAlphaLoc:
                                   (self.fgcmPars.parAlphaLoc+
                                    self.fgcmPars.nCampaignNights)],
                  expNightIndexGOF,
                  2.0 * deltaMagWeightedGOF * innerTermGOF)

        partialArray[self.fgcmPars.nFitPars +
                     self.fgcmPars.parAlphaLoc +
                     uNightIndex] += 1

        ###########
        ## PWV External
        ###########

        if (self.fgcmPars.hasExternalPwv and not self.fgcmPars.useRetrievedPwv):
            hasExtGOF, = np.where(self.fgcmPars.externalPwvFlag[obsExpIndexGOF])
            uNightIndexHasExt = np.unique(expNightIndexGOF[hasExtGOF])
            hasExtGOFG, = np.where(~self.fgcmPars.externalPwvFlag[obsExpIndexGOF[indGOF]])

            # lnPwv Nightly Offset

            sumNumerator[:, :, :] = 0.0
            add_at_3d(sumNumerator,
                      (gsGOF[hasExtGOFG],
                       obsBandIndexGOFI[hasExtGOFG],
                       expNightIndexGOFI[hasExtGOFG]),
                      dLdLnPwvGO[obsFitUseGOI[hasExtGOFG]] /
                      obsMagErr2GOFI[hasExtGOFG])

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
                      2.0 * deltaMagWeightedGOF[hasExtGOF] * innerTermGOF[hasExtGOF])

            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parExternalLnPwvOffsetLoc +
                         uNightIndexHasExt] += 1

            # lnPwv Global Scale

            # NOTE: this may be wrong.  Needs thought.

            partialArray[self.fgcmPars.parExternalLnPwvScaleLoc] = 2.0 * (
                np.sum(deltaMagWeightedGOF[hasExtGOF] * (
                        self.fgcmPars.expLnPwv[obsExpIndexGOF[hasExtGOF]] *
                        dLdLnPwvGO[obsFitUseGO[hasExtGOF]])))

            partialArray[self.fgcmPars.nFitPars +
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

                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parRetrievedLnPwvScaleLoc] += 1

                if self.fgcmPars.useNightlyRetrievedPwv:
                    # lnPwv Retrieved Nightly Offset

                    uNightIndexHasRetrievedPwv = np.unique(expNightIndexGOF[hasRetrievedPwvGOF])

                    sumNumerator[:, :, :] = 0.0
                    add_at_3d(sumNumerator,
                              (gsGOF[hasRetrievedPwvGOFG],
                               obsBandIndexGOFI[hasRetrievedPwvGOFG],
                               expNightIndexGOFI[hasRetrievedPwvGOFG]),
                              dLdLnPwvGO[obsFitUseGOI[hasRetrievedPwvGOFG]] /
                              obsMagErr2GOFI[hasRetrievedPwvGOFG])

                    innerTermGOF[:] = 0.0
                    innerTermGOF[indGOF[hasRetrievedPwvGOFG]] = (dLdLnPwvGO[obsFitUseGOI[hasRetrievedPwvGOFG]] -
                                                       sumNumerator[gsGOF[hasRetrievedPwvGOFG],
                                                                    obsBandIndexGOFI[hasRetrievedPwvGOFG],
                                                                    expNightIndexGOFI[hasRetrievedPwvGOFG]] *
                                                       objMagStdMeanErr2GO[obsFitUseGOI[hasRetrievedPwvGOFG]])

                    add_at_1d(partialArray[self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc:
                                               (self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc+
                                                self.fgcmPars.nCampaignNights)],
                              2.0 * deltaMagWeightedGOF[hasRetrievedPwvGOF] * innerTermGOF[hasRetrievedPwvGOF])

                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc +
                                 uNightIndexHasRetrievedPwv] += 1

                else:
                    # lnPwv Retrieved Global Offset

                    # This may be wrong

                    partialArray[self.fgcmPars.parRetrievedLnPwvOffsetLoc] = 2.0 * (
                        np.sum(deltaMagWeightedGOF[hasRetrievedPwvGOF] * (
                                dLdLnPwvGO[obsFitUseGO[hasRetrievedPwvGOF]])))

                    partialArray[self.fgcmPars.nFitPars +
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
                      dLdLnPwvGO[obsFitUseGOI[noExtGOFG]] /
                      obsMagErr2GOFI[noExtGOFG])

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
                      2.0 * deltaMagWeightedGOF[noExtGOF] * innerTermGOF[noExtGOF])

            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parLnPwvInterceptLoc +
                         uNightIndexNoExt] += 1

            # lnPwv Nightly Slope

            dLdLnPwvSlopeGOFI = self.fgcmPars.expDeltaUT[obsExpIndexGOF[indGOF]] * dLdLnPwvGO[obsFitUseGOI]

            sumNumerator[:, :, :] = 0.0
            add_at_3d(sumNumerator,
                      (gsGOF[noExtGOFG],
                       obsBandIndexGOFI[noExtGOFG],
                       expNightIndexGOFI[noExtGOFG]),
                      (dLdLnPwvSlopeGOFI[noExtGOFG] /
                       obsMagErr2GOFI[noExtGOFG]))

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
                      2.0 * deltaMagWeightedGOF[noExtGOF] * innerTermGOF[noExtGOF])

            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parLnPwvSlopeLoc +
                         uNightIndexNoExt] += 1

            # lnPwv Nightly Quadratic
            if self.useQuadraticPwv:

                dLdLnPwvQuadraticGOFI = self.fgcmPars.expDeltaUT[obsExpIndexGOF[indGOF]]**2. * dLdLnPwvGO[obsFitUseGOI]

                sumNumerator[:, :, :] = 0.0
                add_at_3d(sumNumerator,
                          (gsGOF[noExtGOFG],
                           obsBandIndexGOFI[noExtGOFG],
                           expNightIndexGOFI[noExtGOFG]),
                          (dLdLnPwvQuadraticGOFI[noExtGOFG] /
                           obsMagErr2GOFI[noExtGOFG]))

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
                          2.0 * deltaMagWeightedGOF[noExtGOF] * innerTermGOF[noExtGOF])

                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parLnPwvQuadraticLoc +
                             uNightIndexNoExt] += 1

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
                  dLdLnTauGO[obsFitUseGOI[noExtGOFG]] /
                  obsMagErr2GOFI[noExtGOFG])

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
                  2.0 * deltaMagWeightedGOF[noExtGOF] * innerTermGOF[noExtGOF])

        partialArray[self.fgcmPars.nFitPars +
                     self.fgcmPars.parLnTauInterceptLoc +
                     uNightIndexNoExt] += 1

        # lnTau nightly slope

        dLdLnTauSlopeGOFI = self.fgcmPars.expDeltaUT[obsExpIndexGOF[indGOF]] * dLdLnTauGO[obsFitUseGOI]

        sumNumerator[:, :, :] = 0.0
        add_at_3d(sumNumerator,
                  (gsGOF[noExtGOFG],
                   obsBandIndexGOFI[noExtGOFG],
                   expNightIndexGOFI[noExtGOFG]),
                  (dLdLnTauSlopeGOFI[noExtGOFG] /
                   obsMagErr2GOFI[noExtGOFG]))

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
                  2.0 * deltaMagWeightedGOF[noExtGOF] * innerTermGOF[noExtGOF])

        partialArray[self.fgcmPars.nFitPars +
                     self.fgcmPars.parLnTauSlopeLoc +
                     uNightIndexNoExt] += 1

        ##################
        ## Washes (QE Sys)
        ##################

        # The washes don't need to worry about the limits ... 0.1 mag is 0.1 mag here.

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
                      1.0 / obsMagErr2GOFI)

            innerTermGOF[:] = 0.0
            innerTermGOF[indGOF] = (1.0 - sumNumerator[gsGOF,
                                                       obsBandIndexGOFI,
                                                       ravelIndexGOF[indGOF]] *
                                    objMagStdMeanErr2GO[obsFitUseGOI])

            add_at_1d(partialArray[self.fgcmPars.parQESysInterceptLoc:
                                       (self.fgcmPars.parQESysInterceptLoc +
                                        self.fgcmPars.parQESysIntercept.size)],
                      ravelIndexGOF,
                      2.0 * deltaMagWeightedGOF * innerTermGOF)

            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parQESysInterceptLoc +
                         uWashBandIndex] += 1

        else:
            # We have one gray mirror term for all bands
            uWashIndex = np.unique(expWashIndexGOF)
            #t = time.time()
            sumNumerator = np.zeros((goodStars.size, self.fgcmPars.nBands, self.fgcmPars.nWashIntervals))
            add_at_3d(sumNumerator,
                      (gsGOF, obsBandIndexGOFI, expWashIndexGOF[indGOF]),
                      1.0 / obsMagErr2GOFI)

            innerTermGOF[:] = 0.0
            innerTermGOF[indGOF] = (1.0 - sumNumerator[gsGOF,
                                                       obsBandIndexGOFI,
                                                       expWashIndexGOF[indGOF]] *
                                    objMagStdMeanErr2GO[obsFitUseGOI])

            add_at_1d(partialArray[self.fgcmPars.parQESysInterceptLoc:
                                       (self.fgcmPars.parQESysInterceptLoc +
                                        self.fgcmPars.nWashIntervals)],
                      expWashIndexGOF,
                      2.0 * deltaMagWeightedGOF * innerTermGOF)

            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parQESysInterceptLoc +
                         uWashIndex] += 1

        #################
        ## Filter offset
        #################

        sumNumerator = np.zeros((goodStars.size, self.fgcmPars.nBands, self.fgcmPars.nLUTFilter))
        add_at_3d(sumNumerator,
                  (gsGOF, obsBandIndexGOFI, obsLUTFilterIndexGO[obsFitUseGOI]),
                  1.0 / obsMagErr2GOFI)

        innerTermGOF[:] = 0.0
        innerTermGOF[indGOF] = (1.0 - sumNumerator[gsGOF,
                                                   obsBandIndexGOFI,
                                                   obsLUTFilterIndexGO[obsFitUseGOI]] *
                                objMagStdMeanErr2GO[obsFitUseGOI])

        add_at_1d(partialArray[self.fgcmPars.parFilterOffsetLoc:
                                   (self.fgcmPars.parFilterOffsetLoc +
                                    self.fgcmPars.nLUTFilter)],
                  obsLUTFilterIndexGO[obsFitUseGO],
                  2.0 * deltaMagWeightedGOF * innerTermGOF)

        # Now set those to zero the derivatives we aren't using
        partialArray[self.fgcmPars.parFilterOffsetLoc:
                         (self.fgcmPars.parFilterOffsetLoc +
                          self.fgcmPars.nLUTFilter)][~self.fgcmPars.parFilterOffsetFitFlag] = 0.0
        uOffsetIndex, = np.where(self.fgcmPars.parFilterOffsetFitFlag)
        partialArray[self.fgcmPars.nFitPars +
                     self.fgcmPars.parFilterOffsetLoc +
                     uOffsetIndex] += 1

        totalArr = snmm.getArray(self.totalHandleDict[thisCore])
        totalArr[:] = totalArr[:] + partialArray

        return None

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state

