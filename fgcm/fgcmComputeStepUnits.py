from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import os
import sys
import esutil
import time

from .fgcmUtilities import _pickle_method
from .fgcmUtilities import objFlagDict

import types
try:
    import copy_reg as copyreg
except ImportError:
    import copyreg

import multiprocessing
from multiprocessing import Pool

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

copyreg.pickle(types.MethodType, _pickle_method)

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

        self.fgcmLog.info('Initializing FgcmChisq')

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

        # these are the standard *band* I10s
        self.I10StdBand = fgcmConfig.I10StdBand

        self.illegalValue = fgcmConfig.illegalValue

        self.maxParStepFraction = 0.1 # hard code this

        if (fgcmConfig.useSedLUT and self.fgcmLUT.hasSedLUT):
            self.useSedLUT = True
        else:
            self.useSedLUT = False

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
        self.fgcmLog.info('Found %d good stars for step units' % (goodStars.size))

        if (goodStars.size == 0):
            raise RuntimeError("No good stars to fit!")

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        expFlag = self.fgcmPars.expFlag

        goodStarsSub, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=expFlag)

        self.nSums = 1 # nobs
        # 0: nFitPars -> derivative for step calculation
        # nFitPars: 2*nFitPars -> parameters which are touched
        # Note that we only need nobs, not ref because we assume all observations are
        # going to have one or the other, and this doesn't care which is which
        self.nSums += 2 * self.fgcmPars.nFitPars

        proc = multiprocessing.Process()
        workerIndex = proc._identity[0]+1
        proc = None

        self.totalHandleDict = {}
        for thisCore in xrange(self.nCore):
            self.totalHandleDict[workerIndex + thisCore] = (
                snmm.createArray(self.nSums,dtype='f8'))

        nSections = goodStars.size // self.nStarPerRun + 1
        goodStarsList = np.array_split(goodStars,nSections)

        splitValues = np.zeros(nSections-1,dtype='i4')
        for i in xrange(1,nSections):
            splitValues[i-1] = goodStarsList[i][0]

        splitIndices = np.searchsorted(goodStars[goodStarsSub], splitValues)

        # and split along the indices
        goodObsList = np.split(goodObs,splitIndices)

        workerList = list(zip(goodStarsList,goodObsList))

        # reverse sort so the longest running go first
        workerList.sort(key=lambda elt:elt[1].size, reverse=True)

        # make a pool
        pool = Pool(processes=self.nCore)
        # Compute magnitudes
        pool.map(self._stepWorker, workerList, chunksize=1)

        pool.close()
        pool.join()

        # sum up the partial sums from the different jobs
        partialSums = np.zeros(self.nSums,dtype='f8')
        for thisCore in xrange(self.nCore):
            partialSums[:] += snmm.getArray(
                self.totalHandleDict[workerIndex + thisCore])[:]

        nonZero, = np.where(partialSums[self.fgcmPars.nFitPars: 2*self.fgcmPars.nFitPars] > 0)
        nActualFitPars = nonZero.size

        # Get the number of degrees of freedom
        fitDOF = partialSums[-1] - float(nActualFitPars)

        dChisqdPNZ = partialSums[nonZero] / fitDOF

        # default step is 1.0
        self.fgcmPars.stepUnits[:] = 1.0

        # And the actual step size for good pars
        self.fgcmPars.stepUnits[nonZero] = np.abs(dChisqdPNZ) / self.fitGradientTolerance

        # Leave these in temporarily...
        print('O3:')
        print(self.fgcmPars.stepUnits[self.fgcmPars.parO3Loc:
                                          (self.fgcmPars.parO3Loc +
                                           self.fgcmPars.nCampaignNights)])
        print('Alpha:')
        print(self.fgcmPars.stepUnits[self.fgcmPars.parAlphaLoc:
                                          (self.fgcmPars.parAlphaLoc +
                                           self.fgcmPars.nCampaignNights)])
        print('PWV intercept:')
        print(self.fgcmPars.stepUnits[self.fgcmPars.parLnPwvInterceptLoc:
                                          (self.fgcmPars.parLnPwvInterceptLoc +
                                           self.fgcmPars.nCampaignNights)])
        print('Tau intercept:')
        print(self.fgcmPars.stepUnits[self.fgcmPars.parLnTauInterceptLoc:
                                          (self.fgcmPars.parLnTauInterceptLoc +
                                           self.fgcmPars.nCampaignNights)])
        print('Washes:')
        print(self.fgcmPars.stepUnits[self.fgcmPars.parQESysInterceptLoc:
                                          (self.fgcmPars.parQESysInterceptLoc +
                                           self.fgcmPars.nWashIntervals)])

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

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsLUTFilterIndex = snmm.getArray(self.fgcmStars.obsLUTFilterIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)
        obsSecZenith = snmm.getArray(self.fgcmStars.obsSecZenithHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)

        # cut these down now, faster later
        obsObjIDIndexGO = obsObjIDIndex[goodObs]
        obsBandIndexGO = obsBandIndex[goodObs]
        obsLUTFilterIndexGO = obsLUTFilterIndex[goodObs]
        obsExpIndexGO = obsExpIndex[goodObs]
        obsSecZenithGO = obsSecZenith[goodObs]
        obsCCDIndexGO = obsCCDIndex[goodObs]

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

        ##########
        ## O3
        ##########

        expNightIndexGOF = self.fgcmPars.expNightIndex[obsExpIndexGO[obsFitUseGO]]
        uNightIndex = np.unique(expNightIndexGOF)

        # First, we compute the maximum change in magnitude expected from a (10%) shift
        # in the given parameter
        # And then we say the test summation (to compute units) should assume the configured
        # convergence criteria OR this shift, whichever is smaller.  That way we don't
        # go crazy and try to the ozone to incorrectly move when no amount of ozone is
        # going to get the z-band to budge (for example).

        #maxDeltaO3GO = np.abs(dLdO3GO) * (self.fgcmLUT.o3[-1] - self.fgcmLUT.o3[0]) * self.maxParStepFraction
        #deltaMagGO = np.clip(maxDeltaO3GO, None, self.stepUnitReference)
        #deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

        np.add.at(partialArray[self.fgcmPars.parO3Loc:
                                   (self.fgcmPars.parO3Loc +
                                    self.fgcmPars.nCampaignNights)],
                  expNightIndexGOF,
                  2.0 * (deltaMagWeightedGOF) * (
                dLdO3GO[obsFitUseGO]))

        partialArray[self.fgcmPars.nFitPars +
                     self.fgcmPars.parO3Loc +
                     uNightIndex] += 1

        ###########
        ## Alpha
        ###########

        #maxDeltaAlphaGO = np.abs(dLdAlphaGO) * (self.fgcmLUT.alpha[-1] - self.fgcmLUT.alpha[0]) * self.maxParStepFraction
        #deltaMagGO = np.clip(maxDeltaAlphaGO, None, self.stepUnitReference)
        #deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

        np.add.at(partialArray[self.fgcmPars.parAlphaLoc:
                                   (self.fgcmPars.parAlphaLoc+
                                    self.fgcmPars.nCampaignNights)],
                  expNightIndexGOF,
                  2.0 * (deltaMagWeightedGOF) * (
                dLdAlphaGO[obsFitUseGO]))

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

            # This can be used for both of those below
            #maxDeltaLnPwvGO = np.abs(dLdLnPwvGO) * (self.fgcmLUT.lnPwv[-1] - self.fgcmLUT.lnPwv[0]) * self.maxParStepFraction
            #deltaMagGO = np.clip(maxDeltaLnPwvGO, None, self.stepUnitReference)
            #deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

            np.add.at(partialArray[self.fgcmPars.parExternalLnPwvOffsetLoc:
                                       (self.fgcmPars.parExternalLnPwvOffsetLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF[hasExtGOF],
                      2.0 * deltaMagWeightedGOF[hasExtGOF] * (
                    dLdLnPwvGO[obsFitUseGO[hasExtGOF]]))

            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parExternalLnPwvOffsetLoc +
                         uNightIndexHasExt] += 1

            # PWV Global Scale

            partialArray[self.fgcmPars.parExternalLnPwvScaleLoc] = 2.0 * (
                np.sum(deltaMagWeightedGOF[hasExtGOF] * (
                        self.fgcmPars.expLnPwv[obsExpIndexGO[obsFitUseGO[hasExtGOF]]] *
                        dLdLnPwvGO[obsFitUseGO[hasExtGOF]])))

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

                # This can be used for all of the ones below
                #maxDeltaLnPwvGO = np.abs(dLdLnPwvGO) * (self.fgcmLUT.lnPwv[-1] - self.fgcmLUT.lnPwv[0]) * self.maxParStepFraction
                #deltaMagGO = np.clip(maxDeltaLnPwvGO, None, self.stepUnitReference)
                #deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

                partialArray[self.fgcmPars.parRetrievedLnPwvScaleLoc] = 2.0 * (
                    np.sum(deltaMagWeightedGOF[hasRetrievedPwvGOF] * (
                            self.fgcmPars.expLnPwv[obsExpIndexGO[obsFitUseGO[hasRetreivedPwvGOF]]] *
                            dLdLnPwvGO[obsFitUseGO[hasRetrievedPwvGOF]])))

                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parRetrievedLnPwvScaleLoc] += 1

                if self.fgcmPars.useNightlyRetrievedPwv:
                    # PWV Retrieved Nightly Offset

                    uNightIndexHasRetrievedPwv = np.unique(expNightIndexGOF[hasRetrievedPwvGOF])

                    np.add.at(partialArray[self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc:
                                               (self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc+
                                                self.fgcmPars.nCampaignNights)],
                              expNightIndexGOF[hasRetrievedPwvGOF],
                              2.0 * deltaMagWeightedGOF[hasRetrievedPwvGOF] * (
                            dLdLnPwvGO[obsFitUseGO[hasRetrievedPwvGOF]]))

                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parRetrievedLnPwvNightlyOffsetLoc +
                                 uNightIndexHasRetrievedPwv] += 1

                else:
                    # PWV Retrieved Global Offset

                    partialArray[self.fgcmPars.parRetrievedLnPwvOffsetLoc] = 2.0 * (
                        np.sum(deltaMagWeightedGOF[hasRetrievedPwvGOF] * (
                                dLdLnPwvGO[obsFitUseGO[hasRetrievedPwvGOF]])))

                    partialArray[self.fgcmPars.nFitPars +
                                 self.fgcmPars.parRetrievedLnPwvOffsetLoc] += 1

        else:
            ###########
            ## Pwv No External
            ###########

            noExtGOF, = np.where(~self.fgcmPars.externalPwvFlag[obsExpIndexGO[obsFitUseGO]])
            uNightIndexNoExt = np.unique(expNightIndexGOF[noExtGOF])

            # Pwv Nightly Intercept

            #maxDeltaLnPwvGO = np.abs(dLdLnPwvGO) * (self.fgcmLUT.lnPwv[-1] - self.fgcmLUT.lnPwv[0]) * self.maxParStepFraction
            #deltaMagGO = np.clip(maxDeltaLnPwvGO, None, self.stepUnitReference)
            #deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

            np.add.at(partialArray[self.fgcmPars.parLnPwvInterceptLoc:
                                       (self.fgcmPars.parLnPwvInterceptLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF[noExtGOF],
                      2.0 * deltaMagWeightedGOF[noExtGOF] * (
                    dLdLnPwvGO[obsFitUseGO[noExtGOF]]))

            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parLnPwvInterceptLoc +
                         uNightIndexNoExt] += 1

            # lnPwv Nightly Slope

            #maxDeltaLnPwvSlopeGO = (np.abs(dLdLnPwvGO * self.fgcmPars.expDeltaUT[obsExpIndexGO]) *
            #                        (self.fgcmLUT.lnPwv[-1] - self.fgcmLUT.lnPwv[0]) *
            #                        self.maxParStepFraction)
            #deltaMagGO = np.clip(maxDeltaLnPwvSlopeGO, None, self.stepUnitReference)
            #deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

            np.add.at(partialArray[self.fgcmPars.parLnPwvSlopeLoc:
                                       (self.fgcmPars.parLnPwvSlopeLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF[noExtGOF],
                      2.0 * deltaMagWeightedGOF[noExtGOF] * (
                    (self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]] *
                     dLdLnPwvGO[obsFitUseGO[noExtGOF]])))

            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parLnPwvSlopeLoc +
                         uNightIndexNoExt] += 1

            # lnPwv Nightly Quadratic
            if self.useQuadraticPwv:

                #maxDeltaLnPwvQuadraticGO = (np.abs(dLdLnPwvGO * self.fgcmPars.expDeltaUT[obsExpIndexGO]**2.) *
                #                            (self.fgcmLUT.lnPwv[-1] - self.fgcmLUT.lnPwv[0]) *
                #                            self.maxParStepFraction)
                #deltaMagGO = np.clip(maxDeltaLnPwvQuadraticGO, None, self.stepUnitReference)
                #deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

                np.add.at(partialArray[self.fgcmPars.parLnPwvQuadraticLoc:
                                           (self.fgcmPars.parLnPwvQuadraticLoc+
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF[noExtGOF],
                          2.0 * deltaMagWeightedGOF[noExtGOF] * (
                        (self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]]**2. * dLdLnPwvGO[obsFitUseGO[noExtGOF]])))

                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parLnPwvQuadraticLoc +
                             uNightIndexNoExt] += 1

        #############
        ## Tau External
        #############

        if (self.fgcmPars.hasExternalTau):
            # NOT IMPLEMENTED PROPERLY YET

            hasExtGOF,=np.where(self.fgcmPars.externalTauFlag[obsExpIndexGO[obsFitUseGO]])
            uNightIndexHasExt = np.unique(expNightIndexGOF[hasExtGOF])

            # Tau Nightly Offset

            np.add.at(partialArray[self.fgcmPars.parExternalLnTauOffsetLoc:
                                       (self.fgcmPars.parExternalLnTauOffsetLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF[hasExtGOF],
                      2.0 * deltaMagWeightedGOF[hasExtGOF] * (
                    dLdLnTauGO[obsFitUseGO[hasExtGOF]]))

            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parExternalLnTauOffsetLoc +
                         uNightIndexHasExt] += 1

            # Tau Global Scale
            ## MAYBE: is this correct with the logs?

            partialArray[self.fgcmPars.parExternalLnTauScaleLoc] = 2.0 * (
                np.sum(deltaMagWeightedGOF[hasExtGOF] * (
                        dLdLnTauGO[obsFitUseGO[hasExtGOF]])))

            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parExternalLnTauScaleLoc] += 1

        ###########
        ## Tau No External
        ###########

        noExtGOF, = np.where(~self.fgcmPars.externalTauFlag[obsExpIndexGO[obsFitUseGO]])
        uNightIndexNoExt = np.unique(expNightIndexGOF[noExtGOF])

        # lnTau Nightly Intercept

        #maxDeltaLnTauGO = np.abs(dLdLnTauGO) * (self.fgcmLUT.lnTau[-1] - self.fgcmLUT.lnTau[0]) * self.maxParStepFraction
        #deltaMagGO = np.clip(maxDeltaLnTauGO, None, self.stepUnitReference)
        #deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

        np.add.at(partialArray[self.fgcmPars.parLnTauInterceptLoc:
                                   (self.fgcmPars.parLnTauInterceptLoc+
                                    self.fgcmPars.nCampaignNights)],
                  expNightIndexGOF[noExtGOF],
                  2.0 * deltaMagWeightedGOF[noExtGOF] * (
                dLdLnTauGO[obsFitUseGO[noExtGOF]]))

        partialArray[self.fgcmPars.nFitPars +
                     self.fgcmPars.parLnTauInterceptLoc +
                     uNightIndexNoExt] += 1

        # lnTau nightly slope

        #maxDeltaLnTauSlopeGO = (np.abs(dLdLnTauGO * self.fgcmPars.expDeltaUT[obsExpIndexGO]) *
        #                        (self.fgcmLUT.lnTau[-1] - self.fgcmLUT.lnTau[0]) *
        #                        self.maxParStepFraction)
        #deltaMagGO = np.clip(maxDeltaLnTauSlopeGO, None, self.stepUnitReference)
        #deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

        np.add.at(partialArray[self.fgcmPars.parLnTauSlopeLoc:
                                   (self.fgcmPars.parLnTauSlopeLoc+
                                    self.fgcmPars.nCampaignNights)],
                  expNightIndexGOF[noExtGOF],
                  2.0 * deltaMagWeightedGOF[noExtGOF] * (
                (self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]] *
                 dLdLnTauGO[obsFitUseGO[noExtGOF]])))

        partialArray[self.fgcmPars.nFitPars +
                     self.fgcmPars.parLnTauSlopeLoc +
                     uNightIndexNoExt] += 1

        ##################
        ## Washes (QE Sys)
        ##################

        # The washes don't need to worry about the limits ... 0.1 mag is 0.1 mag here.

        expWashIndexGOF = self.fgcmPars.expWashIndex[obsExpIndexGO[obsFitUseGO]]

        # Wash Intercept

        #deltaMagGO = np.zeros(goodObs.size) + self.stepUnitReference
        #deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] * obsWeightGO[obsFitUseGO]

        if self.instrumentParsPerBand:
            # We have per-band intercepts
            # Non-fit bands will be given the mean of the others (in fgcmParameters),
            # because they aren't in the chi2.
            #uWashBandIndex = np.unique(expWashIndexGOF * self.fgcmPars.nBands +
            #                           obsBandIndexGO[obsFitUseGO])
            uWashBandIndex = np.unique(np.ravel_multi_index((obsBandIndexGO[obsFitUseGO],
                                                             expWashIndexGOF),
                                                            self.fgcmPars.parQESysIntercept.shape))

            np.add.at(partialArray[self.fgcmPars.parQESysInterceptLoc:
                                       (self.fgcmPars.parQESysInterceptLoc +
                                        self.fgcmPars.parQESysIntercept.size)],
                      np.ravel_multi_index((obsBandIndexGO[obsFitUseGO],
                                            expWashIndexGOF),
                                           self.fgcmPars.parQESysIntercept.shape),
                      2.0 * deltaMagWeightedGOF)

            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parQESysInterceptLoc +
                         uWashBandIndex] += 1

        else:
            # We have one gray mirror term for all bands
            uWashIndex = np.unique(expWashIndexGOF)

            np.add.at(partialArray[self.fgcmPars.parQESysInterceptLoc:
                                       (self.fgcmPars.parQESysInterceptLoc +
                                        self.fgcmPars.nWashIntervals)],
                      expWashIndexGOF,
                      2.0 * deltaMagWeightedGOF)

            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parQESysInterceptLoc +
                         uWashIndex] += 1

        #################
        ## Filter offset
        #################

        np.add.at(partialArray[self.fgcmPars.parFilterOffsetLoc:
                                   (self.fgcmPars.parFilterOffsetLoc +
                                    self.fgcmPars.nLUTFilter)],
                  obsLUTFilterIndexGO[obsFitUseGO],
                  2.0 * deltaMagWeightedGOF)

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

