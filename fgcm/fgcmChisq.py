from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import time

from fgcmUtilities import _pickle_method

import types
import copy_reg
import multiprocessing
from multiprocessing import Pool

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmChisq(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmStars,fgcmLUT):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.log('INFO','Initializing FgcmChisq')

        # does this need to be shm'd?
        self.fgcmPars = fgcmPars

        # this is shm'd
        self.fgcmLUT = fgcmLUT

        # also shm'd
        self.fgcmStars = fgcmStars

        # need to configure
        self.nCore = fgcmConfig.nCore
        self.ccdStartIndex = fgcmConfig.ccdStartIndex

        #self.fitChisqs = []
        self.resetFitChisqList()

        # not sure what we need the config for

        #resourceUsage('End of chisq init')

    def resetFitChisqList(self):
        self.fitChisqs = []

    def __call__(self,fitParams,fitterUnits=False,computeDerivatives=False,computeSEDSlopes=False,debug=False,allExposures=False):
        """
        """

        # computeDerivatives: do we want to compute the derivatives?
        # computeSEDSlope: compute SED Slope and recompute mean mags?
        # fitterUnits: units of the fitter or "true" units?

        self.computeDerivatives = computeDerivatives
        self.computeSEDSlopes = computeSEDSlopes
        self.fitterUnits = fitterUnits
        self.allExposures = allExposures

        self.fgcmLog.log('DEBUG','FgcmChisq: computeDerivatives = %d' %
                         (int(computeDerivatives)))
        self.fgcmLog.log('DEBUG','FgcmChisq: computeSEDSlopes = %d' %
                         (int(computeSEDSlopes)))
        self.fgcmLog.log('DEBUG','FgcmChisq: fitterUnits = %d' %
                         (int(fitterUnits)))
        self.fgcmLog.log('DEBUG','FgcmChisq: allExposures = %d' %
                         (int(allExposures)))

        startTime = time.time()

        if (self.allExposures and (self.computeDerivatives or
                                   self.computeSEDSlopes)):
            raise ValueError("Cannot set allExposures and computeDerivatives or computeSEDSlopes")
        self.fgcmPars.reloadParArray(fitParams,fitterUnits=self.fitterUnits)
        self.fgcmPars.parsToExposures()


        # and reset numbers if necessary
        if (not self.allExposures):
            snmm.getArray(self.fgcmStars.objMagStdMeanHandle)[:] = 99.0
            snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)[:] = 99.0

        goodStars,=np.where(snmm.getArray(self.fgcmStars.objFlagHandle) == 0)

        self.fgcmLog.log('INFO','Found %d good stars for chisq' % (goodStars.size))

        if (goodStars.size == 0):
            raise ValueError("No good stars to fit!")

        # we'll be able to parallelize based on where stars are on the sky, I think.
        # and just work on a goodStars.
        # and we won't need locks either, since we're not touching the same stars/memory
        #  (I hope)

        self.nSums = 2  # chisq, nobs
        if (self.computeDerivatives):
            self.nSums += self.fgcmPars.nFitPars

        self.debug = debug
        if (self.debug):
            # debug mode: single core
            self.totalHandleDict = {}
            self.totalHandleDict[0] = snmm.createArray(self.nSums,dtype='f8')

            self._worker(goodStars)

            partialSums = snmm.getArray(self.totalHandleDict[0])[:]
        else:
            # regular multi-core

            self.fgcmLog.log('INFO','Running chisq on %d cores' % (self.nCore))

            # make a dummy process to discover starting child number
            proc = multiprocessing.Process()
            workerIndex = proc._identity[0]+1
            proc = None

            self.totalHandleDict = {}
            for thisCore in xrange(self.nCore):
                self.totalHandleDict[workerIndex + thisCore] = (
                    snmm.createArray(self.nSums,dtype='f8'))

            # split goodStars into a list of arrays of roughly equal size
            goodStarsList = np.array_split(goodStars,self.nCore)

            # make a pool
            pool = Pool(processes=self.nCore)
            pool.map(self._worker,goodStarsList)
            pool.close()
            pool.join()

            # sum up the partial sums from the different jobs
            partialSums = np.zeros(self.nSums,dtype='f8')
            for thisCore in xrange(self.nCore):
                partialSums[:] += snmm.getArray(
                    self.totalHandleDict[workerIndex + thisCore])[:]


        if (not self.allExposures):
            ## FIXME: dof should be actual number of fit parameters

            fitDOF = partialSums[-1] - float(self.fgcmPars.nFitPars)
            if (fitDOF <= 0):
                raise ValueError("Number of parameters fitted is more than number of constraints! (%d > %d)" % (self.fgcmPars.nFitPars,partialSums[-1]))

            fitChisq = partialSums[-2] / fitDOF
            if (self.computeDerivatives):
                dChisqdP = partialSums[0:self.fgcmPars.nFitPars] / fitDOF

            # want to append this...
            self.fitChisqs.append(fitChisq)

            self.fgcmLog.log('INFO','Chisq/dof = %.2f' % (fitChisq))
        else:
            fitChisq = self.fitChisqs[-1]

        # free shared arrays
        for key in self.totalHandleDict.keys():
            snmm.freeArray(self.totalHandleDict[key])

        self.fgcmLog.log('INFO','Chisq computation took %.2f seconds.' %
                         (time.time() - startTime))

        self.fgcmStars.magStdComputed = True
        if (self.allExposures):
            self.fgcmStars.allMagStdComputed = True

        if (self.computeDerivatives):
            return fitChisq, dChisqdP
        else:
            return fitChisq

    def _worker(self,goodStars):
        """
        """

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsSecZenith = snmm.getArray(self.fgcmStars.obsSecZenithHandle)
        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        obsMagADUErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)

        _,goodObs=esutil.numpy_util.match(goodStars,obsObjIDIndex,presorted=True)

        if (not self.allExposures):
            gd,=np.where(self.fgcmPars.expFlag[obsExpIndex[goodObs]] == 0)
            goodObs = goodObs[gd]

        # which observations are used in the fit?
        _,obsFitUseGO = esutil.numpy_util.match(self.fgcmPars.fitBandIndex,
                                                obsBandIndex[goodObs])

        # now refer to obsBandIndex[goodObs]
        # add GO to index names that are cut to goodObs

        lutIndicesGO = self.fgcmLUT.getIndices(obsBandIndex[goodObs],
                                               self.fgcmPars.expPWV[obsExpIndex[goodObs]],
                                               self.fgcmPars.expO3[obsExpIndex[goodObs]],
                                               np.log(self.fgcmPars.expTau[obsExpIndex[goodObs]]),
                                               self.fgcmPars.expAlpha[obsExpIndex[goodObs]],
                                               obsSecZenith[goodObs],
                                               obsCCDIndex[goodObs],
                                               self.fgcmPars.expPmb[obsExpIndex[goodObs]])
        I0GO = self.fgcmLUT.computeI0(obsBandIndex[goodObs],
                                      self.fgcmPars.expPWV[obsExpIndex[goodObs]],
                                      self.fgcmPars.expO3[obsExpIndex[goodObs]],
                                      np.log(self.fgcmPars.expTau[obsExpIndex[goodObs]]),
                                      self.fgcmPars.expAlpha[obsExpIndex[goodObs]],
                                      obsSecZenith[goodObs],
                                      obsCCDIndex[goodObs],
                                      self.fgcmPars.expPmb[obsExpIndex[goodObs]],
                                      lutIndicesGO)
        I10GO = self.fgcmLUT.computeI1(lutIndicesGO) / I0GO


        qeSysGO = self.fgcmPars.expQESys[obsExpIndex[goodObs]]

        obsMagGO = obsMagADU[goodObs] + 2.5*np.log10(I0GO) + qeSysGO

        # this is annoying that we have to do this.
        obsMagErr2GO = obsMagADUErr[goodObs]**2.

        if (self.computeSEDSlopes):
            wtSum = np.zeros_like(objMagStdMean,dtype='f8')
            np.add.at(wtSum,
                      (obsObjIDIndex[goodObs],obsBandIndex[goodObs]),
                      1./obsMagErr2GO)
            gd=np.where(wtSum > 0.0)
            # important: only zero accumulator for our stars
            objMagStdMean[gd] = 0.0

            # note that obsMag is already cut to goodObs
            np.add.at(objMagStdMean,
                      (obsObjIDIndex[goodObs],obsBandIndex[goodObs]),
                      obsMagGO/obsMagErr2GO)

            objMagStdMean[gd] /= wtSum[gd]
            objMagStdMeanErr[gd] = np.sqrt(1./wtSum[gd])

            self.fgcmStars.computeObjectSEDSlopes(goodStars)

        # compute linearized chromatic correction
        deltaStdGO = 2.5 * np.log10((1.0 +
                                   objSEDSlope[obsObjIDIndex[goodObs],
                                               obsBandIndex[goodObs]] * I10GO) /
                                  (1.0 + objSEDSlope[obsObjIDIndex[goodObs],
                                                     obsBandIndex[goodObs]] *
                                   self.fgcmLUT.I10Std[obsBandIndex[goodObs]]))
        # we can only do this for calibration stars.
        obsMagStd[goodObs] = obsMagGO + deltaStdGO

        if (self.allExposures) :
            # kick out
            return None

        ## FIXME: change to only make memory for the current objects under consideration?
        ##        (worried about memory usage...)

        # compute mean mags
        wtSum = np.zeros_like(objMagStdMean,dtype='f8')
        np.add.at(wtSum,
                  (obsObjIDIndex[goodObs],obsBandIndex[goodObs]),
                  1./obsMagErr2GO)
        # only zero out the accumulator where we have observations of objects!
        gd=np.where(wtSum > 0.0)
        objMagStdMean[gd] = 0.0

        np.add.at(objMagStdMean,
                  (obsObjIDIndex[goodObs],obsBandIndex[goodObs]),
                  obsMagStd[goodObs]/obsMagErr2GO)

        objMagStdMean[gd] /= wtSum[gd]
        objMagStdMeanErr[gd] = np.sqrt(1./wtSum[gd])

        # compute delta-mags

        deltaMagGO = (obsMagStd[goodObs] -
                      objMagStdMean[obsObjIDIndex[goodObs],
                                    obsBandIndex[goodObs]])
        deltaMagErr2GO = (obsMagErr2GO +
                          objMagStdMeanErr[obsObjIDIndex[goodObs],
                                           obsBandIndex[goodObs]]**2.)
        deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] / deltaMagErr2GO[obsFitUseGO]

        # and compute chisq
        partialChisq = np.sum(deltaMagGO[obsFitUseGO]**2./deltaMagErr2GO[obsFitUseGO])

        partialArray = np.zeros(self.nSums,dtype='f8')
        partialArray[-2] = partialChisq
        partialArray[-1] = obsFitUseGO.size

        if (self.computeDerivatives):
            unitDict=self.fgcmPars.getUnitDict(fitterUnits=self.fitterUnits)

            # this is going to be ugly.  wow, how many indices and sub-indices?
            #  or does it simplify since we need all the obs on a night?
            #  we shall see!  And speed up!

            (dLdPWVGO,dLdO3GO,dLdTauGO,dLdAlphaGO) = (
                self.fgcmLUT.computeLogDerivatives(lutIndicesGO,
                                                   I0GO,
                                                   self.fgcmPars.expTau[
                        obsExpIndex[goodObs]]))


            # we have objMagStdMeanErr[objIndex,:] = \Sum_{i"} 1/\sigma^2_{i"j}
            #   note that this is summed over all observations of an object in a band
            #   so that this is already done

            # we need magdLdp = \Sum_{i'} (1/\sigma^2_{i'j}) dL(i',j|p)
            #   note that this is summed over all observations in a filter that
            #   touch a given parameter

            # set up arrays
            magdLdPWVIntercept = np.zeros((self.fgcmPars.nCampaignNights,
                                           self.fgcmPars.nFitBands))
            magdLdPWVSlope = np.zeros_like(magdLdPWVIntercept)
            magdLdPWVOffset = np.zeros_like(magdLdPWVIntercept)
            magdLdTauIntercept = np.zeros_like(magdLdPWVIntercept)
            magdLdTauSlope = np.zeros_like(magdLdPWVIntercept)
            magdLdTauOffset = np.zeros_like(magdLdPWVIntercept)
            magdLdAlpha = np.zeros_like(magdLdPWVIntercept)
            magdLdO3 = np.zeros_like(magdLdPWVIntercept)

            magdLdPWVScale = np.zeros(self.fgcmPars.nFitBands,dtype='f4')
            magdLdTauScale = np.zeros_like(magdLdPWVScale)

            magdLdWashIntercept = np.zeros((self.fgcmPars.nWashIntervals,
                                            self.fgcmPars.nFitBands))
            magdLdWashSlope = np.zeros_like(magdLdWashIntercept)

            # precompute object err2...
            #  seems inefficient to compute for all if we're parallel, but
            #   it might not matter.  (Also memory usage worries)
            objMagStdMeanErr2 = objMagStdMeanErr**2.

            ##########
            ## O3
            ##########

            expNightIndexGOF = self.fgcmPars.expNightIndex[obsExpIndex[goodObs[obsFitUseGO]]]
            uNightIndex = np.unique(expNightIndexGOF)

            np.add.at(magdLdO3,
                      (expNightIndexGOF,obsBandIndex[goodObs[obsFitUseGO]]),
                      dLdO3GO[obsFitUseGO] / obsMagErr2GO[obsFitUseGO])
            np.multiply.at(magdLdO3,
                           (expNightIndexGOF,obsBandIndex[goodObs[obsFitUseGO]]),
                           objMagStdMeanErr2[obsObjIDIndex[goodObs[obsFitUseGO]],
                                             obsBandIndex[goodObs[obsFitUseGO]]])
            np.add.at(partialArray[self.fgcmPars.parO3Loc:
                                       (self.fgcmPars.parO3Loc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF,
                      deltaMagWeightedGOF * (
                    (dLdO3GO[obsFitUseGO] -
                     magdLdO3[expNightIndexGOF,obsBandIndex[goodObs[obsFitUseGO]]])))

            partialArray[self.fgcmPars.parO3Loc +
                         uNightIndex] *= (2.0 / unitDict['o3Unit'])

            ###########
            ## Alpha
            ###########

            np.add.at(magdLdAlpha,
                      (expNightIndexGOF,obsBandIndex[goodObs[obsFitUseGO]]),
                      dLdAlphaGO[obsFitUseGO] / obsMagErr2GO[obsFitUseGO])
            np.multiply.at(magdLdAlpha,
                           (expNightIndexGOF,obsBandIndex[goodObs[obsFitUseGO]]),
                           objMagStdMeanErr2[obsObjIDIndex[goodObs[obsFitUseGO]],
                                             obsBandIndex[goodObs[obsFitUseGO]]])
            np.add.at(partialArray[self.fgcmPars.parAlphaLoc:
                                       (self.fgcmPars.parAlphaLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF,
                      deltaMagWeightedGOF * (
                    (dLdAlphaGO[obsFitUseGO] -
                     magdLdAlpha[expNightIndexGOF,obsBandIndex[goodObs[obsFitUseGO]]])))

            partialArray[self.fgcmPars.parAlphaLoc +
                         uNightIndex] *= (2.0 / unitDict['alphaUnit'])


            ###########
            ## PWV External
            ###########

            if (self.fgcmPars.hasExternalPWV):
                hasExtGOF,=np.where(self.fgcmPars.externalPWVFlag[obsExpIndex[goodObs[obsFitUseGO]]])

                # PWV Nightly Offset
                np.add.at(magdLdPWVOffset,
                          (expNightIndexGOF[hasExtGOF],
                           obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]]),
                          dLdPWVGO[obsFitUseGO[hasExtGOF]] /
                          obsMagErr2GO[obsFitUseGO[hasExtGOF]])
                np.multiply.at(magdLdPWVOffset,
                               (expNightIndexGOF[hasExtGOF],
                                obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]]),
                               objMagStdMeanErr2[obsObjIDIndex[goodObs[obsFitUseGO[hasExtGOF]]],
                                                 obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]]])
                np.add.at(partialArray[self.fgcmPars.parExternalPWVOffsetLoc:
                                           (self.fgcmPars.parExternalPWVOffsetLoc+
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF[hasExtGOF],
                          deltaMagWeightedGOF[hasExtGOF] * (
                        (dLdPWVGO[obsFitUseGO[hasExtGOF]] -
                         magdLdPWVOffset[expNightIndexGOF[hasExtGOF],
                                         obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]]])))
                partialArray[self.fgcmPars.parExternalPWVOffsetLoc +
                             uNightIndex] *= (2.0 / unitDict['pwvUnit'])


                # PWV Global Scale
                np.add.at(magdLdPWVScale,
                          obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]],
                          self.fgcmPars.expPWV[obsExpIndex[goodObs[obsFitUseGO[hasExtGOF]]]] *
                          dLdPWVGO[obsFitUseGO[hasExtGOF]] /
                          obsMagErr2GO[obsFitUseGO[hasExtGOF]])
                np.multiply.at(magdLdPWVScale,
                               obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]],
                               objMagStdMeanErr2[obsObjIDIndex[goodObs[obsFitUseGO[hasExtGOF]]],
                                             obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]]])
                partialArray[self.fgcmPars.parExternalPWVScaleLoc] = 2.0 * (
                    np.sum(deltaMagWeightedGOF[hasExtGOF] * (
                            self.fgcmPars.expPWV[obsExpIndex[goodObs[obsFitUseGO[hasExtGOF]]]] *
                            dLdPWVGO[obsFitUseGO[hasExtGOF]] -
                            magdLdPWVScale[obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]]])) /
                    unitDict['pwvUnit'])

            ###########
            ## PWV No External
            ###########

            noExtGOF, = np.where(~self.fgcmPars.externalPWVFlag[obsExpIndex[goodObs[obsFitUseGO]]])
            # PWV Nightly Intercept

            np.add.at(magdLdPWVIntercept,
                      (expNightIndexGOF[noExtGOF],
                       obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]),
                      dLdPWVGO[obsFitUseGO[noExtGOF]] /
                      obsMagErr2GO[obsFitUseGO[noExtGOF]])
            np.multiply.at(magdLdPWVIntercept,
                           (expNightIndexGOF[noExtGOF],
                            obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]),
                           objMagStdMeanErr2[obsObjIDIndex[goodObs[obsFitUseGO[noExtGOF]]],
                                             obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]])
            np.add.at(partialArray[self.fgcmPars.parPWVInterceptLoc:
                                       (self.fgcmPars.parPWVInterceptLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF[noExtGOF],
                      deltaMagWeightedGOF[noExtGOF] * (
                    (dLdPWVGO[obsFitUseGO[noExtGOF]] -
                     magdLdPWVOffset[expNightIndexGOF[noExtGOF],
                                     obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]])))

            partialArray[self.fgcmPars.parPWVInterceptLoc +
                         uNightIndex] *= (2.0 / unitDict['pwvUnit'])

            # PWV Nightly Slope
            np.add.at(magdLdPWVSlope,
                      (expNightIndexGOF[noExtGOF],
                       obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]),
                      self.fgcmPars.expDeltaUT[obsExpIndex[goodObs[obsFitUseGO[noExtGOF]]]] *
                      dLdPWVGO[obsFitUseGO[noExtGOF]] /
                      obsMagErr2GO[obsFitUseGO[noExtGOF]])
            np.multiply.at(magdLdPWVSlope,
                           (expNightIndexGOF[noExtGOF],
                            obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]),
                           objMagStdMeanErr2[obsObjIDIndex[goodObs[obsFitUseGO[noExtGOF]]],
                                             obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]])
            np.add.at(partialArray[self.fgcmPars.parPWVSlopeLoc:
                                       (self.fgcmPars.parPWVSlopeLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF[noExtGOF],
                      deltaMagWeightedGOF[noExtGOF] * (
                    (self.fgcmPars.expDeltaUT[obsExpIndex[goodObs[obsFitUseGO[noExtGOF]]]] *
                     dLdPWVGO[obsFitUseGO[noExtGOF]] -
                     magdLdPWVSlope[expNightIndexGOF[noExtGOF],
                                    obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]])))

            partialArray[self.fgcmPars.parPWVSlopeLoc +
                         uNightIndex] *= (2.0 / unitDict['pwvSlopeUnit'])

            #############
            ## Tau External
            #############

            if (self.fgcmPars.hasExternalTau):
                hasExtGOF,=np.where(self.fgcmPars.externalTauFlag[obsExpIndex[goodObs[obsFitUseGO]]])

                # Tau Nightly Offset
                np.add.at(magdLdTauOffset,
                          (expNightIndexGOF[hasExtGOF],
                           obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]]),
                          dLdTauGO[obsFitUseGO[hasExtGOF]] /
                          obsMagErr2GO[obsFitUseGO[hasExtGOF]])
                np.multiply.at(magdLdTauOffset,
                               (expNightIndexGOF[hasExtGOF],
                                obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]]),
                               objMagStdMeanErr2[obsObjIDIndex[goodObs[obsFitUseGO[hasExtGOF]]],
                                                 obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]]])
                np.add.at(partialArray[self.fgcmPars.parExternalTauOffsetLoc:
                                           (self.fgcmPars.parExternalTauOffsetLoc+
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF[hasExtGOF],
                          deltaMagWeightedGOF[hasExtGOF] * (
                        (dLdTauGO[obsFitUseGO[hasExtGOF]] -
                         magdLdTauOffset[expNightIndexGOF[hasExtGOF],
                                         obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]]])))

                partialArray[self.fgcmPars.parExternalTauOffsetLoc +
                             uNightIndex] *= (2.0 / unitDict['tauUnit'])

                # Tau Global Scale
                ## MAYBE: is this correct with the logs?
                np.add.at(magdLdTauScale,
                          obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]],
                          self.fgcmPars.expTau[obsExpIndex[goodObs[obsFitUseGO[hasExtGOF]]]] *
                          dLdTauGO[obsFitUseGO[hasExtGOF]] /
                          obsMagErr2GO[obsFitUseGO[hasExtGOF]])
                np.multiply.at(magdLdTauScale,
                               obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]],
                               objMagStdMeanErr2[obsObjIDIndex[goodObs[obsFitUseGO[hasExtGOF]]],
                                             obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]]])
                partialArray[self.fgcmPars.parExternalTauScaleLoc] = 2.0 * (
                    np.sum(deltaMagWeightedGOF[hasExtGOF] * (
                            self.fgcmPars.expTau[obsExpIndex[goodObs[obsFitUseGO[hasExtGOF]]]] *
                            dLdTauGO[obsFitUseGO[hasExtGOF]] -
                            magdLdPWVScale[obsBandIndex[goodObs[obsFitUseGO[hasExtGOF]]]])) /
                    unitDict['tauUnit'])

            ###########
            ## Tau No External
            ###########

            noExtGOF, = np.where(~self.fgcmPars.externalTauFlag[obsExpIndex[goodObs[obsFitUseGO]]])

            # Tau Nightly Intercept
            np.add.at(magdLdTauIntercept,
                      (expNightIndexGOF[noExtGOF],
                       obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]),
                      dLdTauGO[obsFitUseGO[noExtGOF]] /
                      obsMagErr2GO[obsFitUseGO[noExtGOF]])
            np.multiply.at(magdLdTauIntercept,
                           (expNightIndexGOF[noExtGOF],
                            obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]),
                           objMagStdMeanErr2[obsObjIDIndex[goodObs[obsFitUseGO[noExtGOF]]],
                                             obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]])
            np.add.at(partialArray[self.fgcmPars.parTauInterceptLoc:
                                       (self.fgcmPars.parTauInterceptLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF[noExtGOF],
                      deltaMagWeightedGOF[noExtGOF] * (
                    (dLdTauGO[obsFitUseGO[noExtGOF]] -
                     magdLdTauOffset[expNightIndexGOF[noExtGOF],
                                     obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]])))

            partialArray[self.fgcmPars.parTauInterceptLoc +
                         uNightIndex] *= (2.0 / unitDict['tauUnit'])

            # Tau Nightly Slope
            np.add.at(magdLdTauSlope,
                      (expNightIndexGOF[noExtGOF],
                       obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]),
                      self.fgcmPars.expDeltaUT[obsExpIndex[goodObs[obsFitUseGO[noExtGOF]]]] *
                      dLdTauGO[obsFitUseGO[noExtGOF]] /
                      obsMagErr2GO[obsFitUseGO[noExtGOF]])
            np.multiply.at(magdLdTauSlope,
                           (expNightIndexGOF[noExtGOF],
                            obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]),
                           objMagStdMeanErr2[obsObjIDIndex[goodObs[obsFitUseGO[noExtGOF]]],
                                             obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]])
            np.add.at(partialArray[self.fgcmPars.parTauSlopeLoc:
                                       (self.fgcmPars.parTauSlopeLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF[noExtGOF],
                      deltaMagWeightedGOF[noExtGOF] * (
                    (self.fgcmPars.expDeltaUT[obsExpIndex[goodObs[obsFitUseGO[noExtGOF]]]] *
                     dLdTauGO[obsFitUseGO[noExtGOF]] -
                     magdLdTauSlope[expNightIndexGOF[noExtGOF],
                                    obsBandIndex[goodObs[obsFitUseGO[noExtGOF]]]])))

            partialArray[self.fgcmPars.parTauSlopeLoc +
                         uNightIndex] *= (2.0 / unitDict['tauSlopeUnit'])

            #############
            ## Washes (QE Sys)
            #############

            expWashIndexGOF = self.fgcmPars.expWashIndex[obsExpIndex[goodObs[obsFitUseGO]]]
            uWashIndex = np.unique(expWashIndexGOF)

            # Wash Intercept
            np.add.at(magdLdWashIntercept,
                      (expWashIndexGOF,obsBandIndex[goodObs[obsFitUseGO]]),
                      1./obsMagErr2GO[obsFitUseGO])
            np.multiply.at(magdLdWashIntercept,
                           (expWashIndexGOF,obsBandIndex[goodObs[obsFitUseGO]]),
                           objMagStdMeanErr2[obsObjIDIndex[goodObs[obsFitUseGO]],
                                             obsBandIndex[goodObs[obsFitUseGO]]])
            np.add.at(partialArray[self.fgcmPars.parQESysInterceptLoc:
                                       (self.fgcmPars.parQESysInterceptLoc +
                                        self.fgcmPars.nWashIntervals)],
                      expWashIndexGOF,
                      deltaMagWeightedGOF * (
                    (1.0 - magdLdWashIntercept[expWashIndexGOF,
                                               obsBandIndex[goodObs[obsFitUseGO]]])))

            partialArray[self.fgcmPars.parQESysInterceptLoc +
                         uWashIndex] *= (2.0 / unitDict['qeSysUnit'])

            # Wash Slope
            np.add.at(magdLdWashSlope,
                      (expWashIndexGOF,obsBandIndex[goodObs[obsFitUseGO]]),
                      (self.fgcmPars.expMJD[obsExpIndex[goodObs[obsFitUseGO]]] -
                       self.fgcmPars.washMJDs[expWashIndexGOF]) /
                       obsMagErr2GO[obsFitUseGO])
            np.multiply.at(magdLdWashSlope,
                           (expWashIndexGOF,obsBandIndex[goodObs[obsFitUseGO]]),
                           objMagStdMeanErr2[obsObjIDIndex[goodObs[obsFitUseGO]],
                                             obsBandIndex[goodObs[obsFitUseGO]]])
            np.add.at(partialArray[self.fgcmPars.parQESysSlopeLoc:
                                       (self.fgcmPars.parQESysSlopeLoc +
                                        self.fgcmPars.nWashIntervals)],
                      expWashIndexGOF,
                      deltaMagWeightedGOF * (
                    (self.fgcmPars.expMJD[obsExpIndex[goodObs[obsFitUseGO]]] -
                     self.fgcmPars.washMJDs[expWashIndexGOF]) -
                    magdLdWashSlope[expWashIndexGOF,
                                    obsBandIndex[goodObs[obsFitUseGO]]]))
            partialArray[self.fgcmPars.parQESysSlopeLoc +
                         uWashIndex] *= (2.0 / unitDict['qeSysSlopeUnit'])


        # note that this store doesn't need locking because we only access
        #  a given array from a singel process
        if self.debug:
            thisCore = 0
        else:
            thisCore = multiprocessing.current_process()._identity[0]

        totalArr = snmm.getArray(self.totalHandleDict[thisCore])
        totalArr[:] = totalArr[:] + partialArray


        # and we're done
        return None
