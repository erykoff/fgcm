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
        self.nStarPerRun = fgcmConfig.nStarPerRun

        self.resetFitChisqList()

        # this is the default number of parameters
        self.nActualFitPars = self.fgcmPars.nFitPars
        self.fgcmLog.log('INFO','Default: fit %d parameters.' % (self.nActualFitPars))


    def resetFitChisqList(self):
        self.fitChisqs = []

    def __call__(self,fitParams,fitterUnits=False,computeDerivatives=False,computeSEDSlopes=False,debug=False,allExposures=False):
        """
        """

        # computeDerivatives: do we want to compute the derivatives?
        # computeSEDSlope: compute SED Slope and recompute mean mags?
        # fitterUnits: units of th fitter or "true" units?

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

        # do global pre-matching before giving to workers, because
        #  it is faster this way

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        preStartTime=time.time()
        self.fgcmLog.log('INFO','Pre-matching stars and observations...')
        goodStarsSub,goodObs = esutil.numpy_util.match(goodStars,
                                                       obsObjIDIndex,
                                                       presorted=True)

        if (goodStarsSub[0] != 0.0):
            raise ValueError("Very strange that the goodStarsSub first element is non-zero.")

        if (not self.allExposures):
            # cut out all bad exposures and bad observations
            gd,=np.where((self.fgcmPars.expFlag[obsExpIndex[goodObs]] == 0) &
                         (obsFlag[goodObs] == 0))
        else:
            # just cut out bad observations
            gd,=np.where(obsFlag[goodObs] == 0)

        goodObs=goodObs[gd]

        self.fgcmLog.log('INFO','Pre-matching done in %.1f sec.' %
                         (time.time() - preStartTime))


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

            prepStartTime = time.time()
            nSections = goodStars.size // self.nStarPerRun + 1
            goodStarsList = np.array_split(goodStars,nSections)


            # is there a better way of getting all the first elements from the list?
            #  note that we need to skip the first which should be zero (checked above)
            #  see also fgcmBrightObs.py
            splitValues = np.zeros(nSections-1,dtype='i4')
            for i in xrange(1,nSections):
                splitValues[i-1] = goodStarsList[i][0]

            # get the indices from the goodStarsSub matched list
            splitIndices = np.searchsorted(goodStarsSub, splitValues)

            # and split along the indices
            goodObsList = np.split(goodObs,splitIndices)

            workerList = zip(goodStarsList,goodObsList)

            # reverse sort so the longest running go first
            workerList.sort(key=lambda elt:elt[1].size, reverse=True)

            self.fgcmLog.log('INFO','Using %d sections (%.1f seconds)' %
                             (nSections,time.time()-prepStartTime))

            # make a pool
            pool = Pool(processes=self.nCore)
            #pool.map(self._worker,goodStarsList)
            pool.map(self._worker,workerList)
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
                self.fgcmLog.log('INFO','Actually fit %d parameters.' % (self.nActualFitPars))

            fitDOF = partialSums[-1] - float(self.nActualFitPars)

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

    #def _worker(self,goodStars):
    def _worker(self,goodStarsAndObs):
        """
        """

        goodStars = goodStarsAndObs[0]
        goodObs = goodStarsAndObs[1]

        if self.debug:
            thisCore = 0
        else:
            thisCore = multiprocessing.current_process()._identity[0]


        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)
        obsSecZenith = snmm.getArray(self.fgcmStars.obsSecZenithHandle)
        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        obsMagADUErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)

        #startTime = time.time()

        #_,goodObs=esutil.numpy_util.match(goodStars,obsObjIDIndex,presorted=True)

        #self.fgcmLog.log('DEBUG','chisq %d: matched stars and obs in %.1f sec.' %
        #                 (thisCore, time.time() - startTime),printOnly=True)

        startTime = time.time()
        if (not self.allExposures):
            # if we aren't doing all exposures, cut to expFlag == 0 exposures
            gd,=np.where((self.fgcmPars.expFlag[obsExpIndex[goodObs]] == 0) &
                         (obsFlag[goodObs] == 0))
            goodObs = goodObs[gd]
        else:
            # we are doing all exposures, still cut out bad observations
            gd,=np.where(obsFlag[goodObs] == 0)
            goodObs = goodObs[gd]
        self.fgcmLog.log('DEBUG','chisq %d: cut to good exps in %.1f sec.' %
                         (thisCore, time.time() - startTime),printOnly=True)

        startTime = time.time()
        # cut these down now, faster later
        obsObjIDIndexGO = obsObjIDIndex[goodObs]
        obsBandIndexGO = obsBandIndex[goodObs]
        obsExpIndexGO = obsExpIndex[goodObs]
        obsSecZenithGO = obsSecZenith[goodObs]
        obsCCDIndexGO = obsCCDIndex[goodObs]
        self.fgcmLog.log('DEBUG','chisq %d: cut GO in %.1f sec.' %
                         (thisCore, time.time() - startTime),printOnly=True)


        startTime = time.time()
        # which observations are used in the fit?
        _,obsFitUseGO = esutil.numpy_util.match(self.fgcmPars.fitBandIndex,
                                                obsBandIndexGO)
        self.fgcmLog.log('DEBUG','chisq %d: obsFitUseGO in %.1f sec.' %
                         (thisCore, time.time() - startTime),printOnly=True)


        # now refer to obsBandIndex[goodObs]
        # add GO to index names that are cut to goodObs
        # add GOF to index names that are cut to goodObs[obsFitUseGO]

        startTime = time.time()
        lutIndicesGO = self.fgcmLUT.getIndices(obsBandIndexGO,
                                               self.fgcmPars.expPWV[obsExpIndexGO],
                                               self.fgcmPars.expO3[obsExpIndexGO],
                                               np.log(self.fgcmPars.expTau[obsExpIndexGO]),
                                               self.fgcmPars.expAlpha[obsExpIndexGO],
                                               obsSecZenithGO,
                                               obsCCDIndexGO,
                                               self.fgcmPars.expPmb[obsExpIndexGO])
        I0GO = self.fgcmLUT.computeI0(obsBandIndexGO,
                                      self.fgcmPars.expPWV[obsExpIndexGO],
                                      self.fgcmPars.expO3[obsExpIndexGO],
                                      np.log(self.fgcmPars.expTau[obsExpIndexGO]),
                                      self.fgcmPars.expAlpha[obsExpIndexGO],
                                      obsSecZenithGO,
                                      obsCCDIndexGO,
                                      self.fgcmPars.expPmb[obsExpIndexGO],
                                      lutIndicesGO)
        I10GO = self.fgcmLUT.computeI1(lutIndicesGO) / I0GO

        self.fgcmLog.log('DEBUG','chisq %d: LUTs in %.1f sec.' %
                         (thisCore, time.time() - startTime),printOnly=True)


        qeSysGO = self.fgcmPars.expQESys[obsExpIndexGO]

        obsMagGO = obsMagADU[goodObs] + 2.5*np.log10(I0GO) + qeSysGO

        # this is annoying that we have to do this.
        obsMagErr2GO = obsMagADUErr[goodObs]**2.

        if (self.computeSEDSlopes):
            wtSum = np.zeros_like(objMagStdMean,dtype='f8')
            np.add.at(wtSum,
                      (obsObjIDIndexGO,obsBandIndexGO),
                      1./obsMagErr2GO)
            gd=np.where(wtSum > 0.0)
            # important: only zero accumulator for our stars
            objMagStdMean[gd] = 0.0

            # note that obsMag is already cut to goodObs
            np.add.at(objMagStdMean,
                      (obsObjIDIndexGO,obsBandIndexGO),
                      obsMagGO/obsMagErr2GO)

            objMagStdMean[gd] /= wtSum[gd]
            objMagStdMeanErr[gd] = np.sqrt(1./wtSum[gd])

            self.fgcmStars.computeObjectSEDSlopes(goodStars)

        # compute linearized chromatic correction
        deltaStdGO = 2.5 * np.log10((1.0 +
                                   objSEDSlope[obsObjIDIndexGO,
                                               obsBandIndexGO] * I10GO) /
                                  (1.0 + objSEDSlope[obsObjIDIndexGO,
                                                     obsBandIndexGO] *
                                   self.fgcmLUT.I10Std[obsBandIndexGO]))

        # we can only do this for calibration stars.
        #  must reference the full array to save
        obsMagStd[goodObs] = obsMagGO + deltaStdGO

        if (self.allExposures) :
            # kick out
            return None

        # this is cut here
        obsMagStdGO = obsMagStd[goodObs]

        ## FIXME: change to only make memory for the current objects under consideration?
        ##        (worried about memory usage...)

        startTime = time.time()
        # compute mean mags
        wtSum = np.zeros_like(objMagStdMean,dtype='f8')
        np.add.at(wtSum,
                  (obsObjIDIndexGO,obsBandIndexGO),
                  1./obsMagErr2GO)
        # only zero out the accumulator where we have observations of objects!
        gd=np.where(wtSum > 0.0)
        objMagStdMean[gd] = 0.0

        np.add.at(objMagStdMean,
                  (obsObjIDIndexGO,obsBandIndexGO),
                  obsMagStdGO/obsMagErr2GO)

        objMagStdMean[gd] /= wtSum[gd]
        objMagStdMeanErr[gd] = np.sqrt(1./wtSum[gd])

        self.fgcmLog.log('DEBUG','chisq %d: mean mags in %.1f sec.' %
                         (thisCore, time.time() - startTime),printOnly=True)

        # compute delta-mags

        startTime = time.time()

        deltaMagGO = (obsMagStdGO -
                      objMagStdMean[obsObjIDIndexGO,
                                    obsBandIndexGO])
        deltaMagErr2GO = (obsMagErr2GO +
                          objMagStdMeanErr[obsObjIDIndexGO,
                                           obsBandIndexGO]**2.)
        deltaMagWeightedGOF = deltaMagGO[obsFitUseGO] / deltaMagErr2GO[obsFitUseGO]

        # and compute chisq
        partialChisq = np.sum(deltaMagGO[obsFitUseGO]**2./deltaMagErr2GO[obsFitUseGO])

        partialArray = np.zeros(self.nSums,dtype='f8')
        partialArray[-2] = partialChisq
        partialArray[-1] = obsFitUseGO.size

        self.fgcmLog.log('DEBUG','chisq %d: deltas and chisq in %.1f sec.' %
                         (thisCore, time.time() - startTime),printOnly=True)

        if (self.computeDerivatives):
            unitDict=self.fgcmPars.getUnitDict(fitterUnits=self.fitterUnits)

            # this is going to be ugly.  wow, how many indices and sub-indices?
            #  or does it simplify since we need all the obs on a night?
            #  we shall see!  And speed up!

            (dLdPWVGO,dLdO3GO,dLdTauGO,dLdAlphaGO) = (
                self.fgcmLUT.computeLogDerivatives(lutIndicesGO,
                                                   I0GO,
                                                   self.fgcmPars.expTau[
                        obsExpIndexGO]))


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

            #expNightIndexGOF = self.fgcmPars.expNightIndex[obsExpIndex[goodObs[obsFitUseGO]]]
            expNightIndexGOF = self.fgcmPars.expNightIndex[obsExpIndexGO[obsFitUseGO]]
            uNightIndex = np.unique(expNightIndexGOF)

            np.add.at(magdLdO3,
                      (expNightIndexGOF,obsBandIndexGO[obsFitUseGO]),
                      dLdO3GO[obsFitUseGO] / obsMagErr2GO[obsFitUseGO])
            np.multiply.at(magdLdO3,
                           (expNightIndexGOF,obsBandIndexGO[obsFitUseGO]),
                           objMagStdMeanErr2[obsObjIDIndexGO[obsFitUseGO],
                                             obsBandIndexGO[obsFitUseGO]])
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
                           objMagStdMeanErr2[obsObjIDIndexGO[obsFitUseGO],
                                             obsBandIndexGO[obsFitUseGO]])
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

            if (self.fgcmPars.hasExternalPWV):
                #hasExtGOF,=np.where(self.fgcmPars.externalPWVFlag[obsExpIndex[goodObs[obsFitUseGO]]])
                hasExtGOF,=np.where(self.fgcmPars.externalPWVFlag[obsExpIndexGO[obsFitUseGO]])
                uNightIndexHasExt = np.unique(expNightIndexGOF[hasExtGOF])

                # PWV Nightly Offset
                np.add.at(magdLdPWVOffset,
                          (expNightIndexGOF[hasExtGOF],
                           obsBandIndexGO[obsFitUseGO[hasExtGOF]]),
                          dLdPWVGO[obsFitUseGO[hasExtGOF]] /
                          obsMagErr2GO[obsFitUseGO[hasExtGOF]])
                np.multiply.at(magdLdPWVOffset,
                               (expNightIndexGOF[hasExtGOF],
                                obsBandIndexGO[obsFitUseGO[hasExtGOF]]),
                               objMagStdMeanErr2[obsObjIDIndexGO[obsFitUseGO[hasExtGOF]],
                                                 obsBandIndexGO[obsFitUseGO[hasExtGOF]]])
                np.add.at(partialArray[self.fgcmPars.parExternalPWVOffsetLoc:
                                           (self.fgcmPars.parExternalPWVOffsetLoc+
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF[hasExtGOF],
                          deltaMagWeightedGOF[hasExtGOF] * (
                        (dLdPWVGO[obsFitUseGO[hasExtGOF]] -
                         magdLdPWVOffset[expNightIndexGOF[hasExtGOF],
                                         obsBandIndexGO[obsFitUseGO[hasExtGOF]]])))
                partialArray[self.fgcmPars.parExternalPWVOffsetLoc +
                             uNightIndexHasExt] *= (2.0 / unitDict['pwvUnit'])
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parExternalPWVOffsetLoc +
                             uNightIndexHasExt] += 1


                # PWV Global Scale
                np.add.at(magdLdPWVScale,
                          obsBandIndexGO[obsFitUseGO[hasExtGOF]],
                          self.fgcmPars.expPWV[obsExpIndexGO[obsFitUseGO[hasExtGOF]]] *
                          dLdPWVGO[obsFitUseGO[hasExtGOF]] /
                          obsMagErr2GO[obsFitUseGO[hasExtGOF]])
                np.multiply.at(magdLdPWVScale,
                               obsBandIndexGO[obsFitUseGO[hasExtGOF]],
                               objMagStdMeanErr2[obsObjIDIndexGO[obsFitUseGO[hasExtGOF]],
                                             obsBandIndexGO[obsFitUseGO[hasExtGOF]]])
                partialArray[self.fgcmPars.parExternalPWVScaleLoc] = 2.0 * (
                    np.sum(deltaMagWeightedGOF[hasExtGOF] * (
                            self.fgcmPars.expPWV[obsExpIndexGO[obsFitUseGO[hasExtGOF]]] *
                            dLdPWVGO[obsFitUseGO[hasExtGOF]] -
                            magdLdPWVScale[obsBandIndexGO[obsFitUseGO[hasExtGOF]]])) /
                    unitDict['pwvUnit'])
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parExternalPWVScaleLoc] += 1

            ###########
            ## PWV No External
            ###########

            #noExtGOF, = np.where(~self.fgcmPars.externalPWVFlag[obsExpIndex[goodObs[obsFitUseGO]]])
            noExtGOF, = np.where(~self.fgcmPars.externalPWVFlag[obsExpIndexGO[obsFitUseGO]])
            uNightIndexNoExt = np.unique(expNightIndexGOF[noExtGOF])

            # PWV Nightly Intercept

            np.add.at(magdLdPWVIntercept,
                      (expNightIndexGOF[noExtGOF],
                       obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                      dLdPWVGO[obsFitUseGO[noExtGOF]] /
                      obsMagErr2GO[obsFitUseGO[noExtGOF]])
            np.multiply.at(magdLdPWVIntercept,
                           (expNightIndexGOF[noExtGOF],
                            obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                           objMagStdMeanErr2[obsObjIDIndexGO[obsFitUseGO[noExtGOF]],
                                             obsBandIndexGO[obsFitUseGO[noExtGOF]]])
            np.add.at(partialArray[self.fgcmPars.parPWVInterceptLoc:
                                       (self.fgcmPars.parPWVInterceptLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF[noExtGOF],
                      deltaMagWeightedGOF[noExtGOF] * (
                    (dLdPWVGO[obsFitUseGO[noExtGOF]] -
                     magdLdPWVOffset[expNightIndexGOF[noExtGOF],
                                     obsBandIndexGO[obsFitUseGO[noExtGOF]]])))

            partialArray[self.fgcmPars.parPWVInterceptLoc +
                         uNightIndexNoExt] *= (2.0 / unitDict['pwvUnit'])
            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parPWVInterceptLoc +
                         uNightIndexNoExt] += 1

            # PWV Nightly Slope
            np.add.at(magdLdPWVSlope,
                      (expNightIndexGOF[noExtGOF],
                       obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                      self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]] *
                      dLdPWVGO[obsFitUseGO[noExtGOF]] /
                      obsMagErr2GO[obsFitUseGO[noExtGOF]])
            np.multiply.at(magdLdPWVSlope,
                           (expNightIndexGOF[noExtGOF],
                            obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                           objMagStdMeanErr2[obsObjIDIndexGO[obsFitUseGO[noExtGOF]],
                                             obsBandIndexGO[obsFitUseGO[noExtGOF]]])
            np.add.at(partialArray[self.fgcmPars.parPWVSlopeLoc:
                                       (self.fgcmPars.parPWVSlopeLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF[noExtGOF],
                      deltaMagWeightedGOF[noExtGOF] * (
                    (self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]] *
                     dLdPWVGO[obsFitUseGO[noExtGOF]] -
                     magdLdPWVSlope[expNightIndexGOF[noExtGOF],
                                    obsBandIndexGO[obsFitUseGO[noExtGOF]]])))

            partialArray[self.fgcmPars.parPWVSlopeLoc +
                         uNightIndex] *= (2.0 / unitDict['pwvSlopeUnit'])
            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parPWVSlopeLoc] += 1

            #############
            ## Tau External
            #############

            if (self.fgcmPars.hasExternalTau):
                #hasExtGOF,=np.where(self.fgcmPars.externalTauFlag[obsExpIndex[goodObs[obsFitUseGO]]])
                hasExtGOF,=np.where(self.fgcmPars.externalTauFlag[obsExpIndexGO[obsFitUseGO]])
                uNightIndexHasExt = np.unique(expNightIndexGOF[hasExtGOF])

                # Tau Nightly Offset
                np.add.at(magdLdTauOffset,
                          (expNightIndexGOF[hasExtGOF],
                           obsBandIndexGO[obsFitUseGO[hasExtGOF]]),
                          dLdTauGO[obsFitUseGO[hasExtGOF]] /
                          obsMagErr2GO[obsFitUseGO[hasExtGOF]])
                np.multiply.at(magdLdTauOffset,
                               (expNightIndexGOF[hasExtGOF],
                                obsBandIndexGO[obsFitUseGO[hasExtGOF]]),
                               objMagStdMeanErr2[obsObjIDIndexGO[obsFitUseGO[hasExtGOF]],
                                                 obsBandIndexGO[obsFitUseGO[hasExtGOF]]])
                np.add.at(partialArray[self.fgcmPars.parExternalTauOffsetLoc:
                                           (self.fgcmPars.parExternalTauOffsetLoc+
                                            self.fgcmPars.nCampaignNights)],
                          expNightIndexGOF[hasExtGOF],
                          deltaMagWeightedGOF[hasExtGOF] * (
                        (dLdTauGO[obsFitUseGO[hasExtGOF]] -
                         magdLdTauOffset[expNightIndexGOF[hasExtGOF],
                                         obsBandIndexGO[obsFitUseGO[hasExtGOF]]])))

                partialArray[self.fgcmPars.parExternalTauOffsetLoc +
                             uNightIndexHasExt] *= (2.0 / unitDict['tauUnit'])
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parExternalTauOffsetLoc +
                             uNightIndexHasExt] += 1

                # Tau Global Scale
                ## MAYBE: is this correct with the logs?
                np.add.at(magdLdTauScale,
                          obsBandIndexGO[obsFitUseGO[hasExtGOF]],
                          self.fgcmPars.expTau[obsExpIndexGO[obsFitUseGO[hasExtGOF]]] *
                          dLdTauGO[obsFitUseGO[hasExtGOF]] /
                          obsMagErr2GO[obsFitUseGO[hasExtGOF]])
                np.multiply.at(magdLdTauScale,
                               obsBandIndexGO[obsFitUseGO[hasExtGOF]],
                               objMagStdMeanErr2[obsObjIDIndexGO[obsFitUseGO[hasExtGOF]],
                                             obsBandIndexGO[obsFitUseGO[hasExtGOF]]])
                partialArray[self.fgcmPars.parExternalTauScaleLoc] = 2.0 * (
                    np.sum(deltaMagWeightedGOF[hasExtGOF] * (
                            self.fgcmPars.expTau[obsExpIndexGO[obsFitUseGO[hasExtGOF]]] *
                            dLdTauGO[obsFitUseGO[hasExtGOF]] -
                            magdLdPWVScale[obsBandIndexGO[obsFitUseGO[hasExtGOF]]])) /
                    unitDict['tauUnit'])
                partialArray[self.fgcmPars.nFitPars +
                             self.fgcmPars.parExternalTauScaleLoc] += 1

            ###########
            ## Tau No External
            ###########

            #noExtGOF, = np.where(~self.fgcmPars.externalTauFlag[obsExpIndex[goodObs[obsFitUseGO]]])
            noExtGOF, = np.where(~self.fgcmPars.externalTauFlag[obsExpIndexGO[obsFitUseGO]])
            uNightIndexNoExt = np.unique(expNightIndexGOF[noExtGOF])

            # Tau Nightly Intercept
            np.add.at(magdLdTauIntercept,
                      (expNightIndexGOF[noExtGOF],
                       obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                      dLdTauGO[obsFitUseGO[noExtGOF]] /
                      obsMagErr2GO[obsFitUseGO[noExtGOF]])
            np.multiply.at(magdLdTauIntercept,
                           (expNightIndexGOF[noExtGOF],
                            obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                           objMagStdMeanErr2[obsObjIDIndexGO[obsFitUseGO[noExtGOF]],
                                             obsBandIndexGO[obsFitUseGO[noExtGOF]]])
            np.add.at(partialArray[self.fgcmPars.parTauInterceptLoc:
                                       (self.fgcmPars.parTauInterceptLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF[noExtGOF],
                      deltaMagWeightedGOF[noExtGOF] * (
                    (dLdTauGO[obsFitUseGO[noExtGOF]] -
                     magdLdTauOffset[expNightIndexGOF[noExtGOF],
                                     obsBandIndexGO[obsFitUseGO[noExtGOF]]])))

            partialArray[self.fgcmPars.parTauInterceptLoc +
                         uNightIndexNoExt] *= (2.0 / unitDict['tauUnit'])
            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parTauInterceptLoc +
                         uNightIndexNoExt] += 1

            # Tau Nightly Slope
            np.add.at(magdLdTauSlope,
                      (expNightIndexGOF[noExtGOF],
                       obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                      self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]] *
                      dLdTauGO[obsFitUseGO[noExtGOF]] /
                      obsMagErr2GO[obsFitUseGO[noExtGOF]])
            np.multiply.at(magdLdTauSlope,
                           (expNightIndexGOF[noExtGOF],
                            obsBandIndexGO[obsFitUseGO[noExtGOF]]),
                           objMagStdMeanErr2[obsObjIDIndexGO[obsFitUseGO[noExtGOF]],
                                             obsBandIndexGO[obsFitUseGO[noExtGOF]]])
            np.add.at(partialArray[self.fgcmPars.parTauSlopeLoc:
                                       (self.fgcmPars.parTauSlopeLoc+
                                        self.fgcmPars.nCampaignNights)],
                      expNightIndexGOF[noExtGOF],
                      deltaMagWeightedGOF[noExtGOF] * (
                    (self.fgcmPars.expDeltaUT[obsExpIndexGO[obsFitUseGO[noExtGOF]]] *
                     dLdTauGO[obsFitUseGO[noExtGOF]] -
                     magdLdTauSlope[expNightIndexGOF[noExtGOF],
                                    obsBandIndexGO[obsFitUseGO[noExtGOF]]])))

            partialArray[self.fgcmPars.parTauSlopeLoc +
                         uNightIndexNoExt] *= (2.0 / unitDict['tauSlopeUnit'])
            partialArray[self.fgcmPars.nFitPars +
                         self.fgcmPars.parTauSlopeLoc +
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
                           objMagStdMeanErr2[obsObjIDIndexGO[obsFitUseGO],
                                             obsBandIndexGO[obsFitUseGO]])
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
                           objMagStdMeanErr2[obsObjIDIndexGO[obsFitUseGO],
                                             obsBandIndexGO[obsFitUseGO]])
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
