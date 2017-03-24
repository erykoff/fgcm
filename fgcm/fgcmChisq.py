from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import time

from fgcmUtilities import _pickle_method
from fgcmUtilities import resourceUsage

import types
import copy_reg
#import sharedmem as shm
import multiprocessing
from multiprocessing import Pool

from sharedNumpyMemManager import SharedNumpyMemManager as snmm


#from fgcmLUT import FgcmLUTSHM

copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmChisq(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmStars,fgcmLUT):

        resourceUsage('Start of chisq init')

        # does this need to be shm'd?
        self.fgcmPars = fgcmPars

        # this is shm'd
        self.fgcmLUT = fgcmLUT

        # also shm'd
        self.fgcmStars = fgcmStars

        # need to configure
        self.nCore = 4

        self.fitChisqs = []

        # not sure what we need the config for

        resourceUsage('End of chisq init')

    def __call__(self,fitParams,fitterUnits=False,computeDerivatives=False,computeSEDSlopes=False,debug=False):
        """
        """

        # computeDerivatives: do we want to compute the derivatives?
        # computeSEDSlope: compute SED Slope and recompute mean mags?
        # fitterUnits: units of the fitter or "true" units?

        self.computeDerivatives = computeDerivatives
        self.computeSEDSlopes = computeSEDSlopes
        self.fitterUnits = fitterUnits

        # for things that need to be changed, we need to create an array *here*
        # I think.  And copy it back out.  Sigh.

        #resourceUsage('Start of call')

        # this is the function that will be called by the fitter, I believe.

        # unpack the parameters and convert units if necessary. These are not
        # currently shared memory, since they should be small enough to not be
        # a problem.  But we can revisit.

        self.fgcmPars.reloadParArray(fitParams,fitterUnits=self.fitterUnits)
        self.fgcmPars.parsToExposures()


        a,b=esutil.numpy_util.match(self.fgcmPars.expArray,
                                    snmm.getArray(self.fgcmStars.obsExpHandle)[:])
        self.obsExpIndexHandle = snmm.createArray(a.size,dtype='i4')
        snmm.getArray(self.obsExpIndexHandle)[b] = a

        # and reset numbers
        snmm.getArray(self.fgcmStars.objMagStdMeanHandle)[:] = 99.0
        snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)[:] = 99.0

        # and select good stars!  These are the ones to map.
        goodStars,=np.where(snmm.getArray(self.fgcmStars.starFlagHandle) == 0)

        # testing
        goodStars=goodStars[0:10000]

        # prepare the return arrays...
        # how many do we have?

        self.nSums = 2   # chisq, nobs
        if (self.computeDerivatives):
            self.nSums += self.fgcmPars.nFitPars  # one for each parameter

        self.debug=debug
        if (self.debug):
            self.totalHandleDict = {}
            self.totalHandleDict[0] = snmm.createArray(self.nSums,dtype='f4')

            for goodStar in goodStars:
                self._worker(goodStar)

            partialSums = snmm.getArray(self.totalHandleDict[0])[:]
        else:
            # make a dummy process to discover starting child number
            proc = multiprocessing.Process()
            workerIndex = proc._identity[0]+1
            proc = None

            self.totalHandleDict = {}
            for thisCore in xrange(self.nCore):
                self.totalHandleDict[workerIndex + thisCore] = snmm.createArray(self.nSums,dtype='f4')

            # will want to make a pool

            pool = Pool(processes=4)
            #resourceUsage('premap')
            pool.map(self._worker,goodStars)
            pool.close()
            pool.join()

            # and return the derivatives + chisq
            partialSums = np.zeros(self.nSums,dtype='f8')
            for thisCore in xrange(self.nCore):
                partialSums[:] += snmm.getArray(self.totalHandleDict[workerIndex + thisCore])[:]

        fitDOF = partialSums[-1] - float(self.fgcmPars.nFitPars)
        if (fitDOF <= 0):
            raise ValueError("Number of parameters fitted is more than number of constraints! (%d > %d)" % (self.fgcmPars.nFitPars,partialSums[-1]))

        fitChisq = partialSums[-2] / fitDOF
        if (self.computeDerivatives):
            dChisqdP = partialSums[0:self.fgcmPars.nFitPars] / fitDOF

        # want to append this...
        self.fitChisqs.append(fitChisq)

        # free shared arrays
        snmm.freeArray(self.obsExpIndexHandle)
        for key in self.totalHandleDict.keys():
            snmm.freeArray(self.totalHandleDict[key])

        #resourceUsage('end')

        print(fitChisq)

        if (self.computeDerivatives):
            return fitChisq, dChisqdP
        else:
            return fitChisq

    def _worker(self,objIndex):
        """
        """

        #print("In worker...")

        # make local pointers to useful arrays...
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        objNobs = snmm.getArray(self.fgcmStars.objNobsHandle)

        thisObsIndex = obsIndex[objObsIndex[objIndex]:objObsIndex[objIndex]+objNobs[objIndex]]
        thisObsExpIndex = snmm.getArray(self.obsExpIndexHandle)[thisObsIndex]

        # cut to good exposures
        #  I think this can be done in the parent more efficiently...but not now.
        gd,=np.where(self.fgcmPars.expFlag[thisObsExpIndex] == 0)
        thisObsIndex=thisObsIndex[gd]
        thisObsExpIndex = thisObsExpIndex[gd]

        thisObsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)[thisObsIndex]
        thisObsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle)[thisObsIndex] - 1

        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        obsMagADUErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)

        # need to know which are fit bands!
        #print("matching to fit bands...")
        _,thisObsFitUse = esutil.numpy_util.match(self.fgcmPars.fitBandIndex,thisObsBandIndex)

        # these IDs check out!
        #print(snmm.getArray(self.fgcmStars.obsObjIDHandle)[thisObsIndex])

        # need to compute secZenith
        #  is this the right place for it?  I don't know!
        thisObjRA = np.radians(snmm.getArray(self.fgcmStars.objRAHandle)[objIndex])
        thisObjDec = np.radians(snmm.getArray(self.fgcmStars.objDecHandle)[objIndex])
        if (thisObjRA > np.pi) :
            thisObjRA -= 2*np.pi
        thisObjHA = (self.fgcmPars.expTelHA[thisObsExpIndex] +
                     self.fgcmPars.expTelRA[thisObsExpIndex] -
                     thisObjRA)
        thisSecZenith = 1./(np.sin(thisObjDec)*self.fgcmPars.sinLatitude +
                        np.cos(thisObjDec)*self.fgcmPars.cosLatitude*np.cos(thisObjHA))

        #print(thisObsBandIndex)
        #print(thisSecZenith)

        # get I0obs values...
        #print("Going to LUT!")
        lutIndices = self.fgcmLUT.getIndices(thisObsBandIndex,
                                             self.fgcmPars.expPWV[thisObsExpIndex],
                                             self.fgcmPars.expO3[thisObsExpIndex],
                                             np.log(self.fgcmPars.expTau[thisObsExpIndex]),
                                             self.fgcmPars.expAlpha[thisObsExpIndex],
                                             thisSecZenith,
                                             thisObsCCDIndex,
                                             self.fgcmPars.expPmb[thisObsExpIndex])

        #print("exp:",self.fgcmPars.expArray[thisObsExpIndex])
        #print("pwv:",self.fgcmPars.expPWV[thisObsExpIndex])
        #print("o3:",self.fgcmPars.expO3[thisObsExpIndex])
        #print("tau:",self.fgcmPars.expTau[thisObsExpIndex])
        #print("alpha:",self.fgcmPars.expAlpha[thisObsExpIndex])
        #print("ccd:",thisObsCCDIndex)
        #print("pmb:",self.fgcmPars.expPmb[thisObsExpIndex])

        # and I10obs values...
        thisI0 = self.fgcmLUT.computeI0(thisObsBandIndex,
                                        self.fgcmPars.expPWV[thisObsExpIndex],
                                        self.fgcmPars.expO3[thisObsExpIndex],
                                        np.log(self.fgcmPars.expTau[thisObsExpIndex]),
                                        self.fgcmPars.expAlpha[thisObsExpIndex],
                                        thisSecZenith,
                                        thisObsCCDIndex,
                                        self.fgcmPars.expPmb[thisObsExpIndex],
                                        lutIndices)
        thisI10 = self.fgcmLUT.computeI1(lutIndices) / thisI0

        #print("I0:",thisI0)
        #print("I10:",thisI10)

        thisQESys = self.fgcmPars.expQESys[thisObsExpIndex]

        # compute thisMagObs
        thisMagObs = obsMagADU[thisObsIndex] + 2.5*np.log10(thisI0) + thisQESys

        thisMagErr2 = obsMagADUErr[thisObsIndex]**2.


        if (self.computeSEDSlopes):
            #print("Computing SED Slopes")
            # use magObs to compute mean mags...
            # compute in all bands here.

            # how much time is the where taking?
            for i in xrange(self.fgcmStars.nBands):
                inBand,=np.where(thisObsBandIndex == i)
                if (inBand.size > 0):
                    wtSum = np.sum(1./thisMagErr2[inBand])
                    objMagStdMeanErr[objIndex,i] = np.sqrt(1./wtSum)
                    objMagStdMean[objIndex,i] = np.sum(obsMagStd[thisObsIndex[inBand]]/
                                                       thisMagErr2[inBand])/wtSum

            self.fgcmStars.computeObjectSEDSlope(objIndex)

            #if (np.max(objMagStdMean[objIndex,:]) > 90.0) :
                # cannot compute
            #    objSEDSlope[objIndex,:] = 0.0
            #else:
                # need to do FIT BANDS
                #   FIXME
            #    S = np.zeros(self.fgcmPars.nBands-1,dtype='f4')
            #    for i in xrange(self.fgcmPars.nBands-1):
            #        S[i] = -0.921 * (objMagStdMean[objIndex,i+1] - objMagStdMean[objIndex,i])/(self.fgcmLUT.lambdaStd[i+1] - self.fgcmLUT.lambdaStd[i])

                # this is hacked for now
            #    objSEDSlope[objIndex,0] = S[0] - 1.0 * ((self.fgcmLUT.lambdaStd[1] - self.fgcmLUT.lambdaStd[0])/(self.fgcmLUT.lambdaStd[2]-self.fgcmLUT.lambdaStd[0])) * (S[1]-S[0])
            #    objSEDSlope[objIndex,1] = (S[0] + S[1])/2.0
            #    objSEDSlope[objIndex,2] = (S[1] + S[2])/2.0
            #    objSEDSlope[objIndex,3] = S[2] + 0.5 * ((self.fgcmLUT.lambdaStd[3]-self.fgcmLUT.lambdaStd[2])/(self.fgcmLUT.lambdStd[3]-self.fgcmLUT.lambdaStd[1])) * (S[2] - S[1])
            #    if ((objMagStdMean[objIndex,4]) < 90.0):
            #        objSEDSlope[objIndex,4] = S[2] + 1.0 * ((self.fgcmLUT.lambdaStd[3]-self.fgcmLUT.lambdaStd[2])/(self.fgcmLUT.lambdaStd[3]-self.fgcmLUT.lambdaStd[1])) * (S[2]-S[1])

        # compute magStd (and record)
        thisDeltaStd = 2.5 * np.log10((1.0 + objSEDSlope[objIndex,thisObsBandIndex] * thisI10) / (1.0 + objSEDSlope[objIndex,thisObsBandIndex] * self.fgcmLUT.I10Std[thisObsBandIndex]))

        obsMagStd[thisObsIndex] = thisMagObs + thisDeltaStd

        # compute mean objMagStdMean
        #print("Computing mean mags...")
        for i in xrange(self.fgcmStars.nBands):
            inBand,=np.where(thisObsBandIndex == i)
            if (inBand.size > 0):
                wtSum = np.sum(1./thisMagErr2[inBand])
                objMagStdMeanErr[objIndex,i] = np.sqrt(1./wtSum)
                objMagStdMean[objIndex,i] = np.sum(obsMagStd[thisObsIndex[inBand]]/
                                                   thisMagErr2[inBand])/wtSum

        # compute deltaMag
        deltaMag = obsMagStd[thisObsIndex] - objMagStdMean[objIndex,thisObsBandIndex]
        deltaMagErr2 = thisMagErr2 + objMagStdMeanErr[objIndex,thisObsBandIndex]**2.
        deltaMagWeighted = deltaMag/deltaMagErr2

        # finally, compute the chisq.  Also need a return array!
        partialChisq = np.sum(deltaMag**2./deltaMagErr2)

        partialArray = np.zeros(self.nSums,dtype='f4')

        # last one is the chisq
        partialArray[-2] = partialChisq
        partialArray[-1] = thisObsIndex.size

        # and compute the derivatives if desired...
        if (self.computeDerivatives):
            #print("Computing derivatives!")

            unitDict=self.fgcmPars.getUnitDict(self.fitterUnits)
            # do I need to loop over all parameters?

            # i,i',i": loop over observations (in a given band)
            # j: loop over objects

            # first, we need dL(i,j|p) = d/dp(2.5*log10(LUT(i,j|p)))
            #                          = 1.086*(LUT'(i,j|p)/LUT(i,j|p))
            (dLdPWV,dLdO3,dLdTau,dLdAlpha) = (
                self.fgcmLUT.computeLogDerivatives(lutIndices, thisI0, self.fgcmPars.expTau[thisObsExpIndex]))

            # we have objMagStdMeanErr[objIndex,:] = \Sum_{i"} 1/\sigma^2_{i"j}
            #   note that this is summed over all observations of an object in a band
            #   so that this is already done

            # we need magdLdp = \Sum_{i'} (1/\sigma^2_{i'j}) dL(i',j|p)
            #   note that this is summed over all observations in a filter that
            #   touch a given parameter

            # set up arrays
            #print("Setting up arrays...")
            nightSizeTuple = (self.fgcmPars.nCampaignNights,self.fgcmPars.nFitBands)
            magdLdPWVIntercept = np.zeros(nightSizeTuple,dtype='f4')
            magdLdPWVSlope = np.zeros(nightSizeTuple,dtype='f4')
            magdLdPWVOffset = np.zeros(nightSizeTuple,dtype='f4')
            magdLdTauIntercept = np.zeros(nightSizeTuple,dtype='f4')
            magdLdTauSlope = np.zeros(nightSizeTuple,dtype='f4')
            magdLdTauOffset = np.zeros(nightSizeTuple,dtype='f4')
            magdLdAlpha = np.zeros(nightSizeTuple,dtype='f4')
            magdLdO3 = np.zeros(nightSizeTuple,dtype='f4')
            magdLdPWVScale = np.zeros(self.fgcmPars.nFitBands,dtype='f4')
            magdLdTauScale = np.zeros(self.fgcmPars.nFitBands,dtype='f4')

            washSizeTuple = (self.fgcmPars.nWashIntervals,self.fgcmPars.nFitBands)
            magdLdWashIntercept = np.zeros(washSizeTuple,dtype='f4')
            magdLdWashSlope = np.zeros(washSizeTuple,dtype='f4')

            # precompute object err2...
            thisObjMagStdMeanErr2 = objMagStdMeanErr[objIndex,:]**2.

            ## FIXME: change to histogram?
            for fitBandCtr in xrange(self.fgcmPars.nFitBands):
                #print("band: %d" % (fitBandCtr))
                bandIndex = self.fgcmPars.fitBandIndex[fitBandCtr]
                inBand,=np.where(thisObsBandIndex == bandIndex)

                # for nightly parameters, loop over all the nights with observations
                hist,rev=esutil.stat.histogram(self.fgcmPars.expNightIndex[thisObsExpIndex[inBand]],min=0,max=self.fgcmPars.nCampaignNights-1,rev=True)
                use,=np.where(hist > 0)

                for i in use:
                    i1a=rev[rev[i]:rev[i+1]]

                    tempExpIndex = thisObsExpIndex[inBand[i1a]]
                    # for PWV we need to know if it has an external value

                    hasExt = np.where(self.fgcmPars.externalPWVFlag[tempExpIndex])
                    magdLdPWVOffset[i,bandIndex] = (
                        np.sum(dLdPWV[inBand[i1a[hasExt]]] /
                               thisMagErr2[inBand[i1a[hasExt]]]) * (
                            thisObjMagStdMeanErr2[bandIndex]))

                    noExt = np.where(~self.fgcmPars.externalPWVFlag[tempExpIndex])
                    magdLdPWVIntercept[i,bandIndex] = (
                        np.sum(dLdPWV[inBand[i1a[noExt]]] /
                               thisMagErr2[inBand[i1a[noExt]]]) * (
                            thisObjMagStdMeanErr2[bandIndex]))
                    magdLdPWVSlope[i,bandIndex] = (
                        np.sum(self.fgcmPars.expDeltaUT[tempExpIndex[noExt]] *
                               dLdPWV[inBand[i1a[noExt]]] /
                               thisMagErr2[inBand[i1a[noExt]]]) * (
                            thisObjMagStdMeanErr2[bandIndex]))

                    # for tau, external or not
                    hasExt = np.where(self.fgcmPars.externalTauFlag[tempExpIndex])
                    magdLdTauOffset[i,bandIndex] = (
                        np.sum(dLdTau[inBand[i1a[hasExt]]] /
                               thisMagErr2[inBand[i1a[hasExt]]]) * (
                            thisObjMagStdMeanErr2[bandIndex]))

                    noExt = np.where(~self.fgcmPars.externalTauFlag[tempExpIndex])
                    magdLdTauIntercept[i,bandIndex] = (
                        np.sum(dLdTau[inBand[i1a[noExt]]] /
                               thisMagErr2[inBand[i1a[noExt]]]) * (
                            thisObjMagStdMeanErr2[bandIndex]))
                    magdLdTauSlope[i,bandIndex] = (
                        np.sum(self.fgcmPars.expDeltaUT[tempExpIndex[noExt]] *
                               dLdTau[inBand[i1a[noExt]]] /
                               thisMagErr2[inBand[i1a[noExt]]]) * (
                            thisObjMagStdMeanErr2[bandIndex]))

                    # O3
                    magdLdO3[i,bandIndex] = (
                        np.sum(dLdO3[inBand[i1a]] /
                               thisMagErr2[inBand[i1a]]) * (
                            thisObjMagStdMeanErr2[bandIndex]))

                    # Alpha
                    magdLdAlpha[i,bandIndex] = (
                        np.sum(dLdAlpha[inBand[i1a]] /
                               thisMagErr2[inBand[i1a]]) * (
                            thisObjMagStdMeanErr2[bandIndex]))

                #print("blah")
                # per-filter survey parameters
                hasExt, = np.where(self.fgcmPars.externalPWVFlag[thisObsExpIndex[inBand]])
                magdLdPWVScale[bandIndex] = (np.sum(self.fgcmPars.expPWV[thisObsExpIndex[inBand[hasExt]]] *
                                                   dLdPWV[inBand[hasExt]] /
                                                   thisMagErr2[inBand[hasExt]]) * (
                        thisObjMagStdMeanErr2[bandIndex]))

                hasExt, = np.where(self.fgcmPars.externalTauFlag[thisObsExpIndex[inBand]])
                magdLdTauScale[bandIndex] = (np.sum(self.fgcmPars.expTau[thisObsExpIndex[inBand[hasExt]]] *
                                                   dLdTau[inBand[hasExt]] /
                                                   thisMagErr2[inBand[hasExt]]) * (
                        thisObjMagStdMeanErr2[bandIndex]))

                # and wash epochs...
                # This has to loop over filters and wash intervals

                hist,rev=esutil.stat.histogram(self.fgcmPars.expWashIndex[thisObsExpIndex[inBand]],
                                               min=0,max=self.fgcmPars.nWashIntervals-1,
                                               rev=True)

                use,=np.where(hist > 0)
                for i in use:
                    #print("wash epoch %d" % (i))
                    i1a=rev[rev[i]:rev[i+1]]

                    tempExpIndex = thisObsExpIndex[inBand[i1a]]

                    magdLdWashIntercept[i,bandIndex] = (
                        np.sum(1./thisMagErr2[inBand[i1a]]) * (
                            thisObjMagStdMeanErr2[bandIndex]))

                    magdLdWashSlope[i,bandIndex] = (
                        np.sum((self.fgcmPars.expMJD[tempExpIndex] -
                                self.fgcmPars.washMJDs[i]) / (
                                thisMagErr2[inBand[i1a]])) * (
                            thisObjMagStdMeanErr2[bandIndex]))

            # and fill the partial structure with sums
            # first the nightly parameters...
            #print("histogramming for partials...")
            hist,rev=esutil.stat.histogram(self.fgcmPars.expNightIndex[thisObsExpIndex[thisObsFitUse]],min=0,max=self.fgcmPars.nCampaignNights-1,rev=True)
            use,=np.where(hist > 0)
            #print(use.size)
            for i in use:
                # i1a is all the observations on a given night
                i1a=rev[rev[i]:rev[i+1]]

                tempObsIndex = thisObsFitUse[i1a]
                tempExpIndex = thisObsExpIndex[tempObsIndex]
                tempBandIndex = thisObsBandIndex[tempObsIndex]

                ## pwv parameters

                if (self.fgcmPars.hasExternalPWV):
                    hasExt, = np.where(self.fgcmPars.externalPWVFlag[tempExpIndex])
                    partialArray[self.fgcmPars.parExternalPWVOffsetLoc + i] = 2.0 * (
                        np.sum(deltaMagWeighted[tempObsIndex[hasExt]] * (
                                dLdPWV[tempObsIndex[hasExt]] -
                                magdLdPWVOffset[i,tempBandIndex[hasExt]])) /
                        unitDict['pwvUnit'])

                noExt, = np.where(~self.fgcmPars.externalPWVFlag[tempExpIndex])
                partialArray[self.fgcmPars.parPWVInterceptLoc + i] = 2.0 * (
                    np.sum(deltaMagWeighted[tempObsIndex[noExt]] * (
                            dLdPWV[tempObsIndex[noExt]] -
                            magdLdPWVIntercept[i,tempBandIndex[noExt]])) / 
                    unitDict['pwvUnit'])
                partialArray[self.fgcmPars.parPWVSlopeLoc + i] = 2.0 * (
                    np.sum(deltaMagWeighted[tempObsIndex[noExt]] * (
                            self.fgcmPars.expDeltaUT[tempExpIndex[noExt]] *
                            dLdPWV[tempObsIndex[noExt]] -
                            magdLdPWVSlope[i,tempBandIndex[noExt]])) /
                    unitDict['pwvSlopeUnit'])

                ## tau parameters
                if (self.fgcmPars.hasExternalTau):
                    hasExt, = np.where(self.fgcmPars.externalTauFlag[tempExpIndex])
                    partialArray[self.fgcmPars.parExternalTauOffsetLoc + i] = 2.0 * (
                        np.sum(deltaMagWeighted[tempObsIndex[hasExt]] *
                               (dLdTau[tempObsIndex[hasExt]] -
                                magdLdTauOffset[i,tempBandIndex[hasExt]])) /
                        unitDict['tauUnit'])

                noExt, = np.where(~self.fgcmPars.externalTauFlag[tempExpIndex])
                partialArray[self.fgcmPars.parTauInterceptLoc + i] = 2.0 * (
                    np.sum(deltaMagWeighted[tempObsIndex[noExt]] * (
                            dLdTau[tempObsIndex[noExt]] -
                            magdLdTauIntercept[i,tempBandIndex[noExt]])) /
                    unitDict['tauUnit'])
                partialArray[self.fgcmPars.parTauSlopeLoc + i] = 2.0 * (
                    np.sum(deltaMagWeighted[tempObsIndex[noExt]] * (
                            self.fgcmPars.expDeltaUT[tempExpIndex[noExt]] *
                            dLdTau[tempObsIndex[noExt]] -
                            magdLdTauSlope[i,tempBandIndex[noExt]])) /
                    unitDict['tauSlopeUnit'])

                # alpha
                partialArray[self.fgcmPars.parAlphaLoc + i] = 2.0 * (
                    np.sum(deltaMagWeighted[tempObsIndex] * (
                            dLdAlpha[tempObsIndex] -
                            magdLdAlpha[i,tempBandIndex])) /
                    unitDict['alphaUnit'])

                # O3
                partialArray[self.fgcmPars.parO3Loc + i] = 2.0 * (
                    np.sum(deltaMagWeighted[tempObsIndex] * (
                            dLdO3[tempObsIndex] -
                            magdLdO3[i,tempBandIndex])) /
                    unitDict['o3Unit'])

            # and the washes...
            hist,rev=esutil.stat.histogram(
                self.fgcmPars.expWashIndex[thisObsExpIndex[thisObsFitUse]],
                min=0,max=self.fgcmPars.nWashIntervals-1,
                rev=True)

            use,=np.where(hist > 0)
            for i in use:
                i1a = rev[rev[i]:rev[i+1]]

                tempObsIndex = thisObsFitUse[i1a]
                tempExpIndex = thisObsExpIndex[tempObsIndex]
                tempBandIndex = thisObsBandIndex[tempObsIndex]

                partialArray[self.fgcmPars.parQESysInterceptLoc + i] = 2.0 * (
                    np.sum(deltaMagWeighted[tempObsIndex] * (
                            1.0 -
                            magdLdWashIntercept[i,tempBandIndex])) /
                    unitDict['qeSysUnit'])
                partialArray[self.fgcmPars.parQESysSlopeLoc + i] = 2.0 * (
                    np.sum(deltaMagWeighted[tempObsIndex] * (
                            (self.fgcmPars.expMJD[tempExpIndex] -
                             self.fgcmPars.washMJDs[i]) -
                            magdLdWashSlope[i,tempBandIndex])) /
                    unitDict['qeSysSlopeUnit'])


            # and the global parameters...
            if (self.fgcmPars.hasExternalPWV):
                hasExt, = np.where(self.fgcmPars.externalPWVFlag[thisObsExpIndex[thisObsFitUse]])
                tempObsIndex = thisObsFitUse[hasExt]
                partialArray[self.fgcmPars.parExternalPWVScaleLoc] = 2.0 * (
                    np.sum(deltaMagWeighted[tempObsIndex] * (
                            self.fgcmPars.expPWV[thisObsExpIndex[tempObsIndex]] *
                            dLdPWV[tempObsIndex] -
                            magdLdPWVScale[thisObsBandIndex[tempObsIndex]])) /
                    unitDict['pwvUnit'])

            if (self.fgcmPars.hasExternalTau):
                hasExt, = np.where(self.fgcmPars.externalTauFlag[thisObsExpIndex[thisObsFitUse]])
                tempObsIndex = thisObsFitUse[hasExt]
                partialArray[self.fgcmPars.parExternalTauScaleLoc] = 2.0 * (
                    np.sum(deltaMagWeighted[tempObsIndex] * (
                            self.fgcmPars.expTau[thisObsExpIndex[tempObsIndex]] *
                            dLdTau[tempObsIndex] -
                            magdLdTauScale[thisObsBandIndex[tempObsIndex]])) /
                    unitDict['tauUnit'])


        # note that this doesn't need locking because we are only accessing
        #   a single array from a single process
        if self.debug:
            thisCore = 0
        else:
            thisCore = multiprocessing.current_process()._identity[0]
        totalArr = snmm.getArray(self.totalHandleDict[thisCore])
        totalArr[:] = totalArr[:] + partialArray

        #print("done")
        # no return
        return None
