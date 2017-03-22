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

    def __call__(self,fitParams,computeDerivatives=False,computeSEDSlopes=False,fitterUnits=False):
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

        resourceUsage('Start of call')

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
        #goodStars=goodStars[0:1000]

        # prepare the return arrays...
        # how many do we have?

        self.nSums = 2   # chisq, nobs
        if (self.computeDerivatives):
            self.nSums += self.fgcmPars.nFitPars  # one for each parameter

        # make a dummy process to discover starting child number
        proc = multiprocessing.Process()
        workerIndex = proc._identity[0]+1
        proc = None

        self.totalHandleDict = {}
        for thisCore in xrange(self.nCore):
            self.totalHandleDict[workerIndex + thisCore] = snmm.createArray(self.nSums,dtype='f4')


        # will want to make a pool

        pool = Pool(processes=4)
        resourceUsage('premap')
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

        resourceUsage('end')

        if (self.computeDerivatives):
            return fitChisq, dChisqdP
        else:
            return fitChisq

    def _worker(self,objIndex):
        """
        """

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
            # use magObs to compute mean mags...
            # compute in all bands here.

            # how much time is the where taking?
            for i in xrange(self.fgcmStars.nBands):
                #use,=np.where((obsExpFlag[thisObsIndex] == 0) &
                #              (obsBandIndex[thisObsIndex] == i))
                #use,=np.where((self.fgcmPars.expFlag[thisObsExpIndex] == 0) &
                #              (thisObsBandIndex == i))
                use,=np.where(thisObsBandIndex == i)
                if (use.size > 0):
                    wtSum = np.sum(1./thisMagErr2[use])
                    objMagStdMeanErr[objIndex,i] = np.sqrt(1./wtSum)
                    objMagStdMean[objIndex,i] = np.sum(obsMagStd[thisObsIndex[use]]/
                                                       thisMagErr2[use])/wtSum

            if (np.max(objMagStdMean[objIndex,:]) > 90.0) :
                # cannot compute
                objSEDSlope[objIndex,:] = 0.0
            else:
                # need to do FIT BANDS
                #   FIXME
                S = np.zeros(self.fgcmPars.nBands-1,dtype='f4')
                for i in xrange(self.fgcmPars.nBands-1):
                    S[i] = -0.921 * (objMagStdMean[objIndex,i+1] - objMagStdMean[objIndex,i])/(self.fgcmLUT.lambdaStd[i+1] - self.fgcmLUT.lambdaStd[i])

                # this is hacked for now
                objSEDSlope[objIndex,0] = S[0] - 1.0 * ((self.fgcmLUT.lambdaStd[1] - self.fgcmLUT.lambdaStd[0])/(self.fgcmLUT.lambdaStd[2]-self.fgcmLUT.lambdaStd[0])) * (S[1]-S[0])
                objSEDSlope[objIndex,1] = (S[0] + S[1])/2.0
                objSEDSlope[objIndex,2] = (S[1] + S[2])/2.0
                objSEDSlope[objIndex,3] = S[2] + 0.5 * ((self.fgcmLUT.lambdaStd[3]-self.fgcmLUT.lambdaStd[2])/(self.fgcmLUT.lambdStd[3]-self.fgcmLUT.lambdaStd[1])) * (S[2] - S[1])
                if ((objMagStdMean[objIndex,4]) < 90.0):
                    objSEDSlope[objIndex,4] = S[2] + 1.0 * ((self.fgcmLUT.lambdaStd[3]-self.fgcmLUT.lambdaStd[2])/(self.fgcmLUT.lambdaStd[3]-self.fgcmLUT.lambdaStd[1])) * (S[2]-S[1])

        # compute magStd (and record)
        thisDeltaStd = 2.5 * np.log10((1.0 + objSEDSlope[objIndex,thisObsBandIndex] * thisI10) / (1.0 + objSEDSlope[objIndex,thisObsBandIndex] * self.fgcmLUT.I10Std[thisObsBandIndex]))

        obsMagStd[thisObsIndex] = thisMagObs + thisDeltaStd

        # compute mean objMagStdMean
        for i in xrange(self.fgcmStars.nBands):
            #use,=np.where((obsExpFlag[thisObsIndex] == 0) &
            #              (thisObsBandIndex == i))
            #use,=np.where((self.fgcmPars.expFlag[thisObsExpIndex] == 0) &
            #              (thisObsBandIndex == i))
            use,=np.where(thisObsBandIndex == i)
            if (use.size > 0):
                wtSum = np.sum(1./thisMagErr2[use])
                objMagStdMeanErr[objIndex,i] = np.sqrt(1./wtSum)
                objMagStdMean[objIndex,i] = np.sum(obsMagStd[thisObsIndex[use]]/
                                                   thisMagErr2[use])/wtSum

        # compute deltaMag
        deltaMag = obsMagStd[thisObsIndex] - objMagStdMean[objIndex,thisObsBandIndex]
        deltaMagErr2 = thisMagErr2 + objMagStdMeanErr[objIndex,thisObsBandIndex]**2.

        # finally, compute the chisq.  Also need a return array!
        partialChisq = np.sum(deltaMag**2./deltaMagErr2)

        partialArray = np.zeros(self.nSums,dtype='f4')

        # last one is the chisq
        partialArray[-2] = partialChisq
        partialArray[-1] = thisObsIndex.size

        # and compute the derivatives if desired...
        if (self.computeDerivatives):
            pass

        # note that this doesn't need locking because we are only accessing
        #   a single array from a single process
        thisCore = multiprocessing.current_process()._identity[0]
        totalArr = snmm.getArray(self.totalHandleDict[thisCore])
        totalArr[:] = totalArr[:] + partialArray

        # no return
        return None
