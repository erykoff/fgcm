from __future__ import print_function

import numpy as np
import os
import sys
import esutil
import time
import scipy.optimize

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cmx

from sharedNumpyMemManager import SharedNumpyMemManager as snmm
from fgcmUtilities import poly2dFunc

class FgcmSuperStarFlat(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmStars,fgcmGray):

        self.fgcmLog = fgcmConfig.fgcmLog
        self.fgcmLog.info('Initializing FgcmSuperStarFlat')

        self.fgcmPars = fgcmPars

        self.fgcmStars = fgcmStars
        self.fgcmGray = fgcmGray

        self.illegalValue = fgcmConfig.illegalValue
        self.minStarPerCCD = fgcmConfig.minStarPerCCD
        self.ccdOffsets = fgcmConfig.ccdOffsets
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.epochNames = fgcmConfig.epochNames
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.ccdGrayMaxStarErr = fgcmConfig.ccdGrayMaxStarErr


        self.superStarSubCCD = fgcmConfig.superStarSubCCD

    def computeSuperStarFlats(self, doPlots=True, doNotUseSubCCD=False):
        """
        """

        startTime = time.time()
        self.fgcmLog.info('Computing superstarflats')

        # New version, use the stars directly
        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsSuperStarApplied = snmm.getArray(self.fgcmStars.obsSuperStarAppliedHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        # make sure we have enough obervations per band
        #  (this may be redundant)
        minObs = objNGoodObs[:,self.fgcmStars.bandRequiredIndex].min(axis=1)

        # select good stars...
        goodStars, = np.where((minObs >= self.fgcmStars.minPerBand) &
                              (objFlag == 0))

        # match the good stars to the observations
        _,goodObs = esutil.numpy_util.match(goodStars,
                                            obsObjIDIndex,
                                            presorted=True)

        # and filter out bad observations, non-photometric, etc
        gd,=np.where((obsFlag[goodObs] == 0) &
                     (self.fgcmPars.expFlag[obsExpIndex[goodObs]] == 0))
        goodObs = goodObs[gd]

        # we need to compute E_gray == <mstd> - mstd for each observation
        # compute EGray, GO for Good Obs
        EGrayGO = (objMagStdMean[obsObjIDIndex[goodObs],obsBandIndex[goodObs]] -
                   obsMagStd[goodObs])

        # take into account correlated average mag error
        EGrayErr2GO = (obsMagErr[goodObs]**2. -
                       objMagStdMeanErr[obsObjIDIndex[goodObs],obsBandIndex[goodObs]]**2.)

        # one more cut on the maximum error
        gd,=np.where(EGrayErr2GO < self.ccdGrayMaxStarErr)
        goodObs=goodObs[gd]
        # unapply input superstar correction here (note opposite sign)
        EGrayGO=EGrayGO[gd] + obsSuperStarApplied[goodObs]
        EGrayErr2GO=EGrayErr2GO[gd]

        # and record the deltas (per ccd)
        prevSuperStarFlatCenter = np.zeros((self.fgcmPars.nEpochs,
                                            self.fgcmPars.nLUTFilter,
                                            self.fgcmPars.nCCD))
        superStarFlatCenter = np.zeros_like(prevSuperStarFlatCenter)
        superStarNGoodStars = np.zeros_like(prevSuperStarFlatCenter, dtype=np.int32)

        # and the mean and sigma over the focal plane for reference
        superStarFlatFPMean = np.zeros((self.fgcmPars.nEpochs,
                                        self.fgcmPars.nLUTFilter))
        superStarFlatFPSigma = np.zeros_like(superStarFlatFPMean)
        deltaSuperStarFlatFPMean = np.zeros_like(superStarFlatFPMean)
        deltaSuperStarFlatFPSigma = np.zeros_like(superStarFlatFPMean)


        # Note that we use the poly2dFunc even when the previous numbers
        #  were just an offset, because the other terms are zeros
        prevSuperStarFlatCenter[:,:,:] = self.fgcmPars.superStarFlatCenter

        if not self.superStarSubCCD or doNotUseSubCCD:
            # do not use subCCD x/y information (or x/y not available)

            # Next, we sort by epoch, band
            superStarWt = np.zeros_like(superStarFlatCenter)
            superStarOffset = np.zeros_like(superStarWt)

            # need separate for required bands and extra bands for which stars to use
            _,reqBandUse = esutil.numpy_util.match(self.fgcmStars.bandRequiredIndex,
                                                   obsBandIndex[goodObs])

            np.add.at(superStarWt,
                      (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs[reqBandUse]]],
                       self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs[reqBandUse]]],
                       obsCCDIndex[goodObs[reqBandUse]]),
                      1./EGrayErr2GO[reqBandUse])
            np.add.at(superStarOffset,
                      (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs[reqBandUse]]],
                       self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs[reqBandUse]]],
                       obsCCDIndex[goodObs[reqBandUse]]),
                      EGrayGO[reqBandUse]/EGrayErr2GO[reqBandUse])
            np.add.at(superStarNGoodStars,
                      (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs[reqBandUse]]],
                       self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs[reqBandUse]]],
                       obsCCDIndex[goodObs[reqBandUse]]),
                      1)

            for extraBandIndex in self.fgcmStars.bandExtraIndex:
                extraBandUse, = np.where((obsBandIndex[goodObs] == extraBandIndex) &
                                         (objNGoodObs[obsObjIDIndex[goodObs],extraBandIndex] >=
                                          self.fgcmStars.minPerBand))
                np.add.at(superStarWt,
                          (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs[extraBandUse]]],
                           self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs[extraBandUse]]],
                           obsCCDIndex[goodObs[extraBandUse]]),
                          1./EGrayErr2GO[extraBandUse])
                np.add.at(superStarOffset,
                          (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs[extraBandUse]]],
                           self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs[extraBandUse]]],
                           obsCCDIndex[goodObs[extraBandUse]]),
                          EGrayGO[extraBandUse]/EGrayErr2GO[extraBandUse])
                np.add.at(superStarNGoodStars,
                          (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs[extraBandUse]]],
                           self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs[extraBandUse]]],
                           obsCCDIndex[goodObs[extraBandUse]]),
                      1)

            gd = np.where(superStarNGoodStars > 2)
            superStarOffset[gd] /= superStarWt[gd]

            # and this is the same as the numbers for the center
            superStarFlatCenter[:,:,:] = superStarOffset[:,:,:]

            # and record...

            self.fgcmPars.parSuperStarFlat[:,:,:,0] = superStarOffset

        else:
            # with x/y, new sub-ccd

            # we will need the ccd offset signs
            self._computeCCDOffsetSigns(goodObs)

            obsX = snmm.getArray(self.fgcmStars.obsXHandle)
            obsY = snmm.getArray(self.fgcmStars.obsYHandle)

            obsXGO = obsX[goodObs]
            obsYGO = obsY[goodObs]

            # need to histogram this all up.  Watch for extra bands

            epochFilterHash = (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs]]*
                               (self.fgcmPars.nLUTFilter+1)*(self.fgcmPars.nCCD+1) +
                               self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs]]*
                               (self.fgcmPars.nCCD+1) +
                               obsCCDIndex[goodObs])

            h, rev = esutil.stat.histogram(epochFilterHash, rev=True)

            for i in xrange(h.size):
                if h[i] == 0: continue

                i1a = rev[rev[i]:rev[i+1]]

                # get the indices for this epoch/filter/ccd
                epInd = self.fgcmPars.expEpochIndex[obsExpIndex[goodObs[i1a[0]]]]
                fiInd = self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs[i1a[0]]]]
                cInd = obsCCDIndex[goodObs[i1a[0]]]

                # check if this is an extra band and needs caution
                bInd = obsBandIndex[goodObs[i1a[0]]]
                if bInd in self.fgcmStars.bandExtraIndex:
                    extraBandUse, = np.where(objNGoodObs[obsObjIDIndex[goodObs[i1a]], bInd] >=
                                             self.fgcmStars.minPerBand)
                    if extraBandUse.size == 0:
                        continue

                    i1a = i1a[extraBandUse]

                try:
                    fit, cov = scipy.optimize.curve_fit(poly2dFunc,
                                                        np.vstack((obsXGO[i1a],
                                                                   obsYGO[i1a])),
                                                        EGrayGO[i1a],
                                                        p0=[0.0,0.0,0.0,0.0,0.0,0.0],
                                                        sigma=np.sqrt(EGrayErr2GO[i1a]))
                except:
                    print("Warning: fit failed to converge (%d, %d, %d), setting to mean"
                          % (epInd, fiInd, cInd))
                    fit = np.zeros(6)
                    fit[0] = (np.sum(EGrayGO[i1a]/EGrayErr2GO[i1a]) /
                              np.sum(1./EGrayErr2GO[i1a]))

                superStarNGoodStars[epInd, fiInd, cInd] = i1a.size


                # compute the central value for use with the delta
                xy = np.vstack((self.ccdOffsets['X_SIZE'][cInd]/2.,
                               self.ccdOffsets['Y_SIZE'][cInd]/2.))
                superStarFlatCenter[epInd, fiInd, cInd] = poly2dFunc(xy,
                                                                     *fit)

                # and record the fit

                self.fgcmPars.parSuperStarFlat[epInd, fiInd, cInd, :] = fit

        # compute the delta...
        deltaSuperStarFlatCenter = superStarFlatCenter - prevSuperStarFlatCenter

        # and the overall stats...
        for e in xrange(self.fgcmPars.nEpochs):
            for f in xrange(self.fgcmPars.nLUTFilter):
                use,=np.where(superStarNGoodStars[e, f, :] > 0)

                if use.size < 3:
                    continue

                superStarFlatFPMean[e, f] = np.mean(superStarFlatCenter[e, f, use])
                superStarFlatFPSigma[e, f] = np.std(superStarFlatCenter[e, f, use])
                deltaSuperStarFlatFPMean[e, f] = np.mean(deltaSuperStarFlatCenter[e, f, use])
                deltaSuperStarFlatFPSigma[e, f] = np.std(deltaSuperStarFlatCenter[e, f, use])

                self.fgcmLog.info('Superstar epoch %d filter %s: %.4f +/- %.4f  Delta: %.4f +/- %.4f' %
                                  (e, self.fgcmPars.lutFilterNames[f],
                                   superStarFlatFPMean[e, f], superStarFlatFPSigma[e, f],
                                   deltaSuperStarFlatFPMean[e, f], deltaSuperStarFlatFPSigma[e, f]))

        self.fgcmLog.info('Computed SuperStarFlats in %.2f seconds.' %
                          (time.time() - startTime))

        if doPlots:
            self.fgcmLog.info('Making SuperStarFlat plots')

            # can we do a combined plot?  Two panel?  I think that would be
            #  better, but I'm worried about the figure sizes
            self.plotSuperStarFlatsAndDelta(self.fgcmPars.parSuperStarFlat,
                                            deltaSuperStarFlatCenter,
                                            superStarNGoodStars,
                                            superStarFlatFPMean, superStarFlatFPSigma,
                                            deltaSuperStarFlatFPMean, deltaSuperStarFlatFPSigma)

    def plotSuperStarFlatsAndDelta(self, superStarPars, deltaSuperStar, superStarNGoodStars,
                                   superStarFlatFPMean, superStarFlatFPSigma,
                                   deltaSuperStarFlatFPMean, deltaSuperStarFlatFPSigma):
        """
        """

        from fgcmUtilities import plotCCDMap
        from fgcmUtilities import plotCCDMapPoly2d

        for e in xrange(self.fgcmPars.nEpochs):
            for f in xrange(self.fgcmPars.nLUTFilter):
                use, = np.where(superStarNGoodStars[e, f, :] > 0)

                if use.size == 0:
                    continue

                # double-wide.  Don't give number because that was dumb
                fig=plt.figure(figsize=(16,6))
                fig.clf()

                # left side plot the map with x/y
                ax=fig.add_subplot(121)

                if not self.superStarSubCCD:
                    plotCCDMap(ax, self.ccdOffsets[use], superStarPars[e, f, use, 0],
                               'SuperStar (mag)')
                else:
                    plotCCDMapPoly2d(ax, self.ccdOffsets[use], superStarPars[e, f, use, :],
                                     'SuperStar (mag)')

                # and annotate

                text = r'$(%s)$' % (self.fgcmPars.lutFilterNames[f]) + '\n' + \
                    r'%.4f +/- %.4f' % (superStarFlatFPMean[e,f],
                                        superStarFlatFPSigma[e,f])
                ax.annotate(text,
                            (0.1,0.93),xycoords='axes fraction',
                            ha='left',va='top',fontsize=18)

                # right side plot the deltas
                ax=fig.add_subplot(122)

                plotCCDMap(ax, self.ccdOffsets[use], deltaSuperStar[e,f,use],
                           'Central Delta-SuperStar (mag)')

                # and annotate
                text = r'$(%s)$' % (self.fgcmPars.lutFilterNames[f]) + '\n' + \
                    r'%.4f +/- %.4f' % (deltaSuperStarFlatFPMean[e,f],
                                        deltaSuperStarFlatFPSigma[e,f])
                ax.annotate(text,
                            (0.1,0.93),xycoords='axes fraction',
                            ha='left',va='top',fontsize=18)

                fig.tight_layout()

                fig.savefig('%s/%s_%s_%s_%s.png' % (self.plotPath,
                                                    self.outfileBaseWithCycle,
                                                    'superstar',
                                                    self.fgcmPars.lutFilterNames[f],
                                                    self.epochNames[e]))
                plt.close()


    def _computeCCDOffsetSigns(self, goodObs):

        import scipy.stats

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)

        obsX = snmm.getArray(self.fgcmStars.obsXHandle)
        obsY = snmm.getArray(self.fgcmStars.obsYHandle)
        objRA = snmm.getArray(self.fgcmStars.objRAHandle)
        objDec = snmm.getArray(self.fgcmStars.objDecHandle)

        h, rev = esutil.stat.histogram(obsCCDIndex[goodObs], rev=True)

        for i in xrange(h.size):
            if h[i] == 0: continue

            i1a = rev[rev[i]:rev[i+1]]

            cInd = obsCCDIndex[goodObs[i1a[0]]]

            if self.ccdOffsets['RASIGN'][cInd] == 0:
                # choose a good exposure to work with
                hTest, revTest = esutil.stat.histogram(obsExpIndex[goodObs[i1a]], rev=True)
                maxInd = np.argmax(hTest)
                testStars = revTest[revTest[maxInd]:revTest[maxInd+1]]

                testRA = objRA[obsObjIDIndex[goodObs[i1a[testStars]]]]
                testDec = objDec[obsObjIDIndex[goodObs[i1a[testStars]]]]
                testX = obsX[goodObs[i1a[testStars]]]
                testY = obsY[goodObs[i1a[testStars]]]

                corrXRA,_ = scipy.stats.pearsonr(testX,testRA)
                corrYRA,_ = scipy.stats.pearsonr(testY,testRA)

                if (corrXRA > corrYRA):
                    self.ccdOffsets['XRA'][cInd] = True
                else:
                    self.ccdOffsets['XRA'][cInd] = False

                if self.ccdOffsets['XRA'][cInd]:
                    # x is correlated with RA
                    if corrXRA < 0:
                        self.ccdOffsets['RASIGN'][cInd] = -1
                    else:
                        self.ccdOffsets['RASIGN'][cInd] = 1

                    corrYDec,_ = scipy.stats.pearsonr(testY,testDec)
                    if corrYRA < 0:
                        self.ccdOffsets['DECSIGN'][cInd] = -1
                    else:
                        self.ccdOffsets['DECSIGN'][cInd] = 1
                else:
                    # y is correlated with RA
                    if corrYRA < 0:
                        self.ccdOffsets['RASIGN'][cInd] = -1
                    else:
                        self.ccdOffsets['RASIGN'][cInd] = 1

                    corrXDec,_ = scipy.stats.pearsonr(testX,testDec)
                    if corrXDec < 0:
                        self.ccdOffsets['DECSIGN'][cInd] = -1
                    else:
                        self.ccdOffsets['DECSIGN'][cInd] = 1

    def computeSuperStarFlatsOrig(self,doPlots=True):
        """
        """

        startTime = time.time()
        self.fgcmLog.info('Computing superstarflats')

        ## FIXME: need to filter out SN (deep) exposures.  Hmmm.

        #deltaSuperStarFlat = np.zeros_like(self.fgcmPars.parSuperStarFlat)
        #deltaSuperStarFlatNCCD = np.zeros_like(self.fgcmPars.parSuperStarFlat,dtype='i4')
        deltaSuperStarFlat = np.zeros((self.fgcmPars.nEpochs,
                                       self.fgcmPars.nLUTFilter,
                                       self.fgcmPars.nCCD))
        deltaSuperStarFlatNCCD = np.zeros_like(deltaSuperStarFlat, dtype='i4')

        ccdGray = snmm.getArray(self.fgcmGray.ccdGrayHandle)
        ccdGrayErr = snmm.getArray(self.fgcmGray.ccdGrayErrHandle)
        ccdNGoodStars = snmm.getArray(self.fgcmGray.ccdNGoodStarsHandle)

        # only select those CCDs that we have an adequate gray calculation
        expIndexUse,ccdIndexUse=np.where((ccdNGoodStars >= self.minStarPerCCD))

        # and only select exposures that should go into the SuperStarFlat
        gd,=np.where(self.fgcmPars.expFlag[expIndexUse] == 0)
        expIndexUse=expIndexUse[gd]
        ccdIndexUse=ccdIndexUse[gd]

        self.fgcmLog.info('SuperStarFlats based on %d exposures' % (gd.size))

        # sum up ccdGray values
        #  note that this is done per *filter* not per *band*
        np.add.at(deltaSuperStarFlat,
                  (self.fgcmPars.expEpochIndex[expIndexUse],
                   self.fgcmPars.expLUTFilterIndex[expIndexUse],
                   ccdIndexUse),
                  ccdGray[expIndexUse,ccdIndexUse])
        np.add.at(deltaSuperStarFlatNCCD,
                  (self.fgcmPars.expEpochIndex[expIndexUse],
                   self.fgcmPars.expLUTFilterIndex[expIndexUse],
                   ccdIndexUse),
                  1)

        # only use exp/ccd where we have at least one observation
        gd=np.where(deltaSuperStarFlatNCCD > 0)
        deltaSuperStarFlat[gd] /= deltaSuperStarFlatNCCD[gd]

        # this accumulates onto the input parameters
        # self.fgcmPars.parSuperStarFlat += deltaSuperStarFlat
        self.fgcmPars.parSuperStarFlat[:,:,:,0] += deltaSuperStarFlat

        ## MAYBE: change fgcmGray to remove the deltaSuperStarFlat?
        ##  or we can rely on the iterations.  Try that first.

        ## FIXME
        self.deltaSuperStarFlatMean = np.zeros((self.fgcmPars.nEpochs,
                                                self.fgcmPars.nLUTFilter),dtype='f8')
        self.deltaSuperStarFlatSigma = np.zeros_like(self.deltaSuperStarFlatMean)
        self.superStarFlatMean = np.zeros_like(self.deltaSuperStarFlatMean)
        self.superStarFlatSigma = np.zeros_like(self.deltaSuperStarFlatMean)

        for i in xrange(self.fgcmPars.nEpochs):
            for j in xrange(self.fgcmPars.nLUTFilter):
                use,=np.where(deltaSuperStarFlatNCCD[i,j,:] > 0)

                if use.size < 3:
                    continue

                self.deltaSuperStarFlatMean[i,j] = np.mean(deltaSuperStarFlat[i,j,use])
                self.deltaSuperStarFlatSigma[i,j] = np.std(deltaSuperStarFlat[i,j,use])
                self.superStarFlatMean[i,j] = np.mean(self.fgcmPars.parSuperStarFlat[i,j,use,0])
                self.superStarFlatSigma[i,j] = np.std(self.fgcmPars.parSuperStarFlat[i,j,use,0])
                self.fgcmLog.info('Superstar epoch %d filter %s: %.4f +/- %.4f' %
                                 (i,self.fgcmPars.lutFilterNames[j],
                                  self.superStarFlatMean[i,j],
                                  self.superStarFlatSigma[i,j]))
                self.fgcmLog.info('DeltaSuperStar epoch %d filter %s: %.4f +/- %.4f' %
                                 (i,self.fgcmPars.lutFilterNames[j],
                                  self.deltaSuperStarFlatMean[i,j],
                                  self.deltaSuperStarFlatSigma[i,j]))

        self.fgcmLog.info('Computed SuperStarFlats in %.2f seconds.' %
                         (time.time() - startTime))

        if (doPlots):
            self.fgcmLog.info('Making SuperStarFlat plots')
            self.plotSuperStarFlats(deltaSuperStarFlat,
                                    self.deltaSuperStarFlatMean,
                                    self.deltaSuperStarFlatSigma,
                                    nCCDArray=deltaSuperStarFlatNCCD,
                                    name='deltasuperstar')
            self.plotSuperStarFlats(self.fgcmPars.parSuperStarFlat[:,:,:,0],
                                    self.superStarFlatMean,
                                    self.superStarFlatSigma,
                                    nCCDArray=deltaSuperStarFlatNCCD,
                                    name='superstar')

        # and we're done.

    def plotSuperStarFlats(self, superStarArray, superStarMean, superStarSigma,
                            nCCDArray=None, name='superstar'):
        """
        """
        from fgcmUtilities import plotCCDMap

        for i in xrange(self.fgcmPars.nEpochs):
            for j in xrange(self.fgcmPars.nLUTFilter):
                # only do those that had a non-zero number of CCDs to fit in this epoch
                if (nCCDArray is not None):
                    use,=np.where(nCCDArray[i,j,:] > 0)
                else:
                    use,=np.where(superStarArray[i,j,:] > self.illegalValue)

                if use.size == 0:
                    continue

                fig=plt.figure(1,figsize=(8,6))
                fig.clf()

                ax=fig.add_subplot(111)

                plotCCDMap(ax, self.ccdOffsets[use], superStarArray[i,j,use],
                           'Superflat Correction (mag)')

                text = r'$(%s)$' % (self.fgcmPars.lutFilterNames[j]) + '\n' + \
                    r'%.4f +/- %.4f' % (superStarMean[i,j],superStarSigma[i,j])
                ax.annotate(text,
                            (0.1,0.93),xycoords='axes fraction',
                            ha='left',va='top',fontsize=18)

                fig.savefig('%s/%s_%s_%s_%s.png' % (self.plotPath,
                                                    self.outfileBaseWithCycle,
                                                    name,
                                                    self.fgcmPars.lutFilterNames[j],
                                                    self.epochNames[i]))


