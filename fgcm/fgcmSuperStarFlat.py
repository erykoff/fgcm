from __future__ import division, absolute_import, print_function
from past.builtins import xrange

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

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm
from .fgcmUtilities import Cheb2dField

class FgcmSuperStarFlat(object):
    """
    Class to compute the SuperStarFlat.

    parameters
    ----------
    fgcmConfig: FgcmConfig
    fgcmPars: FgcmParameters
    fgcmStars: FgcmStars

    Config variables
    ----------------
    ccdGrayMaxStarErr: float
       Maximum error for any star observation to be used to compute superStar
    superStarSubCCD: bool, default=False
       Compute superStar flats on sub-ccd scale?
    """

    def __init__(self,fgcmConfig,fgcmPars,fgcmStars):

        self.fgcmLog = fgcmConfig.fgcmLog
        self.fgcmLog.info('Initializing FgcmSuperStarFlat')

        self.fgcmPars = fgcmPars

        self.fgcmStars = fgcmStars

        self.illegalValue = fgcmConfig.illegalValue
        self.minStarPerCCD = fgcmConfig.minStarPerCCD
        self.ccdOffsets = fgcmConfig.ccdOffsets
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.epochNames = fgcmConfig.epochNames
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.ccdGrayMaxStarErr = fgcmConfig.ccdGrayMaxStarErr

        self.superStarSubCCD = fgcmConfig.superStarSubCCD
        self.superStarSubCCDChebyshevOrder = fgcmConfig.superStarSubCCDChebyshevOrder
        self.superStarSubCCDTriangular = fgcmConfig.superStarSubCCDTriangular
        self.superStarSigmaClip = fgcmConfig.superStarSigmaClip

    def computeSuperStarFlats(self, doPlots=True, doNotUseSubCCD=False, onlyObsErr=False):
        """
        Compute the SuperStar Flats

        parameters
        ----------
        doPlots: bool, default=True
        doNotUseSubCCD: bool, default=False
           Override any setting of superStarSubCCD (used for initial guess)
        onlyObsErr: bool, default=False
           Only use observation error (used for initial guess)
        """

        startTime = time.time()
        self.fgcmLog.info('Computing superstarflats')

        # New version, use the stars directly
        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsSuperStarApplied = snmm.getArray(self.fgcmStars.obsSuperStarAppliedHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        # Flag bad observations here...
        self.fgcmStars.performSuperStarOutlierCuts(self.fgcmPars)

        goodStars = self.fgcmStars.getGoodStarIndices(checkMinObs=True)
        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=self.fgcmPars.expFlag, checkBadMag=True)

        # we need to compute E_gray == <mstd> - mstd for each observation
        # compute EGray, GO for Good Obs
        #EGrayGO = (objMagStdMean[obsObjIDIndex[goodObs],obsBandIndex[goodObs]] -
        #           obsMagStd[goodObs])

        #if onlyObsErr:
            # only observational error.  Use this option when doing initial guess at superstarFlat
        #    EGrayErr2GO = obsMagErr[goodObs]**2.
        #else:
            # take into account correlated average mag error
        #    EGrayErr2GO = (obsMagErr[goodObs]**2. -
        #                   objMagStdMeanErr[obsObjIDIndex[goodObs],obsBandIndex[goodObs]]**2.)
        EGrayGO, EGrayErr2GO = self.fgcmStars.computeEGray(goodObs, onlyObsErr=onlyObsErr)

        # one more cut on the maximum error
        # as well as making sure that it didn't go below zero
        gd,=np.where((EGrayErr2GO < self.ccdGrayMaxStarErr) & (EGrayErr2GO > 0.0) &
                     (np.abs(EGrayGO) < 50.0))

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

        # Note that we use the cheb2dFunc even when the previous numbers
        #  were just an offset, because the other terms are zeros
        prevSuperStarFlatCenter[:, :, :] = self.fgcmPars.superStarFlatCenter

        if not self.superStarSubCCD or doNotUseSubCCD:
            # do not use subCCD x/y information (or x/y not available)

            mark = np.ones(goodObs.size, dtype=np.bool)

            # Next, we sort by epoch, band
            superStarWt = np.zeros_like(superStarFlatCenter)
            superStarOffset = np.zeros_like(superStarWt)

            goodObs2 = goodObs[mark]

            np.add.at(superStarWt,
                      (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs2]],
                       self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs2]],
                       obsCCDIndex[goodObs2]),
                      1./EGrayErr2GO[mark])
            np.add.at(superStarOffset,
                      (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs2]],
                       self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs2]],
                       obsCCDIndex[goodObs2]),
                      EGrayGO[mark]/EGrayErr2GO[mark])
            np.add.at(superStarNGoodStars,
                      (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs2]],
                       self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs2]],
                       obsCCDIndex[goodObs[mark]]),
                      1)

            # We need to make sure we set the bad ones to zero, or else we get
            # crazy statistics
            # This cut maybe should be larger, huh.
            gd = (superStarNGoodStars > 2)
            superStarOffset[gd] /= superStarWt[gd]
            superStarOffset[~gd] = 0.0

            # The superstar for the center is always in magnitude space
            superStarFlatCenter[:,:,:] = superStarOffset[:, :, :]

            # And the central parameter should be in flux or mag space depending
            self.fgcmPars.parSuperStarFlat[:, :, :, 0] = 10.**(superStarOffset / (-2.5))

        else:
            # with x/y, new sub-ccd

            # we will need the ccd offset signs
            self._computeCCDOffsetSigns(goodObs)

            # Scale X and Y
            # We assume that ccd goes from 0 to X_SIZE, 0 to Y_SIZE
            #xSize = self.ccdOffsets['X_SIZE'][obsCCDIndex]
            #ySize = self.ccdOffsets['Y_SIZE'][obsCCDIndex]
            #obsXScaled = (snmm.getArray(self.fgcmStars.obsXHandle) - xSize/2.) / (xSize/2.)
            #obsYScaled = (snmm.getArray(self.fgcmStars.obsYHandle) - ySize/2.) / (ySize/2.)

            #obsXScaledGO = obsXScaled[goodObs]
            #obsYScaledGO = obsYScaled[goodObs]

            obsXGO = snmm.getArray(self.fgcmStars.obsXHandle)[goodObs]
            obsYGO = snmm.getArray(self.fgcmStars.obsYHandle)[goodObs]

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

                computeMean = False
                try:
                    # New chebyshev method
                    order = self.superStarSubCCDChebyshevOrder
                    pars = np.zeros((order + 1, order + 1))
                    pars[0, 0] = 1.0
                    lowBounds = np.zeros_like(pars) - np.inf
                    highBounds = np.zeros_like(pars) + np.inf

                    # Check that we have enough stars to constrain this...
                    # In general, let's demand we have 10 times as many stars as
                    # parameters (which is actually quite thin), or else we'll
                    # just compute the mean
                    if (i1a.size < 10 * pars.size):
                        self.fgcmLog.info("Warning: insufficient stars for chebyshev fit (%d, %d, %d), setting to mean"
                                          % (epInd, fiInd, cInd))
                        fit = pars.flatten()
                        computeMean = True
                    else:
                        if self.superStarSubCCDTriangular:
                            iind = np.repeat(np.arange(order + 1), order + 1)
                            jind = np.tile(np.arange(order + 1), order + 1)
                            high, = np.where((iind + jind) > order)
                            # Cannot set exactly to zero or curve_fit will complain
                            lowBounds[iind[high], jind[high]] = -1e-50
                            highBounds[iind[high], jind[high]] = 1e-50

                        FGrayGOInd = 10.**(EGrayGO[i1a] / (-2.5))
                        FGrayErrGOInd = (np.log(10.) / 2.5) * np.sqrt(EGrayErr2GO[i1a]) * FGrayGOInd
                        field = Cheb2dField(self.ccdOffsets['X_SIZE'][cInd],
                                            self.ccdOffsets['Y_SIZE'][cInd],
                                            pars)
                        fit, cov = scipy.optimize.curve_fit(field,
                                                            np.vstack((obsXGO[i1a],
                                                                       obsYGO[i1a])),
                                                            FGrayGOInd,
                                                            p0=list(pars.flatten()),
                                                            sigma=FGrayErrGOInd,
                                                            bounds=list(np.vstack((lowBounds.flatten(),
                                                                                   highBounds.flatten()))))

                        if self.superStarSubCCDTriangular:
                            # Force these to be identically zero (which they probably are)
                            fit[high] = 0.0

                        if (fit[0] == 0.0 or fit[0] == 1.0 or
                            (fit[0] < 0.0)):
                            self.fgcmLog.info("Warning: fit failed on (%d, %d, %d), setting to mean"
                                              % (epInd, fiInd, cInd))
                            fit = pars.flatten()
                            computeMean = True

                except (ValueError, RuntimeError, TypeError):
                    self.fgcmLog.info("Warning: fit failed to converge (%d, %d, %d), setting to mean"
                                      % (epInd, fiInd, cInd))
                    fit = pars.flatten()
                    computeMean = True

                if computeMean:
                    fit = np.zeros(self.fgcmPars.superStarNPar)
                    fit[0] = (np.sum(EGrayGO[i1a]/EGrayErr2GO[i1a]) /
                              np.sum(1./EGrayErr2GO[i1a]))
                    fit[0] = 10.**(fit[0] / (-2.5))

                superStarNGoodStars[epInd, fiInd, cInd] = i1a.size

                # compute the central value for use with the delta
                field = Cheb2dField(self.ccdOffsets['X_SIZE'][cInd],
                                    self.ccdOffsets['Y_SIZE'][cInd],
                                    fit)
                superStarFlatCenter[epInd, fiInd, cInd] = -2.5 * np.log10(field.evaluateCenter())

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
        Plot SuperStar flats and deltas

        parameters
        ----------
        superStarPars: float array (nEpochs, nLUTFilter, nCCD, superStarNPar)
           Parameters per epoch, filter, ccd
        deltaSuperStar: float array (nEpochs, nLUTFilter, nCCD)
           Central superstar offset
        superStarNGoodStars: int array (nEpochs, nLUTFilter, nCCD)
           Number of good stars per superstar
        superStarFlatFPMean: float array (nEpochs, nLUTFilter)
           Focal plane mean for plot scaling
        superStarFlatFPSigma: float array (nEpochs, nLUTFilter)
           Focal plane sigma for plot scaling
        deltaSuperStarFlatFPMean: float array (nEpochs, nLUTFilter)
           Focal plane mean of delta for plot scaling
        deltaSuperStarFlatFPSigma: float array (nEpochs, nLUTFilter)
           Focal plane sigma of delta for plot scaling
        """

        from .fgcmUtilities import plotCCDMap
        from .fgcmUtilities import plotCCDMap2d

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
                    # New flux parameters
                    plotCCDMap(ax, self.ccdOffsets[use], -2.5 * np.log10(superStarPars[e, f, use, 0]),
                               'SuperStar (mag)')
                else:
                    plotCCDMap2d(ax, self.ccdOffsets[use], superStarPars[e, f, use, :],
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
        """
        Internal method to figure out plotting signs

        parameters
        ----------
        goodObs: int array
           Array of good observations
        """

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

                if (np.abs(corrXRA) > np.abs(corrYRA)):
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
                    if corrYDec < 0:
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

    def computeSuperStarFlatsOrig(self,fgcmGray,doPlots=True):
        """
        Old-school superStarFlat computed from CCDGray

        parameters
        ----------
        fgcmGray: FgcmGray
        doPlots: bool, default=True
        """

        startTime = time.time()
        self.fgcmLog.info('Computing superstarflats')

        ## FIXME: need to filter out SN (deep) exposures.  Hmmm.

        deltaSuperStarFlat = np.zeros((self.fgcmPars.nEpochs,
                                       self.fgcmPars.nLUTFilter,
                                       self.fgcmPars.nCCD))
        deltaSuperStarFlatNCCD = np.zeros_like(deltaSuperStarFlat, dtype='i4')

        ccdGray = snmm.getArray(fgcmGray.ccdGrayHandle)
        ccdGrayErr = snmm.getArray(fgcmGray.ccdGrayErrHandle)
        ccdNGoodStars = snmm.getArray(fgcmGray.ccdNGoodStarsHandle)

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
        Old-school plotting of superStarFlats

        parameters
        ----------
        superStarArray: float array (nEpochs, nLUTFilter, nCCD)
        superStarMean: float array (nEpochs, nLUTFilter)
           Focal plane average for plot scaling
        superStarSigma: float array (nEpochs, nLUTFilter)
           Focal plane sigma for plot scaling
        nCCDArray: float array, optional (nEpochs, nLUTFilter, nCCD)
           Number of good CCDs per superstar
        name: string, default='superstar'
           What name to put on the files/plots
        """

        from .fgcmUtilities import plotCCDMap

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
                plt.close(fig)


