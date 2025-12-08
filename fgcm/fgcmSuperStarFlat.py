import numpy as np
import os
import sys
import esutil
import time
import scipy.optimize
from scipy.stats import median_abs_deviation

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm
from .fgcmUtilities import Cheb2dField
from .fgcmUtilities import makeFigure, putButlerFigure


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

    def __init__(self, fgcmConfig, fgcmPars, fgcmStars, butlerQC=None, plotHandleDict=None):

        self.fgcmLog = fgcmConfig.fgcmLog
        self.fgcmLog.debug('Initializing FgcmSuperStarFlat')

        self.fgcmPars = fgcmPars

        self.fgcmStars = fgcmStars

        self.butlerQC = butlerQC
        self.plotHandleDict = plotHandleDict

        self.illegalValue = fgcmConfig.illegalValue
        self.minStarPerCCD = fgcmConfig.minStarPerCCD
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber
        self.epochNames = fgcmConfig.epochNames
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.ccdGrayMaxStarErr = fgcmConfig.ccdGrayMaxStarErr
        self.quietMode = fgcmConfig.quietMode

        self.superStarSubCCD = fgcmConfig.superStarSubCCD
        self.superStarSubCCDChebyshevOrder = fgcmConfig.superStarSubCCDChebyshevOrder
        self.superStarSubCCDTriangular = fgcmConfig.superStarSubCCDTriangular
        self.superStarSigmaClip = fgcmConfig.superStarSigmaClip
        self.superStarPlotCCDResiduals = fgcmConfig.superStarPlotCCDResiduals

    def setDeltaMapperDefault(self, deltaMapperDefault):
        """
        Set the deltaMapperDefault array.

        Parameters
        ----------
        deltaMapperDefault : `np.recarray`
        """
        self.deltaMapperDefault = deltaMapperDefault

    def computeSuperStarFlats(self, doPlots=True, doNotUseSubCCD=False, onlyObsErr=False, forceZeroMean=False):
        """
        Compute the SuperStar Flats

        parameters
        ----------
        doPlots: bool, default=True
        doNotUseSubCCD: bool, default=False
           Override any setting of superStarSubCCD (used for initial guess)
        onlyObsErr: bool, default=False
           Only use observation error (used for initial guess)
        forceZeroMean: bool, default=False
           Force the mean superstar to be zero in each epoch/band
        """

        startTime = time.time()
        self.fgcmLog.debug('Computing superstarflats')

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

        if not np.any(self.superStarSubCCD) or doNotUseSubCCD:
            # do not use subCCD x/y information (or x/y not available)

            mark = np.ones(goodObs.size, dtype=bool)

            # Next, we sort by epoch, band
            superStarWt = np.zeros_like(superStarFlatCenter)
            superStarOffset = np.zeros_like(superStarWt)

            goodObs2 = goodObs[mark]

            np.add.at(superStarWt,
                      (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs2]],
                       self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs2]],
                       obsCCDIndex[goodObs2]),
                      (1./EGrayErr2GO[mark]).astype(superStarWt.dtype))
            np.add.at(superStarOffset,
                      (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs2]],
                       self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs2]],
                       obsCCDIndex[goodObs2]),
                      (EGrayGO[mark]/EGrayErr2GO[mark]).astype(superStarOffset.dtype))
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
            superStarOffset[~gd] = self.illegalValue

            # The superstar for the center is always in magnitude space
            # And this can be set to "illegal value"
            superStarFlatCenter[:, :, :] = superStarOffset[:, :, :]

            # And for signaling here, we're going to set to a large value
            superStarOffset[~gd] = 100.0

            # And the central parameter should be in flux space
            self.fgcmPars.parSuperStarFlat[:, :, :, 0] = 10.**(superStarOffset / (-2.5))

        else:
            # with x/y, new sub-ccd

            # Note that the ccd offset signs are now computed in fgcmFitCycle.

            obsXGO = snmm.getArray(self.fgcmStars.obsXHandle)[goodObs]
            obsYGO = snmm.getArray(self.fgcmStars.obsYHandle)[goodObs]

            # need to histogram this all up.  Watch for extra bands

            epochFilterHash = (self.fgcmPars.expEpochIndex[obsExpIndex[goodObs]]*
                               (self.fgcmPars.nLUTFilter+1)*(self.fgcmPars.nCCD+1) +
                               self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs]]*
                               (self.fgcmPars.nCCD+1) +
                               obsCCDIndex[goodObs])

            h, rev = esutil.stat.histogram(epochFilterHash, rev=True)

            for i in range(h.size):
                if h[i] == 0: continue

                i1a = rev[rev[i]:rev[i+1]]

                # get the indices for this epoch/filter/ccd
                epInd = self.fgcmPars.expEpochIndex[obsExpIndex[goodObs[i1a[0]]]]
                fiInd = self.fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs[i1a[0]]]]
                cInd = obsCCDIndex[goodObs[i1a[0]]]
                bInd = self.fgcmPars.expBandIndex[obsExpIndex[goodObs[i1a[0]]]]

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
                    if not self.superStarSubCCD[bInd]:
                        # Just compute mean for this band
                        fit = pars.flatten()
                        computeMean = True
                    elif (i1a.size < 10 * pars.size):
                        self.fgcmLog.warning("Insufficient stars for chebyshev fit (%d, %d, %d), setting to mean"
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
                        field = Cheb2dField(self.deltaMapperDefault['x_size'][cInd],
                                            self.deltaMapperDefault['y_size'][cInd],
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
                            self.fgcmLog.warning("Fit failed on (%d, %d, %d), setting to mean"
                                                 % (epInd, fiInd, cInd))
                            fit = pars.flatten()
                            computeMean = True

                except (ValueError, RuntimeError, TypeError):
                    self.fgcmLog.warning("Fit failed to converge (%d, %d, %d), setting to mean"
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
                field = Cheb2dField(self.deltaMapperDefault['x_size'][cInd],
                                    self.deltaMapperDefault['y_size'][cInd],
                                    fit)
                superStarFlatCenter[epInd, fiInd, cInd] = -2.5 * np.log10(field.evaluateCenter())

                # and record the fit
                self.fgcmPars.parSuperStarFlat[epInd, fiInd, cInd, :] = 0
                self.fgcmPars.parSuperStarFlat[epInd, fiInd, cInd, 0: fit.size] = fit

                if doPlots and i1a.size > 0 and self.superStarPlotCCDResiduals:
                    # Compute the residuals and plot them.
                    superStar = -2.5 * np.log10(np.clip(field.evaluate(obsXGO[i1a], obsYGO[i1a]), 0.1, None))
                    resid = EGrayGO[i1a] - superStar

                    # Choose a gridsize appropriate for the number of stars.
                    gridsize = int(np.clip(np.sqrt(i1a.size/5), 2, 150))

                    vmin, vmax = np.percentile(resid*1000, [25, 75])

                    fig = makeFigure(figsize=(8, 6))
                    fig.clf()
                    ax = fig.add_subplot(111)

                    hb = ax.hexbin(obsXGO[i1a], obsYGO[i1a], C=resid*1000, gridsize=gridsize, vmin=vmin, vmax=vmax)
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_title("%s %s %s" % (self.fgcmPars.lutFilterNames[fiInd],
                                               self.epochNames[epInd],
                                               str(cInd)))
                    ax.set_aspect("equal")
                    fig.colorbar(hb, label="SuperStar Residual (mmag)")
                    fig.tight_layout()

                    # These plots will only be used for local debugging because
                    # of the large number of plots that may be produced
                    # (one per filter per epoch per detector).

                    if self.butlerQC is not None:
                        putButlerFigure(
                            self.fgcmLog,
                            self.butlerQC,
                            self.plotHandleDict,
                            "SuperstarResidual",
                            self.cycleNumber,
                            fig,
                            filterName=self.fgcmPars.lutFilterNames[fiInd],
                            epoch=self.epochNames[epInd],
                            detector=str(cInd),
                        )
                    elif self.plotPath is not None:
                        fig.savefig("%s/%s_superstar_resid_%s_%s_%s.png" % (self.plotPath,
                                                                            self.outfileBaseWithCycle,
                                                                            self.fgcmPars.lutFilterNames[fiInd],
                                                                            self.epochNames[epInd],
                                                                            str(cInd)))

                    def std_func(x):
                        return median_abs_deviation(x, scale="normal")

                    fig = makeFigure(figsize=(8, 6))
                    fig.clf()
                    ax = fig.add_subplot(111)

                    hb = ax.hexbin(
                        obsXGO[i1a],
                        obsYGO[i1a],
                        C=resid*1000,
                        gridsize=gridsize,
                        reduce_C_function=std_func,
                    )
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_title("%s %s %s" % (self.fgcmPars.lutFilterNames[fiInd],
                                               self.epochNames[epInd],
                                               str(cInd)))
                    ax.set_aspect("equal")
                    fig.colorbar(hb, label="SuperStar Residual Std Dev (mmag)")
                    fig.tight_layout()
                    if self.butlerQC is not None:
                        putButlerFigure(
                            self.fgcmLog,
                            self.butlerQC,
                            self.plotHandleDict,
                            "SuperstarResidualStd",
                            self.cycleNumber,
                            fig,
                            filterName=self.fgcmPars.lutFilterNames[fiInd],
                            epoch=self.epochNames[epInd],
                            detector=str(cInd),
                        )
                    elif self.plotPath is not None:
                        fig.savefig("%s/%s_superstar_residstd_%s_%s_%s.png" % (self.plotPath,
                                                                               self.outfileBaseWithCycle,
                                                                               self.fgcmPars.lutFilterNames[fiInd],
                                                                               self.epochNames[epInd],
                                                                               str(cInd)))

            # And we need to flag those that have bad observations
            bad = np.where(superStarNGoodStars == 0)
            if bad[0].size > 0:
                superStarFlatCenter[bad] = self.illegalValue
                self.fgcmPars.parSuperStarFlat[bad[0], bad[1], bad[2], 0] = 100.0

        # compute the delta...
        deltaSuperStarFlatCenter = superStarFlatCenter - prevSuperStarFlatCenter

        # and the overall stats...
        for e in range(self.fgcmPars.nEpochs):
            for f in range(self.fgcmPars.nLUTFilter):
                use,=np.where(superStarNGoodStars[e, f, :] > 0)

                if use.size < 3:
                    continue

                superStarFlatFPMean[e, f] = np.mean(superStarFlatCenter[e, f, use])
                superStarFlatFPSigma[e, f] = np.std(superStarFlatCenter[e, f, use])

                if forceZeroMean:
                    # Subtract off the mag
                    superStarFlatCenter[e, f, use] -= superStarFlatFPMean[e, f]
                    # Divide the flux parameter
                    self.fgcmPars.parSuperStarFlat[e, f, use, 0] /= 10.**(superStarFlatFPMean[e, f] / (-2.5))
                    # And reset the mean and delta
                    deltaSuperStarFlatCenter[e, f, use] -= superStarFlatFPMean[e, f]
                    superStarFlatFPMean[e, f] -= superStarFlatFPMean[e, f]

                deltaSuperStarFlatFPMean[e, f] = np.mean(deltaSuperStarFlatCenter[e, f, use])
                deltaSuperStarFlatFPSigma[e, f] = np.std(deltaSuperStarFlatCenter[e, f, use])

                self.fgcmLog.info('Superstar epoch %d filter %s: %.2f +/- %.2f  Delta: %.2f +/- %.2f mmag' %
                                  (e, self.fgcmPars.lutFilterNames[f],
                                   superStarFlatFPMean[e, f] * 1000.0, superStarFlatFPSigma[e, f] * 1000.0,
                                   deltaSuperStarFlatFPMean[e, f] * 1000.0, deltaSuperStarFlatFPSigma[e, f] * 1000.0))

        if not self.quietMode:
            self.fgcmLog.info('Computed SuperStarFlats in %.2f seconds.' %
                              (time.time() - startTime))

        if doPlots:
            self.fgcmLog.debug('Making SuperStarFlat plots')

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
        from matplotlib import colormaps

        # We want to use a red/blue diverging colormap, centered at 0.
        cmap = colormaps.get_cmap("bwr")

        for e in range(self.fgcmPars.nEpochs):
            for f in range(self.fgcmPars.nLUTFilter):
                use, = np.where(superStarNGoodStars[e, f, :] > 0)

                if use.size == 0:
                    continue

                # double-wide.  Don't give number because that was dumb
                fig = makeFigure(figsize=(16, 6))
                fig.clf()

                # left side plot the map with x/y
                ax=fig.add_subplot(121)

                if not self.superStarSubCCD:
                    # New flux parameters
                    plotCCDMap(ax, self.deltaMapperDefault[use], -2.5 * np.log10(superStarPars[e, f, use, 0]) * 1000.0,
                               'SuperStar (mmag)', cmap=cmap, symmetric=True)
                else:
                    plotCCDMap2d(ax, self.deltaMapperDefault[use], superStarPars[e, f, use, :],
                                 'SuperStar (mmag)', cmap=cmap, symmetric=True)

                # and annotate

                text = '(%s)' % (self.fgcmPars.lutFilterNames[f]) + '\n' + \
                    r'%.2f +/- %.2f' % (superStarFlatFPMean[e,f]*1000.0,
                                        superStarFlatFPSigma[e,f]*1000.0)
                ax.annotate(text,
                            (0.1,0.93),xycoords='axes fraction',
                            ha='left',va='top',fontsize=18)

                # right side plot the deltas
                ax=fig.add_subplot(122)

                plotCCDMap(ax, self.deltaMapperDefault[use], deltaSuperStar[e, f, use]*1000.0,
                           'Central Delta-SuperStar (mmag)', cmap=cmap, symmetric=True)

                # and annotate
                text = '(%s)' % (self.fgcmPars.lutFilterNames[f]) + '\n' + \
                    r'%.2f +/- %.2f' % (deltaSuperStarFlatFPMean[e,f]*1000.0,
                                        deltaSuperStarFlatFPSigma[e,f]*1000.0)
                ax.annotate(text,
                            (0.1,0.93),xycoords='axes fraction',
                            ha='left',va='top',fontsize=18)

                fig.tight_layout()

                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "Superstar",
                                    self.cycleNumber,
                                    fig,
                                    filterName=self.fgcmPars.lutFilterNames[f],
                                    epoch=self.epochNames[e])
                elif self.plotPath is not None:
                    fig.savefig("%s/%s_%s_%s_%s.png" % (self.plotPath,
                                                        self.outfileBaseWithCycle,
                                                        "superstar",
                                                        self.fgcmPars.lutFilterNames[f],
                                                        self.epochNames[e]))
