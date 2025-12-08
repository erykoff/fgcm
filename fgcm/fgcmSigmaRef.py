import numpy as np
import os
import sys
import esutil
import time
import scipy.optimize
from scipy.stats import median_abs_deviation

from .fgcmUtilities import gaussFunction
from .fgcmUtilities import histoGauss
from .fgcmUtilities import objFlagDict
from .fgcmUtilities import makeFigure, putButlerFigure

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmSigmaRef(object):
    """
    Class to compute reference catalog statistics for stars.

    Parameters
    ----------
    fgcmConfig: FgcmConfig
    fgcmPars: FgcmParameters
    fgcmStars: FgcmStars
    """

    def __init__(self, fgcmConfig, fgcmPars, fgcmStars, butlerQC=None, plotHandleDict=None):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.debug('Initializing FgcmSigmaRef')

        self.fgcmPars = fgcmPars
        self.fgcmStars = fgcmStars

        self.butlerQC = butlerQC
        self.plotHandleDict = plotHandleDict

        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber
        self.colorSplitIndices = fgcmConfig.colorSplitIndices
        self.refStarOutlierNSig = fgcmConfig.refStarOutlierNSig
        self.quietMode = fgcmConfig.quietMode

        if not self.fgcmStars.hasRefstars:
            raise RuntimeError("Cannot use FgcmSigmaRef without reference stars!")

    def computeSigmaRef(self, doPlots=False):
        """
        Compute sigmaRef for all bands

        Parameters
        ----------
        doPlots : `bool`, optional
        """

        startTime = time.time()
        self.fgcmLog.debug('Computing sigmaRef')

        # Input numbers
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)

        objRefIDIndex = snmm.getArray(self.fgcmStars.objRefIDIndexHandle)
        refMag = snmm.getArray(self.fgcmStars.refMagHandle)
        refMagErr = snmm.getArray(self.fgcmStars.refMagErrHandle)

        # FIXME: at the moment, use all stars
        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=True, checkMinObs=True)

        # Select only stars that have reference magnitudes
        # and that are not flagged as outliers
        # We allow all the reference colors including reserved in
        # to get color range statistics.
        mask = objFlagDict['REFSTAR_OUTLIER']
        use, = np.where((objRefIDIndex[goodStars] >= 0) &
                        ((objFlag[goodStars] & mask) == 0))
        goodRefStars = goodStars[use]

        # We need to have a branch of "small-number" and "large number" of reference stars
        offsetRef = np.zeros(self.fgcmStars.nBands)
        sigmaRef = np.zeros(self.fgcmStars.nBands)

        if goodRefStars.size < 50:
            # Arbitrarily do 50 as the cutoff between small and large number...

            self.fgcmLog.info('Found %d refstars (< 50), so computing "small-number" statistics:' % (goodRefStars.size))

            for bandIndex, band in enumerate(self.fgcmStars.bands):
                if not self.fgcmPars.hasExposuresInBand[bandIndex]:
                    continue

                # Filter on previous bad refstars
                refUse, = np.where((refMag[objRefIDIndex[goodRefStars], bandIndex] < 90.0) &
                                   (objMagStdMean[goodRefStars, bandIndex] < 90.0))

                if refUse.size == 0:
                    self.fgcmLog.info("No reference stars in %s band." % (band))
                    continue

                delta = (objMagStdMean[goodRefStars[refUse], bandIndex] -
                         refMag[objRefIDIndex[goodRefStars[refUse]], bandIndex])

                offsetRef[bandIndex] = np.median(delta)
                sigmaRef[bandIndex] = 1.4826 * np.median(np.abs(delta - offsetRef[bandIndex]))

                # We don't look for outliers with small-number statistics (for now)

                self.fgcmLog.info('offsetRef (%s) = %.2f +/- %.2f mmag' % (band, offsetRef[bandIndex] * 1000.0, sigmaRef[bandIndex] * 1000.0))

        else:
            # Large numbers
            self.fgcmLog.debug('More than 50 reference stars, so computing "large-number" statistics.')

            # and we do 4 runs: full, blue 25%, middle 50%, red 25%

            # Compute "g-i" based on the configured colorSplitIndices
            gmiGRS = (objMagStdMean[goodRefStars, self.colorSplitIndices[0]] -
                      objMagStdMean[goodRefStars, self.colorSplitIndices[1]])

            okColor, = np.where((objMagStdMean[goodRefStars, self.colorSplitIndices[0]] < 90.0) &
                                (objMagStdMean[goodRefStars, self.colorSplitIndices[1]] < 90.0))

            # sort these
            st = np.argsort(gmiGRS[okColor])
            gmiCutLow = np.array([0.0,
                                  gmiGRS[okColor[st[0]]],
                                  gmiGRS[okColor[st[int(0.25 * st.size)]]],
                                  gmiGRS[okColor[st[int(0.75 * st.size)]]]])
            gmiCutHigh = np.array([0.0,
                                   gmiGRS[okColor[st[int(0.25 * st.size)]]],
                                   gmiGRS[okColor[st[int(0.75 * st.size)]]],
                                   gmiGRS[okColor[st[-1]]]])
            gmiCutNames = ['All', 'Blue25', 'Middle50', 'Red25']

            message = None

            for bandIndex, band in enumerate(self.fgcmStars.bands):
                if not self.fgcmPars.hasExposuresInBand[bandIndex]:
                    continue

                # start the figure which will have 4 panels
                # (the figure may not be drawn and written if not configured)
                # Note that we use the histogram fit/plot code to get the
                # fit coefficients even if we are not persisting the plots.
                # Fortunately, the makeFigure code now ensures that this
                # does not have any side effects.
                fig = makeFigure(figsize=(9, 6))
                fig.clf()

                started = False
                for c, name in enumerate(gmiCutNames):
                    if c == 0:
                        # This is the "All"
                        refUse, = np.where((refMag[objRefIDIndex[goodRefStars], bandIndex] < 90.0) &
                                           (objMagStdMean[goodRefStars, bandIndex] < 90.0))
                    else:
                        refUse, = np.where((refMag[objRefIDIndex[goodRefStars[okColor]],
                                                   bandIndex] < 90.0) &
                                           (objMagStdMean[goodRefStars[okColor],
                                                          bandIndex] < 90.0) &
                                           (gmiGRS[okColor] > gmiCutLow[c]) &
                                           (gmiGRS[okColor] < gmiCutHigh[c]))
                        refUse = okColor[refUse]

                    if refUse.size == 0:
                        self.fgcmLog.info("No reference stars in %s band (color cut %d)." % (band, c))
                        continue

                    delta = (objMagStdMean[goodRefStars[refUse], bandIndex] -
                             refMag[objRefIDIndex[goodRefStars[refUse]], bandIndex])

                    # Precut for fit to ensure crazy outliers don't mess things up.
                    med = np.median(delta)
                    sigmaMad = median_abs_deviation(delta, scale="normal")
                    ok = (np.abs(delta - med) < 10*sigmaMad)

                    ax = fig.add_subplot(2, 2, c + 1)

                    try:
                        coeff = histoGauss(ax, delta[ok]*1000.0)
                        coeff[1] /= 1000.0
                        coeff[2] /= 1000.0
                        if coeff[3] > 0:
                            # This was a failed fit ... set to inf
                            coeff = np.array([np.inf, np.inf, np.inf, coeff[3]])
                    except Exception as inst:
                        coeff = np.array([np.inf, np.inf, np.inf])

                    if not np.isfinite(coeff[2]):
                        self.fgcmLog.info("Failed to compute sigmaRef (%s) (%s)." %
                                          (band, name))

                    offsetRef[bandIndex] = coeff[1]
                    sigmaRef[bandIndex] = coeff[2]

                    self.fgcmLog.info("offsetRef (%s) (%s) = %.2f +/- %0.2f mmag" %
                                      (band, name, offsetRef[bandIndex]*1000.0, sigmaRef[bandIndex]*1000.0))

                    # Compute outliers, if desired.
                    if (c == 0) and (self.refStarOutlierNSig > 0.0):
                        bad, = np.where((np.abs(delta - offsetRef[bandIndex]) >
                                         self.refStarOutlierNSig * sigmaRef[bandIndex]))
                        if bad.size > 0:
                            message = "Marked %d reference stars as REFSTAR_OUTLIER from observations in the %s band." % (bad.size, band)
                            objFlag[goodRefStars[refUse[bad]]] |= objFlagDict['REFSTAR_OUTLIER']
                        else:
                            message = None

                    if not doPlots:
                        continue

                    ax.tick_params(axis='both', which='major', labelsize=14)

                    text=r'$(%s)$' % (band) + '\n' + \
                        r'$\mathrm{Cycle\ %d}$' % (self.cycleNumber) + '\n' + \
                        r'$\mu = %.2f$' % (coeff[1]*1000.0) + '\n' + \
                        r'$\sigma_\mathrm{ref} = %.2f$' % (coeff[2]*1000.0) + '\n' + \
                        name

                    ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=14)
                    ax.set_xlabel(r'$\overline{m_\mathrm{std}} - m_\mathrm{ref}\,(\mathrm{mmag})$', fontsize=14)
                    if not started:
                        started = True
                        plotXRange = ax.get_xlim()
                    else:
                        ax.set_xlim(plotXRange)

                if doPlots:
                    fig.tight_layout()

                    if self.butlerQC is not None:
                        putButlerFigure(self.fgcmLog,
                                        self.butlerQC,
                                        self.plotHandleDict,
                                        "SigmaRef",
                                        self.cycleNumber,
                                        fig,
                                        band=band)
                    elif self.plotPath is not None:
                        fig.savefig('%s/%s_sigmaref_%s.png' % (self.plotPath,
                                                               self.outfileBaseWithCycle,
                                                               band))

                if message is not None:
                    self.fgcmLog.info(message)

        # Record these numbers because they are useful to have saved and
        # not just logged.
        self.fgcmPars.compRefOffset[:] = offsetRef
        self.fgcmPars.compRefSigma[:] = sigmaRef

        if not self.quietMode:
            self.fgcmLog.info('Done computing sigmaRef in %.2f sec.' %
                              (time.time() - startTime))

