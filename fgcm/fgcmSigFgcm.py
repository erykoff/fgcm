import numpy as np
import os
import sys
import esutil
import time
import scipy.optimize

from .fgcmUtilities import makeFigure, putButlerFigure

from .fgcmUtilities import gaussFunction
from .fgcmUtilities import histoGauss

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmSigFgcm(object):
    """
    Class to compute repeatability statistics for stars.

    parameters
    ----------
    fgcmConfig: FgcmConfig
    fgcmPars: FgcmParameters
    fgcmStars: FgcmStars

    Config variables
    ----------------
    sigFgcmMaxEGray: list
       Maxmimum m_std - <m_std> to consider to compute sigFgcm
    sigFgcmMaxErr: float
       Maxmimum error on m_std - <m_std> to consider to compute sigFgcm
    """

    def __init__(self, fgcmConfig, fgcmPars, fgcmStars, butlerQC=None, plotHandleDict=None):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.debug('Initializing FgcmSigFgcm')

        # need fgcmPars because it has the sigFgcm
        self.fgcmPars = fgcmPars

        # need fgcmStars because it has the stars (duh)
        self.fgcmStars = fgcmStars

        self.butlerQC = butlerQC
        self.plotHandleDict = plotHandleDict

        # and config numbers
        self.sigFgcmMaxEGray = fgcmConfig.sigFgcmMaxEGray
        self.sigFgcmMaxErr = fgcmConfig.sigFgcmMaxErr
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber
        self.colorSplitIndices = fgcmConfig.colorSplitIndices
        self.bandRequiredIndex = fgcmConfig.bandRequiredIndex
        self.bandNotRequiredIndex = fgcmConfig.bandNotRequiredIndex
        self.quietMode = fgcmConfig.quietMode

    def computeSigFgcm(self, reserved=False, save=True, crunch=False, nonphotometric=False):
        """
        Compute sigFgcm for all bands

        parameters
        ----------
        reserved : `bool`, optional
            Use reserved stars instead of fit stars?
        save : `bool`, optional
            Save computed values to fgcmPars?
        crunch : `bool`, optional
            Compute based on ccd-crunched values?
        nonphotometric : `bool`, optional
            Include non-photometric exposures?
        """

        if (not self.fgcmStars.magStdComputed):
            raise ValueError("Must run FgcmChisq to compute magStd before computeCCDAndExpGray")

        startTime = time.time()
        self.fgcmLog.debug('Computing sigFgcm.')

        # input numbers
        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)

        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        if reserved:
            goodStars = self.fgcmStars.getGoodStarIndices(onlyReserve=True, checkMinObs=True)
        else:
            goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=True, checkMinObs=True)

        if nonphotometric:
            expFlagTemp = None
        else:
            expFlagTemp = self.fgcmPars.expFlag

        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=expFlagTemp, checkBadMag=True)

        # we need to compute E_gray == <mstd> - mstd for each observation
        # this statistic is internal only, no reference stars.
        # see fgcmSigmaRef for reference stars
        EGrayGO, EGrayErr2GO = self.fgcmStars.computeEGray(goodObs, ignoreRef=True)

        EGrayPullGO = EGrayGO / np.sqrt(EGrayErr2GO)

        # now we can compute sigFgcm

        sigFgcm = np.zeros(self.fgcmStars.nBands)

        # and we do 4 runs: full, blue 25%, middle 50%, red 25%

        # Compute "g-i" based on the configured colorSplitIndices
        gmiGO = (objMagStdMean[obsObjIDIndex[goodObs], self.colorSplitIndices[0]] -
                 objMagStdMean[obsObjIDIndex[goodObs], self.colorSplitIndices[1]])

        # Note that not every star has a valid g-i color, so we need to check for that.
        okColor, = np.where((objMagStdMean[obsObjIDIndex[goodObs], self.colorSplitIndices[0]] < 90.0) &
                            (objMagStdMean[obsObjIDIndex[goodObs], self.colorSplitIndices[1]] < 90.0))
        # sort these
        st = np.argsort(gmiGO[okColor])
        gmiCutLow = np.array([0.0,
                              gmiGO[okColor[st[0]]],
                              gmiGO[okColor[st[int(0.25 * st.size)]]],
                              gmiGO[okColor[st[int(0.75 * st.size)]]]])
        gmiCutHigh = np.array([0.0,
                               gmiGO[okColor[st[int(0.25 * st.size)]]],
                               gmiGO[okColor[st[int(0.75 * st.size)]]],
                               gmiGO[okColor[st[-1]]]])
        gmiCutNames = ['All', 'Blue25', 'Middle50', 'Red25']

        sigTypes = []
        if reserved:
            sigTypes.append('reserved')
            extraName = 'reserved-stars'
            extraNameButler = "ReservedStars"
        else:
            sigTypes.append('fit')
            extraName = 'all-stars'
            extraNameButler = "AllStars"
        if crunch:
            sigTypes.append('crunched')
            extraName += '_crunched'
            extraNameButler += "Crunched"
        if nonphotometric:
            sigTypes.append('nonphotometric')
            extraName += "_nonphot"
            extraNameButler += "NonPhotom"

        sigType = '/'.join(sigTypes)

        for bandIndex, band in enumerate(self.fgcmStars.bands):
            if not self.fgcmPars.hasExposuresInBand[bandIndex]:
                continue

            # start the figure which will have 4 panels
            fig = makeFigure(figsize=(9, 6))
            fig.clf()

            started=False
            for c, name in enumerate(gmiCutNames):
                if c == 0:
                    # This is the "All"
                    # There shouldn't be any need any additional checks on if these
                    # stars were actually observed in this band, because the goodObs
                    # selection takes care of that.
                    sigUse, = np.where((np.abs(EGrayGO) < self.sigFgcmMaxEGray[bandIndex]) &
                                       (EGrayErr2GO > 0.0) &
                                       (EGrayErr2GO < self.sigFgcmMaxErr**2.) &
                                       (EGrayGO != 0.0) &
                                       (obsBandIndex[goodObs] == bandIndex))
                else:
                    sigUse, = np.where((np.abs(EGrayGO[okColor]) < self.sigFgcmMaxEGray[bandIndex]) &
                                       (EGrayErr2GO[okColor] > 0.0) &
                                       (EGrayErr2GO[okColor] < self.sigFgcmMaxErr**2.) &
                                       (EGrayGO[okColor] != 0.0) &
                                       (obsBandIndex[goodObs[okColor]] == bandIndex) &
                                       (gmiGO[okColor] > gmiCutLow[c]) &
                                       (gmiGO[okColor] < gmiCutHigh[c]))
                    sigUse = okColor[sigUse]

                if (sigUse.size == 0):
                    self.fgcmLog.info('sigFGCM: No good observations in %s band (color cut %d).' %
                                     (self.fgcmPars.bands[bandIndex],c))
                    continue

                ax=fig.add_subplot(2,2,c+1)

                try:
                    coeff = histoGauss(ax, EGrayGO[sigUse]*1000.0)
                    coeff[1] /= 1000.0
                    coeff[2] /= 1000.0
                except Exception as inst:
                    coeff = np.array([np.inf, np.inf, np.inf])

                if not np.isfinite(coeff[2]):
                    self.fgcmLog.info("Failed to compute sigFgcm (%s) (%s).  Setting to 0.05" %
                                     (self.fgcmPars.bands[bandIndex], name))
                    sigFgcm[bandIndex] = 0.05
                elif (np.median(EGrayErr2GO[sigUse]) > coeff[2]**2.):
                    self.fgcmLog.info("Typical error is larger than width (%s) (%s).  Setting to 0.001" %
                                      (self.fgcmPars.bands[bandIndex], name))
                    sigFgcm[bandIndex] = 0.001
                else:
                    sigFgcm[bandIndex] = np.sqrt(coeff[2]**2. -
                                                 np.median(EGrayErr2GO[sigUse]))

                self.fgcmLog.info("%s sigFgcm (%s) (%s) = %.2f mmag" % (
                        sigType,
                        self.fgcmPars.bands[bandIndex],
                        name,
                        sigFgcm[bandIndex]*1000.0))

                if (save and (c==0)):
                    # only save if we're doing the full color range
                    self.fgcmPars.compSigFgcm[bandIndex] = sigFgcm[bandIndex]

                if (reserved and not crunch and (c == 0) and not nonphotometric):
                    # Save the reserved raw repeatability.  Used for
                    # convergence testing in LSST stack.
                    self.fgcmPars.compReservedRawRepeatability[bandIndex] = coeff[2]
                elif (reserved and crunch and (c == 0)) and not nonphotometric:
                    self.fgcmPars.compReservedRawCrunchedRepeatability[bandIndex] = coeff[2]

                if self.plotPath is None:
                    continue

                ax.tick_params(axis='both',which='major',labelsize=14)

                text=r'$(%s)$' % (self.fgcmPars.bands[bandIndex]) + '\n' + \
                    r'$\mathrm{Cycle\ %d}$' % (self.cycleNumber) + '\n' + \
                    r'$\mu = %.2f$' % (coeff[1]*1000.0) + '\n' + \
                    r'$\sigma_\mathrm{tot} = %.2f$' % (coeff[2]*1000.0) + '\n' + \
                    r'$\sigma_\mathrm{FGCM} = %.2f$' % (sigFgcm[bandIndex]*1000.0) + '\n' + \
                    name

                ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=14)
                ax.set_xlabel(r'$E^{\mathrm{gray}}\,(\mathrm{mmag})$',fontsize=14)

                ax.set_title(extraName)

                if (not started):
                    started=True
                    plotXRange = ax.get_xlim()
                else:
                    ax.set_xlim(plotXRange)

            if self.plotPath is not None:
                fig.tight_layout()

                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    f"SigmaFgcm{extraNameButler}",
                                    self.cycleNumber,
                                    fig,
                                    band=self.fgcmPars.bands[bandIndex])
                else:
                    fig.savefig('%s/%s_sigfgcm_%s_%s.png' % (self.plotPath,
                                                             self.outfileBaseWithCycle,
                                                             extraName,
                                                             self.fgcmPars.bands[bandIndex]))

        # Make pulls plots as above.
        for bandIndex, band in enumerate(self.fgcmStars.bands):
            if not self.fgcmPars.hasExposuresInBand[bandIndex]:
                continue

            # start the figure which will have 4 panels
            fig = makeFigure(figsize=(9, 6))
            fig.clf()

            started = False
            for c, name in enumerate(gmiCutNames):
                if c == 0:
                    # This is the "All"
                    # There shouldn't be any need any additional checks on if these
                    # stars were actually observed in this band, because the goodObs
                    # selection takes care of that.
                    sigUse, = np.where((np.abs(EGrayGO) < self.sigFgcmMaxEGray[bandIndex]) &
                                       (EGrayErr2GO > 0.0) &
                                       (EGrayErr2GO < self.sigFgcmMaxErr**2.) &
                                       (EGrayGO != 0.0) &
                                       (obsBandIndex[goodObs] == bandIndex))
                else:
                    sigUse, = np.where((np.abs(EGrayGO[okColor]) < self.sigFgcmMaxEGray[bandIndex]) &
                                       (EGrayErr2GO[okColor] > 0.0) &
                                       (EGrayErr2GO[okColor] < self.sigFgcmMaxErr**2.) &
                                       (EGrayGO[okColor] != 0.0) &
                                       (obsBandIndex[goodObs[okColor]] == bandIndex) &
                                       (gmiGO[okColor] > gmiCutLow[c]) &
                                       (gmiGO[okColor] < gmiCutHigh[c]))
                    sigUse = okColor[sigUse]

                if (sigUse.size == 0):
                    self.fgcmLog.info('sigFGCMPulls: No good observations in %s band (color cut %d).' %
                                     (self.fgcmPars.bands[bandIndex],c))
                    continue

                ax = fig.add_subplot(2, 2, c + 1)

                try:
                    coeff = histoGauss(ax, EGrayPullGO[sigUse])
                except Exception as inst:
                    coeff = np.array([np.inf, np.inf, np.inf])

                if not np.isfinite(coeff[2]):
                    self.fgcmLog.info("Failed to compute sigFgcmPulls (%s) (%s)." %
                                      (self.fgcmPars.bands[bandIndex], name))
                    coeff[2] = -10.0

                self.fgcmLog.info("%s sigFgcmPull (%s) (%s) = %.2f" % (
                        sigType,
                        self.fgcmPars.bands[bandIndex],
                        name,
                        coeff[2]))

                if self.plotPath is None:
                    continue

                ax.tick_params(axis='both',which='major',labelsize=14)

                text=r'$(%s)$' % (self.fgcmPars.bands[bandIndex]) + '\n' + \
                    r'$\mathrm{Cycle\ %d}$' % (self.cycleNumber) + '\n' + \
                    r'$\mu = %.2f$' % (coeff[1]) + '\n' + \
                    r'$\sigma_\mathrm{pull} = %.2f$' % (coeff[2]) + '\n' + \
                    name

                ax.annotate(text, (0.95, 0.93), xycoords='axes fraction', ha='right', va='top', fontsize=14)
                ax.set_xlabel(r'$E^{\mathrm{gray}}/\sigma_{E^{\mathrm{gray}}}$', fontsize=14)

                ax.set_title(extraName)

                if (not started):
                    started = True
                    plotXRange = ax.get_xlim()
                else:
                    ax.set_xlim(plotXRange)

            if self.plotPath is not None:
                fig.tight_layout()

                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    f"SigmaFgcmPulls{extraNameButler}",
                                    self.cycleNumber,
                                    fig,
                                    band=self.fgcmPars.bands[bandIndex])
                else:
                    fig.savefig('%s/%s_sigfgcmpulls_%s_%s.png' % (self.plotPath,
                                                                  self.outfileBaseWithCycle,
                                                                  extraName,
                                                                  self.fgcmPars.bands[bandIndex]))




        if not self.quietMode:
            self.fgcmLog.info('Done computing sigFgcm in %.2f sec.' %
                              (time.time() - startTime))



