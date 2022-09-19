import numpy as np
import os
import sys
import esutil
import time
import matplotlib.pyplot as plt
import scipy.optimize as optimize

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmMirrorChromaticity(object):
    """
    Class which computes mirror chromaticity corrections.

    Only run when fgcmConfig.fitMirrorChromaticity is True.

    Parameters
    ----------
    fgcmConfig: `FgcmConfig`
       Config object
    fgcmPars: `FgcmParameters`
       Parameter object
    fgcmStars: `FgcmStars`
       Stars object
    fgcmLUT: `FgcmLUT`
       Look-up table object

    """
    def __init__(self, fgcmConfig, fgcmPars, fgcmStars, fgcmLUT):
        self.fgcmLog = fgcmConfig.fgcmLog
        self.fgcmLog.info('Initializing FgcmMirrorChromaticity')

        self.fgcmPars = fgcmPars
        self.fgcmStars = fgcmStars
        self.fgcmLUT = fgcmLUT

        self.colorSplitIndices = fgcmConfig.colorSplitIndices
        self.illegalValue = fgcmConfig.illegalValue
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle

        self.I0StdBand = fgcmConfig.I0StdBand
        self.I1StdBand = fgcmConfig.I1StdBand
        self.I2StdBand = fgcmConfig.I2StdBand
        self.I10StdBand = fgcmConfig.I10StdBand

        self.magConst = 2.5 / np.log(10.0)

    def computeMirrorChromaticity(self):
        """
        Compute the mirror chromaticity terms.
        """

        startTime = time.time()
        self.fgcmLog.info("Fitting mirror chromaticity terms...")

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsLUTFilterIndex = snmm.getArray(self.fgcmStars.obsLUTFilterIndexHandle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)

        # First, we need to compute g-i and take the 25% red and blue ends

        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=False, checkMinObs=True, checkHasColor=True)
        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, checkBadMag=True, expFlag=self.fgcmPars.expFlag)

        gmiGO = (objMagStdMean[obsObjIDIndex[goodObs], self.colorSplitIndices[0]] -
                 objMagStdMean[obsObjIDIndex[goodObs], self.colorSplitIndices[1]])

        st = np.argsort(gmiGO)

        blueCut = gmiGO[st[int(0.25 * st.size)]]
        redCut = gmiGO[st[int(0.75 * st.size)]]

        # Unapply corrections locally
        corrections = self.fgcmStars.applyMirrorChromaticityCorrection(self.fgcmPars, self.fgcmLUT, returnCorrections=True)

        self.obsMagStdGO = obsMagStd[goodObs] - corrections[goodObs]

        self.obsExpIndexGO = obsExpIndex[goodObs]

        self.deltaTGO = self.fgcmPars.expMJD[self.obsExpIndexGO] - self.fgcmPars.mirrorChromaticityPivot[self.fgcmPars.expCoatingIndex[self.obsExpIndexGO]]

        self.objMagStdMeanGO = objMagStdMean[obsObjIDIndex[goodObs], obsBandIndex[goodObs]]
        self.objSEDSlopeGO = objSEDSlope[obsObjIDIndex[goodObs], obsBandIndex[goodObs]]
        self.EGrayErr2GO = obsMagErr[goodObs]**2. - objMagStdMeanErr[obsObjIDIndex[goodObs], obsBandIndex[goodObs]]**2.

        maxDt = np.max(np.abs(self.deltaTGO))

        bounds = [(-0.5 / maxDt, 0.5 / maxDt),
                  (0.0, 0.0)]

        if self.fgcmPars.nCoatingIntervals > 1:
            for i in range(self.fgcmPars.nCoatingIntervals - 1):
                bounds.append((-0.5, 0.5))

        for filterIndex in range(self.fgcmPars.nLUTFilter):
            use, = np.where((obsLUTFilterIndex[goodObs] == filterIndex) &
                            (self.fgcmPars.expFlag[obsExpIndex[goodObs]] == 0) &
                            (self.EGrayErr2GO > 0.0))
            redTemp, = np.where(gmiGO[use] > redCut)
            self.redStarsInFilter = use[redTemp]
            blueTemp, = np.where(gmiGO[use] < blueCut)
            self.blueStarsInFilter = use[blueTemp]

            parZeros = np.zeros(len(bounds))
            parInitial = self.fgcmPars.compMirrorChromaticity[filterIndex, :]
            self.lutFilterIndex = filterIndex

            fun = optimize.optimize.MemoizeJac(self)
            jac = fun.derivative

            res = optimize.minimize(fun,
                                    parInitial,
                                    method='L-BFGS-B',
                                    bounds=bounds,
                                    jac=jac,
                                    options={'maxfun': 50,
                                             'maxiter': 50,
                                             'maxcor': 20},
                                    callback=None)

            parFinal = res.x

            self.fgcmLog.info("Chromaticity slope in %s filter is %.6f" % (self.fgcmPars.lutFilterNames[filterIndex], parFinal[0]))
            if self.fgcmPars.nCoatingIntervals > 1:
                self.fgcmLog.info("Chromaticity intercepts for %s filter are %s" % (self.fgcmPars.lutFilterNames[filterIndex], np.array2string(parFinal[1: ], separator=', ')))

            # Record the values
            self.fgcmPars.compMirrorChromaticity[filterIndex, :] = parFinal

            if self.plotPath is not None:

                firstMJD = np.floor(np.min(self.fgcmPars.expMJD))

                chisqRaw, derivRaw, expGrayColorSplitRaw, expGrayColorSplitWtRaw = self(parZeros, returnExpGray=True)
                chisqBest, derivBest, expGrayColorSplit, expGrayColorSplitWt = self(parFinal, returnExpGray=True)

                delta = ((expGrayColorSplit[:, 1] - expGrayColorSplit[:, 0]) -
                         (expGrayColorSplitRaw[:, 1] - expGrayColorSplitRaw[:, 0]))

                ok, = np.where((expGrayColorSplitWt[:, 0] > 0.0) &
                               (expGrayColorSplitWt[:, 1] > 0.0))

                if ok.size > 0:
                    deltaColorRaw = (expGrayColorSplitRaw[ok, 0] - expGrayColorSplitRaw[ok, 1]) * 1000.0
                    st = np.argsort(deltaColorRaw)
                    extent = [np.min(self.fgcmPars.expMJD[ok] - firstMJD),
                              np.max(self.fgcmPars.expMJD[ok] - firstMJD),
                              deltaColorRaw[st[int(0.01*deltaColorRaw.size)]],
                              deltaColorRaw[st[int(0.99*deltaColorRaw.size)]]]

                    plt.set_cmap('viridis')

                    fig = plt.figure(1, figsize=(8, 6))
                    fig.clf()

                    ax = fig.add_subplot(111)
                    ax.hexbin(self.fgcmPars.expMJD[ok] - firstMJD,
                              (expGrayColorSplitRaw[ok, 0] - expGrayColorSplitRaw[ok, 1]) * 1000.0,
                              bins='log', extent=extent)

                    ylim = ax.get_ylim()
                    for i in range(self.fgcmPars.nCoatingIntervals):
                        ax.plot([self.fgcmPars.coatingMJDs[i] - firstMJD,
                                 self.fgcmPars.coatingMJDs[i] - firstMJD],
                                ylim, 'r--')
                        u, = np.where(self.fgcmPars.expCoatingIndex[ok] == i)
                        ax.plot(self.fgcmPars.expMJD[ok[u]] - firstMJD, delta[ok[u]] * 1000.0, 'r--')
                    ax.set_xlabel('MJD - %.0f' % (firstMJD))
                    ax.set_ylabel('EXP_GRAY (%s) (red25) - EXP_GRAY (%s) (blue25) (mmag)' %
                                  (self.fgcmPars.lutFilterNames[filterIndex],
                                   self.fgcmPars.lutFilterNames[filterIndex]))

                    text=r'$(%s)$' % (self.fgcmPars.lutFilterNames[filterIndex])
                    ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=16)

                    plt.savefig('%s/%s_compare-redblue-mirrorchrom_%s.png' % (self.plotPath,
                                                                              self.outfileBaseWithCycle,
                                                                              self.fgcmPars.lutFilterNames[filterIndex]))

                    plt.close(fig)

        # Clear things out of memory
        self.blueStarsInFilter = None
        self.redStarsInFilter = None
        self.obsMagStdGO = None
        self.obsExpIndexGO = None
        self.deltaTGO = None
        self.EGrayErr2GO = None
        self.objMagStdMeanGO = None
        self.objSEDSlopeGO = None

        self.fgcmLog.info("Done fitting mirror chromaticity in %.1f s" % (time.time() - startTime))

    def __call__(self, fitPars, returnExpGray=False):
        c1 = fitPars[0]
        c0s = np.array(fitPars[1: ])

        nColorSplits = 2

        expGrayColorSplit = np.zeros((self.fgcmPars.nExp, nColorSplits))
        expGrayColorSplitWt = np.zeros((self.fgcmPars.nExp, nColorSplits))
        dExpGraydC0ColorSplit = np.zeros((self.fgcmPars.nExp, c0s.size, nColorSplits))
        dExpGraydC1ColorSplit = np.zeros((self.fgcmPars.nExp, nColorSplits))

        for k, sel in enumerate([self.redStarsInFilter, self.blueStarsInFilter]):
            cAlGOS = np.clip(c0s[self.fgcmPars.expCoatingIndex[self.obsExpIndexGO[sel]]] +
                             c1 * self.deltaTGO[sel],
                             -1.0, 1.0)

            termOneGOS = 1.0 + (cAlGOS / self.fgcmLUT.lambdaStd[self.lutFilterIndex]) * self.fgcmLUT.I10Std[self.lutFilterIndex]
            termTwoGOS = 1.0 + (((cAlGOS / self.fgcmLUT.lambdaStd[self.lutFilterIndex]) * (self.fgcmLUT.I1Std[self.lutFilterIndex] + self.objSEDSlopeGO[sel] * self.fgcmLUT.I2Std[self.lutFilterIndex])) /
                                (self.fgcmLUT.I0Std[self.lutFilterIndex] + self.objSEDSlopeGO[sel] * self.fgcmLUT.I1Std[self.lutFilterIndex]))

            deltaGOS = -2.5 * np.log10(termOneGOS) + 2.5 * np.log10(termTwoGOS)

            EGrayGOS = self.objMagStdMeanGO[sel] - (self.obsMagStdGO[sel] + deltaGOS)

            np.add.at(expGrayColorSplit[:, k],
                      self.obsExpIndexGO[sel],
                      EGrayGOS / self.EGrayErr2GO[sel])
            np.add.at(expGrayColorSplitWt[:, k],
                      self.obsExpIndexGO[sel],
                      1. / self.EGrayErr2GO[sel])

            ok, = np.where(expGrayColorSplitWt[:, k] > 0.0)
            expGrayColorSplit[ok, k] /= expGrayColorSplitWt[ok, k]

            # Note that this is negative the because the delta is subtracted
            dDeltadC0GOS = (self.magConst * (1. / termOneGOS) * (self.fgcmLUT.I1Std[self.lutFilterIndex] / self.fgcmLUT.lambdaStd[self.lutFilterIndex]) -
                            self.magConst * (1. / termTwoGOS) * ((self.fgcmLUT.I1Std[self.lutFilterIndex] + self.objSEDSlopeGO[sel] * self.fgcmLUT.I2Std[self.lutFilterIndex]) / self.fgcmLUT.lambdaStd[self.lutFilterIndex]) / (self.fgcmLUT.I0Std[self.lutFilterIndex] + self.objSEDSlopeGO[sel] * self.fgcmLUT.I1Std[self.lutFilterIndex]))

            np.add.at(dExpGraydC0ColorSplit[:, :, k],
                      (self.obsExpIndexGO[sel], self.fgcmPars.expCoatingIndex[self.obsExpIndexGO[sel]]),
                      dDeltadC0GOS / self.EGrayErr2GO[sel])

            for i in range(c0s.size):
                dExpGraydC0ColorSplit[ok, i, k] /= expGrayColorSplitWt[ok, k]

            np.add.at(dExpGraydC1ColorSplit[:, k],
                      self.obsExpIndexGO[sel],
                      dDeltadC0GOS * self.deltaTGO[sel] / self.EGrayErr2GO[sel])

            dExpGraydC1ColorSplit[ok, k] /= expGrayColorSplitWt[ok, k]

        ok, = np.where((expGrayColorSplitWt[:, 0] > 0.0) &
                       (expGrayColorSplitWt[:, 1] > 0.0))

        chisq = np.sum((expGrayColorSplit[ok, 0] - expGrayColorSplit[ok, 1])**2. * (expGrayColorSplitWt[ok, 0] + expGrayColorSplitWt[ok, 1])) / (ok.size - len(fitPars))

        deriv = np.zeros(len(fitPars))
        # The c1 derivative
        deriv[0] = np.sum((2.0 * (expGrayColorSplitWt[ok, 0] + expGrayColorSplitWt[ok, 1]) *
                           (expGrayColorSplit[ok, 0] - expGrayColorSplit[ok, 1]) *
                           (dExpGraydC1ColorSplit[ok, 0] - dExpGraydC1ColorSplit[ok, 1])))
        np.add.at(deriv[1: ],
                  self.fgcmPars.expCoatingIndex[ok],
                  (2.0 * (expGrayColorSplitWt[ok, 0] + expGrayColorSplitWt[ok, 1]) *
                   (expGrayColorSplit[ok, 0] - expGrayColorSplit[ok, 1]) *
                   (dExpGraydC0ColorSplit[ok, self.fgcmPars.expCoatingIndex[ok], 0] -
                    dExpGraydC0ColorSplit[ok, self.fgcmPars.expCoatingIndex[ok], 1])))

        deriv /= (ok.size - len(fitPars))

        if returnExpGray:
            return chisq, deriv, expGrayColorSplit, expGrayColorSplitWt
        else:
            return chisq, deriv
