import numpy as np
import os
import sys
import esutil
import scipy.optimize

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm
from .fgcmUtilities import makeFigure, putButlerFigure
from matplotlib import colormaps


class MagErrorModelFitter(object):
    """
    """
    def __init__(self, mag, magErr, fwhm, sky, fwhmPivot, skyPivot):
        self.mag = mag.astype(np.float64)
        self.logErr = np.log10(magErr.astype(np.float64))
        self.mag2 = self.mag**2.
        self.fwhmPivot = fwhmPivot
        self.logFwhm = np.log10(fwhm.astype(np.float64) / self.fwhmPivot)
        self.skyPivot = skyPivot
        self.logSky = np.log10(sky.astype(np.float64) / self.skyPivot)

    def __call__(self, pars):
        yMod = (pars[0] + pars[1] * self.mag + pars[2] * self.mag2 + pars[3] * self.logFwhm +
                pars[4] * self.logSky + pars[5] * self.mag * self.logFwhm +
                pars[6] * self.mag * self.logSky)

        return np.sum(np.abs(yMod - self.logErr))


class FgcmModelMagErrors(object):
    """
    Class which models the magnitude errors.
    """

    def __init__(self, fgcmConfig, fgcmPars, fgcmStars, butlerQC=None, plotHandleDict=None):
        self.fgcmLog = fgcmConfig.fgcmLog
        self.fgcmLog.debug('Initializing FgcmModelMagErrors')

        self.fgcmPars = fgcmPars
        self.fgcmStars = fgcmStars

        self.butlerQC = butlerQC
        self.plotHandleDict = plotHandleDict

        self.sigma0Phot = fgcmConfig.sigma0Phot
        self.minObsPerBand = fgcmConfig.minObsPerBand
        self.modelMagErrorNObs = fgcmConfig.modelMagErrorNObs
        self.modelMagErrors = fgcmConfig.modelMagErrors
        self.illegalValue = fgcmConfig.illegalValue
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber
        self.quietMode = fgcmConfig.quietMode
        self.rng = fgcmConfig.rng

    def computeMagErrorModel(self, fitName, doPlots=False):
        """
        Compute magnitude error model

        parameters
        ----------
        fitName : `str`
            Name of the fit to put in plot labeling
        doPlots : `bool`, optional
        """

        if not self.modelMagErrors:
            if not self.quietMode:
                self.fgcmLog.info('No magnitude error model will be computed')
            return

        if not self.quietMode:
            self.fgcmLog.info('Computing magnitude error model parameters...')

        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        obsMagADUErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)

        obsExptime = self.fgcmPars.expExptime[obsExpIndex]
        obsFwhm = self.fgcmPars.expFwhm[obsExpIndex]
        obsSkyBrightness = self.fgcmPars.expSkyBrightness[obsExpIndex]

        goodStars = self.fgcmStars.getGoodStarIndices(checkMinObs=True)
        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=self.fgcmPars.expFlag, checkBadMag=True)

        # And loop over bands
        for bandIndex in range(self.fgcmPars.nBands):
            if not self.fgcmPars.hasExposuresInBand[bandIndex]:
                continue
            use0, = np.where((obsBandIndex[goodObs] == bandIndex) &
                             (objNGoodObs[obsObjIDIndex[goodObs], bandIndex] >= self.minObsPerBand))
            # Sample down to the number of observations in config (for speed)
            if (use0.size > self.modelMagErrorNObs):
                use = self.rng.choice(use0, replace=False, size=self.modelMagErrorNObs)
            else:
                use = use0

            if use.size < 5000:
                # This is arbitrary, but necessary.
                self.fgcmLog.info('Not enough star observations to model errors in %s band' % (self.fgcmPars.bands[bandIndex]))
                continue

            # Compute medians exposure time for scaling
            medExptime = np.median(obsExptime[goodObs[use]])

            # Compute quantities that will be used
            obsMagADUGOu = obsMagADU[goodObs[use]] - 2.5 * np.log10(obsExptime[goodObs[use]] / medExptime)
            obsMagADUErrGOu = np.sqrt(obsMagADUErr[goodObs[use]]**2. - self.sigma0Phot**2.)
            obsDeltaStdGOu = obsMagStd[goodObs[use]] - obsMagADU[goodObs[use]]
            obsMagADUMeanGOu = (objMagStdMean[obsObjIDIndex[goodObs[use]], bandIndex] -
                                obsDeltaStdGOu - 2.5 * np.log10(obsExptime[goodObs[use]] / medExptime))
            obsFwhmGOu = obsFwhm[goodObs[use]]
            # We also need to scale the sky brightness to normalize it
            obsSkyBrightnessGOu = obsSkyBrightness[goodObs[use]] * (obsExptime[goodObs[use]] / medExptime)

            # And medians for pivot
            medSkyBrightness = np.median(obsSkyBrightnessGOu)
            medFwhm = np.median(obsFwhmGOu)

            # Fit model is:
            # log10(err) = a + b * MAG + c * MAG**2. +
            #              d * log10(FWHM / <FWHM>) +
            #              e * log10(sky / <sky>) +
            #              f * MAG * log10(FWHM / <FWHM>) +
            #              g * MAG * log10(sky / <sky>)

            # The fit is happier with better starting values, so we break
            # things apart with simple fits
            # The assumption is that higher seeing and higher sky only make the
            # depth worse, so we start by fitting those observations that have
            # less than the median

            okFwhm = (obsFwhmGOu < medFwhm)
            okSky = (obsSkyBrightnessGOu < medSkyBrightness)

            # Start with the quadratic model
            ok = okFwhm & okSky

            if okFwhm.sum() < 1000 or okSky.sum() < 1000 or ok.sum() < 1000:
                self.fgcmLog.info('Not enough quality star observations to model errors in %s band' % (self.fgcmPars.bands[bandIndex]))
                continue

            quadFit = np.polyfit(obsMagADUMeanGOu[ok].astype(np.float64),
                                 np.log10(obsMagADUErrGOu[ok].astype(np.float64)), 2)

            quadModel = (quadFit[0] * obsMagADUMeanGOu**2. +
                         quadFit[1] * obsMagADUMeanGOu +
                         quadFit[2])

            # And the dependence on sky brightness -- selected on less than median fwhm
            skyFit = np.polyfit(np.log10(obsSkyBrightnessGOu[okFwhm].astype(np.float64) / medSkyBrightness),
                                np.log10(obsMagADUErrGOu[okFwhm]) - quadModel[okFwhm], 1)

            # And the dependence on fwhm -- selected on less than median sky
            fwhmFit = np.polyfit(np.log10(obsFwhmGOu[okSky].astype(np.float64) / medFwhm),
                                 np.log10(obsMagADUErrGOu[okSky]) - quadModel[okSky], 1)

            fitFn = MagErrorModelFitter(obsMagADUMeanGOu,
                                        obsMagADUErrGOu,
                                        obsFwhmGOu,
                                        obsSkyBrightnessGOu,
                                        medFwhm,
                                        medSkyBrightness)

            p0 = np.array([quadFit[2], quadFit[1], quadFit[0], fwhmFit[0], skyFit[0], 0.0, 0.0])

            # Use nelder-mead simplex
            pars = scipy.optimize.fmin(fitFn, p0, maxiter=5000, disp=False)
            # And it looks like it can use another run
            pars = scipy.optimize.fmin(fitFn, pars, maxiter=5000, disp=False)

            # And store the values
            self.fgcmPars.compModelErrExptimePivot[bandIndex] = medExptime
            self.fgcmPars.compModelErrFwhmPivot[bandIndex] = medFwhm
            self.fgcmPars.compModelErrSkyPivot[bandIndex] = medSkyBrightness
            self.fgcmPars.compModelErrPars[:, bandIndex] = pars

            # And also plots (if necessary)
            if doPlots:
                ymod = (pars[0] + pars[1] * obsMagADUMeanGOu + pars[2] * obsMagADUMeanGOu**2. +
                        pars[3] * np.log10(obsFwhmGOu / medFwhm) +
                        pars[4] * np.log10(obsSkyBrightnessGOu / medSkyBrightness) +
                        pars[5] * obsMagADUMeanGOu * np.log10(obsFwhmGOu / medFwhm) +
                        pars[6] * obsMagADUMeanGOu * np.log10(obsSkyBrightnessGOu / medSkyBrightness))

                # This is going to be a 3x2 figure
                fig = makeFigure(figsize=(10, 10))
                fig.clf()

                # First row: observed error vs magADU and model error vs mean mag
                extent = (np.min(obsMagADUMeanGOu), np.max(obsMagADUMeanGOu),
                          np.min(ymod), np.max(ymod))

                ax = fig.add_subplot(321)
                ax.hexbin(obsMagADUGOu, np.log10(obsMagADUErrGOu), bins='log', extent=extent, cmap=colormaps.get_cmap("viridis"))
                ax.set_xlabel(r'Obs mag')
                ax.set_ylabel(r'log(ObsErr)')

                ax = fig.add_subplot(322)
                ax.hexbin(obsMagADUMeanGOu, ymod, bins='log', extent=extent, cmap=colormaps.get_cmap("viridis"))
                ax.set_xlabel(r'Mean Mag (Decorr.)')
                ax.set_ylabel(r'log1(ModErr)')

                # Second row: Error vs seeing (slice)
                st = np.argsort(obsMagADUMeanGOu)
                lo = obsMagADUMeanGOu[st[int(0.02*st.size)]]
                hi = obsMagADUMeanGOu[st[int(0.98*st.size)]]
                mid = (lo + hi) / 2. + 0.5
                slit = np.where((obsMagADUMeanGOu > (mid - 0.2)) &
                                (obsMagADUMeanGOu < (mid + 0.2)))

                if slit[0].size == 0:
                    continue

                extent = (np.min(np.log10(obsFwhmGOu[slit])), np.max(np.log10(obsFwhmGOu[slit])),
                          np.min(ymod[slit]), np.max(ymod[slit]))

                ax = fig.add_subplot(323)
                ax.hexbin(np.log10(obsFwhmGOu[slit]), np.log10(obsMagADUErrGOu[slit]),
                          bins='log', extent=extent, cmap=colormaps.get_cmap("viridis"))
                ax.set_xlabel(r'log(FWHM)')
                ax.set_ylabel(r'log(ObsErr)')
                ax.set_title(r'Mag ~ %.1f' % (mid), fontsize=10)

                ax = fig.add_subplot(324)
                ax.hexbin(np.log10(obsFwhmGOu[slit]), ymod[slit], bins='log', extent=extent, cmap=colormaps.get_cmap("viridis"))
                ax.set_xlabel(r'log(FWHM)')
                ax.set_ylabel(r'log(ModErr)')
                ax.set_title(r'Mag ~ %.1f' % (mid), fontsize=10)

                # Third row: Error vs sky brightness (slice)
                extent = (np.min(np.log10(obsSkyBrightnessGOu[slit])),
                          np.max(np.log10(obsSkyBrightnessGOu[slit])),
                          np.min(ymod[slit]), np.max(ymod[slit]))

                ax = fig.add_subplot(325)
                ax.hexbin(np.log10(obsSkyBrightnessGOu[slit]), np.log10(obsMagADUErrGOu[slit]),
                          bins='log', extent=extent, cmap=colormaps.get_cmap("viridis"))
                ax.set_xlabel(r'log(Sky)')
                ax.set_ylabel(r'log(ObsErr)')
                ax.set_title(r'Mag ~ %.1f' % (mid), fontsize=10)

                ax = fig.add_subplot(326)
                ax.hexbin(np.log10(obsSkyBrightnessGOu[slit]), ymod[slit], bins='log', extent=extent)
                ax.set_xlabel(r'log(Sky)')
                ax.set_ylabel(r'log(ModErr)')
                ax.set_title(r'Mag ~ %.1f' % (mid), fontsize=10)

                # And finish up and save
                fig.suptitle('%s: %s band' % (fitName, self.fgcmPars.bands[bandIndex]))
                fig.tight_layout()

                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    f"ModelMagerr{fitName.title()}",
                                    self.cycleNumber,
                                    fig,
                                    band=self.fgcmPars.bands[bandIndex])
                elif self.plotPath is not None:
                    fig.savefig('%s/%s_%s_modelmagerr_%s.png' % (self.plotPath,
                                                                 self.outfileBaseWithCycle,
                                                                 fitName,
                                                                 self.fgcmPars.bands[bandIndex]))
