import numpy as np
import os
import sys
import esutil
import time

from .fgcmUtilities import retrievalFlagDict

import multiprocessing

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm


class FgcmSigmaCal(object):
    """
    Class which calibrates the calibration error floor.

    parameters
    ----------
    fgcmConfig: FgcmConfig
       Config object
    fgcmPars: FgcmParameters
       Parameter object
    fgcmStars: FgcmStars
       Stars object
    """

    def __init__(self, fgcmConfig, fgcmPars, fgcmStars, fgcmGray):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.debug('Initializing FgcmSigmaCal')

        self.fgcmPars = fgcmPars
        self.fgcmStars = fgcmStars
        self.fgcmGray = fgcmGray

        self.nCore = fgcmConfig.nCore
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.nStarPerRun = fgcmConfig.nStarPerRun
        self.sigma0Phot = fgcmConfig.sigma0Phot
        self.sigmaCalRange = fgcmConfig.sigmaCalRange
        self.sigmaCalFitPercentile = fgcmConfig.sigmaCalFitPercentile
        self.sigmaCalPlotPercentile = fgcmConfig.sigmaCalPlotPercentile
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.quietMode = fgcmConfig.quietMode

        # these are the standard *band* I10s
        self.I10StdBand = fgcmConfig.I10StdBand

        self.illegalValue = fgcmConfig.illegalValue

        self._prepareSigmaCalArrays()

    def _prepareSigmaCalArrays(self):
        """
        """

        self.objChi2Handle = snmm.createArray((self.fgcmStars.nStars, self.fgcmPars.nBands), dtype='f8')

    def run(self, applyGray=True):
        """
        Run the sigma_cal computation code.
        """

        self.applyGray = applyGray

        # Select only reserve stars for the good stars...

        goodStars = self.fgcmStars.getGoodStarIndices(onlyReserve=True)

        if not self.quietMode:
            self.fgcmLog.info('Found %d good reserve stars for SigmaCal' % (goodStars.size))

        if goodStars.size == 0:
            raise ValueError("No good reserve stars to fit!")

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        preStartTime=time.time()
        self.fgcmLog.debug('Pre-matching stars and observations...')

        expFlag = self.fgcmPars.expFlag
        goodStarsSub, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=expFlag)

        self.fgcmLog.debug('Pre-matching done in %.1f sec.' %
                           (time.time() - preStartTime))

        nSections = goodStars.size // self.nStarPerRun + 1
        goodStarsList = np.array_split(goodStars, nSections)

        splitValues = np.zeros(nSections-1,dtype='i4')
        for i in range(1, nSections):
            splitValues[i-1] = goodStarsList[i][0]

        # get the indices from the goodStarsSub matched list (matched to goodStars)
        splitIndices = np.searchsorted(goodStars[goodStarsSub], splitValues)

        # and split along the indices
        goodObsList = np.split(goodObs, splitIndices)

        workerList = list(zip(goodStarsList, goodObsList))

        # reverse sort so the longest running go first
        workerList.sort(key=lambda elt:elt[1].size, reverse=True)

        if not self.quietMode:
            self.fgcmLog.info('Running SigmaCal on %d cores' % (self.nCore))

        # Plan:
        # Do 50 steps in the range; if the range is 0, then just set that.
        # Compute all the chi2 for each sigmaCal in the range
        # Take the percentile range of stars by magnitudes
        # Take the one that is closest to zero.
        # Plot all 50, with a color gradient.
        # Overplot in black the one that has the best fit.
        # Label this, put a color bar, etc.
        # One plot/fit per band

        objChi2 = snmm.getArray(self.objChi2Handle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)

        if self.sigmaCalRange[0] == self.sigmaCalRange[1]:
            nStep = 1
        else:
            nStep = 50
        nPlotBin = 10

        sigmaCals = np.linspace(self.sigmaCalRange[0], self.sigmaCalRange[1], nStep)

        if self.plotPath is not None:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import matplotlib.cm as cmx
            use_inset = False
            try:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                use_inset = True
            except ImportError:
                pass

            cm = plt.get_cmap('rainbow')
            plt.set_cmap('rainbow')

            Z = [[0, 0], [0, 0]]
            if self.sigmaCalRange[0] == self.sigmaCalRange[1]:
                useRange = [self.sigmaCalRange[0] - 1e-5, self.sigmaCalRange[1] + 1e-5]
            else:
                useRange = self.sigmaCalRange
            levels = np.linspace(useRange[0], useRange[1], 256)
            CS3 = plt.contourf(Z, levels, cmap=cm)

            cNorm = colors.Normalize(vmin=self.sigmaCalRange[0], vmax=self.sigmaCalRange[1])
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

        medChi2s = np.ones((nStep, self.fgcmPars.nBands))

        # Get the indices to use for the fit for each band
        indices = {}
        plotIndices = {}
        for bandIndex, band in enumerate(self.fgcmPars.bands):
            if not self.fgcmPars.hasExposuresInBand[bandIndex]:
                continue

            ok, = np.where((objMagStdMean[goodStars, bandIndex] < 90.0) &
                           (objMagStdMean[goodStars, bandIndex] != 0.0) &
                           (objNGoodObs[goodStars, bandIndex] > 2))
            st = np.argsort(objMagStdMean[goodStars[ok], bandIndex])
            indices[band] = ok[st[int(self.sigmaCalFitPercentile[0] * st.size):
                                      int(self.sigmaCalFitPercentile[1] * st.size)]]
            plotIndices[band] = ok[st[int(self.sigmaCalPlotPercentile[0] * st.size):
                                          int(self.sigmaCalPlotPercentile[1] * st.size)]]

        if self.plotPath is not None:
            plotMags = np.zeros((sigmaCals.size, self.fgcmPars.nBands, nPlotBin))
            plotChi2s = np.zeros_like(plotMags)

        # And do all the sigmaCals:
        for i, s in enumerate(sigmaCals):
            self.sigmaCal = s

            mp_ctx = multiprocessing.get_context("fork")
            pool = mp_ctx.Pool(processes=self.nCore)
            pool.map(self._worker, workerList, chunksize=1)
            pool.close()
            pool.join()

            for bandIndex, band in enumerate(self.fgcmPars.bands):
                if not self.fgcmPars.hasExposuresInBand[bandIndex]:
                    continue

                ok, = np.where((objChi2[goodStars[indices[band]], bandIndex] > 0.001) &
                               (objChi2[goodStars[indices[band]], bandIndex] < 1000.0))
                if ok.size > 0:
                    medChi2s[i, bandIndex] = np.median(objChi2[goodStars[indices[band][ok]], bandIndex])

            if self.plotPath is not None:
                for bandIndex, band in enumerate(self.fgcmPars.bands):
                    if not self.fgcmPars.hasExposuresInBand[bandIndex]:
                        continue

                    ok, = np.where((objChi2[goodStars[plotIndices[band]], bandIndex] > 0.001) &
                                   (objChi2[goodStars[plotIndices[band]], bandIndex] < 1000.0))

                    if ok.size < 100:
                        self.fgcmLog.warn('Not enough stars with decent chi2 to compute sigmaCal for band %s' % (band))
                        continue

                    # These have already been limited to the plot percentile range
                    h, rev = esutil.stat.histogram(objMagStdMean[goodStars[plotIndices[band][ok]], bandIndex],
                                                   nbin=nPlotBin, rev=True)
                    for j, nInBin in enumerate(h):
                        if nInBin < 100:
                            continue
                        i1a = rev[rev[j]: rev[j + 1]]
                        plotMags[i, bandIndex, j] = np.median(objMagStdMean[goodStars[plotIndices[band][ok[i1a]]], bandIndex])
                        plotChi2s[i, bandIndex, j] = np.median(objChi2[goodStars[plotIndices[band][ok[i1a]]], bandIndex])

        # And get the minima...
        mininds = np.zeros(self.fgcmPars.nBands, dtype=np.int32)
        for bandIndex, band in enumerate(self.fgcmPars.bands):
            if not self.fgcmPars.hasExposuresInBand[bandIndex]:
                continue

            mininds[bandIndex] = np.argmin(np.abs(np.log10(medChi2s[:, bandIndex])))
            self.fgcmPars.compSigmaCal[bandIndex] = sigmaCals[mininds[bandIndex]]
            self.fgcmLog.info('Best sigmaCal (%s band) = %.2f mmag' % (band, sigmaCals[mininds[bandIndex]]*1000.0))

        # And do the plots if desired
        if self.plotPath is not None:
            for bandIndex, band in enumerate(self.fgcmPars.bands):
                if not self.fgcmPars.hasExposuresInBand[bandIndex]:
                    continue

                fig = plt.figure(figsize=(9, 6))
                fig.clf()

                ax = fig.add_subplot(111)

                # Only plot those that are > 0!

                # for each sigmaCal, plot with a new color...
                for i, s in enumerate(sigmaCals):
                    oktoplot, = np.where(plotChi2s[i, bandIndex, :] > 0)
                    ax.plot(plotMags[i, bandIndex, oktoplot],
                            np.log10(plotChi2s[i, bandIndex, oktoplot]),
                            '-', color=scalarMap.to_rgba(s))

                # And the best one
                oktoplot, = np.where(plotChi2s[mininds[bandIndex], bandIndex, :] > 0)
                ax.plot(plotMags[mininds[bandIndex], bandIndex, oktoplot],
                        np.log10(plotChi2s[mininds[bandIndex], bandIndex, oktoplot]), 'k-')

                # and a reference line
                ax.plot([plotMags[i, bandIndex, 0], plotMags[i, bandIndex, -1]], [0, 0], 'k--')

                ax.set_xlabel('Magnitude (%s band)' % (band), fontsize=14)
                ax.set_ylabel('log10(chi2)', fontsize=14)
                ax.set_title('%s band, sigma_cal = %.2f' % (band, self.fgcmPars.compSigmaCal[bandIndex]*1000.0))
                ax.tick_params(axis='both', which='major', labelsize=14)

                if use_inset:
                    axins=inset_axes(ax, width='45%',height='5%',loc=4)
                    plt.colorbar(CS3,cax=axins,orientation='horizontal',format='%.3f',
                                 ticks=[self.sigmaCalRange[0], (self.sigmaCalRange[0] + self.sigmaCalRange[1]) / 2.,
                                        self.sigmaCalRange[1]])
                    axins.xaxis.set_ticks_position('top')
                    axins.tick_params(axis='both',which='major',labelsize=12)

                fig.savefig('%s/%s_sigmacal_%s.png' % (self.plotPath,
                                                       self.outfileBaseWithCycle,
                                                       band))

                plt.close()

    def _worker(self, goodStarsAndObs):
        """
        """

        workerStartTime = time.time()

        goodStars = goodStarsAndObs[0]
        goodObs = goodStarsAndObs[1]

        # We need to make sure we don't overwrite anything we care about!!!!
        # This will be a challenge to keep the memory okay...

        # We already have the mean...
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objChi2 = snmm.getArray(self.objChi2Handle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)

        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsLUTFilterIndex = snmm.getArray(self.fgcmStars.obsLUTFilterIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)
        obsSecZenith = snmm.getArray(self.fgcmStars.obsSecZenithHandle)
        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsMagADUErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        # We already have obsMagStd
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)

        # We apply the gray corrections here!
        # Note that obsMagStd does not have the gray corrections applied
        ccdGray = snmm.getArray(self.fgcmGray.ccdGrayHandle)
        ccdGrayErr = snmm.getArray(self.fgcmGray.ccdGrayErrHandle)
        ccdNGoodTilings = snmm.getArray(self.fgcmGray.ccdNGoodTilingsHandle)

        # Cut down goodObs to valid values
        gd, = np.where((ccdGray[obsExpIndex[goodObs], obsCCDIndex[goodObs]] > self.illegalValue) &
                       (ccdNGoodTilings[obsExpIndex[goodObs], obsCCDIndex[goodObs]] >= 2.0) &
                       (objNGoodObs[obsObjIDIndex[goodObs], obsBandIndex[goodObs]] >= 2) &
                       (objMagStdMean[obsObjIDIndex[goodObs], obsBandIndex[goodObs]] < 99.0))

        goodObs = goodObs[gd]

        # cut these down now, faster later
        obsObjIDIndexGO = obsObjIDIndex[goodObs]
        obsBandIndexGO = obsBandIndex[goodObs]
        obsLUTFilterIndexGO = obsLUTFilterIndex[goodObs]
        obsExpIndexGO = obsExpIndex[goodObs]
        obsSecZenithGO = obsSecZenith[goodObs]
        obsCCDIndexGO = obsCCDIndex[goodObs]

        # We make a sub-copy here that we can overwrite
        obsMagStdGO = obsMagStd[goodObs]

        if self.applyGray:
            # NOTE: This only applies the mean gray per ccd for speed
            # (since this is approximate anyway)
            obsMagStdGO += ccdGray[obsExpIndexGO, obsCCDIndexGO]

        # chi2 = 1. / (N - 1) * Sum ((m_i - mbar)**2. / (sigma_i**2.))
        # N is the number of good observations of the star objNGoodObs[obsObjIDIndexGO, obsBandIndexGO]
        # m_i is obsMagStdGO (after adjustment by ccdGray)
        # mbar is objMagStdMean[obsObjIDIndexGO, obsBandIndexGO]
        # sigma_i**2 = sigma_obs**2. + sig2fgcm / (ntile - 1) + zptvar + sigma_cal**2.
        # It needs to be computed, it is based one
        # - obsMagADUModelErr (this is sigma_obs with an additional sigma0Phot)
        # - self.sigma0Phot
        # - sig2Fgcm (self.fgcmPars.compSigFgcm[self.fgcmPars.expBandIndex[obsExpIndexGO]])
        # - Ntile (ccdNGoodTilings[obsExpIndexGO, obsCCDIndexGO])
        # - zptvar (ccdGrayErr[obsExpIndexGO, obsCCDIndexGO]**2.)
        # - sigma_cal (self.sigmaCal)

        # And recompute the errors...
        nTilingsM1 = np.clip(ccdNGoodTilings[obsExpIndexGO, obsCCDIndexGO] - 1.0, 1.0, None)

        obsMagErr2GO = ((obsMagADUModelErr[goodObs]**2. - self.sigma0Phot**2.) +
                        (self.fgcmPars.compSigFgcm[self.fgcmPars.expBandIndex[obsExpIndexGO]]**2. / nTilingsM1) +
                        (ccdGrayErr[obsExpIndexGO, obsCCDIndexGO]**2.) +
                        (self.sigmaCal**2.))

        # Now we need the per-object chi2...

        objChi2[:, :] = 0.0

        np.add.at(objChi2,
                  (obsObjIDIndexGO, obsBandIndexGO),
                  ((obsMagStdGO - objMagStdMean[obsObjIDIndexGO, obsBandIndexGO])**2. /
                   obsMagErr2GO))
        # There are duplicate indices here, but that's fine because we only want to divide once
        objChi2[obsObjIDIndexGO, obsBandIndexGO] /= (objNGoodObs[obsObjIDIndexGO, obsBandIndexGO] - 1.0)

        # And we're done

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state


