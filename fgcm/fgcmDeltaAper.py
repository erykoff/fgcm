from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import healpy as hp
import os
import sys
import esutil
import time

import matplotlib.pyplot as plt

from .fgcmUtilities import dataBinner

import multiprocessing
from multiprocessing import Pool

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm


class FgcmDeltaAper(object):
    """
    Class which computes delta aperture background offsets.

    Parameters
    ----------
    fgcmConfig: FgcmConfig
       Config object
    fgcmPars: FgcmParameters
       Parameter object
    fgcmStars: FgcmStars
       Star object
    """
    def __init__(self, fgcmConfig, fgcmPars, fgcmStars):
        self.fgcmLog = fgcmConfig.fgcmLog

        if not fgcmStars.hasDeltaAper:
            self.fgcmLog.info("Cannot compute delta aperture parameters without measurements.")
            return

        self.fgcmPars = fgcmPars
        self.fgcmStars = fgcmStars

        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle

        self.deltaMapper = fgcmConfig.focalPlaneProjector(int(fgcmConfig.defaultCameraOrientation))
        self.minStarPerExp = fgcmConfig.minStarPerExp
        self.illegalValue = fgcmConfig.illegalValue
        self.quietMode = fgcmConfig.quietMode
        self.nCore = fgcmConfig.nCore
        self.nStarPerRun = fgcmConfig.nStarPerRun
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.deltaAperFitPerCcdNx = fgcmConfig.deltaAperFitPerCcdNx
        self.deltaAperFitPerCcdNy = fgcmConfig.deltaAperFitPerCcdNy
        self.deltaAperFitSpatialNside = fgcmConfig.deltaAperFitSpatialNside
        self.deltaAperFitSpatialMinStar = fgcmConfig.deltaAperFitSpatialMinStar
        self.deltaAperInnerRadiusArcsec = fgcmConfig.deltaAperInnerRadiusArcsec
        self.deltaAperOuterRadiusArcsec = fgcmConfig.deltaAperOuterRadiusArcsec
        self.deltaAperFitMinNgoodObs = fgcmConfig.deltaAperFitMinNgoodObs

        self.epsilonNormalized = True
        if self.deltaAperOuterRadiusArcsec == 0.0 and self.deltaAperInnerRadiusArcsec == 0.0:
            self.fgcmLog.warn('No aperture radii set.  Epsilon is unnormalized.')
            self.epsilonNormalized = False

        self.njyZp = 48.6 - 9*2.5
        self.k = 2.5/np.log(10.)

    def computeDeltaAperExposures(self):
        """
        Compute deltaAper per-exposure quantities
        """
        if not self.quietMode:
            self.fgcmLog.info('Computing deltaAper per exposure')

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsDeltaAper = snmm.getArray(self.fgcmStars.obsDeltaAperHandle)

        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)

        # Use only good observations of good stars
        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=False, checkMinObs=True)
        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars)

        self.fgcmPars.compMedDeltaAper[:] = self.illegalValue
        self.fgcmPars.compEpsilon[:] = self.illegalValue

        h, rev = esutil.stat.histogram(obsExpIndex[goodObs], rev=True, min=0)
        expIndices, = np.where(h >= self.minStarPerExp)

        for expIndex in expIndices:
            i1a = rev[rev[expIndex]: rev[expIndex + 1]]
            mag = objMagStdMean[obsObjIDIndex[goodObs[i1a]],
                                obsBandIndex[goodObs[i1a]]]

            deltaAper = obsDeltaAper[goodObs[i1a]]
            err = obsMagADUModelErr[goodObs[i1a]]

            # First, we take the brightest half and compute the median
            ok, = np.where((mag < 90.0) & (np.abs(deltaAper) < 0.5))
            if ok.size < (self.minStarPerExp // 2):
                continue

            # Use 25% brightest
            st = np.argsort(mag[ok])
            cutMag = mag[ok[st[int(0.25*st.size)]]]
            bright, = np.where(mag[ok] < cutMag)
            self.fgcmPars.compMedDeltaAper[expIndex] = np.median(deltaAper[ok[bright]])

            # Next, we take the full thing and compute epsilon (nJy)
            flux = 10.**((mag[ok] - self.njyZp)/(-2.5))
            x = (2.5/np.log(10.)) / flux
            y = deltaAper[ok]
            yerr = err[ok]

            # Will need to check for warnings here...
            fit, nStar = self._fitEpsilonWithOutlierRejection(x, y, yerr)

            self.fgcmPars.compEpsilon[expIndex] = self._normalizeEpsilon(fit)

    def computeDeltaAperStars(self, debug=False):
        """
        Compute deltaAper per-star quantities
        """
        self.debug = debug

        startTime=time.time()
        if not self.quietMode:
            self.fgcmLog.info('Compute per-star deltaAper')

        # Reset numbers
        snmm.getArray(self.fgcmStars.objDeltaAperMeanHandle)[:] = 99.0

        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=True)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        goodStarsSub, goodObs = self.fgcmStars.getGoodObsIndices(goodStars)

        if self.debug:
            self._starWorker((goodStars, goodObs))
        else:
            if not self.quietMode:
                self.fgcmLog.info('Running DeltaAper on %d cores' % (self.nCore))

            nSections = goodStars.size // self.nStarPerRun + 1
            goodStarsList = np.array_split(goodStars, nSections)

            splitValues = np.zeros(nSections - 1,dtype='i4')
            for i in xrange(1, nSections):
                splitValues[i - 1] = goodStarsList[i][0]

            splitIndices = np.searchsorted(goodStars[goodStarsSub], splitValues)
            goodObsList = np.split(goodObs, splitIndices)

            workerList = list(zip(goodStarsList,goodObsList))

            # reverse sort so the longest running go first
            workerList.sort(key=lambda elt:elt[1].size, reverse=True)

            pool = Pool(processes=self.nCore)
            pool.map(self._starWorker, workerList, chunksize=1)
            pool.close()
            pool.join()

        if not self.quietMode:
            self.fgcmLog.info('Finished DeltaAper in %.2f seconds.' %
                              (time.time() - startTime))

    def computeEpsilonMap(self):
        """
        Compute global epsilon and local maps.
        """
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)
        objRA = snmm.getArray(self.fgcmStars.objRAHandle)
        objDec = snmm.getArray(self.fgcmStars.objDecHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objDeltaAperMean = snmm.getArray(self.fgcmStars.objDeltaAperMeanHandle)

        globalEpsilon = np.zeros(self.fgcmStars.nBands) + self.illegalValue
        globalOffset = np.zeros(self.fgcmPars.nBands) + self.illegalValue

        # We start with the global fits
        for i, band in enumerate(self.fgcmStars.bands):
            use, = np.where((objNGoodObs[:, i] >= self.deltaAperFitMinNgoodObs) &
                            (np.abs(objDeltaAperMean[:, i]) < 0.5) &
                            (objFlag == 0))
            if use.size == 0:
                continue

            mag_std = objMagStdMean[use, i]
            delta = objDeltaAperMean[use, i]

            # Sample for efficiency if necessary
            nsample = 1000000
            if mag_std.size > nsample:
                r = np.random.choice(mag_std.size, size=nsample, replace=False)
            else:
                r = np.arange(mag_std.size)

            st = np.argsort(mag_std[r])
            mag_min = mag_std[r[st[int(0.01*st.size)]]]
            mag_max = mag_std[r[st[int(0.95*st.size)]]]
            st = np.argsort(delta[r])
            delta_min = delta[r[st[int(0.02*st.size)]]]
            delta_max = delta[r[st[int(0.98*st.size)]]]

            bin_struct = dataBinner(mag_std[r], delta[r], 0.2, [mag_min, mag_max])
            u, = np.where(bin_struct['Y_ERR'] > 0.0)

            # Do nJy
            bin_flux = 10.**((bin_struct['X'] - self.njyZp)/(-2.5))
            fit = np.polyfit((2.5/np.log(10.0))/bin_flux[u],
                             bin_struct['Y'][u],
                             1, w=1./bin_struct['Y_ERR'][u])
            globalEpsilon[i] = self._normalizeEpsilon(fit)
            globalOffset[i] = fit[1]

            # Store the value of njy_per_arcsec2
            self.fgcmPars.compGlobalEpsilon[i] = globalEpsilon[i]
            self.fgcmLog.info('Global background offset in %s band: %.5f nJy/arcsec2' %
                              (band, globalEpsilon[i]))

            if self.plotPath is not None:
                # Do plots
                fig = plt.figure(1, figsize=(8, 6))
                fig.clf()
                ax = fig.add_subplot(111)
                ax.hexbin(mag_std, delta, extent=[mag_min, mag_max,
                                                  delta_min, delta_max], bins='log')
                ax.errorbar(bin_struct['X'][u], bin_struct['Y'][u],
                            yerr=bin_struct['Y_ERR'][u], fmt='r.', markersize=10)
                xplotvals = np.linspace(mag_min, mag_max, 100)
                xplotfluxvals = 10.**((xplotvals - self.njyZp)/(-2.5))
                yplotvals = fit[0]*((2.5/np.log(10.0))/xplotfluxvals) + fit[1]
                ax.plot(xplotvals, yplotvals, 'r-')
                ax.set_xlabel('mag_std_%s' % (band))
                ax.set_ylabel('delta_aper_%s' % (band))
                ax.set_title('%s: %.4f nJy/arcsec2' % (band, globalEpsilon[i]))
                fig.savefig('%s/%s_epsilon_global_%s.png' % (self.plotPath,
                                                             self.outfileBaseWithCycle,
                                                             band))
                plt.close(fig)

        # And then do the mapping
        ipring = hp.ang2pix(self.deltaAperFitSpatialNside,
                            objRA, objDec, lonlat=True)
        h, rev = esutil.stat.histogram(ipring, min=0, max=hp.nside2npix(self.deltaAperFitSpatialNside) - 1, rev=True)

        offsetMap = np.zeros(h.size, dtype=[('nstar_fit', 'i4', (self.fgcmStars.nBands, )),
                                            ('epsilon', 'f4', (self.fgcmStars.nBands, ))])
        upix, = np.where(h >= self.deltaAperFitSpatialMinStar)
        for i in upix:
            i1a = rev[rev[i]: rev[i + 1]]

            for j, band in enumerate(self.fgcmStars.bands):
                use, = np.where((objNGoodObs[i1a, j] >= self.deltaAperFitMinNgoodObs) &
                                (np.abs(objDeltaAperMean[i1a, j]) < 0.5) &
                                (objFlag[i1a] == 0))
                if use.size < self.deltaAperFitSpatialMinStar:
                    continue

                mag_std = objMagStdMean[i1a[use], j]
                magerr_std = objMagStdMeanErr[i1a[use], j]
                delta = objDeltaAperMean[i1a[use], j]

                x_flux = 10.**((mag_std - self.njyZp)/(-2.5))
                delta_err = magerr_std

                xvals = (2.5/np.log(10.)) / x_flux
                yvals = delta
                yerr = delta_err

                fit, nStar = self._fitEpsilonWithOutlierRejection(xvals, yvals, yerr)
                offsetMap['nstar_fit'][i, j] = nStar
                offsetMap['epsilon'][i, j] = self._normalizeEpsilon(fit)

        # Store the offsetmap in njy_per_arcsec2
        self.fgcmPars.compEpsilonMap[:, :] = offsetMap['epsilon']
        self.fgcmPars.compEpsilonNStarMap[:, :] = offsetMap['nstar_fit']

        if self.plotPath is not None:
            for j, band in enumerate(self.fgcmStars.bands):
                hpix, = np.where(offsetMap['nstar_fit'][:, j] >= self.deltaAperFitSpatialMinStar)
                if hpix.size < 2:
                    continue
                ra, dec = hp.pix2ang(self.deltaAperFitSpatialNside, hpix, lonlat=True)
                st = np.argsort(offsetMap['epsilon'][hpix, j])
                vmin = offsetMap['epsilon'][hpix[st[int(0.02*st.size)]], j]
                vmax = offsetMap['epsilon'][hpix[st[int(0.98*st.size)]], j]

                self.fgcmLog.info('Background offset in %s band 2%% to 98%%: %.5f, %.5f nJy/arcsec2' %
                                  (band, vmin, vmax))

                # Rotate RA, and flip
                hi, = np.where(ra > 180.0)
                ra[hi] -= 360.0

                Z = [[0, 0], [0, 0]]
                levels = np.linspace(vmin, vmax, num=150)
                CS3 = plt.contourf(Z, levels)

                fig = plt.figure(1, figsize=(10, 6))
                fig.clf()
                ax = fig.add_subplot(111)
                ax.hexbin(ra, dec, offsetMap['epsilon'][hpix, j], vmin=vmin, vmax=vmax)
                ax.set_xlabel('RA')
                ax.set_ylabel('Dec')
                ax.set_title('%s band' % (band))
                ax.set_aspect('equal')
                xlim = ax.get_xlim()
                ax.set_xlim(xlim[1], xlim[0])
                cb = plt.colorbar(CS3, ticks=np.linspace(vmin, vmax, 5))
                cb.set_label('epsilon (nJy/arcsec2)')
                fig.savefig('%s/%s_epsilon_map_%s.png' % (self.plotPath,
                                                          self.outfileBaseWithCycle,
                                                          band))
                plt.close(fig)

    def computeEpsilonPerCcd(self):
        """
        Compute epsilon binned per ccd.
        """
        if not self.fgcmStars.hasXY:
            self.fgcmLog.info("Cannot compute background x/y correlations without x/y information")
            return

        from .fgcmUtilities import plotCCDMapBinned2d

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsDeltaAper = snmm.getArray(self.fgcmStars.obsDeltaAperHandle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsLUTFilterIndex = snmm.getArray(self.fgcmStars.obsLUTFilterIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsX = snmm.getArray(self.fgcmStars.obsXHandle)
        obsY = snmm.getArray(self.fgcmStars.obsYHandle)

        # Use only good observations of good stars
        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=False, checkMinObs=True)
        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars)

        magGO = objMagStdMean[obsObjIDIndex[goodObs], obsBandIndex[goodObs]]
        deltaAperGO = obsDeltaAper[goodObs] - self.fgcmPars.compMedDeltaAper[obsExpIndex[goodObs]]
        gd, = np.where((magGO < 90.0) &
                       (np.abs(deltaAperGO) < 0.5))

        goodObs = goodObs[gd]
        magGO = magGO[gd]
        deltaAperGO = deltaAperGO[gd]
        magErrGO = objMagStdMeanErr[obsObjIDIndex[goodObs], obsBandIndex[goodObs]]
        deltaAperErrGO = obsMagADUModelErr[goodObs]
        xGO = obsX[goodObs]
        yGO = obsY[goodObs]
        ccdIndexGO = obsCCDIndex[goodObs]
        bandIndexGO = obsBandIndex[goodObs]
        lutFilterIndexGO = obsLUTFilterIndex[goodObs]

        xBin = np.floor((xGO*self.deltaAperFitPerCcdNx)/self.deltaMapper[ccdIndexGO]['x_size']).astype(np.int32)
        yBin = np.floor((yGO*self.deltaAperFitPerCcdNy)/self.deltaMapper[ccdIndexGO]['y_size']).astype(np.int32)

        filterCcdHash = ccdIndexGO*(self.fgcmPars.nLUTFilter + 1) + lutFilterIndexGO

        h, rev = esutil.stat.histogram(filterCcdHash, rev=True)

        # Arbitrary minimum number here
        gdHash, = np.where(h > 10)

        epsilonCcdMap = np.zeros((self.fgcmPars.nLUTFilter, self.fgcmPars.nCCD,
                                  self.deltaAperFitPerCcdNx, self.deltaAperFitPerCcdNy),
                                 dtype=np.float32) + self.illegalValue
        epsilonCcdNStarMap = np.zeros((self.fgcmPars.nLUTFilter, self.fgcmPars.nCCD,
                                       self.deltaAperFitPerCcdNx, self.deltaAperFitPerCcdNy),
                                      dtype=np.int32)


        for i in gdHash:
            i1a = rev[rev[i]: rev[i + 1]]
            cInd = ccdIndexGO[i1a[0]]
            fInd = lutFilterIndexGO[i1a[0]]

            # Some quantities here...
            flux = 10.**((magGO[i1a] - self.njyZp)/(-2.5))
            fluxErr = (2.5/np.log(10.))*magErrGO[i1a]

            # Per filter/ccd we need to compute the bright-end offset
            # In the future, look at adding this information above?
            st = np.argsort(magGO[i1a])
            cutMag = magGO[i1a[st[int(0.25*st.size)]]]
            bright, = np.where(magGO[i1a] < cutMag)
            offset = np.median(deltaAperGO[i1a[bright]])
            c = 10.**(offset/2.5)

            if self.epsilonNormalized:
                norm = self.k*np.pi*(self.deltaAperOuterRadiusArcsec**2. -
                                     c*self.deltaAperInnerRadiusArcsec**2.)/c
            else:
                norm = 1.0

            epsilonApprox = (deltaAperGO[i1a] - offset)*flux/norm
            relativeFluxErr2 = (fluxErr/flux)**2.
            relativeDeltaAperErr2 = (deltaAperErrGO[i1a]/np.clip(deltaAperGO[i1a] - offset, 0.001, None))**2.
            epsilonErrApprox = np.abs(epsilonApprox)*np.sqrt(relativeFluxErr2 +
                                                             relativeDeltaAperErr2)
            epsilonMed = np.median(epsilonApprox)

            xyBinHash = xBin[i1a]*(self.deltaAperFitPerCcdNy + 1) + yBin[i1a]

            h2, rev2 = esutil.stat.histogram(xyBinHash, rev=True)

            gdHash2, = np.where(h2 > 10)
            for j in gdHash2:
                i2a = rev2[rev2[j]: rev2[j + 1]]

                xInd = xBin[i1a[i2a[0]]]
                yInd = yBin[i1a[i2a[0]]]

                if i2a.size >= 500:
                    # We can do the full fit
                    xvals = (2.5/np.log(10.))/flux[i2a]
                    yvals = deltaAperGO[i1a[i2a]] - offset
                    yerr = deltaAperErrGO[i1a[i2a]]

                    fit, nStar = self._fitEpsilonWithOutlierRejection(xvals, yvals, yerr)
                    epsilonCcdMap[fInd, cInd, xInd, yInd] = self._normalizeEpsilon(fit)
                    epsilonCcdNStarMap[fInd, cInd, xInd, yInd] = nStar
                else:
                    # Do the "weighted mean" epsilon with quick outlier rejection
                    ok2, = np.where(np.abs(epsilonApprox[i2a] - epsilonMed) < 3.0*epsilonErrApprox[i2a])
                    wt = 1./epsilonErrApprox[i2a[ok2]]**2.
                    wmean = np.sum(epsilonApprox[i2a[ok2]]*wt)/np.sum(wt)
                    epsilonCcdMap[fInd, cInd, xInd, yInd] = wmean
                    epsilonCcdNStarMap[fInd, cInd, xInd, yInd] = ok2.size

        self.fgcmPars.compEpsilonCcdMap[:] = epsilonCcdMap[:]
        self.fgcmPars.compEpsilonCcdNStarMap[:] = epsilonCcdNStarMap[:]

        if self.plotPath is not None:
            for j, filterName in enumerate(self.fgcmPars.lutFilterNames):
                binnedArray = epsilonCcdMap[j, :, :, :]

                fig = plt.figure(figsize=(8, 6))
                fig.clf()

                ax = fig.add_subplot(111)
                plotCCDMapBinned2d(ax, self.deltaMapper, binnedArray, 'Epsilon (nJy/arcsec2)')

                text = r'$(%s)$' % (filterName)
                ax.annotate(text,
                            (0.1, 0.93), xycoords='axes fraction',
                            ha='left', va='top', fontsize=18)
                fig.tight_layout()

                fig.savefig('%s/%s_epsilon_perccd_%s.png' % (self.plotPath,
                                                             self.outfileBaseWithCycle,
                                                             filterName))
                plt.close(fig)

    def _starWorker(self, goodStarsAndObs):
        """
        Multiprocessing worker for FgcmDeltaAper.  Not to be called on its own.

        Parameters
        ----------
        goodStarsAndObs: tuple[2]
           (goodStars, goodObs)
        """
        # NOTE: No logging is allowed in the _magWorker method

        goodStars = goodStarsAndObs[0]
        goodObs = goodStarsAndObs[1]

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objDeltaAperMean = snmm.getArray(self.fgcmStars.objDeltaAperMeanHandle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsDeltaAper = snmm.getArray(self.fgcmStars.obsDeltaAperHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        # Cut to good exposures
        gd, = np.where((self.fgcmPars.expFlag[obsExpIndex[goodObs]] == 0) &
                       (obsFlag[goodObs] == 0) &
                       (np.abs(obsDeltaAper[goodObs]) < 0.5))
        goodObs = goodObs[gd]

        obsMagErr2GO = obsMagADUModelErr[goodObs]**2.

        wtSum = np.zeros_like(objMagStdMean, dtype='f8')
        objDeltaAperMeanTemp = np.zeros_like(objMagStdMean, dtype='f8')

        np.add.at(objDeltaAperMeanTemp,
                  (obsObjIDIndex[goodObs], obsBandIndex[goodObs]),
                  (obsDeltaAper[goodObs] - self.fgcmPars.compMedDeltaAper[obsExpIndex[goodObs]])/obsMagErr2GO)
        np.add.at(wtSum,
                  (obsObjIDIndex[goodObs], obsBandIndex[goodObs]),
                  1./obsMagErr2GO)

        gd = np.where(wtSum > 0.0)

        objDeltaAperMeanLock = snmm.getArrayBase(self.fgcmStars.objDeltaAperMeanHandle).get_lock()
        objDeltaAperMeanLock.acquire()

        objDeltaAperMean[gd] = objDeltaAperMeanTemp[gd] / wtSum[gd]

        objDeltaAperMeanLock.release()

    def _fitEpsilonWithOutlierRejection(self, xvals, yvals, yerr, madCut=3.0, errCut=5.0):
        """
        """
        # First outlier rejection based on MAD
        med = np.median(yvals)
        sigma_mad = 1.4826*np.median(np.abs(yvals - med))
        ok, = np.where(np.abs(yvals - med) < madCut*sigma_mad)

        fit = np.polyfit(xvals[ok], yvals[ok], 1, w=1./yerr[ok])

        # Second better rejection with residuals
        resid = yvals - (fit[0]*xvals + fit[1])
        ok, = np.where(np.abs(resid) < errCut*yerr)
        fit = np.polyfit(xvals[ok], yvals[ok], 1, w=1./yerr[ok])

        return fit, ok.size

    def _normalizeEpsilon(self, fit):
        """Compute normalized epsilon value.

        Returns raw fit slope if no aperture radii were set.

        Parameters
        ----------
        fit : `iterable`
            Fit parameters (slope, intercept).

        Returns
        -------
        epsilon : `float`
        """
        if not self.epsilonNormalized:
            return fit[0]

        c = 10.**(fit[1]/2.5)
        return (c*fit[0])/(self.k*np.pi*(self.deltaAperOuterRadiusArcsec**2.
                                         - c*self.deltaAperInnerRadiusArcsec**2.))

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state
