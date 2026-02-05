import numpy as np
import hpgeom as hpg
import os
import sys
import esutil
import time
import skyproj

from .fgcmUtilities import dataBinner
from .fgcmUtilities import objFlagDict
from .fgcmUtilities import makeFigure, putButlerFigure

import multiprocessing

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
    def __init__(self, fgcmConfig, fgcmPars, fgcmStars, butlerQC=None, plotHandleDict=None):
        self.fgcmLog = fgcmConfig.fgcmLog

        if not fgcmStars.hasDeltaAper:
            self.fgcmLog.info("Cannot compute delta aperture parameters without measurements.")
            return

        self.fgcmPars = fgcmPars
        self.fgcmStars = fgcmStars

        self.butlerQC = butlerQC
        self.plotHandleDict = plotHandleDict

        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber

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

        self.njyZp = 8.9 + 9*2.5
        self.k = 2.5/np.log(10.)

        self.rng = fgcmConfig.rng

        self.nside_density_cut = 32
        self.density_percentile = 50

    def computeDeltaAperExposures(self, doFullFit=False, doPlots=False):
        """
        Compute deltaAper per-exposure quantities

        Parameters
        ----------
        doFullFit : `bool`, optional
           Do the full (expensive + slow) fit?
        doPlots : `bool`, optional
        """
        if not self.quietMode:
            if doFullFit:
                self.fgcmLog.info('Computing deltaAper per exposure')
            else:
                self.fgcmLog.info('Computing deltaAper offset per exposure')

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

        h, rev = esutil.stat.histogram(obsExpIndex[goodObs], min=0, rev=True)
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

            if not doFullFit:
                continue

            fit, _ = self._fitEpsilonWithDataBinner(mag[ok], deltaAper[ok], nTrial=20)

            if fit is not None:
                self.fgcmPars.compEpsilon[expIndex] = self._normalizeEpsilon(fit)

    def computeDeltaAperStars(self, debug=False, doPlots=False):
        """
        Compute deltaAper per-star quantities.

        Parameters
        ----------
        debug : `bool`, optional
            Debugging (no multiprocessing) mode.
        doPlots : `bool`, optional
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
            for i in range(1, nSections):
                splitValues[i - 1] = goodStarsList[i][0]

            splitIndices = np.searchsorted(goodStars[goodStarsSub], splitValues)
            goodObsList = np.split(goodObs, splitIndices)

            workerList = list(zip(goodStarsList,goodObsList))

            # reverse sort so the longest running go first
            workerList.sort(key=lambda elt:elt[1].size, reverse=True)

            mp_ctx = multiprocessing.get_context('fork')
            pool = mp_ctx.Pool(processes=self.nCore)
            pool.map(self._starWorker, workerList, chunksize=1)
            pool.close()
            pool.join()

        if not self.quietMode:
            self.fgcmLog.info('Finished DeltaAper in %.2f seconds.' %
                              (time.time() - startTime))

        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objDeltaAperMean = snmm.getArray(self.fgcmStars.objDeltaAperMeanHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)
        objRA = snmm.getArray(self.fgcmStars.objRAHandle)
        objDec = snmm.getArray(self.fgcmStars.objDecHandle)
        globalEpsilon = np.zeros(self.fgcmStars.nBands) + self.illegalValue
        globalOffset = np.zeros(self.fgcmPars.nBands) + self.illegalValue

        mask = (objFlagDict['TOO_FEW_OBS'] |
                objFlagDict['BAD_COLOR'] |
                objFlagDict['VARIABLE'] |
                objFlagDict['TEMPORARY_BAD_STAR'] |
                objFlagDict['RESERVED'])

        # Compute global offsets here.
        for i, band in enumerate(self.fgcmStars.bands):
            use, = np.where((objNGoodObs[:, i] >= self.deltaAperFitMinNgoodObs) &
                            (np.abs(objDeltaAperMean[:, i]) < 0.5) &
                            ((objFlag & mask) == 0))
            if use.size == 0:
                continue

            mag_std = objMagStdMean[use, i]
            delta = objDeltaAperMean[use, i]

            pix = hpg.angle_to_pixel(self.nside_density_cut, objRA[use], objDec[use])
            count = np.zeros(hpg.nside_to_npixel(self.nside_density_cut), dtype=np.int32)
            np.add.at(count, pix, 1)

            covered_pixels, = np.where(count > 0)
            density_cut = np.percentile(count[covered_pixels], self.density_percentile)

            # Now we can quickly down-select those that are too dense.
            sub_use = (count[pix] <= density_cut)

            if sub_use.sum() == 0:
                self.fgcmLog.warning("Down-selected density to none?  Seems not possible.")
                continue

            mag_std = mag_std[sub_use]
            delta = delta[sub_use]

            # Sample for efficiency if necessary
            nsample = 1000000
            if mag_std.size > nsample:
                r = self.rng.choice(mag_std.size, size=nsample, replace=False)
            else:
                r = np.arange(mag_std.size)

            fit, bin_struct = self._fitEpsilonWithDataBinner(mag_std[r], delta[r])

            if fit is not None:
                globalEpsilon[i] = self._normalizeEpsilon(fit)
                globalOffset[i] = fit[1]

            # Store the value of njy_per_arcsec2
            self.fgcmPars.compGlobalEpsilon[i] = globalEpsilon[i]
            self.fgcmLog.info('Global background offset in %s band: %.5f nJy/arcsec2' %
                              (band, globalEpsilon[i]))

            if doPlots and fit is not None:
                # Do plots

                st = np.argsort(mag_std[r])
                mag_min = mag_std[r[st[int(0.01*st.size)]]]
                mag_max = mag_std[r[st[int(0.95*st.size)]]]
                st = np.argsort(delta[r])
                delta_min = delta[r[st[int(0.02*st.size)]]]
                delta_max = delta[r[st[int(0.98*st.size)]]]

                fig = makeFigure(figsize=(8, 6))
                fig.clf()
                ax = fig.add_subplot(111)
                ax.hexbin(mag_std, delta, extent=[mag_min, mag_max,
                                                  delta_min, delta_max], bins='log')
                ax.errorbar(bin_struct['X'], bin_struct['Y'],
                            yerr=bin_struct['Y_ERR'], fmt='r.', markersize=10)
                xplotvals = np.linspace(mag_min, mag_max, 100)
                xplotfluxvals = 10.**((xplotvals - self.njyZp)/(-2.5))
                yplotvals = fit[0]*((2.5/np.log(10.0))/xplotfluxvals) + fit[1]
                ax.plot(xplotvals, yplotvals, 'r-')
                ax.set_xlabel('mag_std_%s (mag)' % (band))
                ax.set_ylabel('Normalized delta_aper_%s (mag)' % (band))
                ax.set_title('Global offset %s: %.4f nJy/arcsec2' % (band, globalEpsilon[i]))

                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "EpsilonGlobal",
                                    self.cycleNumber,
                                    fig,
                                    band=band)
                elif self.plotPath is not None:
                    fig.savefig('%s/%s_epsilon_global_%s.png' % (self.plotPath,
                                                                 self.outfileBaseWithCycle,
                                                                 band))

    def computeEpsilonMap(self, doPlots=False):
        """
        Compute global epsilon and local maps.

        Parameters
        ----------
        doPlots : `bool`, optional
        """
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)
        objRA = snmm.getArray(self.fgcmStars.objRAHandle)
        objDec = snmm.getArray(self.fgcmStars.objDecHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objDeltaAperMean = snmm.getArray(self.fgcmStars.objDeltaAperMeanHandle)

        # Do the mapping
        self.fgcmLog.info("Computing delta-aper epsilon spatial map at nside %d" % self.deltaAperFitSpatialNside)
        ipring = hpg.angle_to_pixel(self.deltaAperFitSpatialNside, objRA, objDec, nest=False)
        h, rev = esutil.stat.histogram(
            ipring,
            min=0,
            max=hpg.nside_to_npixel(self.deltaAperFitSpatialNside) - 1,
            rev=True,
        )

        mask = (objFlagDict['TOO_FEW_OBS'] |
                objFlagDict['BAD_COLOR'] |
                objFlagDict['VARIABLE'] |
                objFlagDict['TEMPORARY_BAD_STAR'] |
                objFlagDict['RESERVED'])

        offsetMap = np.zeros(h.size, dtype=[('nstar_fit', 'i4', (self.fgcmStars.nBands, )),
                                            ('epsilon', 'f4', (self.fgcmStars.nBands, ))])
        upix, = np.where(h >= self.deltaAperFitSpatialMinStar)
        for i in upix:
            i1a = rev[rev[i]: rev[i + 1]]

            for j, band in enumerate(self.fgcmStars.bands):
                use, = np.where((objNGoodObs[i1a, j] >= self.deltaAperFitMinNgoodObs) &
                                (np.abs(objDeltaAperMean[i1a, j]) < 0.5) &
                                ((objFlag[i1a] & mask) == 0))
                if use.size < self.deltaAperFitSpatialMinStar:
                    continue

                mag_std = objMagStdMean[i1a[use], j]
                delta = objDeltaAperMean[i1a[use], j]

                nsample = 10000
                if mag_std.size > nsample:
                    r = self.rng.choice(mag_std.size, size=nsample, replace=False)
                else:
                    r = np.arange(mag_std.size)

                fit, _ = self._fitEpsilonWithDataBinner(mag_std[r], delta[r], nTrial=20)

                if fit is not None:
                    offsetMap['nstar_fit'][i, j] = len(mag_std)
                    offsetMap['epsilon'][i, j] = self._normalizeEpsilon(fit)

        # Store the offsetmap in njy_per_arcsec2
        self.fgcmPars.compEpsilonMap[:, :] = offsetMap['epsilon']
        self.fgcmPars.compEpsilonNStarMap[:, :] = offsetMap['nstar_fit']

        if doPlots:
            for j, band in enumerate(self.fgcmStars.bands):
                hpix, = np.where(offsetMap['nstar_fit'][:, j] >= self.deltaAperFitSpatialMinStar)
                if hpix.size < 2:
                    self.fgcmLog.info("Not enough sky coverage for epsilon map in %s band" % (band))
                    continue

                st = np.argsort(offsetMap['epsilon'][hpix, j])
                vmin = offsetMap['epsilon'][hpix[st[int(0.02*st.size)]], j]
                vmax = offsetMap['epsilon'][hpix[st[int(0.98*st.size)]], j]

                self.fgcmLog.info('Background offset in %s band 2%% to 98%%: %.5f, %.5f nJy/arcsec2' %
                                  (band, vmin, vmax))

                fig = makeFigure(figsize=(10, 6))
                fig.clf()
                ax = fig.add_subplot(111)

                sp = skyproj.McBrydeSkyproj(ax=ax)
                sp.draw_hpxpix(
                    self.deltaAperFitSpatialNside,
                    hpix,
                    offsetMap['epsilon'][hpix, j],
                    nest=False,
                    vmin=vmin,
                    vmax=vmax,
                )
                sp.draw_colorbar(label="epsilon (nJy/arcsec2)")
                fig.suptitle("%s band" % (band))

                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "EpsilonMap",
                                    self.cycleNumber,
                                    fig,
                                    band=band)
                elif self.plotPath is not None:
                    fig.savefig('%s/%s_epsilon_map_%s.png' % (self.plotPath,
                                                              self.outfileBaseWithCycle,
                                                              band))

    def computeEpsilonPerCcd(self, doPlots=False):
        """
        Compute epsilon binned per ccd.

        Parameters
        ----------
        doPlots : `bool`, optional
        """
        if not self.fgcmStars.hasXY:
            self.fgcmLog.info("Cannot compute background x/y correlations without x/y information")
            return

        from .fgcmUtilities import plotCCDMapBinned2d

        objRA = snmm.getArray(self.fgcmStars.objRAHandle)
        objDec = snmm.getArray(self.fgcmStars.objDecHandle)
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

        # Downsample to less dense regions before continuing.
        pix = hpg.angle_to_pixel(self.nside_density_cut, objRA[goodStars], objDec[goodStars])
        count = np.zeros(hpg.nside_to_npixel(self.nside_density_cut), dtype=np.int32)
        np.add.at(count, pix, 1)

        coveredPixels, = np.where(count > 0)
        densityCut = np.percentile(count[coveredPixels], self.density_percentile)
        goodStars = goodStars[count[pix] <= densityCut]

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

        xBin = np.floor((xGO*self.deltaAperFitPerCcdNx)/self.deltaMapper['x_size'][ccdIndexGO]).astype(np.int32)
        yBin = np.floor((yGO*self.deltaAperFitPerCcdNy)/self.deltaMapper['y_size'][ccdIndexGO]).astype(np.int32)

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

            # Default normalization
            norm = 1.0

            xyBinHash = xBin[i1a]*(self.deltaAperFitPerCcdNy + 1) + yBin[i1a]

            h2, rev2 = esutil.stat.histogram(xyBinHash, rev=True)

            gdHash2, = np.where(h2 > 10)
            for j in gdHash2:
                i2a = rev2[rev2[j]: rev2[j + 1]]

                if len(i2a) == 0:
                    continue

                xInd = xBin[i1a[i2a[0]]]
                yInd = yBin[i1a[i2a[0]]]

                # Use median statistics for this experimental mode
                st = np.argsort(magGO[i1a[i2a]])
                cutMag = magGO[i1a[i2a[st[int(0.25*st.size)]]]]
                offset = np.median(deltaAperGO[i1a[i2a[magGO[i1a[i2a]] < cutMag]]])
                c = 10.**(offset/2.5)

                if self.epsilonNormalized:
                    norm = self.k*np.pi*(self.deltaAperOuterRadiusArcsec**2. -
                                         c*self.deltaAperInnerRadiusArcsec**2.)/c

                epsilonCcdMap[fInd, cInd, xInd, yInd] = np.median((deltaAperGO[i1a[i2a]] - offset)*flux[i2a]/norm)
                epsilonCcdNStarMap[fInd, cInd, xInd, yInd] = i2a.size

        self.fgcmPars.compEpsilonCcdMap[:] = epsilonCcdMap[:]
        self.fgcmPars.compEpsilonCcdNStarMap[:] = epsilonCcdNStarMap[:]

        scaleRange = np.zeros((self.fgcmPars.nLUTFilter, 2))
        scaleMedian = np.zeros(self.fgcmPars.nLUTFilter)
        for j in range(self.fgcmPars.nLUTFilter):
            flatArray = epsilonCcdMap[j, :, :, :].ravel()
            gd, = np.where(flatArray > self.illegalValue)
            if gd.size == 0:
                continue
            st = np.argsort(flatArray[gd])
            scaleMedian[j] = flatArray[gd[st[int(0.5*st.size)]]]
            scaleRange[j, 0] = flatArray[gd[st[int(0.05*st.size)]]]
            scaleRange[j, 1] = flatArray[gd[st[int(0.95*st.size)]]]

        # ignore bands that have a range > 10
        deltaRange = scaleRange[:, 1] - scaleRange[:, 0]
        filtersToMatchRange, = np.where(deltaRange < 20.0)
        if filtersToMatchRange.size > 0:
            matchedDelta = np.max(deltaRange[filtersToMatchRange])
        else:
            # If they all have big variance, just use that.
            matchedDelta = np.max(deltaRange)

        if doPlots:
            for j, filterName in enumerate(self.fgcmPars.lutFilterNames):
                if self.fgcmPars.filterToBand[filterName] not in self.fgcmPars.bands:
                    continue

                binnedArray = epsilonCcdMap[j, :, :, :]

                fig = makeFigure(figsize=(8, 6))
                fig.clf()

                ax = fig.add_subplot(111)
                plotCCDMapBinned2d(ax, self.deltaMapper, binnedArray, 'Epsilon (nJy/arcsec2)')

                text = '(%s)' % (filterName)
                ax.annotate(text,
                            (0.1, 0.93), xycoords='axes fraction',
                            ha='left', va='top', fontsize=18)
                fig.tight_layout()

                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "EpsilonDetector",
                                    self.cycleNumber,
                                    fig,
                                    filterName=filterName)
                elif self.plotPath is not None:
                    fig.savefig('%s/%s_epsilon_perccd_%s.png' % (self.plotPath,
                                                                 self.outfileBaseWithCycle,
                                                                 filterName))

                # And replot with matched scale
                loHi = [scaleMedian[j] - matchedDelta - 1e-7,
                        scaleMedian[j] + matchedDelta + 1e-7]

                fig = makeFigure(figsize=(8, 6))
                fig.clf()

                ax = fig.add_subplot(111)
                plotCCDMapBinned2d(ax,
                                   self.deltaMapper,
                                   binnedArray,
                                   'Epsilon (nJy/arcsec2)',
                                   loHi=loHi)

                text = '(%s)' % (filterName)
                ax.annotate(text,
                            (0.1, 0.93), xycoords='axes fraction',
                            ha='left', va='top', fontsize=18)
                fig.tight_layout()

                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "EpsilonDetectorMatchscale",
                                    self.cycleNumber,
                                    fig,
                                    filterName=filterName)
                elif self.plotPath is not None:
                    fig.savefig('%s/%s_epsilon_perccd_%s_matchscale.png' % (self.plotPath,
                                                                            self.outfileBaseWithCycle,
                                                                            filterName))

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
                  ((obsDeltaAper[goodObs] - self.fgcmPars.compMedDeltaAper[obsExpIndex[goodObs]])/obsMagErr2GO).astype(objDeltaAperMeanTemp.dtype))
        np.add.at(wtSum,
                  (obsObjIDIndex[goodObs], obsBandIndex[goodObs]),
                  (1./obsMagErr2GO).astype(wtSum.dtype))

        gd = np.where(wtSum > 0.0)

        objDeltaAperMeanLock = snmm.getArrayBase(self.fgcmStars.objDeltaAperMeanHandle).get_lock()
        objDeltaAperMeanLock.acquire()

        objDeltaAperMean[gd] = objDeltaAperMeanTemp[gd] / wtSum[gd]

        objDeltaAperMeanLock.release()

    def _fitEpsilonWithOutlierRejection(self, xvals, yvals, yerr, madCut=3.0, errCut=5.0):
        """
        Fit epsilon with outlier rejection.

        This doesn't work so good.

        Parameters
        ----------
        xvals : `np.ndarray`
            x values.
        yvals : `np.ndarray`
            y values.
        yerr : `np.ndarray`
            y error values.
        madCut : `float`, optional
            Initial median-absolute-deviation cut.
        errCut : `float`, optional
            Secondary nsigma error cut.

        Returns
        -------
        fit : `tuple`
            Fit parameters.
        size : `int`
            Number of stars in final fit.
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

    def _fitEpsilonWithDataBinner(self, mag, delta_aper, binsize=0.2, nTrial=100):
        """
        Fit epsilon with binned data.

        This works much better.

        Parameters
        ----------
        mag : `np.ndarray`
            Magnitude values
        delta_aper : `np.ndarray`
            Delta-aper values.
        binsize : `float`, optional
            Magnitude bin size.
        nTrial : `int`, optional
            Number of bootstrap trials for data binner.

        Returns
        -------
        fit : `tuple`
            Fit parameters
        bin_struct : `np.recarray`
            Binned data structure.
        """
        st = np.argsort(mag)
        mag_min = mag[st[int(0.01*st.size)]]
        mag_max = mag[st[int(0.95*st.size)]]

        bin_struct = dataBinner(mag, delta_aper, binsize, [mag_min, mag_max], rng=self.rng, nTrial=nTrial)
        u, = np.where(bin_struct['Y_ERR'] > 0.0)

        if u.size < 5:
            return None, None

        bin_flux = 10.**((bin_struct['X'] - self.njyZp)/(-2.5))
        fit = np.polyfit((2.5/np.log(10.0))/bin_flux[u],
                         bin_struct['Y'][u],
                         1, w=1./bin_struct['Y_ERR'][u])
        return fit, bin_struct[u]

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
        del state['butlerQC']
        del state['plotHandleDict']
        return state
