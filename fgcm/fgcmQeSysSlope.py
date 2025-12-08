import numpy as np
import os
import sys
import esutil
import time
import scipy.optimize
from astropy.time import Time

from .fgcmUtilities import makeFigure, putButlerFigure
from matplotlib import colormaps

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmQeSysSlope(object):
    """
    Class which computes the slope of the system QE degredation.

    Parameters
    ----------
    fgcmConfig: FgcmConfig
       Config object
    fgcmPars: FgcmParameters
       Parameter object
    fgcmStars: FgcmStars
       Stars object
    initialCycle: `bool`
       Is this the initial cycle? (Force gray computation)
    """
    def __init__(self, fgcmConfig, fgcmPars, fgcmStars, butlerQC=None, plotHandleDict=None):
        self.fgcmLog = fgcmConfig.fgcmLog
        self.fgcmLog.debug('Initializing FgcmQeSysSlope')

        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.plotPath = fgcmConfig.plotPath
        self.cycleNumber = fgcmConfig.cycleNumber

        self.fgcmPars = fgcmPars
        self.fgcmStars = fgcmStars

        self.butlerQC = butlerQC
        self.plotHandleDict = plotHandleDict

        self.bandFitIndex = fgcmConfig.bandFitIndex
        self.instrumentParsPerBand = fgcmConfig.instrumentParsPerBand
        self.instrumentSlopeMinDeltaT = fgcmConfig.instrumentSlopeMinDeltaT
        self.ccdGrayMaxStarErr = fgcmConfig.ccdGrayMaxStarErr

    def computeQeSysSlope(self, name, doPlots=False):
        """
        Compute QE system slope

        Parameters
        ----------
        name : `str`
            Name to put on filenames
        doPlots : `bool`, optional
        """

        objID = snmm.getArray(self.fgcmStars.objIDHandle)

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsMagADUModelErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)

        # Select good stars and good observations of said stars
        goodStars = self.fgcmStars.getGoodStarIndices(checkMinObs=True)
        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=self.fgcmPars.expFlag, checkBadMag=True)

        # Further filter good observations
        ok, = np.where((obsMagADUModelErr[goodObs] < self.ccdGrayMaxStarErr) &
                       (obsMagADUModelErr[goodObs] > 0.0) &
                       (obsMagStd[goodObs] < 90.0))

        goodObs = goodObs[ok]

        # Make copies so we don't overwrite anything
        obsMagStdGO = obsMagStd[goodObs]
        obsMagErr2GO = obsMagADUModelErr[goodObs]**2.
        obsExpIndexGO = obsExpIndex[goodObs]

        # Remove the previously applied slope
        deltaQESlopeGO = (self.fgcmPars.compQESysSlopeApplied[obsBandIndex[goodObs], self.fgcmPars.expWashIndex[obsExpIndexGO]] *
                          (self.fgcmPars.expMJD[obsExpIndexGO] -
                           self.fgcmPars.washMJDs[self.fgcmPars.expWashIndex[obsExpIndexGO]]))
        obsMagStdGO -= deltaQESlopeGO

        # split per wash interval
        washH, washRev = esutil.stat.histogram(self.fgcmPars.expWashIndex[obsExpIndexGO], min=0, rev=True)
        washIndices, = np.where(washH > 0)

        for washIndex in washIndices:
            i1a = washRev[washRev[washIndex]: washRev[washIndex + 1]]

            # Split per band, and compute the delta-T and delta-Mag

            bandH, bandRev = esutil.stat.histogram(obsBandIndex[goodObs[i1a]], min=0, rev=True)
            bandIndices, = np.where(bandH > 0)

            deltaTAll = None

            for bandIndex in bandIndices:
                if not self.fgcmPars.hasExposuresInBand[bandIndex]:
                    continue
                i2a = bandRev[bandRev[bandIndex]: bandRev[bandIndex + 1]]

                # Now lump the stars together

                thisObjID = objID[obsObjIDIndex[goodObs[i1a[i2a]]]]
                thisMjd = self.fgcmPars.expMJD[obsExpIndexGO[i1a[i2a]]]
                thisMag = obsMagStdGO[i1a[i2a]]
                thisMagErr2 = obsMagErr2GO[i1a[i2a]]

                minID = thisObjID.min()
                maxID = thisObjID.max()

                # we need to sort and take unique to get the index of the first mjd
                st = np.argsort(thisMjd)

                minMjd = np.zeros(maxID - minID + 1)
                starIndices, firstIndex = np.unique(thisObjID[st] - minID, return_index=True)
                minMjd[starIndices] = thisMjd[st[firstIndex]]

                firstMag = np.zeros_like(minMjd, dtype=np.float32)
                firstMag[starIndices] = thisMag[st[firstIndex]]

                firstMagErr2 = np.zeros_like(firstMag)
                firstMagErr2[starIndices] = thisMagErr2[st[firstIndex]]

                deltaT = thisMjd - minMjd[thisObjID - minID]
                deltaMag = thisMag - firstMag[thisObjID - minID]
                deltaMagErr2 = thisMagErr2 + firstMagErr2[thisObjID - minID]

                okDelta, = np.where((deltaT > self.instrumentSlopeMinDeltaT) &
                                    (deltaMagErr2 < self.ccdGrayMaxStarErr))

                deltaT = deltaT[okDelta]
                deltaMag = deltaMag[okDelta]
                deltaMagErr2 = deltaMagErr2[okDelta]

                # Check if we are doing one band at a time or lumping together.
                if not self.instrumentParsPerBand:
                    # Lump all together.  Not the most efficient, may need to update.
                    if deltaTAll is None:
                        deltaTAll = deltaT
                        deltaMagAll = deltaMag
                        deltaMagErr2All = deltaMagErr2
                    elif bandIndex in self.bandFitIndex:
                        # only add if this is one of the fit bands
                        deltaTAll = np.append(deltaTAll, deltaT)
                        deltaMagAll = np.append(deltaMagAll, deltaMag)
                        deltaMagErr2All = np.append(deltaMagErr2All, deltaMagErr2)
                else:
                    # Do per band

                    if deltaT.size < 500:
                        # Just do no slope
                        extraString = ' (Not enough observations)'
                        slopeMean = 0.0
                        slopeMeanErr = 0.0
                    else:
                        extraString = ''
                        slope = deltaMag / deltaT
                        slopeErr2 = deltaMagErr2 / np.abs(deltaT)**2.
                        slopeMean = np.clip(-1 * np.sum(slope / slopeErr2) / np.sum(1. / slopeErr2), -0.001, 0.0)
                        slopeMeanErr = np.sqrt(1. / np.sum(1. / slopeErr2))

                    self.fgcmLog.info("Wash interval %d, computed qe slope in %s band: %.6f +/- %.6f mmag/day%s" %
                                      (washIndex, self.fgcmPars.bands[bandIndex], slopeMean*1000.0, slopeMeanErr*1000.0, extraString))
                    self.fgcmPars.compQESysSlope[bandIndex, washIndex] = slopeMean

            if not self.instrumentParsPerBand:
                # Compute all together

                if deltaTAll.size < 500:
                    extraString = ' (Not enough observations)'
                    slopeMeanAll = 0.0
                    slopeMeanErrAll = 0.0
                else:
                    extraString = ''
                    slopeAll = deltaMagAll / deltaTAll
                    slopeErr2All = deltaMagErr2All / np.abs(deltaTAll)**2.
                    slopeMeanAll = np.clip(-1 * np.sum(slopeAll / slopeErr2All) / np.sum(1. / slopeErr2All), -0.001, 0.0)
                    slopeMeanErrAll = np.sqrt(1. / np.sum(1. / slopeErr2All))

                self.fgcmLog.info("Wash interval %d, computed qe slope in all bands: %.6f +/- %.6f mmag/day%s" %
                                  (washIndex, slopeMeanAll*1000.0, slopeMeanErrAll*1000.0, extraString))
                self.fgcmPars.compQESysSlope[:, washIndex] = slopeMeanAll

        if doPlots:
            # Make the plots
            firstMJD = np.floor(np.min(self.fgcmPars.expMJD))

            fig = makeFigure(figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)

            colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']
            started = False

            for i in range(self.fgcmPars.nWashIntervals):
                use, = np.where(self.fgcmPars.expWashIndex == i)
                if use.size == 0:
                    # There are none in this interval, that's fine
                    continue

                washMJDRange = [np.min(self.fgcmPars.expMJD[use]), np.max(self.fgcmPars.expMJD[use])]

                if self.instrumentParsPerBand:
                    # Need to plot all of them one-by-one
                    for j in range(self.fgcmPars.nBands):
                        if not self.fgcmPars.hasExposuresInBand[j]:
                            continue
                        label = self.fgcmPars.bands[j] if not started else None
                        ax.plot(washMJDRange - firstMJD,
                                1000.0*((washMJDRange - self.fgcmPars.washMJDs[i])*self.fgcmPars.compQESysSlope[j, i] +
                                        self.fgcmPars.parQESysIntercept[j, i]), linestyle='--', color=colors[j % len(colors)], linewidth=2, label=label)
                else:
                    ax.plot(washMJDRange - firstMJD,
                            1000.0*((washMJDRange - self.fgcmPars.washMJDs[i])*self.fgcmPars.compQESysSlope[0, i] +
                            self.fgcmPars.parQESysIntercept[0, i]), 'r--', linewidth=3)
                started = True

            if self.instrumentParsPerBand:
                ax.legend(loc=3)

            ax.set_xlabel(r'$\mathrm{MJD}\ -\ %.0f$' % (firstMJD), fontsize=16)
            ax.set_ylabel(r'$2.5 \log_{10} (S^{\mathrm{optics}})\,(\mathrm{mmag})$', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)

            # Make the vertical wash markers
            ylim = ax.get_ylim()
            for i in range(self.fgcmPars.nWashIntervals):
                ax.plot([self.fgcmPars.washMJDs[i] - firstMJD, self.fgcmPars.washMJDs[i]-firstMJD],
                        ylim, 'k--')

            if self.butlerQC is not None:
                putButlerFigure(self.fgcmLog,
                                self.butlerQC,
                                self.plotHandleDict,
                                f"QESysWashes{name.title()}",
                                self.cycleNumber,
                                fig)
            elif self.plotPath is not None:
                fig.savefig('%s/%s_qesys_washes_%s.png' % (self.plotPath,
                                                           self.outfileBaseWithCycle,
                                                           name))

    def plotQeSysRefStars(self, name, doPlots=False):
        """
        Plot reference stars (if available).  Compare residuals.

        Parameters
        ----------
        name : `str`
            Name to give the files.
        doPlots : `bool`, optional
        """

        if not self.fgcmStars.hasRefstars:
            self.fgcmLog.info("No reference stars for QE sys plots.")
            return

        if not doPlots:
            return

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)

        objRefIDIndex = snmm.getArray(self.fgcmStars.objRefIDIndexHandle)
        refMag = snmm.getArray(self.fgcmStars.refMagHandle)

        goodStars = self.fgcmStars.getGoodStarIndices(checkMinObs=True)
        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=self.fgcmPars.expFlag, checkBadMag=True)

        # Take out the previous slope...
        obsMagStdGO = obsMagStd[goodObs]
        obsExpIndexGO = obsExpIndex[goodObs]

        deltaQESysGO = (self.fgcmPars.parQESysIntercept[obsBandIndex[goodObs], self.fgcmPars.expWashIndex[obsExpIndexGO]] +
                        self.fgcmPars.compQESysSlopeApplied[obsBandIndex[goodObs], self.fgcmPars.expWashIndex[obsExpIndexGO]] *
                        (self.fgcmPars.expMJD[obsExpIndexGO] -
                         self.fgcmPars.washMJDs[self.fgcmPars.expWashIndex[obsExpIndexGO]]))

        obsMagObsGO = obsMagStdGO - deltaQESysGO

        goodRefObsGO, = np.where(objRefIDIndex[obsObjIDIndex[goodObs]] >= 0)
        if goodRefObsGO.size < 100:
            self.fgcmLog.info("Not enough reference star information to make QE sys plots.")
            return

        obsUse, = np.where((obsMagStd[goodObs[goodRefObsGO]] < 90.0) &
                           (refMag[objRefIDIndex[obsObjIDIndex[goodObs[goodRefObsGO]]],
                                   obsBandIndex[goodObs[goodRefObsGO]]] < 90.0))
        if obsUse.size < 100:
            self.fgcmLog.info("Not enough good reference star information to make QE sys plots.")
            return
        goodRefObsGO = goodRefObsGO[obsUse]

        EGrayGRO = (refMag[objRefIDIndex[obsObjIDIndex[goodObs[goodRefObsGO]]],
                           obsBandIndex[goodObs[goodRefObsGO]]] -
                    obsMagStdGO[goodRefObsGO])

        EGrayObsGRO = (refMag[objRefIDIndex[obsObjIDIndex[goodObs[goodRefObsGO]]],
                           obsBandIndex[goodObs[goodRefObsGO]]] -
                       obsMagObsGO[goodRefObsGO])

        mjdGRO = self.fgcmPars.expMJD[obsExpIndexGO[goodRefObsGO]]
        minMjd = mjdGRO.min() - 5.0
        maxMjd = mjdGRO.max() + 5.0
        mjdGRO -= minMjd

        # And get a human-readable time out
        t = Time(minMjd, format='mjd')
        t.format = 'datetime'
        startString = '%04d-%02d-%02d' % (t.value.year, t.value.month, t.value.day)

        st = np.argsort(EGrayGRO)

        xMin = 0
        xMax = maxMjd - minMjd
        yMin = EGrayGRO[st[int(0.001 * st.size)]]*1000.0
        yMax = EGrayGRO[st[int(0.999 * st.size)]]*1000.0

        st = np.argsort(EGrayObsGRO)
        yMinObs = EGrayObsGRO[st[int(0.001 * st.size)]]*1000.0
        yMaxObs = EGrayObsGRO[st[int(0.999 * st.size)]]*1000.0

        # which wash dates are within the range...
        washInRange, = np.where((self.fgcmPars.washMJDs >= minMjd) &
                                (self.fgcmPars.washMJDs <= maxMjd))

        for i, band in enumerate(self.fgcmPars.bands):
            if not self.fgcmPars.hasExposuresInBand[i]:
                continue

            use, = np.where(obsBandIndex[goodObs[goodRefObsGO]] == i)
            if use.size < 100:
                self.fgcmLog.info("Not enough good reference star information in %s band to make QE sys plots." % (band))
                continue

            fig = makeFigure(figsize=(8, 6))
            ax = fig.add_subplot(111)

            ax.hexbin(
                mjdGRO[use],
                EGrayGRO[use]*1000.0,
                bins='log',
                extent=[xMin, xMax, yMin, yMax],
                cmap=colormaps.get_cmap("viridis"),
            )

            for washIndex in washInRange:
                ax.plot([self.fgcmPars.washMJDs[washIndex] - minMjd, self.fgcmPars.washMJDs[washIndex] - minMjd], [yMin, yMax], 'r--', linewidth=2)

            ax.set_title('%s band' % (band))
            ax.set_xlabel('Days since %s (%.0f)' % (startString, minMjd), fontsize=14)
            ax.set_ylabel('m_ref - m_std (mmag)', fontsize=14)

            if self.butlerQC is not None:
                putButlerFigure(self.fgcmLog,
                                self.butlerQC,
                                self.plotHandleDict,
                                f"QESysRefstarsStd{name.title()}",
                                self.cycleNumber,
                                fig,
                                band=band)
            elif self.plotPath is not None:
                fig.savefig('%s/%s_qesys_refstars-std_%s_%s.png' % (self.plotPath,
                                                                    self.outfileBaseWithCycle,
                                                                    name, band))

            fig = makeFigure(figsize=(8, 6))
            ax = fig.add_subplot(111)

            ax.hexbin(
                mjdGRO[use],
                EGrayObsGRO[use]*1000.0,
                bins='log',
                extent=[xMin, xMax, yMinObs, yMaxObs],
                cmap=colormaps.get_cmap("viridis")
            )

            for washIndex in washInRange:
                ax.plot([self.fgcmPars.washMJDs[washIndex] - minMjd, self.fgcmPars.washMJDs[washIndex] - minMjd], [yMinObs, yMaxObs], 'r--', linewidth=2)

            ax.set_title('%s band' % (band))
            ax.set_xlabel('Days since %s (%.0f)' % (startString, minMjd), fontsize=14)
            ax.set_ylabel('m_ref - m_obs (mmag)', fontsize=14)

            if self.butlerQC is not None:
                putButlerFigure(self.fgcmLog,
                                self.butlerQC,
                                self.plotHandleDict,
                                f"QESysRefstarsObs{name.title()}",
                                self.cycleNumber,
                                fig,
                                band=band)
            elif self.plotPath is not None:
                fig.savefig('%s/%s_qesys_refstars-obs_%s_%s.png' % (self.plotPath,
                                                                    self.outfileBaseWithCycle,
                                                                    name, band))
