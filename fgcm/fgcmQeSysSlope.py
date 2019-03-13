from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import os
import sys
import esutil
import time
import matplotlib.pyplot as plt
import scipy.optimize
from astropy.time import Time

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
    def __init__(self, fgcmConfig, fgcmPars, fgcmStars):
        self.fgcmLog = fgcmConfig.fgcmLog
        self.fgcmLog.info('Initializing FgcmQeSysSlope')

        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.plotPath = fgcmConfig.plotPath

        self.fgcmPars = fgcmPars
        self.fgcmStars = fgcmStars

        self.bandFitIndex = fgcmConfig.bandFitIndex
        self.instrumentParsPerBand = fgcmConfig.instrumentParsPerBand
        self.instrumentSlopeMinDeltaT = fgcmConfig.instrumentSlopeMinDeltaT
        self.ccdGrayMaxStarErr = fgcmConfig.ccdGrayMaxStarErr

    def computeQeSysSlope(self, name, doPlots=True):
        """
        Compute QE system slope

        Parameters
        ----------
        name: `str`
           Name to put on filenames
        doPlots: `bool`, optional
           Make the plots.  Default is True.
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
        deltaQESlopeGO = (self.fgcmPars.compQESysSlopeApplied[self.fgcmPars.expWashIndex[obsExpIndexGO], obsBandIndex[goodObs]] *
                          (self.fgcmPars.expMJD[obsExpIndexGO] -
                           self.fgcmPars.washMJDs[self.fgcmPars.expWashIndex[obsExpIndexGO]]))
        obsMagStdGO -= deltaQESlopeGO

        # split per wash interval
        washH, washRev = esutil.stat.histogram(self.fgcmPars.expWashIndex[obsExpIndexGO], rev=True, min=0)
        washIndices, = np.where(washH > 0)

        for washIndex in washIndices:
            i1a = washRev[washRev[washIndex]: washRev[washIndex + 1]]

            # Split per band, and compute the delta-T and delta-Mag

            bandH, bandRev = esutil.stat.histogram(obsBandIndex[goodObs[i1a]], rev=True, min=0)
            bandIndices, = np.where(bandH > 0)

            deltaTAll = None

            for bandIndex in bandIndices:
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
                        slopeMean = 0.0
                        slopeMeanErr = 0.0
                    else:
                        slope = deltaMag / deltaT
                        slopeErr2 = deltaMagErr2 / np.abs(deltaT)**2.
                        slopeMean = np.clip(-1 * np.sum(slope / slopeErr2) / np.sum(1. / slopeErr2), -0.001, 0.0)
                        slopeMeanErr = np.sqrt(1. / np.sum(1. / slopeErr2))

                    """
                    # Make an output file here...
                    if deltaT.size > 500:
                        import fitsio
                        tempCat = np.zeros(slope.size, dtype=[('deltaMag', 'f4'),
                                                              ('deltaT', 'f4'),
                                                              ('deltaMagErr2', 'f4')])
                        tempCat['deltaMag'] = deltaMag
                        tempCat['deltaT'] = deltaT
                        tempCat['deltaMagErr2'] = deltaMagErr2

                        fitsio.write('%s_slopestuff_%s_%s_%02d.fits' % (self.outfileBaseWithCycle,
                                                                        name,
                                                                        self.fgcmPars.bands[bandIndex],
                                                                        washIndex), tempCat)
                                                                        """
                    self.fgcmLog.info("Wash interval %d, computed qe slope in %s band: %.8f +/- %.8f" %
                                      (washIndex, self.fgcmPars.bands[bandIndex], slopeMean, slopeMeanErr))
                    self.fgcmPars.compQESysSlope[washIndex, bandIndex] = slopeMean

            if not self.instrumentParsPerBand:
                # Compute all together

                if deltaTAll.size < 500:
                    slopeMeanAll = 0.0
                    slopeMeanErrAll = 0.0
                else:
                    slopeAll = deltaMagAll / deltaTAll
                    slopeErr2All = deltaMagErr2All / np.abs(deltaTAll)**2.
                    slopeMeanAll = np.clip(-1 * np.sum(slopeAll / slopeErr2All) / np.sum(1. / slopeErr2All), -0.001, 0.0)
                    slopeMeanErrAll = np.sqrt(1. / np.sum(1. / slopeErr2All))

                self.fgcmLog.info("Wash interval %d, computed qe slope in all bands: %.8f +/- %.8f" %
                                  (washIndex, slopeMeanAll, slopeMeanErrAll))
                self.fgcmPars.compQESysSlope[washIndex, :] = slopeMeanAll

        if doPlots:
            # Make the plots
            firstMJD = np.floor(np.min(self.fgcmPars.expMJD))

            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)

            colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']
            started = False

            if self.instrumentParsPerBand:
                parQESysIntercept = self.fgcmPars.parQESysIntercept.reshape((self.fgcmPars.nWashIntervals, self.fgcmPars.nBands))

            for i in xrange(self.fgcmPars.nWashIntervals):
                use, = np.where(self.fgcmPars.expWashIndex == i)
                washMJDRange = [np.min(self.fgcmPars.expMJD[use]), np.max(self.fgcmPars.expMJD[use])]

                if self.instrumentParsPerBand:
                    # Need to plot all of them one-by-one
                    for j in xrange(self.fgcmPars.nBands):
                        label = self.fgcmPars.bands[j] if not started else None
                        ax.plot(washMJDRange - firstMJD,
                                (washMJDRange - self.fgcmPars.washMJDs[i])*self.fgcmPars.compQESysSlope[i, j] +
                                parQESysIntercept[i, j], linestyle='--', color=colors[j], linewidth=2, label=label)
                else:
                    ax.plot(washMJDRange - firstMJD,
                            (washMJDRange - self.fgcmPars.washMJDs[i])*self.fgcmPars.compQESysSlope[i, 0] +
                            self.fgcmPars.parQESysIntercept[i, 0], 'r--', linewidth=3)
                started = True

            if self.instrumentParsPerBand:
                ax.legend(loc=3)

            ax.set_xlabel(r'$\mathrm{MJD}\ -\ %.0f$' % (firstMJD), fontsize=16)
            ax.set_ylabel('$2.5 \log_{10} (S^{\mathrm{optics}})$', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)

            # Make the vertical wash markers
            ylim = ax.get_ylim()
            for i in xrange(self.fgcmPars.nWashIntervals):
                ax.plot([self.fgcmPars.washMJDs[i] - firstMJD, self.fgcmPars.washMJDs[i]-firstMJD],
                        ylim, 'k--')

            fig.savefig('%s/%s_qesys_washes_%s.png' % (self.plotPath,
                                                       self.outfileBaseWithCycle,
                                                       name))

            plt.close(fig)

    def plotQeSysRefStars(self, name):
        """
        Plot reference stars (if available) and compare to QE sys slopes.

        Parameters
        ----------
        name: `str`
           name to give the files
        """

        return

        if not self.fgcmStars.hasRefStars:
            self.fgcmLog.info("No reference stars for QE sys plots.")
            return

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)

        objRefIDIndex = snmm.getArray(self.objRefIDIndexHandle)
        refMag = snmm.getArray(self.refMagHandle)

        goodStars = self.fgcmStars.getGoodStarIndices(checkMinObs=True)
        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=self.fgcmPars.expFlag, checkBadMag=True)

        # Take out the previous slope...
        obsMagStdGO = obsMagStd[goodObs]
        obsExpIndexGO = obsExpIndex[goodObs]

        deltaQESlopeGO = (self.fgcmPars.compQESysSlopeApplied[self.fgcmPars.expWashIndex[obsExpIndexGO], obsBandIndex[goodObs]] *
                          (self.fgcmPars.expMJD[obsExpIndexGO] -
                           self.fgcmPars.washMJDs[self.fgcmPars.expWashIndex[obsExpIndexGO]]))
        obsMagStdGO -= deltaQESlopeGO

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

        mjdGRO = self.fgcmPars.expMJD[obsExpIndexGO[goodRefObsGO]]
        minMjd = mjdGRO.min() - 5.0
        maxMjd = mjdGRO.max() + 5.0
        mjdGRO -= minMjd

        # And get a human-readable time out
        t = Time(minMjd, format='mjd')
        t.format = 'datetime'
        startString = '%04d-%02d-%02d' % (t.value.year, t.value.month, t.value.day)

        st = np.argsort(EGrayGRO)

        xMin = minMjd
        xMax = maxMjd
        yMin = EGrayGRO[st[int(0.001 * st.size)]]
        yMax = EGrayGRO[st[int(0.999 * st.size)]]

        # which wash dates are within the range...
        washInRange, = np.where((self.fgcmPars.washMJDs >= minMjd) &
                                (self.fgcmPars.washMJDs <= maxMjd))

        tempWashMJDs = self.fgcmPars.washMJDs
        tempWashMJDs = np.append(self.fgcmPars.washMJDs, xMax)

        for i, band in enumerate(self.fgcmPars.bands):

            use, = np.where(obsBandIndex[goodObs[goodRefObsGO]] == i)
            if use.size < 100:
                self.fgcmLog.info("Not enough good reference star information in %s band to make QE sys plots." % (band))
                continue

            fig = plt.figure(1, figsize=(8, 6))
            ax = fig.add_subplot(111)

            ax.hexbin(mjdGRO[use], EGrayGRO[use], bins='log', extent=[xMin, xMax, yMin, yMax])

            for washIndex in washInRange:
                ax.plot([self.fgcmPars.washMJDs[washIndex] - minMjd, self.fgcmPars.washMJDs[washIndex] - minMjd], [yMin, yMax], 'r--', linewidth=2)

                # And plot the corresponding slope line thing
                washMJDRange = np.clip([tempWashMJDs[washIndex], tempWashMJDs[washIndex + 1]], xMin, xMax)
                slope = self.fgcmPars.compQESysSlope[washIndex, i]
                if self.instrumentParsPerBand:
                    intercept = self.fgcmPars.parQESysIntercept[washIndex, i]
                else:
                    intercept = self.fgcmPars.parQESysIntercept[washIndex]

                ax.plot(washMJDRange - minMjd,
                        (washMJDRange -
                         self.fgcmPars.washMJDs[washIndex]) * slope +
                        intercept,
                        linestyle='-', color='b', linewidth=2)

            ax.set_title('%s band' % (band))
            ax.set_xlabel('Days since %s' % (startString), fontsize=14)
            ax.set_ylabel('m_ref - m_obs', fontsize=14)

            fig.savefig('%s/%s_qesys_refstars_%s_%s.png' % (self.plotPath,
                                                            self.outfileBaseWithCycle,
                                                            name, band))
            plt.close(fig)

