import numpy as np
import os
import sys
import esutil
import scipy.interpolate
import scipy.optimize

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

from .fgcmUtilities import retrievalFlagDict
from .fgcmUtilities import makeFigure, putButlerFigure
from matplotlib import colormaps


class FgcmRetrieveAtmosphere(object):
    """
    Class to convert retrieved integrals into atmosphere parameters.  Experimental.

    parameters
    ----------
    fgcmConfig: FgcmConfig
    fgcmLUT: FgcmLUT
    fgcmPars: FgcmPars
    """

    def __init__(self, fgcmConfig, fgcmLUT, fgcmPars, butlerQC=None, plotHandleDict=None):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.debug('Initializing fgcmRetrieveAtmosphere')

        self.fgcmLUT = fgcmLUT
        self.fgcmPars = fgcmPars

        self.butlerQC = butlerQC
        self.plotHandleDict = plotHandleDict

        self.pwvRetrievalSmoothBlock = fgcmConfig.pwvRetrievalSmoothBlock
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber
        self.minCCDPerExp = fgcmConfig.minCCDPerExp
        self.illegalValue = fgcmConfig.illegalValue
        self.useNightlyRetrievedPwv = fgcmConfig.useNightlyRetrievedPwv
        self.tauRetrievalMinCCDPerNight = fgcmConfig.tauRetrievalMinCCDPerNight
        self.quietMode = fgcmConfig.quietMode

    def r1ToPwv(self, fgcmRetrieval):
        """
        Convert R1 values to Pwv.  Experimental.

        parameters
        ----------
        fgcmRetrieval: FgcmRetrieval
        """

        if not self.quietMode:
            self.fgcmLog.info('Retrieving PWV Values...')

        parArray = self.fgcmPars.getParArray(fitterUnits=False)
        # Note that at this point it doesn't matter if we use the input or not
        self.fgcmPars.parsToExposures(retrievedInput=True)

        expIndexArray = np.repeat(np.arange(self.fgcmPars.nExp), self.fgcmPars.nCCD)
        ccdIndexArray = np.tile(np.arange(self.fgcmPars.nCCD),self.fgcmPars.nExp)

        r0 = snmm.getArray(fgcmRetrieval.r0Handle)
        r10 = snmm.getArray(fgcmRetrieval.r10Handle)

        # Reset values
        self.fgcmPars.compRetrievedLnPwvRaw[:] = self.illegalValue
        # set this to be the default
        self.fgcmPars.compRetrievedLnPwvRaw[:] = self.fgcmPars.lnPwvStd
        self.fgcmPars.compRetrievedLnPwv[:] = self.fgcmPars.lnPwvStd
        self.fgcmPars.compRetrievedLnPwvFlag[:] = retrievalFlagDict['EXPOSURE_STANDARD']

        # FIXME: check that there are actually z-band images...etc.
        # Also: allow retrieval from other bands, configurable.
        try:
            zBandIndex = self.fgcmPars.bands.index('z')
        except ValueError:
            self.fgcmLog.info("No z band, so no PWV retrieval.")
            return

        zUse,=np.where((self.fgcmPars.expBandIndex[expIndexArray] == zBandIndex) &
                       (self.fgcmPars.expFlag[expIndexArray] == 0) &
                       (np.abs(r0[expIndexArray, ccdIndexArray]) < 1000.0) &
                       (np.abs(r10[expIndexArray, ccdIndexArray]) < 1000.0))

        if zUse.size == 0:
            self.fgcmLog.info("Could not find any good z-band exposures for PWV retrieval.")
            return

        o3ZU = self.fgcmPars.expO3[expIndexArray[zUse]]
        lnTauZU = self.fgcmPars.expLnTau[expIndexArray[zUse]]
        alphaZU = self.fgcmPars.expAlpha[expIndexArray[zUse]]
        secZenithZU = 1./(np.sin(self.fgcmPars.expTelDec[expIndexArray[zUse]]) *
                          self.fgcmPars.sinLatitude +
                          np.cos(self.fgcmPars.expTelDec[expIndexArray[zUse]]) *
                          self.fgcmPars.cosLatitude *
                          np.cos(self.fgcmPars.expTelHA[expIndexArray[zUse]]))
        pmbZU = self.fgcmPars.expPmb[expIndexArray[zUse]]

        r1ZU = (r10[expIndexArray[zUse], ccdIndexArray[zUse]] *
                r0[expIndexArray[zUse], ccdIndexArray[zUse]])

        rLnPwvZU = np.zeros(zUse.size)

        lnPwvVals = self.fgcmLUT.lnPwv
        I1Arr = np.zeros((lnPwvVals.size, zUse.size))

        for i, lnPwv in enumerate(lnPwvVals):
            indices = self.fgcmLUT.getIndices(np.repeat(zBandIndex, zUse.size),
                                              np.repeat(lnPwv, zUse.size),
                                              o3ZU, lnTauZU, alphaZU, secZenithZU,
                                              ccdIndexArray[zUse], pmbZU)
            I1Arr[i, :] = self.fgcmLUT.computeI1(np.repeat(lnPwv, zUse.size),
                                                 o3ZU, lnTauZU, alphaZU, secZenithZU,
                                                 pmbZU, indices)

        for i in range(zUse.size):
            interpolator = scipy.interpolate.interp1d(I1Arr[:,i], lnPwvVals)
            rLnPwvZU[i] = interpolator(np.clip(r1ZU[i], I1Arr[:,i].min() + 0.0001,
                                             I1Arr[:,i].max() - 0.0001))


        # next, we median together each exposure...
        minExpIndex = np.min(expIndexArray[zUse])
        h, rev = esutil.stat.histogram(expIndexArray[zUse], min=minExpIndex, rev=True)

        gd, = np.where(h >= self.minCCDPerExp)

        rLnPwvStruct = np.zeros(gd.size, dtype=[('EXPINDEX', 'i4'),
                                                ('RLNPWV_MED', 'f8'),
                                                ('RLNPWV_SMOOTH', 'f8'),
                                                ('MJD', 'f8')])

        rLnPwvStruct['EXPINDEX'] = minExpIndex + gd

        for i in range(gd.size):
            i1a = rev[rev[gd[i]]:rev[gd[i]+1]]

            rLnPwvStruct['RLNPWV_MED'][i] = np.mean(rLnPwvZU[i1a])

        rLnPwvStruct['MJD'] = self.fgcmPars.expMJD[rLnPwvStruct['EXPINDEX']]

        # next, we do the median smoothing using pwvRetrievalSmoothBlock
        # self.pwvRetrievalSmoothBlock = fgcmConfig.pwvRetrievalSmoothBlock

        h, rev = esutil.stat.histogram(self.fgcmPars.expNightIndex[rLnPwvStruct['EXPINDEX']], rev=True)

        # we do this on any night that we have at least 1
        gd, = np.where(h > 0)

        for i in range(gd.size):
            i1a = rev[rev[gd[i]]:rev[gd[i]+1]]

            if (i1a.size == 1):
                rLnPwvStruct['RLNPWV_SMOOTH'][i1a[0]] = rLnPwvStruct['RLNPWV_MED'][i1a[0]]
            else:
                # these are not sorted out of expNightIndex
                st = np.argsort(rLnPwvStruct['MJD'][i1a])
                i1a = i1a[st]

                for j in range(i1a.size):
                    u = np.arange(np.clip(j - self.pwvRetrievalSmoothBlock/2, 0, i1a.size-1),
                                  np.clip(j + self.pwvRetrievalSmoothBlock/2, 0, i1a.size-1),
                                  1, dtype=np.int32)
                    rLnPwvStruct['RLNPWV_SMOOTH'][i1a[j]] = np.median(rLnPwvStruct['RLNPWV_MED'][i1a[u]])


        # Record these values and set a flag...
        self.fgcmPars.compRetrievedLnPwvRaw[rLnPwvStruct['EXPINDEX']] = rLnPwvStruct['RLNPWV_MED']
        self.fgcmPars.compRetrievedLnPwv[rLnPwvStruct['EXPINDEX']] = rLnPwvStruct['RLNPWV_SMOOTH']
        # unset standard and set that it's been retrieved
        self.fgcmPars.compRetrievedLnPwvFlag[rLnPwvStruct['EXPINDEX']] &= ~retrievalFlagDict['EXPOSURE_STANDARD']
        self.fgcmPars.compRetrievedLnPwvFlag[rLnPwvStruct['EXPINDEX']] |= retrievalFlagDict['EXPOSURE_RETRIEVED']

        # and finally we do interpolation to all the exposures...

        nightIndexWithLnPwv = np.unique(self.fgcmPars.expNightIndex[rLnPwvStruct['EXPINDEX']])

        a, b = esutil.numpy_util.match(nightIndexWithLnPwv, self.fgcmPars.expNightIndex)
        h, rev = esutil.stat.histogram(self.fgcmPars.expNightIndex[b], rev=True)

        gd, = np.where(h > 0)

        for i in range(gd.size):
            i1a = b[rev[rev[gd[i]]:rev[gd[i]+1]]]

            # sort by MJD
            st = np.argsort(self.fgcmPars.expMJD[i1a])
            i1a = i1a[st]

            # base interpolation on the ones with RPWV fits
            hasLnPwv, = np.where((self.fgcmPars.compRetrievedLnPwvFlag[i1a] & retrievalFlagDict['EXPOSURE_RETRIEVED']) > 0)

            noLnPwv, = np.where((self.fgcmPars.compRetrievedLnPwvFlag[i1a] & retrievalFlagDict['EXPOSURE_RETRIEVED']) == 0)

            # if we have < 3, we have a special branch -- night average
            if hasLnPwv.size < 3:
                self.fgcmPars.compRetrievedLnPwv[i1a[noLnPwv]] = np.mean(self.fgcmPars.compRetrievedLnPwv[i1a[hasLnPwv]])
            else:
                # regular interpolation

                interpolator = scipy.interpolate.interp1d(self.fgcmPars.expMJD[i1a[hasLnPwv]],
                                                          self.fgcmPars.compRetrievedLnPwv[i1a[hasLnPwv]],
                                                          bounds_error=False,
                                                          fill_value=(self.fgcmPars.compRetrievedLnPwv[i1a[hasLnPwv[0]]],
                                                                      self.fgcmPars.compRetrievedLnPwv[i1a[hasLnPwv[-1]]]))
                self.fgcmPars.compRetrievedLnPwv[i1a[noLnPwv]] = interpolator(self.fgcmPars.expMJD[i1a[noLnPwv]])
            # Flagging is the same
            self.fgcmPars.compRetrievedLnPwvFlag[i1a[noLnPwv]] &= ~retrievalFlagDict['EXPOSURE_STANDARD']
            self.fgcmPars.compRetrievedLnPwvFlag[i1a[noLnPwv]] |= retrievalFlagDict['EXPOSURE_INTERPOLATED']

        if self.plotPath is not None:
            # if there are fewer than ... 3000 do points, more than do hexbin
            hasPwv, = np.where((self.fgcmPars.compRetrievedLnPwvFlag & retrievalFlagDict['EXPOSURE_RETRIEVED']) > 0)

            #  RPWV_SMOOTH vs RPWV_SMOOTH_INPUT  (fgcmPars.compRetrievedPWVInput)
            #   (this checks for convergence on the actual measured values)
            fig = makeFigure(figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)

            hadPwv, = np.where((self.fgcmPars.compRetrievedLnPwvInput[hasPwv] != self.fgcmPars.pwvStd))

            if (hadPwv.size >= 3000):
                ax.hexbin(np.exp(self.fgcmPars.compRetrievedLnPwvInput[hasPwv[hadPwv]]),
                          np.exp(self.fgcmPars.compRetrievedLnPwv[hasPwv[hadPwv]]),
                          bins='log', cmap=colormaps.get_cmap("viridis"))
            else:
                ax.plot(np.exp(self.fgcmPars.compRetrievedLnPwvInput[hasPwv[hadPwv]]),
                        np.exp(self.fgcmPars.compRetrievedLnPwv[hasPwv[hadPwv]]), 'b.')
            plotRange = np.array([np.exp(self.fgcmPars.compRetrievedLnPwv[hasPwv].min())+0.001,
                                  np.exp(self.fgcmPars.compRetrievedLnPwv[hasPwv].max())-0.001])
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('RPWV_INPUT (mm)')
            ax.set_ylabel('RPWV (mm)')

            if self.butlerQC is not None:
                putButlerFigure(self.fgcmLog,
                                self.butlerQC,
                                self.plotHandleDict,
                                "RpwvVsRpwvInput",
                                self.cycleNumber,
                                fig)
            else:
                fig.savefig('%s/%s_rpwv_vs_rpwv_in.png' % (self.plotPath,
                                                           self.outfileBaseWithCycle))

            #  RPWV_RAW vs RPWV_SMOOTH (current calculation, just to make sure)
            fig = makeFigure(figsize=(8, 6))
            fig.clf()

            ax = fig.add_subplot(111)

            if hasPwv.size >= 3000:
                # we can use hexbin; this is arbitrary.
                ax.hexbin(np.exp(self.fgcmPars.compRetrievedLnPwv[hasPwv]),
                          np.exp(self.fgcmPars.compRetrievedLnPwvRaw[hasPwv]), bins='log',
                          cmap=colormaps.get_cmap("viridis"))
            else:
                ax.plot(np.exp(self.fgcmPars.compRetrievedLnPwv[hasPwv]),
                        np.exp(self.fgcmPars.compRetrievedLnPwvRaw[hasPwv]), 'b.')
            plotRange = np.array([np.exp(self.fgcmPars.compRetrievedLnPwvRaw[hasPwv].min())+0.001,
                                  np.exp(self.fgcmPars.compRetrievedLnPwvRaw[hasPwv].max())-0.001])
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('RPWV_SMOOTH (mm)')
            ax.set_ylabel('RPWV_RAW (mm)')

            if self.butlerQC is not None:
                putButlerFigure(self.fgcmLog,
                                self.butlerQC,
                                self.plotHandleDict,
                                "RpwvVsRpwvSmooth",
                                self.cycleNumber,
                                fig)
            else:
                fig.savefig('%s/%s_rpwv_vs_rpwv_smooth.png' % (self.plotPath,
                                                               self.outfileBaseWithCycle))

            fig = makeFigure(figsize=(8, 6))
            fig.clf()

            ax = fig.add_subplot(111)

            if hasPwv.size >= 3000:
                # we can use hexbin; this is arbitrary.
                ax.hexbin(np.exp(self.fgcmPars.compRetrievedLnPwv[hasPwv]),
                          np.exp(self.fgcmPars.expLnPwv[hasPwv]), bins='log',
                          cmap=colormaps.get_cmap("viridis"))
            else:
                ax.plot(np.exp(self.fgcmPars.compRetrievedLnPwv[hasPwv]),
                        np.exp(self.fgcmPars.expLnPwv[hasPwv]), 'b.')
            plotRange = np.array([np.exp(self.fgcmPars.compRetrievedLnPwv[hasPwv].min())+0.001,
                                  np.exp(self.fgcmPars.compRetrievedLnPwv[hasPwv].max())-0.001])
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('RPWV (mm)')
            ax.set_ylabel('PWV_MODEL (mm)')

            if self.butlerQC is not None:
                putButlerFigure(self.fgcmLog,
                                self.butlerQC,
                                self.plotHandleDict,
                                "ModelPwvVsRpwv",
                                self.cycleNumber,
                                fig)
            else:
                fig.savefig('%s/%s_pwv_vs_rpwv.png' % (self.plotPath,
                                                       self.outfileBaseWithCycle))

        # and we're done!  Everything is filled in!
        self.fgcmLog.debug('Done computing retrieved PWV values')

    def r0ToNightlyTau(self, fgcmRetrieval):
        """
        Convert R0 values to nightly Tau, using airmass.  Experimental.

        parameters
        ----------
        fgcmRetrieval: FgcmRetrieval
        """

        if not self.quietMode:
            self.fgcmLog.info('Retrieving Nightly Tau Values...')

        #bandIndex, = np.where(self.fgcmPars.bands == 'g')[0]

        r0 = snmm.getArray(fgcmRetrieval.r0Handle)

        expIndexArray = np.repeat(np.arange(self.fgcmPars.nExp), self.fgcmPars.nCCD)
        ccdIndexArray = np.tile(np.arange(self.fgcmPars.nCCD), self.fgcmPars.nExp)

        expSecZenith = 1./(np.sin(self.fgcmPars.expTelDec[expIndexArray]) *
                           self.fgcmPars.sinLatitude +
                           np.cos(self.fgcmPars.expTelDec[expIndexArray]) *
                           self.fgcmPars.cosLatitude *
                           np.cos(self.fgcmPars.expTelHA[expIndexArray]))

        # FIXME make this configurable?
        #tauBands = np.array(['g', 'r', 'i'])
        #nTauBands = tauBands.size
        tauBands = ['g', 'r', 'i']
        nTauBands = len(tauBands)

        tauRetrievedBands = np.zeros((nTauBands, self.fgcmPars.nCampaignNights)) + self.illegalValue

        for i in range(nTauBands):
            #bandIndex, = np.where(self.fgcmPars.bands == tauBands[i])[0]
            bandIndex = self.fgcmPars.bands.index(tauBands[i])

            tauScale = (self.fgcmLUT.lambdaStd[bandIndex] / self.fgcmLUT.lambdaNorm) ** (-self.fgcmLUT.alphaStd)

            use,=np.where((self.fgcmPars.expBandIndex[expIndexArray] == bandIndex) &
                          (self.fgcmPars.expFlag[expIndexArray] == 0) &
                          (r0[expIndexArray, ccdIndexArray] > 0.0))

            indices = self.fgcmLUT.getIndices(np.repeat(bandIndex, use.size),
                                              np.repeat(self.fgcmLUT.pwvStd, use.size),
                                              np.repeat(self.fgcmLUT.o3Std, use.size),
                                              np.repeat(np.log(0.00001), use.size),
                                              np.repeat(self.fgcmLUT.alphaStd, use.size),
                                              expSecZenith[use],
                                              ccdIndexArray[use],
                                              np.repeat(self.fgcmLUT.pmbStd, use.size))
            I0Ref = self.fgcmLUT.computeI0(np.repeat(self.fgcmLUT.pwvStd, use.size),
                                           np.repeat(self.fgcmLUT.o3Std, use.size),
                                           np.repeat(np.log(0.00001), use.size),
                                           np.repeat(self.fgcmLUT.alphaStd, use.size),
                                           expSecZenith[use],
                                           np.repeat(self.fgcmLUT.pmbStd, use.size),
                                           indices)

            extDelta = (-2.5*np.log10(r0[expIndexArray[use], ccdIndexArray[use]]) +
                         2.5*np.log10(I0Ref))

            h, rev = esutil.stat.histogram(self.fgcmPars.expNightIndex[expIndexArray[use]], min=0, rev=True)

            gd, = np.where(h > self.tauRetrievalMinCCDPerNight)
            if not self.quietMode:
                self.fgcmLog.info('Found %d nights to retrieve tau in %s band' %
                                  (gd.size, tauBands[i]))

            for j in range(gd.size):
                i1a = rev[rev[gd[j]]:rev[gd[j] + 1]]
                fit=np.polyfit(expSecZenith[use[i1a]], extDelta[i1a], 1.0)

                tauRetrievedBands[i, gd[j]] = fit[0] / tauScale

        # now loop over nights and take the average of good ones...
        self.fgcmPars.compRetrievedTauNight[:] = self.fgcmPars.tauStd
        for i in range(self.fgcmPars.nCampaignNights):
            u, = np.where(tauRetrievedBands[:, i] > self.illegalValue)
            if u.size > 0:
                self.fgcmPars.compRetrievedTauNight[i] = np.mean(tauRetrievedBands[u, i])

        # and clip to bounds
        self.fgcmPars.compRetrievedTauNight[:] = np.clip(self.fgcmPars.compRetrievedTauNight,
                                                         self.fgcmLUT.tau[0]+0.0001,
                                                         self.fgcmLUT.tau[-1]-0.0001)

        # And the plots
        # The nightlyTau code is not currently used because it is
        # not fully developed.  I have commented out the plots because
        # this code is not used/tested currently.  I am leaving
        # in the code in case it gets resurrected in the future.
        """
        if self.plotPath is not None:
            hasTau, = np.where((self.fgcmPars.compRetrievedTauNight != self.fgcmPars.tauStd) &
                               (self.fgcmPars.compRetrievedTauNightInput != self.fgcmPars.tauStd))
            fig = makeFigure(figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)

            ax.plot(self.fgcmPars.compRetrievedTauNightInput[hasTau],
                    self.fgcmPars.compRetrievedTauNight[hasTau], 'b.')
            plotRange = np.array([self.fgcmPars.compRetrievedTauNight.min() + 0.001,
                                  self.fgcmPars.compRetrievedTauNight.max() - 0.001])
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('RTAU_NIGHT_INPUT')
            ax.set_ylabel('RTAU_NIGHT')

            fig.savefig('%s/%s_rtaunight_vs_rtaunight_in.png' % (self.plotPath,
                                                                 self.outfileBaseWithCycle))

            hasTau, = np.where(self.fgcmPars.compRetrievedTauNight != self.fgcmPars.tauStd)

            fig = makeFigure(figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)

            ax.plot(np.exp(self.fgcmPars.parLnTauIntercept[hasTau]),
                    self.fgcmPars.compRetrievedTauNight[hasTau], 'b.')
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('TAU_INTERCEPT_MODEL')
            ax.set_ylabel('RTAU_NIGHT')

            fig.savefig('%s/%s_rtaunight_vs_tauint.png' % (self.plotPath,
                                                           self.outfileBaseWithCycle))
        """

    def expGrayToNightlyTau(self, fgcmGray):
        """
        Convert exposure gray to nightly Tau, using airmass.  Experimental.

        parameters
        ----------
        fgcmGray: FgcmGray
        """

        if not self.quietMode:
            self.fgcmLog.info('Retrieving nightly Tau values...')

        expGray = snmm.getArray(fgcmGray.expGrayHandle)

        expIndexArray = np.repeat(np.arange(self.fgcmPars.nExp), self.fgcmPars.nCCD)
        ccdIndexArray = np.tile(np.arange(self.fgcmPars.nCCD), self.fgcmPars.nExp)

        expSecZenith = 1./(np.sin(self.fgcmPars.expTelDec[expIndexArray]) *
                           self.fgcmPars.sinLatitude +
                           np.cos(self.fgcmPars.expTelDec[expIndexArray]) *
                           self.fgcmPars.cosLatitude *
                           np.cos(self.fgcmPars.expTelHA[expIndexArray]))

        lutIndices = self.fgcmLUT.getIndices(self.fgcmPars.expBandIndex[expIndexArray],
                                             self.fgcmPars.expLnPwv[expIndexArray],
                                             self.fgcmPars.expO3[expIndexArray],
                                             self.fgcmPars.expLnTau[expIndexArray],
                                             self.fgcmPars.expAlpha[expIndexArray],
                                             expSecZenith,
                                             ccdIndexArray,
                                             self.fgcmPars.expPmb[expIndexArray])
        i0 = self.fgcmLUT.computeI0(self.fgcmPars.expLnPwv[expIndexArray],
                                    self.fgcmPars.expO3[expIndexArray],
                                    self.fgcmPars.expLnTau[expIndexArray],
                                    self.fgcmPars.expAlpha[expIndexArray],
                                    expSecZenith,
                                    self.fgcmPars.expPmb[expIndexArray],
                                    lutIndices)

        r0Gray = np.zeros(expIndexArray.size)
        ok, = np.where(np.isfinite(expGray[expIndexArray]) &
                       (np.abs(expGray[expIndexArray]) < 1.0) &
                       (self.fgcmPars.expFlag[expIndexArray] == 0))
        r0Gray[ok] = 10.**((expGray[expIndexArray[ok]]/2.5)) * i0[ok]

        # FIXME make configurable
        tauBands = np.array(['g', 'r', 'i'])
        nTauBands = tauBands.size

        def slopeFunc(X, slope):
            return slope*X

        tauRetrievedBands = np.zeros((tauBands.size, self.fgcmPars.nCampaignNights)) + self.illegalValue
        tauModelBands = np.zeros_like(tauRetrievedBands)

        for i in range(nTauBands):
            #bandIndex, = np.where(self.fgcmPars.bands == tauBands[i])[0]
            bandIndex = self.fgcmPars.bands.index(tauBands[i])

            # investigate
            tauScale = (self.fgcmLUT.lambdaStd[bandIndex] / self.fgcmLUT.lambdaNorm) ** (-self.fgcmLUT.alphaStd)

            use,=np.where((self.fgcmPars.expBandIndex[expIndexArray] == bandIndex) &
                          (self.fgcmPars.expFlag[expIndexArray] == 0) &
                          (r0Gray > 0.0))

            indices = self.fgcmLUT.getIndices(np.repeat(bandIndex, use.size),
                                              self.fgcmPars.expLnPwv[expIndexArray[use]],
                                              self.fgcmPars.expO3[expIndexArray[use]],
                                              np.repeat(np.log(0.00001), use.size),
                                              #np.repeat(self.fgcmLUT.alphaStd, use.size),
                                              self.fgcmPars.expAlpha[expIndexArray[use]],
                                              expSecZenith[use],
                                              ccdIndexArray[use],
                                              self.fgcmPars.expPmb[expIndexArray[use]])
            I0Ref = self.fgcmLUT.computeI0(self.fgcmPars.expLnPwv[expIndexArray[use]],
                                           self.fgcmPars.expO3[expIndexArray[use]],
                                           np.repeat(np.log(0.00001), use.size),
                                           #np.repeat(self.fgcmLUT.alphaStd, use.size),
                                           self.fgcmPars.expAlpha[expIndexArray[use]],
                                           expSecZenith[use],
                                           self.fgcmPars.expPmb[expIndexArray[use]],
                                           indices)

            extDelta = (-2.5*np.log10(r0Gray[use]) +
                         2.5*np.log10(I0Ref))

            h, rev = esutil.stat.histogram(self.fgcmPars.expNightIndex[expIndexArray[use]], min=0, rev=True)

            gd, = np.where(h > self.tauRetrievalMinCCDPerNight)
            if not self.quietMode:
                self.fgcmLog.info('Found %d nights to retrieve tau in %s band' %
                                  (gd.size, tauBands[i]))

            for j in range(gd.size):
                i1a = rev[rev[gd[j]]:rev[gd[j] + 1]]
                fit, cov = scipy.optimize.curve_fit(slopeFunc, expSecZenith[use[i1a]],
                                                    extDelta[i1a])
                tauRetrievedBands[i, gd[j]] = fit[0] / tauScale

                tauModelBands[i, gd[j]] = np.exp(np.mean(self.fgcmPars.expLnTau[expIndexArray[use[i1a]]]))

        # now loop over nights and take the average of good ones...
        self.fgcmPars.compRetrievedTauNight[:] = self.fgcmPars.tauStd
        modelTauNight = np.zeros_like(self.fgcmPars.compRetrievedTauNight)
        for i in range(self.fgcmPars.nCampaignNights):
            u, = np.where(tauRetrievedBands[:, i] > self.illegalValue)
            if u.size > 0:
                self.fgcmPars.compRetrievedTauNight[i] = np.mean(tauRetrievedBands[u, i])
                modelTauNight[i] = np.mean(tauModelBands[u, i])

        # and clip to bounds
        self.fgcmPars.compRetrievedTauNight[:] = np.clip(self.fgcmPars.compRetrievedTauNight,
                                                         self.fgcmLUT.tau[0]+0.0001,
                                                         self.fgcmLUT.tau[-1]-0.0001)

        # And the plots
        """
        if self.plotPath is not None:
            hasTau, = np.where((self.fgcmPars.compRetrievedTauNight != self.fgcmPars.tauStd) &
                               (self.fgcmPars.compRetrievedTauNightInput != self.fgcmPars.tauStd))
            fig = makeFigure(figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)

            ax.plot(self.fgcmPars.compRetrievedTauNightInput[hasTau],
                    self.fgcmPars.compRetrievedTauNight[hasTau], 'b.')
            plotRange = np.array([self.fgcmPars.compRetrievedTauNight.min() + 0.001,
                                  self.fgcmPars.compRetrievedTauNight.max() - 0.001])
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('RTAU_NIGHT_INPUT')
            ax.set_ylabel('RTAU_NIGHT')

            fig.savefig('%s/%s_rtaunight_vs_rtaunight_in.png' % (self.plotPath,
                                                                 self.outfileBaseWithCycle))


            hasTau, = np.where(self.fgcmPars.compRetrievedTauNight != self.fgcmPars.tauStd)

            fig = makeFigure(figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)

            ax.plot(modelTauNight[hasTau],
                    self.fgcmPars.compRetrievedTauNight[hasTau], 'b.')
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('TAU_MEAN_MODEL')
            ax.set_ylabel('RTAU_NIGHT')

            fig.savefig('%s/%s_rtaunight_vs_tauint.png' % (self.plotPath,
                                                           self.outfileBaseWithCycle))
        """

