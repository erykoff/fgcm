from __future__ import print_function

import numpy as np
import os
import sys
import esutil
import scipy.interpolate
import scipy.optimize

import matplotlib.pyplot as plt

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

from fgcmUtilities import retrievalFlagDict

class FgcmRetrieveAtmosphere(object):
    """
    """

    def __init__(self, fgcmConfig, fgcmLUT, fgcmPars):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.log('INFO','Initializing fgcmRetrieveAtmosphere')

        self.fgcmLUT = fgcmLUT
        self.fgcmPars = fgcmPars

        self.pwvRetrievalSmoothBlock = fgcmConfig.pwvRetrievalSmoothBlock
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.minCCDPerExp = fgcmConfig.minCCDPerExp
        self.illegalValue = fgcmConfig.illegalValue
        self.useNightlyRetrievedPWV = fgcmConfig.useNightlyRetrievedPWV
        self.tauRetrievalMinCCDPerNight = fgcmConfig.tauRetrievalMinCCDPerNight
        self.ccdOffsets = fgcmConfig.ccdOffsets

    def r1ToPWV(self, fgcmRetrieval, doPlots=True):
        """
        """
        self.fgcmLog.log('INFO','Retrieving PWV Values...')

        parArray = self.fgcmPars.getParArray(fitterUnits=False)
        # Note that at this point it doesn't matter if we use the input or not
        self.fgcmPars.parsToExposures(retrievedInput=True)

        expIndexArray = np.repeat(np.arange(self.fgcmPars.nExp), self.fgcmPars.nCCD)
        ccdIndexArray = np.tile(np.arange(self.fgcmPars.nCCD),self.fgcmPars.nExp)

        r0 = snmm.getArray(fgcmRetrieval.r0Handle)
        r10 = snmm.getArray(fgcmRetrieval.r10Handle)

        # FIXME: check that there are actually z-band images...etc.
        zBandIndex, = np.where(self.fgcmPars.bands == 'z')[0]
        zUse,=np.where((self.fgcmPars.expBandIndex[expIndexArray] == zBandIndex) &
                       (self.fgcmPars.expFlag[expIndexArray] == 0) &
                       (np.abs(r0[expIndexArray, ccdIndexArray]) < 1000.0) &
                       (np.abs(r10[expIndexArray, ccdIndexArray]) < 1000.0))

        o3ZU = self.fgcmPars.expO3[expIndexArray[zUse]]
        #lnTauZU = np.log(self.fgcmPars.expTau[expIndexArray[zUse]])
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

        rPWVZU = np.zeros(zUse.size)

        pwvVals = self.fgcmLUT.pwv
        I1Arr = np.zeros((pwvVals.size, zUse.size))

        for i,pwv in enumerate(pwvVals):
            indices = self.fgcmLUT.getIndices(np.repeat(zBandIndex, zUse.size),
                                              np.repeat(pwv, zUse.size),
                                              o3ZU, lnTauZU, alphaZU, secZenithZU,
                                              ccdIndexArray[zUse], pmbZU)
            I1Arr[i, :] = self.fgcmLUT.computeI1(np.repeat(pwv, zUse.size),
                                                 o3ZU, lnTauZU, alphaZU, secZenithZU,
                                                 pmbZU, indices)

        for i in xrange(zUse.size):
            interpolator = scipy.interpolate.interp1d(I1Arr[:,i], pwvVals)
            rPWVZU[i] = interpolator(np.clip(r1ZU[i], I1Arr[:,i].min() + 0.0001,
                                             I1Arr[:,i].max() - 0.0001))


        # next, we median together each exposure...
        minExpIndex = np.min(expIndexArray[zUse])
        h, rev = esutil.stat.histogram(expIndexArray[zUse], min=minExpIndex, rev=True)

        gd, = np.where(h >= self.minCCDPerExp)

        rPWVStruct = np.zeros(gd.size, dtype=[('EXPINDEX', 'i4'),
                                              ('RPWV_MED', 'f8'),
                                              ('RPWV_SMOOTH', 'f8'),
                                              ('MJD', 'f8')])

        rPWVStruct['EXPINDEX'] = minExpIndex + gd

        for i in xrange(gd.size):
            i1a = rev[rev[gd[i]]:rev[gd[i]+1]]

            rPWVStruct['RPWV_MED'][i] = np.mean(rPWVZU[i1a])

        rPWVStruct['MJD'] = self.fgcmPars.expMJD[rPWVStruct['EXPINDEX']]

        # next, we do the median smoothing using pwvRetrievalSmoothBlock
        # self.pwvRetrievalSmoothBlock = fgcmConfig.pwvRetrievalSmoothBlock

        h, rev = esutil.stat.histogram(self.fgcmPars.expNightIndex[rPWVStruct['EXPINDEX']],
                                       rev=True)

        # we do this on any night that we have at least 1
        gd, = np.where(h > 0)

        for i in xrange(gd.size):
            i1a = rev[rev[gd[i]]:rev[gd[i]+1]]

            if (i1a.size == 1):
                rPWVStruct['RPWV_SMOOTH'][i1a[0]] = rPWVStruct['RPWV_MED'][i1a[0]]
            else:
                # these are not sorted out of expNightIndex
                st = np.argsort(rPWVStruct['MJD'][i1a])
                i1a = i1a[st]

                for j in xrange(i1a.size):
                    u = np.arange(np.clip(j - self.pwvRetrievalSmoothBlock/2, 0, i1a.size-1),
                                  np.clip(j + self.pwvRetrievalSmoothBlock/2, 0, i1a.size-1),
                                  1, dtype=np.int32)
                    rPWVStruct['RPWV_SMOOTH'][i1a[j]] = np.median(rPWVStruct['RPWV_MED'][i1a[u]])

        # Reset values
        self.fgcmPars.compRetrievedPWVRaw[:] = self.illegalValue
        # set this to be the default
        self.fgcmPars.compRetrievedPWV[:] = self.fgcmPars.pwvStd
        self.fgcmPars.compRetrievedPWVFlag[:] = retrievalFlagDict['EXPOSURE_STANDARD']

        # Record these values and set a flag...
        self.fgcmPars.compRetrievedPWVRaw[rPWVStruct['EXPINDEX']] = rPWVStruct['RPWV_MED']
        self.fgcmPars.compRetrievedPWV[rPWVStruct['EXPINDEX']] = rPWVStruct['RPWV_SMOOTH']
        # unset standard and set that it's been retrieved
        self.fgcmPars.compRetrievedPWVFlag[rPWVStruct['EXPINDEX']] &= ~retrievalFlagDict['EXPOSURE_STANDARD']
        self.fgcmPars.compRetrievedPWVFlag[rPWVStruct['EXPINDEX']] |= retrievalFlagDict['EXPOSURE_RETRIEVED']

        # and finally we do interpolation to all the exposures...

        nightIndexWithPWV = np.unique(self.fgcmPars.expNightIndex[rPWVStruct['EXPINDEX']])

        a, b = esutil.numpy_util.match(nightIndexWithPWV, self.fgcmPars.expNightIndex)
        h, rev = esutil.stat.histogram(self.fgcmPars.expNightIndex[b], rev=True)

        gd, = np.where(h > 0)

        for i in xrange(gd.size):
            i1a = b[rev[rev[gd[i]]:rev[gd[i]+1]]]

            # sort by MJD
            st = np.argsort(self.fgcmPars.expMJD[i1a])
            i1a = i1a[st]

            # base interpolation on the ones with RPWV fits
            hasPWV, = np.where((self.fgcmPars.compRetrievedPWVFlag[i1a] & retrievalFlagDict['EXPOSURE_RETRIEVED']) > 0)

            noPWV, = np.where((self.fgcmPars.compRetrievedPWVFlag[i1a] & retrievalFlagDict['EXPOSURE_RETRIEVED']) == 0)

            # if we have < 3, we have a special branch -- night average
            if hasPWV.size < 3:
                self.fgcmPars.compRetrievedPWV[i1a[noPWV]] = np.mean(self.fgcmPars.compRetrievedPWV[i1a[hasPWV]])
            else:
                # regular interpolation

                interpolator = scipy.interpolate.interp1d(self.fgcmPars.expMJD[i1a[hasPWV]],
                                                          self.fgcmPars.compRetrievedPWV[i1a[hasPWV]],
                                                          bounds_error=False,
                                                          fill_value=(self.fgcmPars.compRetrievedPWV[i1a[hasPWV[0]]],
                                                                      self.fgcmPars.compRetrievedPWV[i1a[hasPWV[-1]]]))
                self.fgcmPars.compRetrievedPWV[i1a[noPWV]] = interpolator(self.fgcmPars.expMJD[i1a[noPWV]])
            # Flagging is the same
            self.fgcmPars.compRetrievedPWVFlag[i1a[noPWV]] &= ~retrievalFlagDict['EXPOSURE_STANDARD']
            self.fgcmPars.compRetrievedPWVFlag[i1a[noPWV]] |= retrievalFlagDict['EXPOSURE_INTERPOLATED']

        if doPlots:
            # if there are fewer than ... 3000 do points, more than do hexbin

            hasPWV, = np.where((self.fgcmPars.compRetrievedPWVFlag & retrievalFlagDict['EXPOSURE_RETRIEVED']) > 0)

            #  RPWV_SMOOTH vs RPWV_SMOOTH_INPUT  (fgcmPars.compRetrievedPWVInput)
            #   (this checks for convergence on the actual measured values)
            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)

            hadPWV, = np.where((self.fgcmPars.compRetrievedPWVInput[hasPWV] != self.fgcmPars.pwvStd))

            if (hadPWV.size >= 3000):
                ax.hexbin(self.fgcmPars.compRetrievedPWVInput[hasPWV[hadPWV]],
                          self.fgcmPars.compRetrievedPWV[hasPWV[hadPWV]],
                          bins='log')
            else:
                ax.plot(self.fgcmPars.compRetrievedPWVInput[hasPWV[hadPWV]],
                          self.fgcmPars.compRetrievedPWV[hasPWV[hadPWV]], 'b.')
            plotRange = np.array([self.fgcmPars.compRetrievedPWV[hasPWV].min()+0.001,
                                  self.fgcmPars.compRetrievedPWV[hasPWV].max()-0.001])
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('RPWV_INPUT')
            ax.set_ylabel('RPWV')

            fig.savefig('%s/%s_rpwv_vs_rpwv_in.png' % (self.plotPath,
                                                       self.outfileBaseWithCycle))

            #  RPWV_RAW vs RPWV_SMOOTH (current calculation, just to make sure)
            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()

            ax = fig.add_subplot(111)

            if hasPWV.size >= 3000:
                # we can use hexbin; this is arbitrary.
                ax.hexbin(self.fgcmPars.compRetrievedPWV[hasPWV],
                          self.fgcmPars.compRetrievedPWVRaw[hasPWV], bins='log')
            else:
                ax.plot(self.fgcmPars.compRetrievedPWV[hasPWV],
                          self.fgcmPars.compRetrievedPWVRaw[hasPWV], 'b.')
            plotRange = np.array([self.fgcmPars.compRetrievedPWVRaw[hasPWV].min()+0.001,
                                  self.fgcmPars.compRetrievedPWVRaw[hasPWV].max()-0.001])
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('RPWV_SMOOTH')
            ax.set_ylabel('RPWV_RAW')

            fig.savefig('%s/%s_rpwv_vs_rpwv_smooth.png' % (self.plotPath,
                                                           self.outfileBaseWithCycle))

            #  PWV vs RPWV_SMOOTH
            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()

            ax = fig.add_subplot(111)

            if hasPWV.size >= 3000:
                # we can use hexbin; this is arbitrary.
                ax.hexbin(self.fgcmPars.compRetrievedPWV[hasPWV],
                          self.fgcmPars.expPWV[hasPWV], bins='log')
            else:
                ax.plot(self.fgcmPars.compRetrievedPWV[hasPWV],
                        self.fgcmPars.expPWV[hasPWV], 'b.')
            plotRange = np.array([self.fgcmPars.compRetrievedPWV[hasPWV].min()+0.001,
                                  self.fgcmPars.compRetrievedPWV[hasPWV].max()-0.001])
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('RPWV')
            ax.set_ylabel('PWV_MODEL')

            fig.savefig('%s/%s_pwv_vs_rpwv.png' % (self.plotPath,
                                                   self.outfileBaseWithCycle))


            #  PWV vs RPWV_SCALED
            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()

            ax = fig.add_subplot(111)

            #scaledRetrievedPWV = (self.fgcmPars.compRetrievedPWV * self.fgcmPars.parRetrievedPWVScale +
            #                      self.fgcmPars.parRetrievedPWVOffset)
            if self.useNightlyRetrievedPWV:
                scaledRetrievedPWV = (self.fgcmPars.parRetrievedPWVNightlyOffset[self.fgcmPars.expNightIndex] +
                                      self.fgcmPars.parRetrievedPWVScale *
                                      self.fgcmPars.compRetrievedPWV)
            else:
                scaledRetrievedPWV = (self.fgcmPars.parRetrievedPWVOffset +
                                      self.fgcmPars.parRetrievedPWVScale *
                                      self.fgcmPars.compRetrievedPWV)

            if hasPWV.size >= 3000:
                # we can use hexbin; this is arbitrary.
                ax.hexbin(scaledRetrievedPWV[hasPWV],
                          self.fgcmPars.expPWV[hasPWV], bins='log')
            else:
                ax.plot(scaledRetrievedPWV[hasPWV],
                        self.fgcmPars.expPWV[hasPWV], 'b.')
            plotRange = np.array([self.fgcmPars.compRetrievedPWV[hasPWV].min()+0.001,
                                  self.fgcmPars.compRetrievedPWV[hasPWV].max()-0.001])
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('RPWV_SCALED')
            ax.set_ylabel('PWV_MODEL')


            fig.savefig('%s/%s_pwv_vs_rpwv_scaled.png' % (self.plotPath,
                                                         self.outfileBaseWithCycle))


        # and we're done!  Everything is filled in!
        self.fgcmLog.log('INFO','Done computing retrieved PWV values')

    def r0ToNightlyTau(self, fgcmRetrieval, doPlots=True):
        """
        """
        self.fgcmLog.log('INFO','Retrieving Nightly Tau Values...')

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
        tauBands = np.array(['g', 'r', 'i'])
        nTauBands = tauBands.size

        tauRetrievedBands = np.zeros((tauBands.size, self.fgcmPars.nCampaignNights)) + self.illegalValue

        for i in xrange(nTauBands):
            bandIndex, = np.where(self.fgcmPars.bands == tauBands[i])[0]

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

            h,rev = esutil.stat.histogram(self.fgcmPars.expNightIndex[expIndexArray[use]],
                                          rev=True, min=0)

            gd, = np.where(h > self.tauRetrievalMinCCDPerNight)
            self.fgcmLog.log('INFO', 'Found %d nights to retrieve tau in %s band' %
                             (gd.size, tauBands[i]))

            for j in xrange(gd.size):
                i1a = rev[rev[gd[j]]:rev[gd[j] + 1]]
                fit=np.polyfit(expSecZenith[use[i1a]], extDelta[i1a], 1.0)

                tauRetrievedBands[i, gd[j]] = fit[0] / tauScale

        # now loop over nights and take the average of good ones...
        self.fgcmPars.compRetrievedTauNight[:] = self.fgcmPars.tauStd
        for i in xrange(self.fgcmPars.nCampaignNights):
            u, = np.where(tauRetrievedBands[:, i] > self.illegalValue)
            if u.size > 0:
                self.fgcmPars.compRetrievedTauNight[i] = np.mean(tauRetrievedBands[u, i])

        # and clip to bounds
        self.fgcmPars.compRetrievedTauNight[:] = np.clip(self.fgcmPars.compRetrievedTauNight,
                                                         self.fgcmLUT.tau[0]+0.0001,
                                                         self.fgcmLUT.tau[-1]-0.0001)

        # And the plots
        if doPlots:
            hasTau, = np.where((self.fgcmPars.compRetrievedTauNight != self.fgcmPars.tauStd) &
                               (self.fgcmPars.compRetrievedTauNightInput != self.fgcmPars.tauStd))
            fig = plt.figure(1, figsize=(8, 6))
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

            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)

            ax.plot(np.exp(self.fgcmPars.parLnTauIntercept[hasTau]),
                    self.fgcmPars.compRetrievedTauNight[hasTau], 'b.')
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('TAU_INTERCEPT_MODEL')
            ax.set_ylabel('RTAU_NIGHT')

            fig.savefig('%s/%s_rtaunight_vs_tauint.png' % (self.plotPath,
                                                           self.outfileBaseWithCycle))


    def expGrayToNightlyTau(self, fgcmGray, doPlots=True):
        """
        """

        self.fgcmLog.log('INFO', 'Retrieving nightly Tau values...')

        expGray = snmm.getArray(fgcmGray.expGrayHandle)

        expIndexArray = np.repeat(np.arange(self.fgcmPars.nExp), self.fgcmPars.nCCD)
        ccdIndexArray = np.tile(np.arange(self.fgcmPars.nCCD), self.fgcmPars.nExp)

        expSecZenith = 1./(np.sin(self.fgcmPars.expTelDec[expIndexArray]) *
                           self.fgcmPars.sinLatitude +
                           np.cos(self.fgcmPars.expTelDec[expIndexArray]) *
                           self.fgcmPars.cosLatitude *
                           np.cos(self.fgcmPars.expTelHA[expIndexArray]))

        lutIndices = self.fgcmLUT.getIndices(self.fgcmPars.expBandIndex[expIndexArray],
                                             self.fgcmPars.expPWV[expIndexArray],
                                             self.fgcmPars.expO3[expIndexArray],
                                             self.fgcmPars.expLnTau[expIndexArray],
                                             self.fgcmPars.expAlpha[expIndexArray],
                                             expSecZenith,
                                             ccdIndexArray,
                                             self.fgcmPars.expPmb[expIndexArray])
        i0 = self.fgcmLUT.computeI0(self.fgcmPars.expPWV[expIndexArray],
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

        for i in xrange(nTauBands):
            bandIndex, = np.where(self.fgcmPars.bands == tauBands[i])[0]

            # investigate
            tauScale = (self.fgcmLUT.lambdaStd[bandIndex] / self.fgcmLUT.lambdaNorm) ** (-self.fgcmLUT.alphaStd)

            use,=np.where((self.fgcmPars.expBandIndex[expIndexArray] == bandIndex) &
                          (self.fgcmPars.expFlag[expIndexArray] == 0) &
                          (r0Gray > 0.0))

            indices = self.fgcmLUT.getIndices(np.repeat(bandIndex, use.size),
                                              self.fgcmPars.expPWV[expIndexArray[use]],
                                              self.fgcmPars.expO3[expIndexArray[use]],
                                              np.repeat(np.log(0.00001), use.size),
                                              #np.repeat(self.fgcmLUT.alphaStd, use.size),
                                              self.fgcmPars.expAlpha[expIndexArray[use]],
                                              expSecZenith[use],
                                              ccdIndexArray[use],
                                              self.fgcmPars.expPmb[expIndexArray[use]])
            I0Ref = self.fgcmLUT.computeI0(self.fgcmPars.expPWV[expIndexArray[use]],
                                           self.fgcmPars.expO3[expIndexArray[use]],
                                           np.repeat(np.log(0.00001), use.size),
                                           #np.repeat(self.fgcmLUT.alphaStd, use.size),
                                           self.fgcmPars.expAlpha[expIndexArray[use]],
                                           expSecZenith[use],
                                           self.fgcmPars.expPmb[expIndexArray[use]],
                                           indices)

            extDelta = (-2.5*np.log10(r0Gray[use]) +
                         2.5*np.log10(I0Ref))

            h,rev = esutil.stat.histogram(self.fgcmPars.expNightIndex[expIndexArray[use]],
                                          rev=True, min=0)

            gd, = np.where(h > self.tauRetrievalMinCCDPerNight)
            self.fgcmLog.log('INFO', 'Found %d nights to retrieve tau in %s band' %
                             (gd.size, tauBands[i]))

            for j in xrange(gd.size):
                i1a = rev[rev[gd[j]]:rev[gd[j] + 1]]
                fit, cov = scipy.optimize.curve_fit(slopeFunc, expSecZenith[use[i1a]],
                                                    extDelta[i1a])
                tauRetrievedBands[i, gd[j]] = fit[0] / tauScale

                tauModelBands[i, gd[j]] = np.exp(np.mean(self.fgcmPars.expLnTau[expIndexArray[use[i1a]]]))

        # now loop over nights and take the average of good ones...
        self.fgcmPars.compRetrievedTauNight[:] = self.fgcmPars.tauStd
        modelTauNight = np.zeros_like(self.fgcmPars.compRetrievedTauNight)
        for i in xrange(self.fgcmPars.nCampaignNights):
            u, = np.where(tauRetrievedBands[:, i] > self.illegalValue)
            if u.size > 0:
                self.fgcmPars.compRetrievedTauNight[i] = np.mean(tauRetrievedBands[u, i])
                modelTauNight[i] = np.mean(tauModelBands[u, i])

        # and clip to bounds
        self.fgcmPars.compRetrievedTauNight[:] = np.clip(self.fgcmPars.compRetrievedTauNight,
                                                         self.fgcmLUT.tau[0]+0.0001,
                                                         self.fgcmLUT.tau[-1]-0.0001)

        # And the plots
        if doPlots:
            hasTau, = np.where((self.fgcmPars.compRetrievedTauNight != self.fgcmPars.tauStd) &
                               (self.fgcmPars.compRetrievedTauNightInput != self.fgcmPars.tauStd))
            fig = plt.figure(1, figsize=(8, 6))
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

            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)

            ax.plot(modelTauNight[hasTau],
                    self.fgcmPars.compRetrievedTauNight[hasTau], 'b.')
            ax.plot(plotRange, plotRange, 'r--')
            ax.set_xlabel('TAU_MEAN_MODEL')
            ax.set_ylabel('RTAU_NIGHT')

            fig.savefig('%s/%s_rtaunight_vs_tauint.png' % (self.plotPath,
                                                           self.outfileBaseWithCycle))

