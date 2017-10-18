from __future__ import print_function

import numpy as np
import os
import sys
import esutil
import scipy.interpolate

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

        self.retrievePWV = fgcmConfig.retrievePWV
        self.pwvRetrievalSmoothBlock = fgcmConfig.pwvRetrievalSmoothBlock
        self.plotPath = fgcmConfig.plotPath
        self.minCCDPerExp = fgcmConfig.minCCDPerExp
        self.illegalValue = fgcmConfig.illegalValue

    def r1ToPWV(self, fgcmRetrieval):
        """
        """

        if (not self.retrievePWV):
            raise RuntimeError("Calling r0ToAtmosphere when not using retrieval ... but maybe this should be allowed (later)")

        parArray = self.fgcmPars.getParArray(fitterUnits=False)
        self.fgcmPars.parsToExposures()

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
        lnTauZU = np.log(self.fgcmPars.expTau[expIndexArray[zUse]])
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
        self.fgcmPars.compRetrievedPWVRaw = rPWVStruct['RPWV'][rPWVStruct['EXPINDEX']]
        self.fgcmPars.compRetrievedPWV = rPWVStruct['RPWV_SMOOTH'][rPWVStruct['EXPINDEX']]
        # unset standard and set that it's been retrieved
        self.fgcmPars.compRetrievedPWVFlag[rPWVStruct['EXPINDEX']] &= ~retrievalFlagDict['EXPOSURE_STANDARD']
        self.fgcmPars.compRetrievedPWVFlag[rPWVStruct['EXPINDEX']] |= retrievalFlagDict['EXPOSURE_RETRIEVED']

        # and finally we do interpolation to all the exposures...

        nightIndexWithPWV = np.unique(fgcmPars.expNightIndex[rPWVStruct['EXPINDEX']])

        a, b = esutil.numpy_util.match(nightIndexWithPWV, fgcmPars.expNightIndex)
        h, rev = esutil.stat.histogram(fgcmPars.expNightIndex[b], rev=True)

        gd, = np.where(h > 0)

        for i in xrange(gd.size):
            i1a = b[rev[rev[gd[i]]:rev[gd[i]+1]]]

            # sort by MJD
            st = np.argsort(fgcmPars.expMJD[i1a])
            i1a = i1a[st]

            # base interpolation on the ones with RPWV fits
            hasPWV, = np.where((fgcmPars.compRetrievedPWVFlag[i1a] & retrievalFlagDict['EXPOSURE_RETRIEVED']) > 0)

            interpolator = scipy.interpolate.interp1d(fgcmPars.expMJD[i1a[hasPWV]],
                                                      fgcmPars.compRetrievedPWV[i1a[hasPWV]],
                                                      bounds_error=False,
                                                      fill_value=(fgcmPars.compRetrievedPWV[i1a[hasPWV[0]]],
                                                                  fgcmPars.compRetrievedPWV[i1a[hasPWV[-1]]]))
            noPWV, = np.where((fgcmPars.compRetrievedPWVFlag[i1a] & retrievalFlagDict['EXPOSURE_RETRIEVED']) == 0)
            fgcmPars.compRetrievedPWV[i1a[noPWV]] = interpolator(fgcmPars.expMJD[i1a[noPWV]])
            fgcmPars.compRetrievedPWVFlag[i1a[noPWV]] &= ~retrievalFlagDict['EXPOSURE_STANDARD']
            fgcmPars.compRetrievedPWVFlag[i1a[noPWV]] |= retrievalFlagDict['EXPOSURE_INTERPOLATED']

        # Unsure what plots to put here...
        # and we're done!  Everything is filled in!
        self.fgcmLog.log('INFO','Done computing retrieved PWV values')
