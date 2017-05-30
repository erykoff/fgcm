from __future__ import print_function

import numpy as np
import os
import sys
import esutil
import time
import scipy.optimize

import matplotlib.pyplot as plt

from fgcmUtilities import gaussFunction
from fgcmUtilities import histoGauss


from sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmSigFgcm(object):
    """
    """

    def __init__(self,fgcmConfig,fgcmPars,fgcmStars):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.log('INFO','Initializing FgcmSigFgcm')

        # need fgcmPars because it has the sigFgcm
        self.fgcmPars = fgcmPars

        # need fgcmStars because it has the stars (duh)
        self.fgcmStars = fgcmStars

        # and config numbers
        self.sigFgcmMaxEGray = fgcmConfig.sigFgcmMaxEGray
        self.sigFgcmMaxErr = fgcmConfig.sigFgcmMaxErr
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber

    def computeSigFgcm(self,doPlots=True):
        """
        """

        if (not self.fgcmStars.magStdComputed):
            raise ValueError("Must run FgcmChisq to compute magStd before computeCCDAndExpGray")

        startTime = time.time()
        self.fgcmLog.log('INFO','Computing sigFgcm.')

        # input numbers
        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)

        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        # make sure we have enough obervations per band
        minObs = objNGoodObs[:,self.fgcmStars.bandRequiredIndex].min(axis=1)

        # select good stars...
        goodStars,=np.where((minObs >= self.fgcmStars.minPerBand) &
                            (objFlag == 0))

        # match the good stars to the observations
        _,goodObs = esutil.numpy_util.match(goodStars,
                                            obsObjIDIndex,
                                            presorted=True)

        # and make sure that we only use good observations from good exposures
        gd,=np.where((self.fgcmPars.expFlag[obsExpIndex[goodObs]] == 0) &
                     (obsFlag[goodObs] == 0))

        goodObs = goodObs[gd]

        # we need to compute E_gray == <mstd> - mstd for each observation
        # compute EGray, GO for Good Obs
        EGrayGO = (objMagStdMean[obsObjIDIndex[goodObs],obsBandIndex[goodObs]] -
                   obsMagStd[goodObs])
        # and need the error for Egray: sum in quadrature of individual and avg errs
        EGrayErr2GO = (objMagStdMeanErr[obsObjIDIndex[goodObs],obsBandIndex[goodObs]]**2. +
                       obsMagErr[goodObs]**2.)

        # now we can compute sigFgcm

        for bandIndex in xrange(self.fgcmStars.nBands):
            if (bandIndex in self.fgcmStars.bandRequiredIndex):
                sigUse,=np.where((np.abs(EGrayGO) < self.sigFgcmMaxEGray) &
                                 (EGrayErr2GO > 0.0) &
                                 (EGrayErr2GO < self.sigFgcmMaxErr**2.) &
                                 (EGrayGO != 0.0) &
                                 (obsBandIndex[goodObs] == bandIndex))
            else:
                sigUse,=np.where((np.abs(EGrayGO) < self.sigFgcmMaxEGray) &
                                 (EGrayErr2GO > 0.0) &
                                 (EGrayErr2GO < self.sigFgcmMaxErr**2.) &
                                 (EGrayGO != 0.0) &
                                 (obsBandIndex[goodObs] == bandIndex) &
                                 (objNGoodObs[obsObjIDIndex[goodObs],bandIndex] >=
                                  self.fgcmStars.minPerBand))

            if (sigUse.size == 0):
                self.fgcmLog.log('INFO','sigFGCM: No good observations in %s band.' %
                                 (self.fgcmPars.bands[bandIndex]))
                continue

            fig = plt.figure(1,figsize=(8,6))
            fig.clf()

            ax=fig.add_subplot(111)

            coeff = histoGauss(ax, EGrayGO[sigUse])

            self.fgcmPars.compSigFgcm[bandIndex] = np.sqrt(coeff[2]**2. -
                                                           np.median(EGrayErr2GO[sigUse]))

            if (not np.isfinite(self.fgcmPars.compSigFgcm[bandIndex])):
                self.fgcmLog.log('INFO',"Failed to compute sigFgcm (%s).  Setting to 0.05?" %
                                 (self.fgcmPars.bands[bandIndex]))
                self.fgcmPars.compSigFgcm[bandIndex] = 0.05

            self.fgcmLog.log('INFO',"sigFgcm (%s) = %.4f" % (
                    self.fgcmPars.bands[bandIndex],
                    self.fgcmPars.compSigFgcm[bandIndex]))

            if (not doPlots):
                continue

            ax.tick_params(axis='both',which='major',labelsize=14)

            text=r'$(%s)$' % (self.fgcmPars.bands[bandIndex]) + '\n' + \
                r'$\mathrm{Cycle\ %d}$' % (self.cycleNumber) + '\n' + \
                r'$\mu = %.5f$' % (coeff[1]) + '\n' + \
                r'$\sigma_\mathrm{tot} = %.4f$' % (coeff[2]) + '\n' + \
                r'$\sigma_\mathrm{FGCM} = %.4f$' % (self.fgcmPars.compSigFgcm[bandIndex])

            ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=16)
            ax.set_xlabel(r'$E^{\mathrm{gray}}$',fontsize=16)

            fig.savefig('%s/%s_sigfgcm_%s.png' % (self.plotPath,
                                                  self.outfileBaseWithCycle,
                                                  self.fgcmPars.bands[bandIndex]))


        self.fgcmLog.log('INFO','Done computing sigFgcm in %.2f sec.' %
                         (time.time() - startTime))



