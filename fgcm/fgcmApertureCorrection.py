from __future__ import print_function

import numpy as np
import os
import sys
import esutil
import time
import matplotlib.pyplot as plt
import scipy.optimize


from sharedNumpyMemManager import SharedNumpyMemManager as snmm
from fgcmUtilities import dataBinner

class FgcmApertureCorrection(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmGray):
        self.fgcmLog = fgcmConfig.fgcmLog
        self.fgcmLog.info('Initializing FgcmApertureCorrection')

        self.fgcmPars = fgcmPars

        self.fgcmGray = fgcmGray

        # and record configuration variables
        ## include plot path...
        self.aperCorrFitNBins = fgcmConfig.aperCorrFitNBins
        self.illegalValue = fgcmConfig.illegalValue
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle

    def computeApertureCorrections(self,doPlots=True):
        """
        """

        if (self.aperCorrFitNBins == 0):
            self.fgcmLog.info('No aperture correction will be computed')
            self.fgcmPars.compAperCorrPivot[:] = 0.0
            self.fgcmPars.compAperCorrSlope[:] = 0.0
            self.fgcmPars.compAperCorrSlopeErr[:] = 0.0
            return

        startTime=time.time()
        self.fgcmLog.info('Computing aperture corrections with %d bins' %
                         (self.aperCorrFitNBins))

        # need to make a local copy since we're modifying
        expGray = snmm.getArray(self.fgcmGray.expGrayHandle)
        expGrayTemp = expGray.copy()

        # first, remove any previous correction if necessary...
        if (np.max(self.fgcmPars.compAperCorrRange[1,:]) >
            np.min(self.fgcmPars.compAperCorrRange[0,:])) :
            self.fgcmLog.info('Removing old aperture corrections')

            expSeeingVariableClipped = np.clip(self.fgcmPars.expSeeingVariable,
                                               self.fgcmPars.compAperCorrRange[0,self.fgcmPars.expBandIndex],
                                               self.fgcmPars.compAperCorrRange[1,self.fgcmPars.expBandIndex])

            oldAperCorr = self.fgcmPars.compAperCorrSlope[self.fgcmPars.expBandIndex] * (
                expSeeingVariableClipped -
                self.fgcmPars.compAperCorrPivot[self.fgcmPars.expBandIndex])

            # Note that EXP^gray = < <mstd>_j - mstd_ij >
            #  the aperture correction is applied to mstd_ij
            #  so do de-apply the aperture correction, we need the same sign as in
            #  FgcmStars.applyApertureCorrection

            expGrayTemp += oldAperCorr

        expIndexUse,=np.where(self.fgcmPars.expFlag == 0)

        for i in xrange(self.fgcmPars.nBands):
            use,=np.where((self.fgcmPars.expBandIndex[expIndexUse] == i) &
                          (self.fgcmPars.expSeeingVariable[expIndexUse] > self.illegalValue) &
                          (np.isfinite(self.fgcmPars.expSeeingVariable[expIndexUse])))

            if (use.size == 0):
                self.fgcmLog.info('ApertureCorrection: No good observations in %s band.' % (self.fgcmPars.bands[i]))
                continue

            # sort to set the range...
            #st=np.argsort(expGrayTemp[use])
            st=np.argsort(self.fgcmPars.expSeeingVariable[expIndexUse[use]])
            use=use[st]

            self.fgcmPars.compAperCorrRange[0,i] = self.fgcmPars.expSeeingVariable[expIndexUse[use[int(0.02*use.size)]]]
            self.fgcmPars.compAperCorrRange[1,i] = self.fgcmPars.expSeeingVariable[expIndexUse[use[int(0.98*use.size)]]]

            # this will make a rounder number
            self.fgcmPars.compAperCorrPivot[i] = np.floor(np.median(self.fgcmPars.expSeeingVariable[expIndexUse[use]])*1000)/1000.

            binSize = (self.fgcmPars.compAperCorrRange[1,i] -
                       self.fgcmPars.compAperCorrRange[0,i]) / self.aperCorrFitNBins

            binStruct = dataBinner(self.fgcmPars.expSeeingVariable[expIndexUse[use]],
                                   expGrayTemp[expIndexUse[use]],
                                   binSize,
                                   self.fgcmPars.compAperCorrRange[:,i])
            # remove any empty bins...
            gd,=np.where(binStruct['Y_ERR'] > 0.0)
            if (gd.size < 3):
                self.fgcmLog.info('Warning: could not compute aperture correction for band %s (too few exposures)' % (self.fgcmPars.bands[i]))
                self.fgcmPars.compAperCorrSlope[i] = 0.0
                self.fgcmPars.compAperCorrSlopeErr[i] = 0.0

                continue

            binStruct=binStruct[gd]

            # this helps in debugging?
            binStruct['Y_ERR'] = np.sqrt(binStruct['Y_ERR']**2. + 0.001**2.)

            fit,cov = np.polyfit(binStruct['X_BIN'] - self.fgcmPars.compAperCorrPivot[i],
                                 binStruct['Y'],
                                 1.0,
                                 w=(1./binStruct['Y_ERR'])**2.,
                                 cov=True)

            if ((cov[0,0] < 0.0) or (not np.isfinite(cov[0,0]))) :
                self.fgcmLog.info('Warning: Aperture correction computation failed for band %s' %
                                 (self.fgcmPars.bands[i]))
                self.fgcmPars.compAperCorrSlope[i] = 0.0
                self.fgcmPars.compAperCorrSlopeErr[i] = 0.0

                continue
            else :
                self.fgcmPars.compAperCorrSlope[i] = fit[0]
                self.fgcmPars.compAperCorrSlopeErr[i] = np.sqrt(cov[0,0])

                self.fgcmLog.info('Aperture correction slope in band %s is %.4f +/- %.4f' %
                                 (self.fgcmPars.bands[i],
                                  self.fgcmPars.compAperCorrSlope[i],
                                  self.fgcmPars.compAperCorrSlopeErr[i]))

            if (doPlots):
                fig=plt.figure(1,figsize=(8,6))
                fig.clf()

                ax=fig.add_subplot(111)

                ax.hexbin(self.fgcmPars.expSeeingVariable[expIndexUse[use]],
                          expGrayTemp[expIndexUse[use]],
                          rasterized=True)

                ax.errorbar(binStruct['X_BIN'],binStruct['Y'],
                            yerr=binStruct['Y_ERR'],fmt='r.',markersize=10)
                ax.set_xlim(self.fgcmPars.compAperCorrRange[0,i],
                            self.fgcmPars.compAperCorrRange[1,i])
                ax.locator_params(axis='x',nbins=6)

                ax.tick_params(axis='both',which='major',labelsize=14)

                ax.set_xlabel(r'$\mathrm{ExpSeeingVariable}$',fontsize=16)
                ax.set_ylabel(r'$\mathrm{EXP}^{\mathrm{gray}}$',fontsize=16)

                text=r'$(%s)$' % (self.fgcmPars.bands[i])
                ax.annotate(text,(0.9,0.93),xycoords='axes fraction',
                            ha='right',va='top',color='r',fontsize=16)

                ax.plot(self.fgcmPars.compAperCorrRange[:,i],
                        self.fgcmPars.compAperCorrSlope[i] *
                        (self.fgcmPars.compAperCorrRange[:,i] -
                         self.fgcmPars.compAperCorrPivot[i]),'r--')

                fig.savefig('%s/%s_apercorr_%s.png' % (self.plotPath,
                                                       self.outfileBaseWithCycle,
                                                       self.fgcmPars.bands[i]))


        ## MAYBE: modify ccd gray and exp gray?
        ##  could rely on the iterations taking care of this.

        self.fgcmLog.info('Computed aperture corrections in %.2f seconds.' %
                         (time.time() - startTime))
