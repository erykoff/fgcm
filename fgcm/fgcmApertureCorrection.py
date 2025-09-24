import numpy as np
import os
import sys
import esutil
import time
import scipy.optimize
import warnings

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm
from .fgcmUtilities import dataBinner, makeFigure, putButlerFigure

class FgcmApertureCorrection(object):
    """
    Class which computes aperture corrections.  Note that this will only
      be run if aperCorrFitNBins > 0 and expSeeingVariable is set in the
      exposure info.

    parameters
    ----------
    fgcmConfig: FgcmConfig
       Config object
    fgcmPars: FgcmParameters
       Parameter object
    fgcmGray: FgcmGray
       Gray residual object

    Config variables
    ----------------
    aperCorrFitNBins: int
       Number of expSeeingVariable bins to use for computing correction slope

    """
    def __init__(self, fgcmConfig, fgcmPars, fgcmGray, butlerQC=None, plotHandleDict=None):
        self.fgcmLog = fgcmConfig.fgcmLog
        self.fgcmLog.debug('Initializing FgcmApertureCorrection')

        self.fgcmPars = fgcmPars
        self.fgcmGray = fgcmGray

        self.butlerQC = butlerQC
        self.plotHandleDict = plotHandleDict

        # and record configuration variables
        ## include plot path...
        self.aperCorrFitNBins = fgcmConfig.aperCorrFitNBins
        self.illegalValue = fgcmConfig.illegalValue
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber
        self.quietMode = fgcmConfig.quietMode

        self.rng = fgcmConfig.rng

    def computeApertureCorrections(self):
        """
        Compute aperture corrections if aperCorrFitNBins > 0
        """

        if (self.aperCorrFitNBins == 0):
            self.fgcmLog.info('No aperture correction will be computed')
            # Only reset to zero if there aren't any input parameters
            # Otherwise, leave them as is and they'll get happily saved
            # for the next fit cycle.
            if self.fgcmPars.aperCorrInputSlopes is None:
                self.fgcmPars.compAperCorrPivot[:] = 0.0
                self.fgcmPars.compAperCorrSlope[:] = 0.0
                self.fgcmPars.compAperCorrSlopeErr[:] = 0.0
            return

        startTime=time.time()
        if not self.quietMode:
            self.fgcmLog.info('Computing aperture corrections with %d bins' %
                              (self.aperCorrFitNBins))

        # need to make a local copy since we're modifying
        expGray = snmm.getArray(self.fgcmGray.expGrayHandle)
        expGrayTemp = expGray.copy()

        # save original pivot and range in case it fails
        originalSlope = self.fgcmPars.compAperCorrSlope.copy()
        originalPivot = self.fgcmPars.compAperCorrPivot.copy()
        originalRange = self.fgcmPars.compAperCorrRange.copy()

        # first, remove any previous correction if necessary...
        if (np.max(self.fgcmPars.compAperCorrRange[1,:]) >
            np.min(self.fgcmPars.compAperCorrRange[0,:])) :
            self.fgcmLog.debug('Removing old aperture corrections')

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

        for i in range(self.fgcmPars.nBands):
            if not self.fgcmPars.hasExposuresInBand[i]:
                continue

            use,=np.where((self.fgcmPars.expBandIndex[expIndexUse] == i) &
                          (self.fgcmPars.expSeeingVariable[expIndexUse] > self.illegalValue) &
                          (np.isfinite(self.fgcmPars.expSeeingVariable[expIndexUse])))

            if (use.size == 0):
                self.fgcmLog.info('ApertureCorrection: No good observations in %s band.' % (self.fgcmPars.bands[i]))
                continue

            # sort to set the range...
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
                                   self.fgcmPars.compAperCorrRange[:,i],
                                   rng=self.rng)
            # remove any empty bins...
            gd,=np.where(binStruct['Y_ERR'] > 0.0)
            if (gd.size < 3):
                self.fgcmLog.warning('Could not compute aperture correction for band %s (too few exposures)' % (self.fgcmPars.bands[i]))
                self.fgcmPars.compAperCorrSlope[i] = originalSlope[i]
                self.fgcmPars.compAperCorrPivot[i] = originalPivot[i]
                self.fgcmPars.compAperCorrRange[:, i] = originalRange[:, i]
                self.fgcmPars.compAperCorrSlopeErr[i] = 0.0

                continue

            binStruct=binStruct[gd]

            # this helps in debugging?
            binStruct['Y_ERR'] = np.sqrt(binStruct['Y_ERR']**2. + 0.001**2.)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # This fit might throw a warning, but we check for bad fits below
                    # so we can ignore these internal warnings
                    # (This should only fail on small test regions, but the
                    # logged warning below should be sufficient for users to know
                    # there was a problem)
                    fit,cov = np.polyfit(binStruct['X_BIN'] - self.fgcmPars.compAperCorrPivot[i],
                                         binStruct['Y'],
                                         1.0,
                                         w=(1./binStruct['Y_ERR'])**2.,
                                         cov=True)
            except Exception as inst:
                self.fgcmLog.info('aperture correction completely failed for band %s' %
                                  (self.fgcmPars.bands[i]))
                self.fgcmPars.compAperCorrSlope[i] = originalSlope[i]
                self.fgcmPars.compAperCorrPivot[i] = originalPivot[i]
                self.fgcmPars.compAperCorrRange[:, i] = originalRange[:, i]
                self.fgcmPars.compAperCorrSlopeErr[i] = 0.0
                continue

            if ((cov[0,0] < 0.0) or (not np.isfinite(cov[0,0]))) :
                self.fgcmLog.warning('Aperture correction computation failed for band %s' %
                                 (self.fgcmPars.bands[i]))
                self.fgcmPars.compAperCorrSlope[i] = originalSlope[i]
                self.fgcmPars.compAperCorrPivot[i] = originalPivot[i]
                self.fgcmPars.compAperCorrRange[:, i] = originalRange[:, i]
                self.fgcmPars.compAperCorrSlopeErr[i] = 0.0

                continue
            else :
                self.fgcmPars.compAperCorrSlope[i] = fit[0]
                self.fgcmPars.compAperCorrSlopeErr[i] = np.sqrt(cov[0,0])

                self.fgcmLog.info('Aperture correction slope in band %s is %.2f +/- %.2f mmag/seeingUnit' %
                                 (self.fgcmPars.bands[i],
                                  self.fgcmPars.compAperCorrSlope[i]*1000.0,
                                  self.fgcmPars.compAperCorrSlopeErr[i]*1000.0))

            if self.plotPath is not None:
                from matplotlib import colormaps

                fig = makeFigure(figsize=(8, 6))
                fig.clf()

                ax=fig.add_subplot(111)

                ax.hexbin(self.fgcmPars.expSeeingVariable[expIndexUse[use]],
                          expGrayTemp[expIndexUse[use]]*1000.0,
                          rasterized=True,
                          cmap=colormaps.get_cmap("viridis"))

                ax.errorbar(binStruct['X_BIN'],binStruct['Y']*1000.0,
                            yerr=binStruct['Y_ERR']*1000.0,fmt='r.',markersize=10)
                ax.set_xlim(self.fgcmPars.compAperCorrRange[0,i],
                            self.fgcmPars.compAperCorrRange[1,i])
                ax.locator_params(axis='x',nbins=6)

                ax.tick_params(axis='both',which='major',labelsize=14)

                ax.set_xlabel(r'$\mathrm{ExpSeeingVariable}$',fontsize=16)
                ax.set_ylabel(r'$\mathrm{EXP}^{\mathrm{gray}}\,(\mathrm{mmag})$',fontsize=16)

                text=r'$(%s)$' % (self.fgcmPars.bands[i])
                ax.annotate(text,(0.9,0.93),xycoords='axes fraction',
                            ha='right',va='top',color='r',fontsize=16)

                ax.plot(self.fgcmPars.compAperCorrRange[:,i],
                        1000.0 * self.fgcmPars.compAperCorrSlope[i] *
                        (self.fgcmPars.compAperCorrRange[:,i] -
                         self.fgcmPars.compAperCorrPivot[i]),'r--')

                fig.tight_layout()

                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "Apercorr",
                                    self.cycleNumber,
                                    fig,
                                    band=self.fgcmPars.bands[i])
                else:
                    fig.savefig('%s/%s_apercorr_%s.png' % (self.plotPath,
                                                           self.outfileBaseWithCycle,
                                                           self.fgcmPars.bands[i]))


        ## MAYBE: modify ccd gray and exp gray?
        ##  could rely on the iterations taking care of this.

        if not self.quietMode:
            self.fgcmLog.info('Computed aperture corrections in %.2f seconds.' %
                              (time.time() - startTime))
