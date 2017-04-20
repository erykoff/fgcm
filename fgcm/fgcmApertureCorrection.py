from __future__ import print_function

import numpy as np
import fitsio
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
        self.fgcmLog.log('INFO','Initializing FgcmApertureCorrection')

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

        startTime=time.time()
        self.fgcmLog.log('INFO','Computing aperture corrections')

        # need to make a local copy since we're modifying
        expGray = snmm.getArray(self.fgcmGray.expGrayHandle)
        expGrayTemp = expGray.copy()

        # first, remove any previous correction if necessary...
        self.fgcmLog.log('DEBUG','Removing old aperture corrections')
        if (np.max(self.fgcmPars.compAperCorrRange[1,:]) >
            np.min(self.fgcmPars.compAperCorrRange[0,:])) :

            expSeeingVariableClipped = np.clip(self.fgcmPars.expSeeingVariable,
                                               self.fgcmPars.compAperCorrRange[0,self.fgcmPars.expBandIndex],
                                               self.fgcmPars.compAperCorrRange[1,self.fgcmPars.expBandIndex])

            oldAperCorr = self.fgcmPars.compAperCorrSlope[self.fgcmPars.expBandIndex] * (
                expSeeingVariableClipped -
                self.fgcmPars.compAperCorrPivot[self.fgcmPars.expBandIndex])

            # Need to check sign here...
            expGrayTemp -= oldAperCorr

        expIndexUse,=np.where(self.fgcmPars.expFlag == 0)

        for i in xrange(self.fgcmPars.nBands):
            use,=np.where((self.fgcmPars.expBandIndex[expIndexUse] == i) &
                          (self.fgcmPars.expSeeingVariable[expIndexUse] > self.illegalValue) &
                          (np.isfinite(self.fgcmPars.expSeeingVariable[expIndexUse])))

            # sort to set the range...
            st=np.argsort(expGrayTemp[use])
            use=use[st]

            self.fgcmPars.compAperCorrRange[0,i] = expSeeingVariable[expIndexUse[use[int(0.02*use.size)]]]
            self.fgcmPars.compAperCorrRange[1,i] = expSeeingVariable[expIndexUse[use[int(0.98*use.size)]]]

            # this will make a rounder number
            self.fgcmPars.compAperCorrPivot[i] = np.floor(np.median(expSeeingVariable[expIndexUse[use]])*1000)/1000.

            binSize = (self.fgcmPars.compAperCorrRange[1,i] -
                       self.fgcmPars.compAperCorrRange[0,i]) / self.aperCorrFitNBins

            binStruct = dataBinner(self.fgcmPars.expSeeingVariable[use],
                                   expGrayTemp[use],
                                   binSize,
                                   self.fgcmPars.compAperCorrRange[:,i])

            # remove any empty bins...
            gd,=np.where(binStruct['Y_ERR'] > 0.0)
            binStruct=binStruct[gd]

            fit,cov = np.polyfit(binStruct['X_BIN'] - self.fgcmPars.compAperCorrPivot[i],
                                 binStruct['Y'],
                                 1.0,
                                 w=(1./binStruct['Y_ERR'])**2.,
                                 cov=True)

            self.fgcmPars.compAperCorrSlope[i] = fit[0]
            self.fgcmPars.compAperCorrSlopeErr[i] = np.sqrt(cov[0,0])

            if (doPlots):
                fig=plt.figure(1)
                fig.clf()

                ax=fig.add_subplot(111)

                ax.hexbin(self.fgcmPars.expSeeingVariable[use],
                          expGrayTemp[use],
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

        ## FIXME: modify ccd gray and exp gray?

        self.fgcmLog.log('INFO','Computed aperture corrections in %.2f seconds.' %
                         (time.time() - startTime))
