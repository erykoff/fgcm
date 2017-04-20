from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cmx


from sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmSuperStarFlat(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmGray):

        self.fgcmLog = fgcmConfig.fgcmLog
        self.fgcmLog.log('INFO','Initializing FgcmSuperStarFlat')

        self.fgcmPars = fgcmPars

        self.fgcmGray = fgcmGray

        self.illegalValue = fgcmConfig.illegalValue
        self.minStarPerCCD = fgcmConfig.minStarPerCCD
        self.ccdOffsets = fgcmConfig.ccdOffsets
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.epochNames = fgcmConfig.epochNames

    def computeSuperStarFlats(self,doPlots=True):
        """
        """

        startTime = time.time()
        self.fgcmLog.log('INFO','Computing superstarflats')

        ## FIXME: need to filter out SN (deep) exposures.  Hmmm.

        deltaSuperStarFlat = np.zeros_like(self.fgcmPars.parSuperStarFlat)
        deltaSuperStarFlatNCCD = np.zeros_like(self.fgcmPars.parSuperStarFlat,dtype='i4')

        ccdGray = snmm.getArray(self.fgcmGray.ccdGrayHandle)
        ccdGrayErr = snmm.getArray(self.fgcmGray.ccdGrayErrHandle)
        ccdNGoodStars = snmm.getArray(self.fgcmGray.ccdNGoodStarsHandle)

        # only select those CCDs that we have an adequate gray calculation
        expIndexUse,ccdIndexUse=np.where((ccdNGoodStars >= self.minStarPerCCD))

        # and only select exposures that should go into the SuperStarFlat
        gd,=np.where(self.fgcmPars.expFlag[expIndexUse] == 0)
        expIndexUse=expIndexUse[gd]
        ccdIndexUse=ccdIndexUse[gd]

        self.fgcmLog.log('INFO','SuperStarFlats based on %d exposures' % (gd.size))

        # sum up ccdGray values
        np.add.at(deltaSuperStarFlat,
                  (self.fgcmPars.expEpochIndex[expIndexUse],
                   self.fgcmPars.expBandIndex[expIndexUse],
                   ccdIndexUse),
                  ccdGray[expIndexUse,ccdIndexUse])
        np.add.at(deltaSuperStarFlatNCCD,
                  (self.fgcmPars.expEpochIndex[expIndexUse],
                   self.fgcmPars.expBandIndex[expIndexUse],
                   ccdIndexUse),
                  1)

        # only use exp/ccd where we have at least one observation
        gd=np.where(deltaSuperStarFlatNCCD > 0)
        deltaSuperStarFlat[gd] /= deltaSuperStarFlatNCCD[gd]

        # this accumulates onto the input parameters
        self.fgcmPars.parSuperStarFlat += deltaSuperStarFlat

        ## MAYBE: change fgcmGray to remove the deltaSuperStarFlat?
        ##  or we can rely on the iterations.  Try that first.

        self.fgcmLog.log('INFO','Computed SuperStarFlats in %.2f seconds.' %
                         (time.time() - startTime))

        if (doPlots):
            self.plotSuperStarFlats(deltaSuperStarFlat,
                                    nCCDArray=deltaSuperStarFlatNCCD,
                                    name='deltasuperstar')
            self.plotSuperStarFlats(self.fgcmPars.parSuperStarFlat,
                                    nCCDArray=deltaSuperStarFlatNCCD,
                                    name='superstar')

        # and we're done.

    def plotSuperStarFlats(self, superStarArray, nCCDArray=None, name='superstar'):
        """
        """

        cm = plt.get_cmap('rainbow')
        plt.set_cmap('rainbow')

        plotRARange = [self.ccdOffsets['DELTA_RA'].min() - self.ccdOffsets['RA_SIZE'].max()/2.,
                       self.ccdOffsets['DELTA_RA'].max() + self.ccdOffsets['RA_SIZE'].max()/2.]
        plotDecRange = [self.ccdOffsets['DELTA_DEC'].min() - self.ccdOffsets['DEC_SIZE'].max()/2.,
                        self.ccdOffsets['DELTA_DEC'].max() + self.ccdOffsets['DEC_SIZE'].max()/2.]

        for i in xrange(self.fgcmPars.nEpochs):
            for j in xrange(self.fgcmPars.nBands):
                # only do those that had a non-zero number of CCDs to fit in this epoch
                if (nCCDArray is not None):
                    use,=np.where(nCCDArray[i,j,:] > 0)
                else:
                    use,=np.where(superStarArray[i,j,:] > self.illegalValue)

                # kick out if there is nothing to plot here
                if use.size == 0:
                    continue

                # first, we plot the total superstar flat
                st=np.argsort(superStarArray[i,j,use])

                # add some padding of 1mmag for all zeros
                lo=superStarArray[i,j,use[st[int(0.02*st.size)]]]-0.001
                hi=superStarArray[i,j,use[st[int(0.98*st.size)]]]+0.001

                cNorm = colors.Normalize(vmin=lo,vmax=hi)
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

                Z=[[0,0],[0,0]]
                levels=np.linspace(lo,hi,num=150)
                CS3=plt.contourf(Z,levels,cmap=cm)

                fig=plt.figure(1)
                fig.clf()

                ax=fig.add_subplot(111)

                ax.set_xlim(plotRARange[0]-0.05,plotRARange[1]+0.05)
                ax.set_ylim(plotDecRange[0]-0.05,plotDecRange[1]+0.05)
                ax.set_xlabel(r'$\delta\,\mathrm{R.A.}$',fontsize=16)
                ax.set_ylabel(r'$\delta\,\mathrm{Dec.}$',fontsize=16)
                ax.tick_params(axis='both',which='major',labelsize=14)

                for k in xrange(use.size):
                    off=[self.ccdOffsets['DELTA_RA'][use[k]],
                         self.ccdOffsets['DELTA_DEC'][use[k]]]

                    ax.add_patch(
                        patches.Rectangle(
                            (off[0]-self.ccdOffsets['RA_SIZE'][use[k]]/2.,
                             off[1]-self.ccdOffsets['DEC_SIZE'][use[k]]/2.),
                            self.ccdOffsets['RA_SIZE'][use[k]],
                            self.ccdOffsets['DEC_SIZE'][use[k]],
                            edgecolor="none",
                            facecolor=scalarMap.to_rgba(superStarArray[i,j,use[k]]))
                        )

                cb=None
                # god damn I hate matplotlib
                #  probably have to specify tick...and need to find round numbers.  blah
                cb = plt.colorbar(CS3,ticks=np.linspace(lo,hi,5))
                #cb = plt.colorbar(CS3)

                cb.set_label('Superflat Correction (mag)',fontsize=14)

                ax.annotate(r'$(%s)$' % (self.fgcmPars.bands[j]),
                            (0.1,0.93),xycoords='axes fraction',
                            ha='left',va='top',fontsize=18)


                fig.savefig('%s/%s_%s_%s_%s.png' % (self.plotPath,
                                                    self.outfileBaseWithCycle,
                                                    name,
                                                    self.fgcmPars.bands[j],
                                                    self.epochNames[i]))
