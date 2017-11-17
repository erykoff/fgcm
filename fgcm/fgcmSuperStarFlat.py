from __future__ import print_function

import numpy as np
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
        self.fgcmLog.info('Initializing FgcmSuperStarFlat')

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
        self.fgcmLog.info('Computing superstarflats')

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

        self.fgcmLog.info('SuperStarFlats based on %d exposures' % (gd.size))

        # sum up ccdGray values
        #  note that this is done per *filter* not per *band*
        np.add.at(deltaSuperStarFlat,
                  (self.fgcmPars.expEpochIndex[expIndexUse],
                   self.fgcmPars.expLUTFilterIndex[expIndexUse],
                   ccdIndexUse),
                  ccdGray[expIndexUse,ccdIndexUse])
        np.add.at(deltaSuperStarFlatNCCD,
                  (self.fgcmPars.expEpochIndex[expIndexUse],
                   self.fgcmPars.expLUTFilterIndex[expIndexUse],
                   ccdIndexUse),
                  1)

        # only use exp/ccd where we have at least one observation
        gd=np.where(deltaSuperStarFlatNCCD > 0)
        deltaSuperStarFlat[gd] /= deltaSuperStarFlatNCCD[gd]

        # this accumulates onto the input parameters
        self.fgcmPars.parSuperStarFlat += deltaSuperStarFlat

        ## MAYBE: change fgcmGray to remove the deltaSuperStarFlat?
        ##  or we can rely on the iterations.  Try that first.

        ## FIXME
        self.deltaSuperStarFlatMean = np.zeros((self.fgcmPars.nEpochs,
                                                self.fgcmPars.nLUTFilter),dtype='f8')
        self.deltaSuperStarFlatSigma = np.zeros_like(self.deltaSuperStarFlatMean)
        self.superStarFlatMean = np.zeros_like(self.deltaSuperStarFlatMean)
        self.superStarFlatSigma = np.zeros_like(self.deltaSuperStarFlatMean)

        for i in xrange(self.fgcmPars.nEpochs):
            for j in xrange(self.fgcmPars.nLUTFilter):
                use,=np.where(deltaSuperStarFlatNCCD[i,j,:] > 0)

                if use.size < 3:
                    continue

                self.deltaSuperStarFlatMean[i,j] = np.mean(deltaSuperStarFlat[i,j,use])
                self.deltaSuperStarFlatSigma[i,j] = np.std(deltaSuperStarFlat[i,j,use])
                self.superStarFlatMean[i,j] = np.mean(self.fgcmPars.parSuperStarFlat[i,j,use])
                self.superStarFlatSigma[i,j] = np.std(self.fgcmPars.parSuperStarFlat[i,j,use])
                self.fgcmLog.info('Superstar epoch %d filter %s: %.4f +/- %.4f' %
                                 (i,self.fgcmPars.lutFilterNames[j],
                                  self.superStarFlatMean[i,j],
                                  self.superStarFlatSigma[i,j]))
                self.fgcmLog.info('DeltaSuperStar epoch %d filter %s: %.4f +/- %.4f' %
                                 (i,self.fgcmPars.lutFilterNames[j],
                                  self.deltaSuperStarFlatMean[i,j],
                                  self.deltaSuperStarFlatSigma[i,j]))

        self.fgcmLog.info('Computed SuperStarFlats in %.2f seconds.' %
                         (time.time() - startTime))

        if (doPlots):
            self.fgcmLog.info('Making SuperStarFlat plots')
            self.plotSuperStarFlats(deltaSuperStarFlat,
                                    self.deltaSuperStarFlatMean,
                                    self.deltaSuperStarFlatSigma,
                                    nCCDArray=deltaSuperStarFlatNCCD,
                                    name='deltasuperstar')
            self.plotSuperStarFlats(self.fgcmPars.parSuperStarFlat,
                                    self.superStarFlatMean,
                                    self.superStarFlatSigma,
                                    nCCDArray=deltaSuperStarFlatNCCD,
                                    name='superstar')

        # and we're done.

    def plotSuperStarFlats(self, superStarArray, superStarMean, superStarSigma,
                            nCCDArray=None, name='superstar'):
        """
        """
        from fgcmUtilities import plotCCDMap

        for i in xrange(self.fgcmPars.nEpochs):
            for j in xrange(self.fgcmPars.nLUTFilter):
                # only do those that had a non-zero number of CCDs to fit in this epoch
                if (nCCDArray is not None):
                    use,=np.where(nCCDArray[i,j,:] > 0)
                else:
                    use,=np.where(superStarArray[i,j,:] > self.illegalValue)

                if use.size == 0:
                    continue

                fig=plt.figure(1,figsize=(8,6))
                fig.clf()

                ax=fig.add_subplot(111)

                plotCCDMap(ax, self.ccdOffsets[use], superStarArray[i,j,use],
                           'Superflat Correction (mag)')

                text = r'$(%s)$' % (self.fgcmPars.lutFilterNames[j]) + '\n' + \
                    r'%.4f +/- %.4f' % (superStarMean[i,j],superStarSigma[i,j])
                ax.annotate(text,
                            (0.1,0.93),xycoords='axes fraction',
                            ha='left',va='top',fontsize=18)

                fig.savefig('%s/%s_%s_%s_%s.png' % (self.plotPath,
                                                    self.outfileBaseWithCycle,
                                                    name,
                                                    self.fgcmPars.lutFilterNames[j],
                                                    self.epochNames[i]))


