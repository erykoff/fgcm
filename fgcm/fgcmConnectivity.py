import numpy as np
import esutil

import matplotlib.pyplot as plt

from .fgcmUtilities import objFlagDict
from .fgcmUtilities import histogram_rev_sorted
from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmConnectivity(object):
    """
    Class to check the star/observation connectivity.

    parameters
    ----------
    fgcmConfig: FgcmConfig
    fgcmPars: FgcmPars
    fgcmStars: FgcmStars
    """

    def __init__(self, fgcmConfig, fgcmPars, fgcmStars):
        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.info('Initializing fgcmConnectivity')

        self.fgcmPars = fgcmPars
        self.fgcmStars = fgcmStars

        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.bands = fgcmConfig.bands

    def plotConnectivity(self):
        """
        Make connectivity plots.

        parameters
        ----------
        None
        """

        colors = ['r', 'b', 'm', 'c', 'k', 'g']

        # get values

        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)
        objRA = snmm.getArray(self.fgcmStars.objRAHandle)
        objDec = snmm.getArray(self.fgcmStars.objDecHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        groupFlag = np.zeros((objID.size, len(self.bands)), dtype=np.int32)

        mask = (objFlagDict['TOO_FEW_OBS'] |
                objFlagDict['BAD_COLOR'] |
                objFlagDict['VARIABLE'] |
                objFlagDict['TEMPORARY_BAD_STAR'])

        goodStars, = np.where((objFlag & mask) == 0)

        for b, band in enumerate(self.bands):
            # start with group 0
            groupNumber = 0
            # The nights that have been grouped are a unique set
            groupedNights = set()

            # Take one star, it's the group seed
            groupFlag[goodStars[0], b] = 2**groupNumber

            # Find all good (photometric) observations of stars in this band
            goodObs, = np.where((obsBandIndex == b) & (self.fgcmPars.expFlag[obsExpIndex] == 0) &
                                ((objFlag[obsObjIDIndex] & mask) == 0))

            # Split into nights with a histogram
            h, rev = histogram_rev_sorted(
                self.fgcmPars.expNightIndex[obsExpIndex[goodObs]],
                min=0,
                max=self.fgcmPars.nCampaignNights,
            )

            done = False
            while not done:
                # Start with all the stars that are in the current group
                test, = np.where((groupFlag[obsObjIDIndex[goodObs], b] & (2**groupNumber)) > 0)

                # Find the unique, extra list of nights which have the stars
                nightIndices = np.unique(self.fgcmPars.expNightIndex[obsExpIndex[goodObs[test]]])
                nightIndices = [nightIndex for nightIndex in nightIndices if nightIndex not in groupedNights]

                if len(nightIndices) == 0:
                    # We're done with this group ...
                    # Check if we're totally done.
                    # Make sure we check if these stars have *any* good observations
                    # (e.g., extra bands)
                    testStars, = np.where((groupFlag[goodStars, b] == 0) &
                                          (objNGoodObs[goodStars, b] > 0))
                    if testStars.size == 0:
                        # We have marked all the stars
                        done = True
                    else:
                        # Flag the first of these unmarked stars
                        # Increment the groupNumber
                        groupNumber += 1
                        # Flag the first seed star for the group
                        groupFlag[goodStars[testStars[0]], b] = 2**groupNumber
                    # Back to the beginning
                    continue

                # Loop over all the nights that are connected
                for nightIndex in nightIndices:
                    i1a = rev[rev[nightIndex]: rev[nightIndex + 1]]
                    groupFlag[obsObjIDIndex[goodObs[i1a]], b] |= 2**groupNumber
                    groupedNights.add(nightIndex)

            # And do the plot of the different groups...
            fig = plt.figure(figsize=(10, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.set_rasterization_zorder(1.0)

            for g in range(groupNumber + 1):
                u, = np.where((groupFlag[:, b] & (2**groupNumber)) > 0)
                u = np.random.choice(u, replace=False, size=np.min([u.size, 1000000]))
                ax.plot(objRA[u], objDec[u], colors[g % (len(colors))] + ',', zorder=0.5)
                ax.plot(objRA[u[0]], objDec[u[0]], colors[g % (len(colors))] + '.', label='Group %d' % (g))
            ax.legend(markerscale=2.0)
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            ax.set_title('%s band' % (band))

            fig.savefig('%s/%s_connectivity_groups_%s.png' % (self.plotPath,
                                                              self.outfileBaseWithCycle,
                                                              band))
            plt.close(fig)

