import numpy as np
import os
import sys
import esutil

import matplotlib.pyplot as plt

from .fgcmConfig import FgcmConfig
from .fgcmParameters import FgcmParameters
from .fgcmStars import FgcmStars
from .fgcmLUT import FgcmLUT
from .fgcmZpsToApply import FgcmZpsToApply
from .fgcmDeltaAper import FgcmDeltaAper
from .fgcmUtilities import FocalPlaneProjectorFromOffsets

from .fgcmUtilities import getMemoryString, expFlagDict, obsFlagDict


class FgcmApplyZeropoints:
    """
    Class which reads in FGCM zeropoints and applies them.

    Parameters
    ----------
    configDict: dict
       dictionary with config variables
    useFits: bool, optional
       Read in files using fitsio?
    noFitsDict: dict, optional
       Dict with lutIndex/lutStd/expInfo/[ccdOffsets or focalPlaneProjector]
       if useFits == False

    Note that at least one of useFits or noFitsDict must be supplied.
    """

    def __init__(self, configDict, useFits=False, noFitsDict=None, noOutput=False):
        # are we in fits mode?
        self.useFits = useFits

        if (not self.useFits):
            if (noFitsDict is None):
                raise ValueError("if useFits is False, must supply noFitsDict")

            if (('lutIndex' not in noFitsDict) or
                ('lutStd' not in noFitsDict) or
                ('expInfo' not in noFitsDict) or
                (('ccdOffsets' not in noFitsDict) and
                 ('focalPlaneProjector' not in noFitsDict))):
                raise ValueError("if useFits is False, must supply lutIndex, lutStd, expInfo, "
                                 "[ccdOffsets or focalPlaneProjector] in noFitsDict")

        if self.useFits:
            # Everything can be loaded from fits
            self.fgcmConfig = FgcmConfig.configWithFits(configDict, noOutput=noOutput)
        else:
            # everything must be passed in.
            self.fgcmConfig = FgcmConfig(configDict,
                                         noFitsDict['lutIndex'],
                                         noFitsDict['lutStd'],
                                         noFitsDict['expInfo'],
                                         noOutput=noOutput,
                                         ccdOffsets=noFitsDict.get('ccdOffsets'),
                                         focalPlaneProjector=noFitsDict.get('focalPlaneProjector'))

        # and set up the log
        self.fgcmLog = self.fgcmConfig.fgcmLog
        self.quietMode = self.fgcmConfig.quietMode

        from .sharedNumpyMemManager2 import SharedNumpyMemManager2

        self.snmm = SharedNumpyMemManager2()

        self.fgcmLUT = None
        self.fgcmPars = None
        self.fgcmStars = None
        self.setupComplete = False

    def runWithFits(self):
        """
        Read in files and run, reading data from fits tables.

        parameters
        ----------
        None
        """

        self._setupWithFits()
        self.run()

    def setStars(self, fgcmStars):
        """
        Record the star information.  This is a separate method to allow
         for memory to be cleared.

        parameters
        ----------
        fgcmStars: FgcmStars
           Object with star information
        """
        # this has to be done outside for memory issues

        self.fgcmStars = fgcmStars

    def _setupWithFits(self):
        """
        Set up with fits files.
        """

        if not self.quietMode:
            self.fgcmLog.info(getMemoryString('Setting up with fits'))

        # Read in the LUT
        # (might not be necessary)
        self.fgcmLUT = FgcmLUT.initFromFits(self.fgcmConfig.lutFile, self.snmm,
                                            filterToBand=self.fgcmConfig.filterToBand)

        try:
            self.fgcmPars = FgcmParameters.loadParsWithFits(self.fgcmConfig)
        except:
            self.fgcmPars = FgcmParameters.newParsWithFits(self.fgcmConfig,
                                                           self.fgcmLUT)

        # Read in the stars
        self.fgcmStars = FgcmStars(self.fgcmConfig, self.snmm)
        self.fgcmStars.loadStarsFromFits(self.fgcmPars, computeNobs=True)
        self.fgcmStars.prepStars(self.fgcmPars)

        self.fgcmZpsToApply = FgcmZpsToApply(self.fgcmConfig, self.fgcmPars, self.fgcmStars, self.fgcmLUT, self.snmm)
        self.fgcmZpsToApply.loadZeropointsFromFits()

        self.finishSetup()

    def finishSetup(self):
        """
        Finish apply zeropoints setup.  Check that all the essential objects have been set.
        """

        if (self.fgcmStars is None):
            raise RuntimeError("Must set fgcmStars")
        if (self.fgcmLUT is None):
            raise RuntimeError("Must set fgcmLUT")
        if (self.fgcmPars is None):
            raise RuntimeError("Must set fgcmPars")

        self.setupComplete = True
        if not self.quietMode:
            self.fgcmLog.info(getMemoryString('ApplyZeropoints Prepared'))

    def run(self):
        """
        Apply the zeropoints.
        """

        if (not self.setupComplete):
            raise RuntimeError("Must complete applyZeropoints setup first!")

        # Compute signs for CCD offsets if necessary
        if isinstance(self.fgcmConfig.focalPlaneProjector, FocalPlaneProjectorFromOffsets):
            self.fgcmConfig.focalPlaneProjector.computeCCDOffsetSigns(self.fgcmStars)

        if self.fgcmStars.hasDeltaMagBkg:
            self.fgcmStars.applyDeltaMagBkg()

        # Select good exposures based on the flag cuts
        # Can do this only for exposures that are ALL above the threshold.

        zpFlag = self.holder.getArray(self.fgcmZpsToApply.zpFlagHandle)
        flagMin = np.min(zpFlag, axis=1)

        bad, = np.where(flagMin > self.fgcmConfig.maxFlagZpsToApply)
        self.fgcmPars.expFlag[bad] |= expFlagDict['BAD_ZPFLAG']

        # And flag observations that do not have zeropoints...

        obsExpIndex = self.holder.getArray(self.fgcmStars.obsExpIndexHandle)
        obsCCDIndex = self.holder.getArray(self.fgcmStars.obsCCDHandle) - self.fgcmConfig.ccdStartIndex
        obsFlag = self.holder.getArray(self.fgcmStars.obsFlagHandle)

        bad, = np.where(zpFlag[obsExpIndex, obsCCDIndex] > self.fgcmConfig.maxFlagZpsToApply)
        obsFlag[bad] |= obsFlagDict['NO_ZEROPOINT']

        # Select stars that have the minimum number of observations
        # This sets objNGoodObs
        goodExpsIndex, = np.where(self.fgcmPars.expFlag == 0)
        self.fgcmStars.selectStarsMinObsExpIndex(goodExpsIndex)

        # Compute m^std and <m^std> using the zeropoints, compute SEDs,
        # no error modeling.

        self.fgcmZpsToApply.applyZeropoints()

        # Compute median SED slopes and apply
        self.fgcmStars.fillMissingSedSlopes(self.fgcmPars)

        # Apply any color cuts
        self.fgcmStars.performColorCuts()

        # I believe we should do the superstar outlier cuts
        self.fgcmStars.performSuperStarOutlierCuts(self.fgcmPars)

        # Apply error models
        self.fgcmStars.applyModelMagErrorModel(self.fgcmPars)

        # Re-select stars that have the minimum number of observations
        # (after adding outlier rejections...)
        # This sets objNGoodObs
        goodExpsIndex, = np.where(self.fgcmPars.expFlag == 0)
        self.fgcmStars.selectStarsMinObsExpIndex(goodExpsIndex)

        # Recompute m^std and <m^std> using the zeropoints, with chromatic corrections
        #  and with error modeling.
        self.fgcmZpsToApply.applyZeropoints()

        if self.fgcmStars.hasDeltaAper:
            self.fgcmDeltaAper = FgcmDeltaAper(self.fgcmConfig, self.fgcmPars,
                                               self.fgcmStars)
            if self.fgcmConfig.doComputeDeltaAperExposures:
                self.fgcmDeltaAper.computeDeltaAperExposures()
            if self.fgcmConfig.doComputeDeltaAperStars:
                self.fgcmDeltaAper.computeDeltaAperStars()
                if self.fgcmConfig.doComputeDeltaAperMap:
                    self.fgcmDeltaAper.computeEpsilonMap()
            if self.fgcmConfig.doComputeDeltaAperPerCcd:
                self.fgcmDeltaAper.computeEpsilonPerCcd()

            if self.useFits:
                outParFile = '%s/%s_applied_parameters.fits' % (self.fgcmConfig.outputPath,
                                                                self.fgcmConfig.outfileBaseWithCycle)
                self.fgcmPars.saveParsFits(outParFile)



        # Output the stars.
        if self.fgcmConfig.outputStars:
            outStarFile = '%s/%s_stdstars.fits' % (self.fgcmConfig.outputPath,
                                                   self.fgcmConfig.outfileBaseWithCycle)
            self.fgcmStars.saveStdStars(outStarFile, self.fgcmPars)

