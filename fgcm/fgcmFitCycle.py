from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import os
import sys
import esutil
import scipy.optimize as optimize

import matplotlib.pyplot as plt

from .fgcmConfig import FgcmConfig
from .fgcmParameters import FgcmParameters
from .fgcmChisq import FgcmChisq
from .fgcmStars import FgcmStars
from .fgcmLUT import FgcmLUT
from .fgcmGray import FgcmGray
from .fgcmZeropoints import FgcmZeropoints
from .fgcmSuperStarFlat import FgcmSuperStarFlat
from .fgcmRetrieval import FgcmRetrieval
from .fgcmApertureCorrection import FgcmApertureCorrection
from .fgcmBrightObs import FgcmBrightObs
from .fgcmExposureSelector import FgcmExposureSelector
from .fgcmSigFgcm import FgcmSigFgcm
from .fgcmFlagVariables import FgcmFlagVariables
from .fgcmRetrieveAtmosphere import FgcmRetrieveAtmosphere
from .fgcmModelMagErrors import FgcmModelMagErrors
from .fgcmConnectivity import FgcmConnectivity
from .fgcmSigmaCal import FgcmSigmaCal
from .fgcmSigmaRef import FgcmSigmaRef
from .fgcmQeSysSlope import FgcmQeSysSlope
from .fgcmComputeStepUnits import FgcmComputeStepUnits

from .fgcmUtilities import zpFlagDict
from .fgcmUtilities import getMemoryString
from .fgcmUtilities import MaxFitIterations

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmFitCycle(object):
    """
    Class which runs the main FGCM fitter.

    parameters
    ----------
    configDict: dict
       dictionary with config variables
    useFits: bool, optional
       Read in files using fitsio?
    noFitsDict: dict, optional
       Dict with lutIndox/lutStd/expInfo/ccdOffsets if useFits == False

    Note that at least one of useFits or noFitsDict must be supplied.
    """

    def __init__(self, configDict, useFits=False, noFitsDict=None):
        # are we in fits mode?
        self.useFits = useFits

        if (not self.useFits):
            if (noFitsDict is None):
                raise ValueError("if useFits is False, must supply noFitsDict")

            if (('lutIndex' not in noFitsDict) or
                ('lutStd' not in noFitsDict) or
                ('expInfo' not in noFitsDict) or
                ('ccdOffsets' not in noFitsDict)):
                raise ValueError("if useFits is False, must supply lutIndex, lutStd, expInfo, ccdOffsets in noFitsDict")


        if self.useFits:
            # Everything can be loaded from fits
            self.fgcmConfig = FgcmConfig.configWithFits(configDict)
        else:
            # everything must be passed in.
            self.fgcmConfig = FgcmConfig(configDict,
                                            noFitsDict['lutIndex'],
                                            noFitsDict['lutStd'],
                                            noFitsDict['expInfo'],
                                            noFitsDict['ccdOffsets'])
        # and set up the log
        self.fgcmLog = self.fgcmConfig.fgcmLog

        # and set up cycle info

        self.initialCycle = False
        if (self.fgcmConfig.cycleNumber == 0):
            self.initialCycle = True

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

    def setLUT(self, fgcmLUT):
        """
        Record the look-up table information.  This is a separate method
         for memory to be cleared.

        parameters
        ----------
        fgcmLUT: FgcmLUT
           Object with LUT information
        """
        # this has to be done outside for memory issues

        self.fgcmLUT = fgcmLUT

    def setPars(self, fgcmPars):
        """
        Record the parameter information.  This is a separate method
         for memory to be cleared.

        parameters
        ----------
        fgcmPars: FgcmParameters
           Object with parameter information
        """
        # this has to be done outside for memory issues

        self.fgcmPars = fgcmPars

    def _setupWithFits(self):
        """
        Set up with fits files.
        """

        self.fgcmLog.info(getMemoryString('Setting up with fits'))

        # read in the LUT
        self.fgcmLUT = FgcmLUT.initFromFits(self.fgcmConfig.lutFile,
                                            filterToBand=self.fgcmConfig.filterToBand)

        # Generate or Read Parameters
        if (self.initialCycle):
            self.fgcmPars = FgcmParameters.newParsWithFits(self.fgcmConfig,
                                                           self.fgcmLUT)
        else:
            self.fgcmPars = FgcmParameters.loadParsWithFits(self.fgcmConfig)

        # Read in the stars
        self.fgcmStars = FgcmStars(self.fgcmConfig)
        self.fgcmStars.loadStarsFromFits(self.fgcmPars, computeNobs=True)

        self.finishSetup()

    def finishSetup(self):
        """
        Finish fit cycle setup.  Check that all the essential objects have been set.

        parameters
        ----------
        None

        """

        if (self.fgcmLUT is None):
            raise RuntimeError("Must set fgcmLUT")
        if (self.fgcmPars is None):
            raise RuntimeError("Must set fgcmPars")
        if (self.fgcmStars is None):
            raise RuntimeError("Must set fgcmStars")

        # these are things that can happen without fits

        # And prepare the chisq function
        self.fgcmChisq = FgcmChisq(self.fgcmConfig,self.fgcmPars,
                                   self.fgcmStars,self.fgcmLUT)

        # The step unit calculator
        self.fgcmComputeStepUnits = FgcmComputeStepUnits(self.fgcmConfig, self.fgcmPars,
                                                         self.fgcmStars, self.fgcmLUT)

        # And the exposure selector
        self.expSelector = FgcmExposureSelector(self.fgcmConfig,self.fgcmPars)

        # And the Gray code
        self.fgcmGray = FgcmGray(self.fgcmConfig,self.fgcmPars,self.fgcmStars)

        # And the qeSysSlope code
        self.fgcmQeSysSlope = FgcmQeSysSlope(self.fgcmConfig, self.fgcmPars, self.fgcmStars)

        self.setupComplete = True
        self.fgcmLog.info(getMemoryString('FitCycle Prepared'))

    def run(self):
        """
        Run the FGCM Fit Cycle.  If setup with useFits==True then files for the
          parameters, zeropoints, flagged stars, and atmosphere will be created.
          Otherwise, these can be accessed as attributes as noted.  In addition,
          a new config file for the next file will be created if useFits==True.
          In all cases a suite of plots will be created in a subdirectory or in
          the path specified by outputPath.

        parameters
        ----------
        None

        output attributes
        ----------
        fgcmZpts: FgcmZeropoints
           Contains zeropoints for exposure/ccd pairs and exposure atmosphere parameters
        fgcmPars: FgcmParameters
           Contains fit parameters
        """

        if (not self.setupComplete):
            raise RuntimeError("Must complete fitCycle setup first!")


        if (self.initialCycle):
            self.fgcmLog.info('Fit initial cycle starting...')
        else:
            self.fgcmLog.info('Fit cycle %d starting...' % (self.fgcmConfig.cycleNumber))

        # Apply aperture corrections and SuperStar if available
        # select exposures...
        if (not self.initialCycle):
            self.fgcmLog.debug('FitCycle is applying SuperStarFlat')
            self.fgcmStars.applySuperStarFlat(self.fgcmPars)
            self.fgcmLog.debug('FitCycle is applying ApertureCorrection')
            self.fgcmStars.applyApertureCorrection(self.fgcmPars)

            # and flag exposures using quantities computed from previous cycle
            self.fgcmLog.debug('FitCycle is running selectGoodExposures()')
            self.expSelector.selectGoodExposures()

        # Flag stars with too few exposures
        goodExpsIndex, = np.where(self.fgcmPars.expFlag == 0)
        self.fgcmLog.debug('FitCycle is finding good stars from %d good exposures' % (goodExpsIndex.size))
        #self.fgcmStars.selectStarsMinObs(goodExpsIndex=goodExpsIndex,doPlots=True)
        self.fgcmStars.selectStarsMinObsExpIndex(goodExpsIndex)
        self.fgcmStars.plotStarMap(mapType='initial')

        # Set up the magnitude error modeler
        self.fgcmModelMagErrs = FgcmModelMagErrors(self.fgcmConfig,
                                                   self.fgcmPars,
                                                   self.fgcmStars)

        # Get m^std, <m^std>, SED for all the stars.
        parArray = self.fgcmPars.getParArray(fitterUnits=False)
        if (not self.initialCycle):
            # get the SED from the chisq function
            self.fgcmChisq(parArray,computeSEDSlopes=True,includeReserve=True)

            # flag stars that are outside the color cuts
            self.fgcmStars.performColorCuts()

        else:
            # need to go through the bright observations

            # first, use fgcmChisq to compute m^std for every observation
            ## FIXME: check that this should be allExposures = True
            self.fgcmChisq(parArray,allExposures=True,includeReserve=True)

            # run the bright observation algorithm, computing SEDs
            brightObs = FgcmBrightObs(self.fgcmConfig,self.fgcmPars,self.fgcmStars,self.fgcmLUT)
            brightObs.brightestObsMeanMag(computeSEDSlopes=True)

            self.fgcmLog.info(getMemoryString('FitCycle Post Bright-Obs'))

            # flag stars that are outside our color cuts
            self.fgcmStars.performColorCuts()

            # get expGray for the initial selection
            self.fgcmGray.computeExpGrayForInitialSelection()

            # and select good exposures/flag bad exposures
            self.expSelector.selectGoodExposuresInitialSelection(self.fgcmGray)

            # reflag bad stars with too few observations
            #  (we don't go back and select exposures at this point)
            goodExpsIndex, = np.where(self.fgcmPars.expFlag == 0)
            self.fgcmStars.selectStarsMinObsExpIndex(goodExpsIndex)

            # Compute absolute magnitude starting points (if appropriate)
            if self.fgcmStars.hasRefstars:
                deltaAbsOffset = self.fgcmStars.computeAbsOffset()
                self.fgcmPars.compAbsThroughput *= 10.**(-deltaAbsOffset / 2.5)
                # and need to apply this offset to the mags...
                self.fgcmStars.applyAbsOffset(deltaAbsOffset)

                for i, band in enumerate(self.fgcmPars.bands):
                    self.fgcmLog.info("Initial abs throughput in %s band = %.4f" %
                                      (band, self.fgcmPars.compAbsThroughput[i]))

            # Compute the slopes (initial guess).  Don't plot here, offsets make no sense.
            self.fgcmQeSysSlope.computeQeSysSlope('initial')
            self.fgcmQeSysSlope.plotQeSysRefStars('initial')

            if (self.fgcmConfig.precomputeSuperStarInitialCycle):
                # we want to precompute the superstar flat here...
                self.fgcmLog.info('Configured to precompute superstar flat on initial cycle')
                # Flag superstar outliers here before computing superstar...
                self.fgcmStars.performSuperStarOutlierCuts(self.fgcmPars)

                # Might need option here for no ref stars!
                # Something with the > 1.0.  WTF?
                preSuperStarFlat = FgcmSuperStarFlat(self.fgcmConfig,self.fgcmPars,self.fgcmStars)
                preSuperStarFlat.computeSuperStarFlats(doPlots=False, doNotUseSubCCD=True, onlyObsErr=True, forceZeroMean=True)

                self.fgcmLog.debug('FitCycle is applying pre-computed SuperStarFlat')
                self.fgcmStars.applySuperStarFlat(self.fgcmPars)

            # Last thing: fit the mag errors (if configured)...
            self.fgcmModelMagErrs.computeMagErrorModel('initial')

        # Select calibratable nights
        self.expSelector.selectCalibratableNights()

        # We need one last selection to cut last of bad stars
        goodExpsIndex, = np.where(self.fgcmPars.expFlag == 0)
        self.fgcmStars.selectStarsMinObsExpIndex(goodExpsIndex)

        # And apply the errors (if configured)
        self.fgcmStars.computeModelMagErrors(self.fgcmPars)

        # Reset the superstar outlier flags and compute them now that we have
        # flagged good exposures, good nights, etc.
        self.fgcmStars.performSuperStarOutlierCuts(self.fgcmPars, reset=True)

        # And compute the step units
        parArray = self.fgcmPars.getParArray(fitterUnits=False)
        self.fgcmComputeStepUnits.run(parArray)

        # Make connectivity maps with what we know about photometric selection
        # This code doesn't work properly, skip it for now.
        # fgcmCon = FgcmConnectivity(self.fgcmConfig, self.fgcmPars, self.fgcmStars)
        # fgcmCon.plotConnectivity()

        # Finally, reset the atmosphere parameters if desired (prior to fitting)
        if self.fgcmConfig.resetParameters:
            self.fgcmPars.resetAtmosphereParameters()

        self.fgcmLog.info(getMemoryString('FitCycle Pre-Fit'))

        # Perform Fit (subroutine)
        if (self.fgcmConfig.maxIter > 0):
            self._doFit(ignoreRef=False)
        else:
            self.fgcmLog.info('FitCycle skipping fit because maxIter == 0')

            # However, we do recompute the absolute offset at this point for
            # total consistency

            if self.fgcmStars.hasRefstars:
                self.fgcmLog.info("Final computation of absolute offset.")
                deltaAbsOffset = self.fgcmStars.computeAbsOffset()
                self.fgcmPars.compAbsThroughput *= 10.**(-deltaAbsOffset / 2.5)
                self.fgcmStars.applyAbsOffset(deltaAbsOffset)

        # Plot the parameters whether or not we did a fit!
        self.fgcmPars.plotParameters()

        self.fgcmLog.info(getMemoryString('FitCycle Post-Fit'))

        # another run to soak up the reserve stars...
        # FIXME: look for more efficient way of doing this
        self.fgcmLog.debug('FitCycle computing FgcmChisq all + reserve stars')
        _ = self.fgcmChisq(self.fgcmPars.getParArray(), includeReserve=True)

        # One last run to compute mstd all observations of all exposures
        #  when allExposures is set, mean mags, etc aren't computed
        self.fgcmLog.debug('FitCycle Computing FgcmChisq all exposures')

        _ = self.fgcmChisq(self.fgcmPars.getParArray(), allExposures=True, includeReserve=True)


        self.fgcmLog.info(getMemoryString('After recomputing chisq for all exposures'))

        # Compute CCD^gray and EXP^gray
        self.fgcmLog.debug('FitCycle computing Exp and CCD Gray')
        self.fgcmGray.computeCCDAndExpGray()
        self.fgcmGray.computeExpGrayColorSplit()
        # This must be here before we modify the selection
        self.updatedPhotometricCut, self.updatedHighCut = self.fgcmGray.computeExpGrayCuts()
        self.fgcmLog.info(getMemoryString('After computing CCD and Exp Gray'))

        # Compute sigFgcm
        self.fgcmLog.debug('FitCycle computing sigFgcm')
        self.fgcmSigFgcm = FgcmSigFgcm(self.fgcmConfig,self.fgcmPars,
                                       self.fgcmStars)
        # first compute with all...(better stats)
        self.fgcmSigFgcm.computeSigFgcm(reserved=False, save=True)
        self.fgcmSigFgcm.computeSigFgcm(reserved=True, save=False)

        self.fgcmLog.info(getMemoryString('After computing sigFGCM'))

        # Flag variables for next cycle
        # NOT USED NOW
        #self.fgcmLog.debug('FitCycle flagging variables')
        #self.fgcmFlagVars = FgcmFlagVariables(self.fgcmConfig,self.fgcmPars,
        #                                      self.fgcmStars)
        #self.fgcmFlagVars.flagVariables()

        #self.fgcmLog.info(getMemoryString('After flagging variables'))

        # Re-flag exposures for superstar, aperture, etc.
        self.fgcmLog.debug('FitCycle re-selecting good exposures')
        self.expSelector.selectGoodExposures()

        # Compute Retrieved chromatic integrals
        self.fgcmLog.debug('FitCycle computing retrieved R0/R1')
        self.fgcmRetrieval = FgcmRetrieval(self.fgcmConfig,self.fgcmPars,
                                           self.fgcmStars,self.fgcmLUT)
        self.fgcmRetrieval.computeRetrievalIntegrals()

        self.fgcmLog.info(getMemoryString('After computing retrieved integrals'))

        # Compute Retrieved PWV -- always because why not?
        self.fgcmLog.debug('FitCycle computing RPWV')
        self.fgcmRetrieveAtmosphere = FgcmRetrieveAtmosphere(self.fgcmConfig, self.fgcmLUT,
                                                             self.fgcmPars)
        self.fgcmRetrieveAtmosphere.r1ToPwv(self.fgcmRetrieval)
        # NOTE that neither of these are correct, nor do I think they help at the moment.
        #self.fgcmRetrieveAtmosphere.r0ToNightlyTau(self.fgcmRetrieval)
        #self.fgcmRetrieveAtmosphere.expGrayToNightlyTau(self.fgcmGray)


        # Compute SuperStar Flats
        self.fgcmLog.debug('FitCycle computing SuperStarFlats')
        superStarFlat = FgcmSuperStarFlat(self.fgcmConfig,self.fgcmPars,self.fgcmStars)
        superStarFlat.computeSuperStarFlats()

        self.fgcmLog.info(getMemoryString('After computing superstar flats'))

        # Compute Aperture Corrections
        self.fgcmLog.debug('FitCycle computing ApertureCorrections')
        aperCorr = FgcmApertureCorrection(self.fgcmConfig,self.fgcmPars,self.fgcmGray)
        aperCorr.computeApertureCorrections()

        self.fgcmLog.info(getMemoryString('After computing aperture corrections'))

        self.fgcmLog.debug('FitCycle computing qe sys slope')
        self.fgcmQeSysSlope.computeQeSysSlope('final')
        self.fgcmQeSysSlope.plotQeSysRefStars('final')

        self.fgcmLog.info(getMemoryString('After computing qe sys slope'))

        # Compute mag error model (if configured)
        self.fgcmModelMagErrs.computeMagErrorModel('postfit')

        ## MAYBE:
        #   apply superstar and aperture corrections to grays
        #   if we don't the zeropoints before convergence will be wrong.

        self.fgcmLog.debug('FitCycle computing SigmaCal')
        sigCal = FgcmSigmaCal(self.fgcmConfig, self.fgcmPars, self.fgcmStars, self.fgcmGray)
        sigCal.run()

        if self.fgcmStars.hasRefstars:
            self.fgcmLog.debug('FitCycle computing SigmaRef')
            sigRef = FgcmSigmaRef(self.fgcmConfig, self.fgcmPars, self.fgcmStars)
            sigRef.computeSigmaRef()

        # Make Zeropoints
        # We always want to compute these because of the plots
        # In the future we might want to streamline if something is bogging down.

        self.fgcmZpts = FgcmZeropoints(self.fgcmConfig, self.fgcmPars,
                                       self.fgcmLUT, self.fgcmGray,
                                       self.fgcmRetrieval, self.fgcmStars)
        self.fgcmLog.debug('FitCycle computing zeropoints.')
        self.fgcmZpts.computeZeropoints()

        # And finally compute the stars and test repeatability *after* the crunch
        self.fgcmLog.info('Using FgcmChisq to compute mags with CCD crunch')
        _ = self.fgcmChisq(self.fgcmPars.getParArray(), includeReserve=True,
                           fgcmGray=self.fgcmGray)

        self.fgcmSigFgcm.computeSigFgcm(reserved=True, save=False, crunch=True)

        self.fgcmLog.info(getMemoryString('After computing zeropoints'))

        if (self.useFits):
            if self.fgcmConfig.outputZeropoints:
                self.fgcmZpts.saveZptFits()
                self.fgcmZpts.saveAtmFits()

            # Save parameters
            outParFile = '%s/%s_parameters.fits' % (self.fgcmConfig.outputPath,
                                                    self.fgcmConfig.outfileBaseWithCycle)
            self.fgcmPars.saveParsFits(outParFile)

            # Save bad stars
            outFlagStarFile = '%s/%s_flaggedstars.fits' % (self.fgcmConfig.outputPath,
                                                           self.fgcmConfig.outfileBaseWithCycle)
            self.fgcmStars.saveFlagStarIndices(outFlagStarFile)

            if self.fgcmConfig.outputStars:
                outStarFile = '%s/%s_stdstars.fits' % (self.fgcmConfig.outputPath,
                                                       self.fgcmConfig.outfileBaseWithCycle)
                self.fgcmStars.saveStdStars(outStarFile, self.fgcmPars)

            # Auto-update photometric cuts
            self.fgcmConfig.expGrayPhotometricCut[:] = self.updatedPhotometricCut
            self.fgcmConfig.expGrayHighCut[:] = self.updatedHighCut

            # Save yaml for input to next fit cycle
            outConfFile = '%s/%s_cycle%02d_config.yml' % (self.fgcmConfig.outputPath,
                                                          self.fgcmConfig.outfileBase,
                                                          self.fgcmConfig.cycleNumber+1)
            self.fgcmConfig.saveConfigForNextCycle(outConfFile,outParFile,outFlagStarFile)


        # and make map of coverage

        self.fgcmLog.info('Making map of coverage')
        goodExpsIndex, = np.where(self.fgcmPars.expFlag == 0)
        self.fgcmStars.selectStarsMinObsExpIndex(goodExpsIndex)
        self.fgcmStars.plotStarMap(mapType='final')

        self.fgcmLog.info(getMemoryString('FitCycle Completed'))

    def _doFit(self, doPlots=True, ignoreRef=False, maxIter=None):
        """
        Internal method to do the fit using fmin_l_bfgs_b
        """

        self.fgcmLog.info('Performing fit with %d iterations.' %
                         (self.fgcmConfig.maxIter))

        if maxIter is None:
            maxIter = self.fgcmConfig.maxIter

        # get the initial parameters
        parInitial = self.fgcmPars.getParArray(fitterUnits=True)
        # and the fit bounds
        parBounds = self.fgcmPars.getParBounds(fitterUnits=True)

        # reset the chisq list (for plotting)
        self.fgcmChisq.resetFitChisqList()
        self.fgcmChisq.clearMatchCache()
        self.fgcmChisq.maxIterations = maxIter

        # In the fit, we want to compute the absolute offset if needed.  Otherwise, no.
        computeAbsThroughput = self.fgcmStars.hasRefstars

        try:
            fun = optimize.optimize.MemoizeJac(self.fgcmChisq)
            jac = fun.derivative

            res = optimize.minimize(fun,
                                    parInitial,
                                    args=(True,True,False,False,computeAbsThroughput,ignoreRef),
                                    method='L-BFGS-B',
                                    jac=jac,
                                    bounds=parBounds,
                                    options={'maxfun': maxIter,
                                             'maxiter': maxIter,
                                             'maxcor': 20,
                                             'gtol': self.fgcmConfig.fitGradientTolerance},
                                    callback=None)
            pars = res.x

            chisq = self.fgcmChisq.fitChisqs[-1]
        except MaxFitIterations:
            # We have exceeded the maximum number of iterations, force a cut
            pars = self.fgcmPars.getParArray(fitterUnits=True)
            chisq = self.fgcmChisq.fitChisqs[-1]
            info = None

        self.fgcmLog.info('Fit completed.  Final chi^2/DOF = %.6f' % (chisq))
        self.fgcmChisq.clearMatchCache()
        self.fgcmChisq.maxIterations = -1

        if (doPlots):
            fig=plt.figure(1,figsize=(8,6))
            fig.clf()
            ax=fig.add_subplot(111)

            chisqValues = np.array(self.fgcmChisq.fitChisqs)

            ax.plot(np.arange(chisqValues.size),chisqValues,'k.')

            ax.set_xlabel(r'$\mathrm{Iteration}$',fontsize=16)
            ax.set_ylabel(r'$\chi^2/\mathrm{DOF}$',fontsize=16)

            ax.set_xlim(-0.5,self.fgcmConfig.maxIter+0.5)
            ax.set_ylim(chisqValues[-1]-0.5,chisqValues[0]+0.5)

            fig.savefig('%s/%s_chisq_fit.png' % (self.fgcmConfig.plotPath,
                                                 self.fgcmConfig.outfileBaseWithCycle))
            plt.close(fig)

        # record new parameters
        self.fgcmPars.reloadParArray(pars, fitterUnits=True)

