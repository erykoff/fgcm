from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import scipy.optimize as optimize

import matplotlib.pyplot as plt

from fgcmConfig import FgcmConfig
from fgcmParameters import FgcmParameters
from fgcmChisq import FgcmChisq
from fgcmStars import FgcmStars
from fgcmLUT import FgcmLUTSHM
from fgcmGray import FgcmGray
from fgcmZeropoints import FgcmZeropoints
from fgcmSuperStarFlat import FgcmSuperStarFlat
from fgcmRetrieval import FgcmRetrieval
from fgcmApertureCorrection import FgcmApertureCorrection
from fgcmBrightObs import FgcmBrightObs
from fgcmExposureSelector import FgcmExposureSelector

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmFitCycle(object):
    """
    """
    def __init__(self,configFile):
        self.fgcmConfig = FgcmConfig(configFile)
        self.fgcmLog = self.fgcmConfig.fgcmLog

    def run(self):
        """
        """

        # unsure whether memory usage should be start or not
        self.fgcmLog.logMemoryUsage('INFO','FitCycle Start')

        # Check if this is the initial cycle
        initialCycle = False
        if (self.fgcmConfig.cycleNumber == 0):
            initialCycle = True

        # Generate or Read Parameters
        if (initialCycle):
            self.fgcmLog.log('INFO','Fit initial cycle starting...')
            self.fgcmPars = FgcmParameters(self.fgcmConfig)
        else:
            self.fgcmLog.log('INFO','Fit cycle %d starting...' % (self.fgcmConfig.cycleNumber))
            self.fgcmPars = FgcmParameters(self.fgcmConfig,
                                           parFile=self.fgcmConfig.inParameterFile)


        # Read in Stars
        self.fgcmLog.log('DEBUG','FitCycle is making FgcmStars')
        self.fgcmStars = FgcmStars(self.fgcmConfig,self.fgcmPars)

        # Read in LUT
        self.fgcmLog.log('DEBUG','FitCycle is making FgcmLUT')
        self.fgcmLUT = FgcmLUTSHM(self.fgcmConfig.lutFile)

        # And prepare the chisq function
        self.fgcmLog.log('DEBUG','FitCycle is making FgcmChisq')
        self.fgcmChisq = FgcmChisq(self.fgcmConfig,self.fgcmPars,
                                   self.fgcmStars,self.fgcmLUT)

        # And the exposure selector
        self.fgcmLog.log('DEBUG','FitCycle is making FgcmExposureSelector')
        self.expSelector = FgcmExposureSelector(self.fgcmConfig,self.fgcmPars)

        # And the Gray code
        self.fgcmLog.log('DEBUG','FitCycle is making FgcmGray')
        self.fgcmGray = FgcmGray(self.fgcmConfig,self.fgcmPars,self.fgcmStars)

        self.fgcmLog.logMemoryUsage('INFO','FitCycle Prepared')

        # Apply aperture corrections and SuperStar if available
        # select exposures...
        if (not initialCycle):
            self.fgcmLog.log('DEBUG','FitCycle is applying SuperStarFlat')
            self.fgcmStars.applySuperStarFlat(self.fgcmPars)
            self.fgcmLog.log('DEBUG','FitCycle is applying ApertureCorrection')
            self.fgcmStars.applyApertureCorrection(self.fgcmPars)

            # and flag exposures using quantities computed from previous cycle
            self.fgcmLog.log('DEBUG','FitCycle is running selectGoodExposures()')
            self.expSelector.selectGoodExposures()

        # Flag stars with too few exposures
        goodExpsIndex, = np.where(self.fgcmPars.expFlag == 0)
        self.fgcmLog.log('DEBUG','FitCycle is finding good stars from %d good exposures' % (goodExpsIndex.size))
        self.fgcmStars.selectStarsMinObs(goodExpsIndex=goodExpsIndex)

        # Get m^std, <m^std>, SED for all the stars.
        parArray = self.fgcmPars.getParArray(fitterUnits=False)
        if (not initialCycle):
            # get the SED from the chisq function
            self.fgcmChisq(parArray,computeSEDSlopes=True)

            # flag stars that are outside the color cuts
            self.fgcmStars.performColorCuts()

        else:
            # need to go through the bright observations

            # first, use fgcmChisq to compute m^std for every observation
            self.fgcmChisq(parArray)

            # run the bright observation algorithm, computing SEDs
            brightObs = FgcmBrightObs(self.fgcmConfig,self.fgcmPars,self.fgcmStars)
            brightObs.brightestObsMeanMag(computeSEDSlopes=True)

            self.fgcmLog.logMemoryUsage('INFO','FitCycle Post Bright-Obs')

            # flag stars that are outside our color cuts
            self.fgcmStars.performColorCuts()

            # get expGray for the initial selection
            self.fgcmGray.computeExpGrayForInitialSelection()

            # and select good exposures/flag bad exposures
            self.expSelector.selectGoodExposuresInitialSelection(self.fgcmGray)

            # reflag bad stars with too few observations
            #  (we don't go back and select exposures at this point)
            goodExpsIndex, = np.where(self.fgcmPars.expFlag == 0)
            self.fgcmStars.selectStarsMinObs(goodExpsIndex=goodExpsIndex)


        # Select calibratable nights
        self.expSelector.selectCalibratableNights()

        # We need one last selection to cut last of bad stars
        goodExpsIndex, = np.where(self.fgcmPars.expFlag == 0)
        self.fgcmStars.selectStarsMinObs(goodExpsIndex=goodExpsIndex)

        self.fgcmLog.logMemoryUsage('INFO','FitCycle Pre-Fit')

        # Perform Fit (subroutine)
        if (self.fgcmConfig.maxIter > 0):
            self._doFit()
            self.fgcmPars.plotParameters()
        else:
            self.fgcmLog.log('INFO','FitCycle skipping fit because maxIter == 0')

        self.fgcmLog.logMemoryUsage('INFO','FitCycle Post-Fit')

        # One last run to compute mstd all observations of all exposures
        self.fgcmLog.log('DEBUG','FitCycle Computing FgcmChisq all exposures')
        _ = self.fgcmChisq(self.fgcmPars.getParArray(),allExposures=True)


        # Compute CCD^gray and EXP^gray
        self.fgcmLog.log('DEBUG','FitCycle computing Exp and CCD Gray')
        self.fgcmGray.computeCCDAndExpGray()

        # Re-flag exposures for superstar, aperture, etc.
        self.fgcmLog.log('DEBUG','FitCycle re-selecting good exposures')
        self.expSelector.selectGoodExposures()

        # Compute Retrieved chromatic integrals
        self.fgcmLog.log('DEBUG','FitCycle computing retrieved R0/R1')
        self.fgcmRetrieval = FgcmRetrieval(self.fgcmConfig,self.fgcmPars,
                                           self.fgcmStars,self.fgcmLUT)
        self.fgcmRetrieval.computeRetrievedIntegrals()

        # Compute SuperStar Flats
        self.fgcmLog.log('DEBUG','FitCycle computing SuperStarFlats')
        superStarFlat = FgcmSuperStarFlat(self.fgcmConfig,self.fgcmPars,self.fgcmGray)
        superStarFlat.computeSuperStarFlats()

        # Compute Aperture Corrections
        self.fgcmLog.log('DEBUG','FitCycle computing ApertureCorrections')
        aperCorr = FgcmApertureCorrection(self.fgcmConfig,self.fgcmPars,self.fgcmGray)
        aperCorr.computeApertureCorrections()

        ## MAYBE:
        #   apply superstar and aperture corrections to grays
        #   if we don't the zeropoints before convergence will be wrong.

        # Make Zeropoints -- save also
        self.fgcmLog.log('DEBUG','FitCycle computing zeropoints.')
        fgcmZpts = FgcmZeropoints(self.fgcmConfig,self.fgcmPars,
                                  self.fgcmLUT,self.fgcmGray,
                                  self.fgcmRetrieval)
        fgcmZpts.computeZeropoints()

        # Save parameters
        outParFile = '%s/%s_parameters.fits' % (self.fgcmConfig.outputPath,
                                                self.fgcmConfig.outfileBaseWithCycle)
        self.fgcmPars.saveParFile(outParFile)

        # Save yaml for input to next fit cycle
        outConfFile = '%s/%s_cycle%02d_config.yml' % (self.fgcmConfig.outputPath,
                                                      self.fgcmConfig.outfileBase,
                                                      self.fgcmConfig.cycleNumber+1)
        self.fgcmConfig.saveConfigForNextCycle(outConfFile,outParFile)


        self.fgcmLog.logMemoryUsage('INFO','FitCycle Completed')

    def _doFit(self,doPlots=True):
        """
        """

        self.fgcmLog.log('INFO','Performing fit with %d iterations.' %
                         (self.fgcmConfig.maxIter))

        # get the initial parameters
        parInitial = self.fgcmPars.getParArray(fitterUnits=True)
        # and the fit bounds
        parBounds = self.fgcmPars.getParBounds(fitterUnits=True)

        # reset the chisq list (for plotting)
        self.fgcmChisq.resetFitChisqList()

        pars, chisq, info = optimize.fmin_l_bfgs_b(self.fgcmChisq,   # chisq function
                                                   parInitial,       # initial guess
                                                   fprime=None,      # in fgcmChisq()
                                                   args=(True,True), # fitterUnits, deriv
                                                   approx_grad=False,# don't approx grad
                                                   bounds=parBounds, # boundaries
                                                   m=10,             # "variable metric conditions"
                                                   factr=1e2,        # highish accuracy
                                                   pgtol=1e-9,       # gradient tolerance
                                                   maxfun=self.fgcmConfig.maxIter,
                                                   maxiter=self.fgcmConfig.maxIter,
                                                   iprint=0,         # only one output
                                                   callback=None)    # no callback

        self.fgcmLog.log('INFO','Fit completed.  Final chi^2/DOF = %.2f' % (chisq))

        if (doPlots):
            fig=plt.figure(1)
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

        # record new parameters
        self.fgcmPars.reloadParArray(pars, fitterUnits=True)
