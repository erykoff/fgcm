from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import scipy.optimize as optimize

from fgcmConfig import FgcmConfig
from fgcmParameters import FgcmParameters
from fgcmChisq import FgcmChisq
from fgcmStars import FgcmStars
from fgcmLUT import FgcmLUTSHM
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
        self.fgcmLog.log('DEBUG','Making FgcmStars')
        self.fgcmStars = FgcmStars(self.fgcmConfig)

        # Read in LUT
        self.fgcmLog.log('DEBUG','Making FgcmLUT')
        self.fgcmLUT = FgcmLUTSHM(self.fgcmConfig.lutFile)

        # And prepare the chisq function
        self.fgcmLog.log('DEBUG','Making FgcmChisq')
        self.fgcmChisq = FgcmChisq(self.fgcmConfig,self.fgcmPars,
                                   self.fgcmStars,self.fgcmLUT)

        # And the exposure selector
        self.fgcmLog.log('DEBUG','Making FgcmExposureSelector')
        self.expSelector = FgcmExposureSelector(self.fgcmConfig,self.fgcmPars)

        # And the Gray code
        self.fgcmLog.log('DEBUG','Making FgcmGray')
        self.fgcmGray = FgcmGray(self.fgcmConfig,self.fgcmPars,self.fgcmStars)

        # Apply aperture corrections and SuperStar if available
        # select exposures...
        if (not initialCycle):
            ## FIXME: write code to apply aperture corrections and superstar flats


            # and flag exposures using quantities computed from previous cycle
            self.fgcmLog.log('DEBUG','Running selectGoodExposures()')
            self.expSelector.selectGoodExposures()



        # Flag stars with too few exposures
        goodExpsIndex, = np.where(self.fgcmPars.expFlag == 0)
        self.fgcmLog.log('DEBUG','Finding good stars from %d good exposures' % (goodExpsIndex.size))
        self.fgcmStars.selectStarsMinObs(goodExpsIndex)

        # Get m^std, <m^std>, SED for all the stars.
        parArray = fgcmPars.getParArray(fitterUnits=False)
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
            brightObs.selectGoodStars(computeSEDSlopes=True)

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

        # Perform Fit (subroutine)
        self._doFit()

        # One last run to compute mstd all observations of all exposures
        _ = self.fgcmChisq(self.fgcmPars.getParArray(),allExposures=True)

        # Compute CCD^gray and EXP^gray
        fgcmGray.computeCCDAndExpGray()

        # Re-flag exposures for superstar, aperture, etc.
        self.expSelector.selectGoodExposures()

        # Compute Retrieved chromatic integrals
        retrieval = FgcmRetrieval(self.fgcmConfig,self.fgcmPars,self.fgcmLUT)
        retrieval.computeRetrievedIntegrals()

        # Compute SuperStar Flats
        superStarFlat = FgcmSuperStarFlat(self.fgcmConfig,self.fgcmPars,self.fgcmGray)
        superStarFlat.computeSuperStarFlats()

        # Compute Aperture Corrections
        aperCorr = FgcmApertureCorrection(self.fgcmConfig,self.fgcmPars,self.fgcmGray)
        aperCorr.computeApertureCorrections()

        ## FIXME:
        # I think we need to apply the superstar flats and aperture corrections to the Grays...automatically?

        # Make Zeropoints
        fgcmZpts = FgcmZeropoints(fgcmConfig,fgcmPars,fgcmLUT,fgcmGray,fgcmRetrieval)
        fgcmZpts.computeZeropoints()

        # Save parameters
        outParFile = '%s/%s_parameters.fits' % (self.fgcmConfig.outputPath,
                                                self.fgcmConfig.outfileBaseWithCycle)
        self.fgcmPars.saveParFile(outParFile)

        # Save yaml for input to next fit cycle
        ## FIXME: Write this code

    def _doFit(self):
        """
        """

        self.fgcmLog.log('INFO','Performing fit with %d iterations.' %
                         (self.fgcmConfig.maxIter))
        
        # get the initial parameters
        parInitial = self.fgcmPars.getParArray(fitterUnits=True)
        # and the fit bounds
        parBounds = self.fgcmPars.getParBounds(fitterUnits=True)

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

        # FIXME: add plotting of chisq

        # save new parameters
        self.fgcmPars.reloadParArray(pars, fitterUnits=True)
