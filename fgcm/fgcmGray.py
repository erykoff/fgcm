from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import os
import sys
import esutil
import time
import scipy.optimize

import matplotlib.pyplot as plt


from .fgcmUtilities import gaussFunction
from .fgcmUtilities import histoGauss
from .fgcmUtilities import Cheb2dField

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmGray(object):
    """
    Class which computes ccd and exposure gray residuals.

    parameters
    ----------
    fgcmConfig: FgcmConfig
       Config object
    fgcmPars: FgcmParameters
       Parameter object
    fgcmStars: FgcmStars
       Star object

    Config variables
    ----------------
    minStarPerCCD: int
       Minimum number of stars on a CCD to compute CCD Gray
    minStarPerExp: int
       Minumum number of stars per exposure for *initial* exposure gray
    maxCCDGrayErr: float
       Maximum CCD gray error to be considered "good" to use in exposure gray
    ccdGrayMaxStarErr: float
       Maximum error for any star observation to be used to compute CCD Gray
    expGrayInitialCut: float
       Maximum initial exp gray to be used in plotting
    expGrayCheckDeltaT: float
       Time difference between exposures to check for correlated residuals (plots only)
    """

    def __init__(self,fgcmConfig,fgcmPars,fgcmStars):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.debug('Initializing fgcmGray')

        # need fgcmPars because it tracks good exposures
        #  also this is where the gray info is stored
        self.fgcmPars = fgcmPars

        # need fgcmStars because it has the stars (duh)
        self.fgcmStars = fgcmStars

        # and record configuration variables...
        self.minStarPerCCD = fgcmConfig.minStarPerCCD
        self.minStarPerExp = fgcmConfig.minStarPerExp
        self.maxCCDGrayErr = fgcmConfig.maxCCDGrayErr
        self.ccdGrayMaxStarErr = fgcmConfig.ccdGrayMaxStarErr
        self.ccdGraySubCCD = fgcmConfig.ccdGraySubCCD
        self.ccdGraySubCCDChebyshevOrder = fgcmConfig.ccdGraySubCCDChebyshevOrder
        self.ccdGraySubCCDTriangular = fgcmConfig.ccdGraySubCCDTriangular
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.illegalValue = fgcmConfig.illegalValue
        self.expGrayInitialCut = fgcmConfig.expGrayInitialCut
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber
        self.expGrayCheckDeltaT = fgcmConfig.expGrayCheckDeltaT
        self.colorSplitIndices = fgcmConfig.colorSplitIndices
        self.bandFitIndex = fgcmConfig.bandFitIndex
        self.bandRequiredIndex = fgcmConfig.bandRequiredIndex
        self.bandNotRequiredIndex = fgcmConfig.bandNotRequiredIndex
        self.ccdOffsets = fgcmConfig.ccdOffsets
        self.quietMode = fgcmConfig.quietMode

        self.expGrayPhotometricCut = fgcmConfig.expGrayPhotometricCut
        self.expGrayHighCut = fgcmConfig.expGrayHighCut
        self.autoPhotometricCutNSig = fgcmConfig.autoPhotometricCutNSig
        self.autoPhotometricCutStep = fgcmConfig.autoPhotometricCutStep
        self.autoHighCutNSig = fgcmConfig.autoHighCutNSig

        self.arraysPrepared = False

        self._prepareGrayArrays()

    def _prepareGrayArrays(self):
        """
        Internal method to create shared-memory arrays.
        """

        # we have expGray for Selection
        self.expGrayForInitialSelectionHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayRMSForInitialSelectionHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expNGoodStarForInitialSelectionHandle = snmm.createArray(self.fgcmPars.nExp,dtype='i4')

        # and the exp/ccd gray for the zeropoints

        self.ccdGrayHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdGrayRMSHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdGrayErrHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdNGoodObsHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='i4')
        self.ccdNGoodStarsHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='i4')
        self.ccdNGoodTilingsHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')

        if self.ccdGraySubCCD:
            order = self.ccdGraySubCCDChebyshevOrder
            self.ccdGraySubCCDParsHandle = snmm.createArray((self.fgcmPars.nExp, self.fgcmPars.nCCD, (order + 1) * (order + 1)), dtype='f8')
            self.ccdGrayNPar = (order + 1) * (order + 1)

        self.expGrayHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayRMSHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayErrHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expNGoodStarsHandle = snmm.createArray(self.fgcmPars.nExp,dtype='i4')
        self.expNGoodCCDsHandle = snmm.createArray(self.fgcmPars.nExp,dtype='i2')
        self.expNGoodTilingsHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')

        self.expGrayColorSplitHandle = snmm.createArray((self.fgcmPars.nExp, 3), dtype='f8')
        self.expGrayRMSColorSplitHandle = snmm.createArray((self.fgcmPars.nExp, 3), dtype='f8')
        self.expGrayErrColorSplitHandle = snmm.createArray((self.fgcmPars.nExp, 3), dtype='f8')
        self.expGrayNGoodStarsColorSplitHandle = snmm.createArray((self.fgcmPars.nExp, 3), dtype='i2')

        self.arraysPrepared = True

    def computeExpGrayForInitialSelection(self):
        """
        Compute exposure gray using bright star magnitudes to get initial estimates.

        """

        if (not self.fgcmStars.magStdComputed):
            raise RuntimeError("Must run FgcmChisq to compute magStd before computeExpGrayForInitialSelection")

        # Note this computes ExpGray for all exposures, good and bad

        startTime = time.time()
        self.fgcmLog.debug('Computing ExpGray for initial selection')

        # useful numbers
        expGrayForInitialSelection = snmm.getArray(self.expGrayForInitialSelectionHandle)
        expGrayRMSForInitialSelection = snmm.getArray(self.expGrayRMSForInitialSelectionHandle)
        expNGoodStarForInitialSelection = snmm.getArray(self.expNGoodStarForInitialSelectionHandle)

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        # first, we need to compute E_gray == <mstd> - mstd for each observation

        # compute all the EGray values

        EGray = np.zeros(self.fgcmStars.nStarObs,dtype='f8')
        EGray[obsIndex] = (objMagStdMean[obsObjIDIndex[obsIndex],obsBandIndex[obsIndex]] -
                           obsMagStd[obsIndex])

        # only use good observations of good stars...

        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=False, checkMinObs=True)

        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars)

        self.fgcmLog.debug('FgcmGray initial exp gray using %d observations from %d good stars.' %
                           (goodObs.size,goodStars.size))

        # Now only observations that have the minimum number of good observations are
        # selected, even in the "off-bands"

        # now group per exposure and sum...

        expGrayForInitialSelection[:] = 0.0
        expGrayRMSForInitialSelection[:] = 0.0
        expNGoodStarForInitialSelection[:] = 0

        np.add.at(expGrayForInitialSelection,
                  obsExpIndex[goodObs],
                  EGray[goodObs])
        np.add.at(expGrayRMSForInitialSelection,
                  obsExpIndex[goodObs],
                  EGray[goodObs]**2.)
        np.add.at(expNGoodStarForInitialSelection,
                  obsExpIndex[goodObs],
                  1)

        gd,=np.where(expNGoodStarForInitialSelection > 0)
        expGrayForInitialSelection[gd] /= expNGoodStarForInitialSelection[gd]
        expGrayRMSForInitialSelection[gd] = np.sqrt((expGrayRMSForInitialSelection[gd]/expNGoodStarForInitialSelection[gd]) -
                                             (expGrayForInitialSelection[gd])**2.)

        if not self.quietMode:
            self.fgcmLog.info('ExpGray for initial selection computed for %d exposures.' %
                              (gd.size))
            self.fgcmLog.info('Computed ExpGray for initial selection in %.2f seconds.' %
                              (time.time() - startTime))

        if self.plotPath is None:
            return

        expUse,=np.where((self.fgcmPars.expFlag == 0) &
                         (expNGoodStarForInitialSelection > self.minStarPerExp) &
                         (expGrayForInitialSelection > self.expGrayInitialCut))

        for i in xrange(self.fgcmPars.nBands):
            self.fgcmLog.debug('Making EXP_GRAY (initial) histogram for %s band' %
                               (self.fgcmPars.bands[i]))
            inBand, = np.where(self.fgcmPars.expBandIndex[expUse] == i)

            if (inBand.size == 0) :
                continue

            fig=plt.figure(1,figsize=(8,6))
            fig.clf()

            ax=fig.add_subplot(111)

            coeff = histoGauss(ax, expGrayForInitialSelection[expUse[inBand]] * 1000.0)
            coeff[1] /= 1000.0
            coeff[2] /= 1000.0

            ax.tick_params(axis='both',which='major',labelsize=14)
            ax.locator_params(axis='x',nbins=5)

            text=r'$(%s)$' % (self.fgcmPars.bands[i]) + '\n' + \
                r'$\mathrm{Cycle\ %d}$' % (self.cycleNumber) + '\n' + \
                r'$\mu = %.2f$' % (coeff[1]*1000.0) + '\n' + \
                r'$\sigma = %.2f$' % (coeff[2]*1000.0)

            ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=16)
            ax.set_xlabel(r'$\mathrm{EXP}^{\mathrm{gray}}\,(\mathrm{initial})\,(\mathrm{mmag})$',fontsize=16)
            ax.set_ylabel(r'# of Exposures',fontsize=14)

            fig.savefig('%s/%s_initial_expgray_%s.png' % (self.plotPath,
                                                          self.outfileBaseWithCycle,
                                                          self.fgcmPars.bands[i]))
            plt.close(fig)

    def computeCCDAndExpGray(self, onlyObsErr=False):
        """
        Compute CCD and exposure gray using calibrated magnitudes.

        parameters
        ----------
        onlyObsErr: bool, default=False
           Only use observational error.  Used when making initial superstarflat estimate.
        """

        if (not self.fgcmStars.allMagStdComputed):
            raise ValueError("Must run FgcmChisq to compute magStd before computeCCDAndExpGray")

        startTime = time.time()
        self.fgcmLog.debug('Computing CCDGray and ExpGray.')

        # Note: this computes the gray values for all exposures, good and bad

        # values to set
        ccdGray = snmm.getArray(self.ccdGrayHandle)
        ccdGrayRMS = snmm.getArray(self.ccdGrayRMSHandle)
        ccdGrayErr = snmm.getArray(self.ccdGrayErrHandle)
        ccdNGoodObs = snmm.getArray(self.ccdNGoodObsHandle)
        ccdNGoodStars = snmm.getArray(self.ccdNGoodStarsHandle)
        ccdNGoodTilings = snmm.getArray(self.ccdNGoodTilingsHandle)

        if self.ccdGraySubCCD:
            ccdGraySubCCDPars = snmm.getArray(self.ccdGraySubCCDParsHandle)

        expGray = snmm.getArray(self.expGrayHandle)
        expGrayRMS = snmm.getArray(self.expGrayRMSHandle)
        expGrayErr = snmm.getArray(self.expGrayErrHandle)
        expNGoodCCDs = snmm.getArray(self.expNGoodCCDsHandle)
        expNGoodStars = snmm.getArray(self.expNGoodStarsHandle)
        expNGoodTilings = snmm.getArray(self.expNGoodTilingsHandle)

        # input numbers
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        # Only use good observations of good stars...
        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=False, checkMinObs=True)

        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, checkBadMag=True)

        # we need to compute E_gray == <mstd> - mstd for each observation

        EGrayGO, EGrayErr2GO = self.fgcmStars.computeEGray(goodObs, onlyObsErr=onlyObsErr)

        # one more cut on the maximum error
        # as well as making sure that it didn't go below zero
        gd,=np.where((EGrayErr2GO < self.ccdGrayMaxStarErr) & (EGrayErr2GO > 0.0))
        goodObs=goodObs[gd]
        EGrayGO=EGrayGO[gd]
        EGrayErr2GO=EGrayErr2GO[gd]

        if self.ccdGraySubCCD:
            obsXGO = snmm.getArray(self.fgcmStars.obsXHandle)[goodObs]
            obsYGO = snmm.getArray(self.fgcmStars.obsYHandle)[goodObs]

        self.fgcmLog.debug('FgcmGray using %d observations from %d good stars.' %
                           (goodObs.size,goodStars.size))

        # group by CCD and sum

        ## ccdGray = Sum(EGray/EGrayErr^2) / Sum(1./EGrayErr^2)
        ## ccdGrayRMS = Sqrt((Sum(EGray^2/EGrayErr^2) / Sum(1./EGrayErr^2)) - ccdGray^2)
        ## ccdGrayErr = Sqrt(1./Sum(1./EGrayErr^2))

        ccdGray[:,:] = 0.0
        ccdGrayRMS[:,:] = 0.0
        ccdGrayErr[:,:] = 0.0
        ccdNGoodObs[:,:] = 0
        ccdNGoodStars[:,:] = 0
        ccdNGoodTilings[:,:] = 0.0

        # These are things we compute no matter what:
        # This is a temporary variable
        ccdGrayWt = np.zeros_like(ccdGray)

        np.add.at(ccdGrayWt,
                  (obsExpIndex[goodObs],obsCCDIndex[goodObs]),
                  1./EGrayErr2GO)
        np.add.at(ccdNGoodStars,
                  (obsExpIndex[goodObs],obsCCDIndex[goodObs]),
                  1)
        np.add.at(ccdNGoodObs,
                  (obsExpIndex[goodObs],obsCCDIndex[goodObs]),
                  objNGoodObs[obsObjIDIndex[goodObs],
                              obsBandIndex[goodObs]])

        if not self.ccdGraySubCCD:
            np.add.at(ccdGray,
                      (obsExpIndex[goodObs],obsCCDIndex[goodObs]),
                      EGrayGO/EGrayErr2GO)
            np.add.at(ccdGrayRMS,
                      (obsExpIndex[goodObs],obsCCDIndex[goodObs]),
                      EGrayGO**2./EGrayErr2GO)

            # need at least 3 or else computation can blow up
            gd = np.where((ccdNGoodStars >= 3) & (ccdGrayWt > 0.0) & (ccdGrayRMS > 0.0))
            ccdGray[gd] /= ccdGrayWt[gd]
            tempRMS2 = np.zeros_like(ccdGrayRMS)
            tempRMS2[gd] = (ccdGrayRMS[gd]/ccdGrayWt[gd]) - (ccdGray[gd]**2.)
            ok = np.where(tempRMS2 > 0.0)
            ccdGrayRMS[ok] = np.sqrt(tempRMS2[ok])
            ccdGrayErr[gd] = np.sqrt(1./ccdGrayWt[gd])

        else:
            # We are computing on the sub-ccd scale

            # But first we need to finish the other stuff
            gd = np.where((ccdNGoodStars >= 3) & (ccdGrayWt > 0.0))
            ccdGrayErr[gd] = np.sqrt(1./ccdGrayWt[gd])
            ccdGrayRMS[gd] = 0.0  # this is unused

            # This will probably have to be parallelized
            # For now, let's write some code to do it.

            order = self.ccdGraySubCCDChebyshevOrder
            pars = np.zeros((order + 1, order + 1))
            pars[0, 0] = 1.0

            if self.ccdGraySubCCDTriangular:
                iind = np.repeat(np.arange(order + 1), order + 1)
                jind = np.tile(np.arange(order + 1), order + 1)
                lowInds, = np.where((iind + jind) <= order)
            else:
                lowInds, = np.arange(pars.size)

            FGrayGO = 10.**(EGrayGO / (-2.5))
            FGrayErrGO = (np.log(10.) / 2.5) * np.sqrt(EGrayErr2GO) * FGrayGO

            # Need to split up...
            # And then do the fit, provided we have enough stars.
            expCcdHash = (obsExpIndex[goodObs]*(self.fgcmPars.nCCD + 1) +
                          obsCCDIndex[goodObs])

            h, rev = esutil.stat.histogram(expCcdHash, rev=True)

            # Anything with 2 or fewer stars will be marked bad
            use, = np.where(h >= 3)
            for i in use:
                i1a = rev[rev[i]: rev[i + 1]]

                eInd = obsExpIndex[goodObs[i1a[0]]]
                cInd = obsCCDIndex[goodObs[i1a[0]]]

                ccdNGoodStars[eInd, cInd] = i1a.size

                computeMean = False

                if i1a.size < 10 * pars.size:
                    # insufficient stars for chebyshev fit
                    fit = pars.flatten()
                    computeMean = True
                else:
                    try:
                        field = Cheb2dField.fit(self.ccdOffsets['X_SIZE'][cInd],
                                                self.ccdOffsets['Y_SIZE'][cInd],
                                                order,
                                                obsXGO[i1a], obsYGO[i1a],
                                                FGrayGO[i1a],
                                                valueErr=FGrayErrGO[i1a],
                                                triangular=self.ccdGraySubCCDTriangular)
                        fit = field.pars.flatten()
                    except (ValueError, RuntimeError, TypeError):
                        fit = pars.flatten()
                        computeMean = True

                    if (fit[0] <= 0.0 or fit[0] == 1.0):
                        # The fit failed...
                        fit = pars.flatten()
                        computeMean = True

                if computeMean:
                    fit = pars.flatten()
                    fit[0] = (np.sum(EGrayGO[i1a]/EGrayErr2GO[i1a]) /
                              np.sum(1./EGrayErr2GO[i1a]))
                    fit[0] = 10.**(fit[0] / (-2.5))

                ccdGraySubCCDPars[eInd, cInd, :] = fit
                # Set the CCD Gray in the center
                # unsure if this should be the mean over all the stars...
                field = Cheb2dField(self.ccdOffsets['X_SIZE'][cInd],
                                    self.ccdOffsets['Y_SIZE'][cInd],
                                    fit)
                ccdGray[eInd, cInd] = -2.5 * np.log10(field.evaluateCenter())

        self.fgcmLog.debug('Computed CCDGray for %d CCDs' % (gd[0].size))

        # set illegalValue for totally bad CCDs
        bad = np.where((ccdNGoodStars <= 2) | (ccdGrayWt <= 0.0))
        ccdGray[bad] = self.illegalValue
        ccdGrayRMS[bad] = self.illegalValue
        ccdGrayErr[bad] = self.illegalValue

        # check for infinities -- these should not be here now that I fixed the weight check
        bad=np.where(~np.isfinite(ccdGrayRMS))
        ccdGrayRMS[bad] = self.illegalValue
        bad=np.where(~np.isfinite(ccdGrayErr))
        ccdGrayErr[bad] = self.illegalValue

        # and the ccdNGoodTilings...
        ccdNGoodTilings[gd] = (ccdNGoodObs[gd].astype(np.float64) /
                               ccdNGoodStars[gd].astype(np.float64))


        # group CCD by Exposure and Sum

        goodCCD = np.where((ccdNGoodStars >= self.minStarPerCCD) &
                           (ccdGrayErr > 0.0) &
                           (ccdGrayErr < self.maxCCDGrayErr))

        self.fgcmLog.debug('For ExpGray, found %d good CCDs' %
                           (goodCCD[0].size))

        # note: goodCCD[0] refers to the expIndex, goodCCD[1] to the CCDIndex

        expGray[:] = 0.0
        expGrayRMS[:] = 0.0
        expGrayErr[:] = 0.0
        expNGoodStars[:] = 0
        expNGoodCCDs[:] = 0
        expNGoodTilings[:] = 0.0

        # temporary
        expGrayWt = np.zeros_like(expGray)

        np.add.at(expGrayWt,
                  goodCCD[0],
                  1./ccdGrayErr[goodCCD]**2.)
        np.add.at(expGray,
                  goodCCD[0],
                  ccdGray[goodCCD]/ccdGrayErr[goodCCD]**2.)
        np.add.at(expGrayRMS,
                  goodCCD[0],
                  ccdGray[goodCCD]**2./ccdGrayErr[goodCCD]**2.)
        np.add.at(expNGoodCCDs,
                  goodCCD[0],
                  1)
        np.add.at(expNGoodTilings,
                  goodCCD[0],
                  ccdNGoodTilings[goodCCD])
        np.add.at(expNGoodStars,
                  goodCCD[0],
                  ccdNGoodStars[goodCCD])

        # need at least 3 or else computation can blow up
        gd, = np.where(expNGoodCCDs >= 3)
        expGray[gd] /= expGrayWt[gd]
        expGrayRMS[gd] = np.sqrt((expGrayRMS[gd]/expGrayWt[gd]) - (expGray[gd]**2.))
        expGrayErr[gd] = np.sqrt(1./expGrayWt[gd])
        expNGoodTilings[gd] /= expNGoodCCDs[gd]

        # set illegal value for non-measurements
        bad, = np.where(expNGoodCCDs <= 2)
        expGray[bad] = self.illegalValue
        expGrayRMS[bad] = self.illegalValue
        expGrayErr[bad] = self.illegalValue
        expNGoodTilings[bad] = self.illegalValue


        self.fgcmPars.compExpGray[:] = expGray
        self.fgcmPars.compVarGray[gd] = expGrayRMS[gd]**2.
        self.fgcmPars.compNGoodStarPerExp = expNGoodStars

        ##  per band we plot the expGray for photometric exposures...

        if not self.quietMode:
            self.fgcmLog.info('ExpGray computed for %d exposures.' % (gd.size))
            self.fgcmLog.info('Computed CCDGray and ExpGray in %.2f seconds.' %
                              (time.time() - startTime))

        self.makeExpGrayPlots()

    def computeExpGrayColorSplit(self):
        """
        Do a comparison of expGray splitting red/blue stars
        """

        if (not self.fgcmStars.magStdComputed):
            raise RuntimeError("Must run FgcmChisq to compute magStd before computeExpGrayColorSplit")

        startTime = time.time()
        self.fgcmLog.debug('Computing ExpGrayColorSplit')

        expGrayColorSplit = snmm.getArray(self.expGrayColorSplitHandle)
        expGrayErrColorSplit = snmm.getArray(self.expGrayErrColorSplitHandle)
        expGrayRMSColorSplit = snmm.getArray(self.expGrayRMSColorSplitHandle)
        expGrayNGoodStarsColorSplit = snmm.getArray(self.expGrayNGoodStarsColorSplitHandle)

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        # This should check that every star used has a valid g-i color
        # We also want to filter only photometric observations, because
        # that's what we're going to be using
        goodStars = self.fgcmStars.getGoodStarIndices(includeReserve=False, checkMinObs=True, checkHasColor=True)
        # Compute this for both good and bad exposures
        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, checkBadMag=True)

        EGrayGO, EGrayErr2GO = self.fgcmStars.computeEGray(goodObs, ignoreRef=True)

        gmiGO = (objMagStdMean[obsObjIDIndex[goodObs], self.colorSplitIndices[0]] -
                 objMagStdMean[obsObjIDIndex[goodObs], self.colorSplitIndices[1]])

        # Not every star will have a valid g-i color, so we need to check for that.
        st = np.argsort(gmiGO)
        gmiCutLow = np.array([gmiGO[st[0]],
                              gmiGO[st[int(0.25*st.size)]],
                              gmiGO[st[int(0.75*st.size)]]])
        gmiCutHigh = np.array([gmiGO[st[int(0.25*st.size)]],
                               gmiGO[st[int(0.75*st.size)]],
                               gmiGO[st[-1]]])
        gmiCutNames = ['Blue25', 'Middle50', 'Red25']

        expGrayColorSplit[:, :] = 0.0
        expGrayErrColorSplit[:, :] = 0.0
        expGrayRMSColorSplit[:, :] = 0.0
        expGrayNGoodStarsColorSplit[:, :] = 0
        expGrayWtColorSplit = np.zeros_like(expGrayColorSplit)

        for c in xrange(gmiCutLow.size):
            use, = np.where((gmiGO > gmiCutLow[c]) &
                            (gmiGO < gmiCutHigh[c]))

            np.add.at(expGrayColorSplit[:, c],
                      obsExpIndex[goodObs[use]],
                      EGrayGO[use] / EGrayErr2GO[use])
            np.add.at(expGrayWtColorSplit[:, c],
                      obsExpIndex[goodObs[use]],
                      1. / EGrayErr2GO[use])
            np.add.at(expGrayRMSColorSplit[:, c],
                      obsExpIndex[goodObs[use]],
                      EGrayGO[use]**2. / EGrayErr2GO[use])
            np.add.at(expGrayNGoodStarsColorSplit[:, c],
                      obsExpIndex[goodObs[use]],
                      1)

            gd, = np.where(expGrayNGoodStarsColorSplit[:, c] >= self.minStarPerExp / 4)
            expGrayColorSplit[gd, c] /= expGrayWtColorSplit[gd, c]
            expGrayErrColorSplit[gd, c] = np.sqrt(1. / expGrayWtColorSplit[gd, c])
            expGrayRMSColorSplit[gd, c] = np.sqrt((expGrayRMSColorSplit[gd, c] / expGrayWtColorSplit[gd, c]) - expGrayColorSplit[gd, c]**2.)

            bd, = np.where(expGrayNGoodStarsColorSplit[:, c] < self.minStarPerExp / 4)
            expGrayColorSplit[bd, c] = self.illegalValue
            expGrayRMSColorSplit[bd, c] = self.illegalValue
            expGrayErrColorSplit[bd, c] = self.illegalValue

        if self.plotPath is not None:
            # main plots:
            #  per band, plot expGray for red vs blue stars!

            plt.set_cmap('viridis')

            #for bandIndex in self.bandFitIndex:
            for bandIndex, band in enumerate(self.fgcmPars.bands):
                use, = np.where((self.fgcmPars.expBandIndex == bandIndex) &
                                (self.fgcmPars.expFlag == 0) &
                                (expGrayColorSplit[:, 0] > self.illegalValue) &
                                (expGrayColorSplit[:, 2] > self.illegalValue))
                if (use.size == 0):
                    self.fgcmLog.info('Could not find photometric exposures in band %d' % (bandIndex))
                    continue

                fig = plt.figure(1, figsize=(8, 6))
                fig.clf()

                ax = fig.add_subplot(111)
                ax.hexbin(expGrayColorSplit[use, 0]*1000.0, expGrayColorSplit[use, 2]*1000.0, bins='log')
                ax.set_xlabel('EXP_GRAY (%s) (%s) (mmag)' % (self.fgcmPars.bands[bandIndex], gmiCutNames[0]))
                ax.set_ylabel('EXP_GRAY (%s) (%s) (mmag)' % (self.fgcmPars.bands[bandIndex], gmiCutNames[2]))
                ax.plot([-10.0, 10.0], [-10.0, 10.0], 'r--')

                text=r'$(%s)$' % (self.fgcmPars.bands[bandIndex])
                ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=16)

                fig.savefig('%s/%s_compare-redblue-expgray_%s.png' % (self.plotPath,
                                                                      self.outfileBaseWithCycle,
                                                                      self.fgcmPars.bands[bandIndex]))
                plt.close(fig)

                # And a plot as function of time

                deltaColor = (expGrayColorSplit[use, 2] - expGrayColorSplit[use, 0]) * 1000.
                firstMJD = np.floor(np.min(self.fgcmPars.expMJD))

                st = np.argsort(deltaColor)
                extent = [np.min(self.fgcmPars.expMJD[use] - firstMJD),
                          np.max(self.fgcmPars.expMJD[use] - firstMJD),
                          deltaColor[st[int(0.01*deltaColor.size)]],
                          deltaColor[st[int(0.99*deltaColor.size)]]]

                fig = plt.figure(1, figsize=(8, 6))
                fig.clf()

                ax = fig.add_subplot(111)
                ax.hexbin(self.fgcmPars.expMJD[use] - firstMJD,
                          (expGrayColorSplit[use, 2] - expGrayColorSplit[use, 0]) * 1000.,
                          bins='log', extent=extent)
                ax.set_xlabel('MJD - %.0f' % (firstMJD))
                ax.set_ylabel('EXP_GRAY (%s) (%s) - EXP_GRAY (%s) (%s) (mmag)' %
                              (self.fgcmPars.bands[bandIndex], gmiCutNames[2],
                               self.fgcmPars.bands[bandIndex], gmiCutNames[0]))
                ax.plot([extent[0], extent[1]],
                        [0.0, 0.0], 'r--')

                text=r'$(%s)$' % (self.fgcmPars.bands[bandIndex])
                ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=16)

                plt.savefig('%s/%s_compare-redblue-expgray-mjd_%s.png' % (self.plotPath,
                                                                          self.outfileBaseWithCycle,
                                                                          self.fgcmPars.bands[bandIndex]))
                plt.close(fig)

        # and we're done...

    def makeExpGrayPlots(self):
        """
        Make exposure gray plots.
        """

        # We run this all the time because it has useful logging, but
        # we might not save the image

        # arrays we need
        expNGoodStars = snmm.getArray(self.expNGoodStarsHandle)
        expGray = snmm.getArray(self.expGrayHandle)

        expUse,=np.where((self.fgcmPars.expFlag == 0) &
                         (expNGoodStars > self.minStarPerExp))

        for i in xrange(self.fgcmPars.nBands):
            inBand, = np.where(self.fgcmPars.expBandIndex[expUse] == i)

            if (inBand.size == 0) :
                continue

            # plot histograms of EXP^gray

            fig=plt.figure(1,figsize=(8,6))
            fig.clf()

            ax=fig.add_subplot(111)

            coeff = histoGauss(ax, expGray[expUse[inBand]] * 1000.0)
            coeff[1] /= 1000.0
            coeff[2] /= 1000.0

            ax.tick_params(axis='both',which='major',labelsize=14)
            ax.locator_params(axis='x',nbins=5)

            text=r'$(%s)$' % (self.fgcmPars.bands[i]) + '\n' + \
                r'$\mathrm{Cycle\ %d}$' % (self.cycleNumber) + '\n' + \
                r'$\mu = %.2f$' % (coeff[1]*1000.0) + '\n' + \
                r'$\sigma = %.2f$' % (coeff[2]*1000.0)

            ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=16)
            ax.set_xlabel(r'$\mathrm{EXP}^{\mathrm{gray}}\,(\mathrm{mmag})$',fontsize=16)
            ax.set_ylabel(r'# of Exposures',fontsize=14)

            if self.plotPath is not None:
                fig.savefig('%s/%s_expgray_%s.png' % (self.plotPath,
                                                      self.outfileBaseWithCycle,
                                                      self.fgcmPars.bands[i]))
            plt.close(fig)

            self.fgcmLog.info("sigExpGray (%s) = %.2f mmag" % (
                    self.fgcmPars.bands[i],
                    coeff[2] * 1000.0))

            # plot EXP^gray as a function of secZenith (airmass)
            secZenith = 1./(np.sin(self.fgcmPars.expTelDec[expUse[inBand]]) *
                            self.fgcmPars.sinLatitude +
                            np.cos(self.fgcmPars.expTelDec[expUse[inBand]]) *
                            self.fgcmPars.cosLatitude *
                            np.cos(self.fgcmPars.expTelHA[expUse[inBand]]))

            # zoom in on 1<secZenith<1.5 for plotting
            ok,=np.where(secZenith < 1.5)

            fig=plt.figure(1,figsize=(8,6))
            fig.clf()

            ax=fig.add_subplot(111)

            ax.hexbin(secZenith[ok],expGray[expUse[inBand[ok]]]*1000.0,rasterized=True)

            text = r'$(%s)$' % (self.fgcmPars.bands[i])
            ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=16)

            ax.set_xlabel(r'$\mathrm{sec}(\mathrm{zd})$',fontsize=16)
            ax.set_ylabel(r'$\mathrm{EXP}^{\mathrm{gray}}\,(\mathrm{mmag})$',fontsize=16)

            if self.plotPath is not None:
                fig.savefig('%s/%s_airmass_expgray_%s.png' % (self.plotPath,
                                                              self.outfileBaseWithCycle,
                                                              self.fgcmPars.bands[i]))
            plt.close(fig)

            # plot EXP^gray as a function of UT

            fig=plt.figure(1,figsize=(8,6))
            fig.clf()

            ax=fig.add_subplot(111)

            ax.hexbin(self.fgcmPars.expDeltaUT[expUse[inBand]],
                      expGray[expUse[inBand]]*1000.0,
                      rasterized=True)
            ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=16)

            ax.set_xlabel(r'$\Delta \mathrm{UT}$',fontsize=16)
            ax.set_ylabel(r'$\mathrm{EXP}^{\mathrm{gray}}\,(\mathrm{mmag})$',fontsize=16)

            if self.plotPath is not None:
                fig.savefig('%s/%s_UT_expgray_%s.png' % (self.plotPath,
                                                         self.outfileBaseWithCycle,
                                                         self.fgcmPars.bands[i]))
            plt.close(fig)

        # and plot EXP^gray vs MJD for all bands for deep fields
        fig = plt.figure(1,figsize=(8,6))
        fig.clf()

        ax=fig.add_subplot(111)

        firstMJD = np.floor(np.min(self.fgcmPars.expMJD))

        deepUse,=np.where(self.fgcmPars.expDeepFlag[expUse] == 1)

        ax.hexbin(self.fgcmPars.expMJD[expUse[deepUse]] - firstMJD,
                  expGray[expUse[deepUse]]*1000.0, bins='log')
        ax.set_xlabel(r'$\mathrm{MJD}\ -\ %.0f$' % (firstMJD),fontsize=16)
        ax.set_ylabel(r'$\mathrm{EXP}^{\mathrm{gray}}\,(\mathrm{mmag})$',fontsize=16)

        ax.set_title(r'$\mathrm{Deep Fields}$')

        if self.plotPath is not None:
            fig.savefig('%s/%s_mjd_deep_expgray.png' % (self.plotPath,
                                                        self.outfileBaseWithCycle))
        plt.close(fig)

        # And plot correlations of EXP^gray between pairs of bands
        for ind, bandIndex0 in enumerate(self.bandFitIndex[:-2]):
            bandIndex1 = self.bandFitIndex[ind + 1]

            use0, = np.where((self.fgcmPars.expBandIndex == bandIndex0) &
                             (self.fgcmPars.expFlag == 0) &
                             (expGray > self.illegalValue))
            use1, = np.where((self.fgcmPars.expBandIndex == bandIndex1) &
                             (self.fgcmPars.expFlag == 0) &
                             (expGray > self.illegalValue))

            if use0.size == 0 or use1.size == 0:
                self.fgcmLog.info('Could not find photometric exposures in bands %d or %d' % (bandIndex0, bandIndex1))
                continue

            matchInd = np.clip(np.searchsorted(self.fgcmPars.expMJD[use0],
                                               self.fgcmPars.expMJD[use1]),
                               0,
                               use0.size-1)

            ok,=np.where(np.abs(self.fgcmPars.expMJD[use0[matchInd]] -
                                self.fgcmPars.expMJD[use1]) < self.expGrayCheckDeltaT)

            if ok.size == 0:
                self.fgcmLog.info('Could not find any matched exposures between bands %s and %s within %.2f minutes' %
                                  (self.fgcmPars.bands[bandIndex0],
                                  self.fgcmPars.bands[bandIndex1],
                                  self.expGrayCheckDeltaT * 24 * 60))
                continue

            fig=plt.figure(1,figsize=(8,6))
            fig.clf()

            ax=fig.add_subplot(111)

            ax.hexbin(expGray[use0[matchInd[ok]]]*1000.0,
                      expGray[use1[ok]]*1000.0, bins='log')
            ax.set_xlabel('EXP_GRAY (%s) (mmag)' % (self.fgcmPars.bands[bandIndex0]))
            ax.set_ylabel('EXP_GRAY (%s) (mmag)' % (self.fgcmPars.bands[bandIndex1]))
            ax.plot([-0.01 * 1000, 0.01 * 1000],[-0.01 * 1000, 0.01 * 1000],'r--')

            if self.plotPath is not None:
                fig.savefig('%s/%s_expgray-compare_%s_%s.png' % (self.plotPath,
                                                                 self.outfileBaseWithCycle,
                                                                 self.fgcmPars.bands[bandIndex0],
                                                                 self.fgcmPars.bands[bandIndex1]))
            plt.close(fig)

    def computeExpGrayCuts(self):
        """
        Compute the exposure gray recommended cuts.

        Returns
        -------
        expGrayPhotometricCut: `np.array`
           Float array (per band) of recommended expGray cuts
        expGrayHighCut: `np.array`
           Float array (per band) of recommended expGray cuts (high side)
        """
        expNGoodStars = snmm.getArray(self.expNGoodStarsHandle)
        expGray = snmm.getArray(self.expGrayHandle)

        expUse,=np.where((self.fgcmPars.expFlag == 0) &
                         (expNGoodStars > self.minStarPerExp))

        expGrayPhotometricCut = np.zeros(self.fgcmPars.nBands)
        expGrayHighCut = np.zeros_like(expGrayPhotometricCut)

        # set defaults to those that were set in the config file
        # These are going to be lists of floats for persistence
        expGrayPhotometricCut[:] = [float(f) for f in self.expGrayPhotometricCut]
        expGrayHighCut[:] = [float(f) for f in self.expGrayHighCut]

        for i in xrange(self.fgcmPars.nBands):
            inBand, = np.where(self.fgcmPars.expBandIndex[expUse] == i)

            if inBand.size == 0:
                continue

            coeff = histoGauss(None, expGray[expUse[inBand]])

            # Use nsig * sigma - mean
            delta = self.autoPhotometricCutNSig * coeff[2] - coeff[1]

            cut = -1 * int(np.ceil(delta / self.autoPhotometricCutStep)) * self.autoPhotometricCutStep
            # Clip the cut to a range from 2 times the input to 5 mmag
            expGrayPhotometricCut[i] = max(expGrayPhotometricCut[i]*2,
                                           min(cut, -0.005))

            delta = self.autoHighCutNSig * coeff[2] + coeff[1]
            cut = int(np.ceil(delta / self.autoPhotometricCutStep)) * self.autoPhotometricCutStep
            expGrayHighCut[i] = max(0.005,
                                    min(cut, expGrayHighCut[i]*2))

        return (expGrayPhotometricCut, expGrayHighCut)

    def computeExpGrayCutsFromRepeatability(self):
        """
        Compute exposure gray recommended cuts, using repeatability info.

        Returns
        -------
        expGrayPhotometricCut: `np.array`
           Float array (per band) of recommended expGray cuts
        expGrayHighCut: `np.array`
           Float array (per band) of recommended expGray cuts (high side)
        """

        expGrayPhotometricCut = np.zeros(self.fgcmPars.nBands)
        expGrayHighCut = np.zeros_like(expGrayPhotometricCut)

        # set defaults to those that were set in the config file
        # These are going to be lists of floats for persistence
        expGrayPhotometricCut[:] = [float(f) for f in self.expGrayPhotometricCut]
        expGrayHighCut[:] = [float(f) for f in self.expGrayHighCut]

        for i in xrange(self.fgcmPars.nBands):
            delta = self.autoPhotometricCutNSig * self.fgcmPars.compReservedRawCrunchedRepeatability[i]

            cut = -1 * int(np.ceil(delta / self.autoPhotometricCutStep)) * self.autoPhotometricCutStep
            # Clip the cut to a range from 2 times the input to 5 mmag
            expGrayPhotometricCut[i] = max(expGrayPhotometricCut[i]*2,
                                           min(cut, -0.005))

            delta = self.autoHighCutNSig * self.fgcmPars.compReservedRawCrunchedRepeatability[i]
            cut = int(np.ceil(delta / self.autoPhotometricCutStep)) * self.autoPhotometricCutStep
            expGrayHighCut[i] = max(0.005,
                                    min(cut, expGrayHighCut[i]*2))

        return (expGrayPhotometricCut, expGrayHighCut)

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state
