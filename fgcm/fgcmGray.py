import numpy as np
import os
import sys
import esutil
import time
import scipy.optimize

from .fgcmUtilities import gaussFunction
from .fgcmUtilities import histoGauss
from .fgcmUtilities import Cheb2dField
from .fgcmUtilities import computeDeltaRA
from .fgcmUtilities import expFlagDict
from .fgcmUtilities import histogram_rev_sorted
from .fgcmUtilities import makeFigure, putButlerFigure
from matplotlib import colormaps

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

    def __init__(self, fgcmConfig, fgcmPars, fgcmStars, butlerQC=None, plotHandleDict=None):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.debug('Initializing fgcmGray')

        # need fgcmPars because it tracks good exposures
        #  also this is where the gray info is stored
        self.fgcmPars = fgcmPars

        # need fgcmStars because it has the stars (duh)
        self.fgcmStars = fgcmStars

        self.butlerQC = butlerQC
        self.plotHandleDict = plotHandleDict

        # and record configuration variables...
        self.minStarPerCCD = fgcmConfig.minStarPerCCD
        self.minStarPerExp = fgcmConfig.minStarPerExp
        self.maxCCDGrayErr = fgcmConfig.maxCCDGrayErr
        self.ccdGrayMaxStarErr = fgcmConfig.ccdGrayMaxStarErr
        self.ccdGraySubCCD = fgcmConfig.ccdGraySubCCD
        self.ccdGraySubCCDChebyshevOrder = fgcmConfig.ccdGraySubCCDChebyshevOrder
        self.ccdGraySubCCDTriangular = fgcmConfig.ccdGraySubCCDTriangular
        self.ccdGrayFocalPlane = fgcmConfig.ccdGrayFocalPlane
        self.ccdGrayFocalPlaneChebyshevOrder = fgcmConfig.ccdGrayFocalPlaneChebyshevOrder
        self.ccdGrayFocalPlaneFitMinCcd = fgcmConfig.ccdGrayFocalPlaneFitMinCcd
        self.ccdGrayFocalPlaneMaxStars = fgcmConfig.ccdGrayFocalPlaneMaxStars
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.illegalValue = fgcmConfig.illegalValue
        self.expGrayInitialCut = fgcmConfig.expGrayInitialCut
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber
        self.deltaMagBkgOffsetPercentile = fgcmConfig.deltaMagBkgOffsetPercentile
        self.expGrayCheckDeltaT = fgcmConfig.expGrayCheckDeltaT
        self.colorSplitIndices = fgcmConfig.colorSplitIndices
        self.bandFitIndex = fgcmConfig.bandFitIndex
        self.bandRequiredIndex = fgcmConfig.bandRequiredIndex
        self.bandNotRequiredIndex = fgcmConfig.bandNotRequiredIndex
        self.quietMode = fgcmConfig.quietMode

        self.expGrayPhotometricCut = fgcmConfig.expGrayPhotometricCut
        self.expGrayHighCut = fgcmConfig.expGrayHighCut
        self.autoPhotometricCutNSig = fgcmConfig.autoPhotometricCutNSig
        self.autoPhotometricCutStep = fgcmConfig.autoPhotometricCutStep
        self.autoHighCutNSig = fgcmConfig.autoHighCutNSig

        self.arraysPrepared = False

        self.focalPlaneProjector = fgcmConfig.focalPlaneProjector
        self.defaultCameraOrientation = fgcmConfig.defaultCameraOrientation
        self.deltaMapperDefault = None

        self.rng = fgcmConfig.rng

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
        self.ccdDeltaStdHandle = snmm.createArray((self.fgcmPars.nExp, self.fgcmPars.nCCD), dtype='f8')
        self.ccdGrayRMSHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdGrayErrHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdNGoodObsHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='i4')
        self.ccdNGoodStarsHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='i4')
        self.ccdNGoodTilingsHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')

        if np.any(self.ccdGraySubCCD):
            order = self.ccdGraySubCCDChebyshevOrder
            self.ccdGraySubCCDParsHandle = snmm.createArray((self.fgcmPars.nExp, self.fgcmPars.nCCD, (order + 1) * (order + 1)), dtype='f8')
            self.ccdGrayNPar = (order + 1) * (order + 1)

        self.expGrayHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expDeltaStdHandle = snmm.createArray(self.fgcmPars.nExp, dtype='f8')
        self.expGrayRMSHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayErrHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expNGoodStarsHandle = snmm.createArray(self.fgcmPars.nExp,dtype='i4')
        self.expNGoodCCDsHandle = snmm.createArray(self.fgcmPars.nExp,dtype='i2')
        self.expNGoodTilingsHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')

        self.expGrayColorSplitHandle = snmm.createArray((self.fgcmPars.nExp, 3), dtype='f8')
        self.expGrayRMSColorSplitHandle = snmm.createArray((self.fgcmPars.nExp, 3), dtype='f8')
        self.expGrayErrColorSplitHandle = snmm.createArray((self.fgcmPars.nExp, 3), dtype='f8')
        self.expGrayNGoodStarsColorSplitHandle = snmm.createArray((self.fgcmPars.nExp, 3), dtype='i2')

        self.ccdDeltaMagBkgHandle = snmm.createArray((self.fgcmPars.nExp, self.fgcmPars.nCCD), dtype='f8')
        self.expDeltaMagBkgHandle = snmm.createArray(self.fgcmPars.nExp, dtype='f8')

        self.arraysPrepared = True

    def setDeltaMapperDefault(self, deltaMapperDefault):
        """
        Set the deltaMapperDefault array.

        Parameters
        ----------
        deltaMapperDefault : `np.recarray`
        """
        self.deltaMapperDefault = deltaMapperDefault

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
                  (EGray[goodObs]).astype(expGrayForInitialSelection.dtype))
        np.add.at(expGrayRMSForInitialSelection,
                  obsExpIndex[goodObs],
                  (EGray[goodObs]).astype(expGrayRMSForInitialSelection.dtype)**2.)
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

        for i in range(self.fgcmPars.nBands):
            self.fgcmLog.debug('Making EXP_GRAY (initial) histogram for %s band' %
                               (self.fgcmPars.bands[i]))
            inBand, = np.where(self.fgcmPars.expBandIndex[expUse] == i)

            if (inBand.size == 0) :
                continue

            fig = makeFigure(figsize=(8, 6))
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

            if self.butlerQC is not None:
                putButlerFigure(self.fgcmLog,
                                self.butlerQC,
                                self.plotHandleDict,
                                "ExpgrayInitial",
                                self.cycleNumber,
                                fig,
                                band=self.fgcmPars.bands[i])
            else:
                fig.savefig('%s/%s_initial_expgray_%s.png' % (self.plotPath,
                                                              self.outfileBaseWithCycle,
                                                              self.fgcmPars.bands[i]))

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
        if not self.quietMode:
            self.fgcmLog.info('Computing CCDGray and ExpGray.')

        # Note: this computes the gray values for all exposures, good and bad

        # values to set
        ccdGray = snmm.getArray(self.ccdGrayHandle)
        ccdDeltaStd = snmm.getArray(self.ccdDeltaStdHandle)
        ccdGrayRMS = snmm.getArray(self.ccdGrayRMSHandle)
        ccdGrayErr = snmm.getArray(self.ccdGrayErrHandle)
        ccdNGoodObs = snmm.getArray(self.ccdNGoodObsHandle)
        ccdNGoodStars = snmm.getArray(self.ccdNGoodStarsHandle)
        ccdNGoodTilings = snmm.getArray(self.ccdNGoodTilingsHandle)

        if np.any(self.ccdGraySubCCD):
            ccdGraySubCCDPars = snmm.getArray(self.ccdGraySubCCDParsHandle)

        expGray = snmm.getArray(self.expGrayHandle)
        expDeltaStd = snmm.getArray(self.expDeltaStdHandle)
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
        obsDeltaStd = snmm.getArray(self.fgcmStars.obsDeltaStdHandle)
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

        obsDeltaStdGO = obsDeltaStd[goodObs]

        if np.any(self.ccdGraySubCCD):
            obsXGO = snmm.getArray(self.fgcmStars.obsXHandle)[goodObs]
            obsYGO = snmm.getArray(self.fgcmStars.obsYHandle)[goodObs]

        if np.any(self.ccdGrayFocalPlane):
            obsRAGO = snmm.getArray(self.fgcmStars.obsRAHandle)[goodObs]
            obsDecGO = snmm.getArray(self.fgcmStars.obsDecHandle)[goodObs]

        self.fgcmLog.debug('FgcmGray using %d observations from %d good stars.' %
                           (goodObs.size,goodStars.size))

        # group by CCD and sum

        ## ccdGray = Sum(EGray/EGrayErr^2) / Sum(1./EGrayErr^2)
        ## ccdGrayRMS = Sqrt((Sum(EGray^2/EGrayErr^2) / Sum(1./EGrayErr^2)) - ccdGray^2)
        ## ccdGrayErr = Sqrt(1./Sum(1./EGrayErr^2))

        ccdGray[:,:] = 0.0
        ccdDeltaStd[:, :] = 0.0
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
                  (1./EGrayErr2GO).astype(ccdGrayWt.dtype))
        np.add.at(ccdNGoodStars,
                  (obsExpIndex[goodObs],obsCCDIndex[goodObs]),
                  1)
        np.add.at(ccdNGoodObs,
                  (obsExpIndex[goodObs],obsCCDIndex[goodObs]),
                  objNGoodObs[obsObjIDIndex[goodObs],
                              obsBandIndex[goodObs]])
        np.add.at(ccdDeltaStd,
                  (obsExpIndex[goodObs], obsCCDIndex[goodObs]),
                  (obsDeltaStdGO/EGrayErr2GO).astype(ccdDeltaStd.dtype))

        gd = np.where((ccdNGoodStars >= 3) & (ccdGrayWt > 0.0))
        ccdDeltaStd[gd] /= ccdGrayWt[gd]
        # This is set for everything here
        ccdGrayErr[gd] = np.sqrt(1./ccdGrayWt[gd])
        ccdGrayRMS[gd] = 0.0  # only used for per-ccd checks

        if np.any(self.ccdGrayFocalPlane):
            self.fgcmLog.info("Computing CCDGray full focal-plane fits.")

            # We are doing our full focal plane fits for bands that are set.
            order = self.ccdGrayFocalPlaneChebyshevOrder
            pars = np.zeros((order + 1, order + 1))
            pars[0, 0] = 1.0

            FGrayGO = 10.**(EGrayGO/(-2.5))
            FGrayErrGO = (np.log(10.)/2.5)*np.sqrt(EGrayErr2GO)*FGrayGO

            h, rev = histogram_rev_sorted(obsExpIndex[goodObs])

            use, = np.where(h >= 3)
            for i in use:
                i1a = rev[rev[i]: rev[i + 1]]

                eInd = obsExpIndex[goodObs[i1a[0]]]
                bInd = obsBandIndex[goodObs[i1a[0]]]

                if not self.ccdGrayFocalPlane[bInd]:
                    # We are not fitting the focal plane for this, skip.
                    continue

                if ((i % 100) == 0) and not self.quietMode:
                    self.fgcmLog.info("Working on full focal plane fit for %d" % (self.fgcmPars.expArray[eInd]))

                # Downsample if too many.
                if len(i1a) > self.ccdGrayFocalPlaneMaxStars:
                    i1a = self.rng.choice(i1a, replace=False, size=self.ccdGrayFocalPlaneMaxStars)
                    i1a = np.sort(i1a)

                deltaMapper = self.focalPlaneProjector(int(self.fgcmPars.expTelRot[eInd]))

                raCent = np.rad2deg(self.fgcmPars.expTelRA[eInd])
                decCent = np.rad2deg(self.fgcmPars.expTelDec[eInd])

                deltaRA = computeDeltaRA(obsRAGO[i1a], raCent, dec=decCent, degrees=True)
                deltaDec = obsDecGO[i1a] - decCent
                offsetRA = np.min(deltaRA)
                offsetDec = np.min(deltaDec)

                ccdGrayEvalNStars = np.zeros(self.fgcmPars.nCCD, dtype=np.int32)
                np.add.at(ccdGrayEvalNStars,
                          obsCCDIndex[goodObs[i1a]],
                          1)
                okCcd, = np.where(ccdGrayEvalNStars > 0)

                fitFailed = False

                if okCcd.size < self.ccdGrayFocalPlaneFitMinCcd:
                    # Too few good ccds, use the fitFailed flag
                    fitFailed = True
                else:
                    try:
                        field = Cheb2dField.fit(np.max(deltaRA - offsetRA),
                                                np.max(deltaDec - offsetDec),
                                                order,
                                                deltaRA - offsetRA, deltaDec - offsetDec,
                                                FGrayGO[i1a],
                                                valueErr=FGrayErrGO[i1a],
                                                triangular=False)
                    except (ValueError, RuntimeError, TypeError):
                        # Log a warn and set to a single value...
                        self.fgcmLog.warning("Full focal-plane fit failed on exposure %d" %
                                             (self.fgcmPars.expArray[eInd]))
                        fitFailed = True

                if fitFailed:
                    # Compute the means per-ccd.
                    # This is used when the fit fails (ugh) or when there is not
                    # enough coverage

                    ccdGrayTemp = np.zeros(self.fgcmPars.nCCD)
                    ccdGrayRMSTemp = np.zeros_like(ccdGrayTemp)

                    np.add.at(ccdGrayTemp,
                              obsCCDIndex[goodObs[i1a]],
                              (EGrayGO[i1a]/EGrayErr2GO[i1a]).astype(ccdGrayTemp.dtype))
                    np.add.at(ccdGrayRMSTemp,
                              obsCCDIndex[goodObs[i1a]],
                              (EGrayGO[i1a]**2./EGrayErr2GO[i1a]).astype(ccdGrayRMSTemp.dtype))

                    gdTemp, = np.where((ccdNGoodStars[eInd, :] >= 3) &
                                       (ccdGrayWt[eInd, :] > 0.0) &
                                       (ccdGrayRMSTemp > 0.0))
                    ccdGrayTemp[gdTemp] /= ccdGrayWt[eInd, gdTemp]
                    tempRMS2 = np.zeros_like(ccdGrayRMSTemp)
                    tempRMS2[gdTemp] = (ccdGrayRMSTemp[gdTemp]/ccdGrayWt[eInd, gdTemp]) - ccdGrayTemp[gdTemp]**2.
                    ok, = np.where(tempRMS2 > 0.0)

                    ccdGray[eInd, ok] = ccdGrayTemp[ok]
                    ccdGrayRMS[eInd, ok] = np.sqrt(tempRMS2[ok])

                    # If we have any sub-ccd pars, we need to store the flux parametrized
                    # version as well
                    if np.any(self.ccdGraySubCCD):
                        ccdGraySubCCDPars[eInd, ok, 0] = 10.**(ccdGray[eInd, ok]/(-2.5))
                else:
                    # Sucessful fit

                    # Eval for all the points, and take the mean of the model for the overall
                    # value

                    ccdGrayEvalStars = field.evaluate(deltaRA - offsetRA,
                                                      deltaDec - offsetDec)
                    ccdGrayEval = np.zeros(self.fgcmPars.nCCD)
                    ccdGrayEvalNStars = np.zeros(self.fgcmPars.nCCD, dtype=np.int32)
                    np.add.at(ccdGrayEval,
                              obsCCDIndex[goodObs[i1a]],
                              ccdGrayEvalStars.astype(ccdGrayEval.dtype))
                    np.add.at(ccdGrayEvalNStars,
                              obsCCDIndex[goodObs[i1a]],
                              1)
                    ok, = np.where((ccdGrayEvalNStars > 0) & np.isfinite(ccdGrayEval))
                    ccdGrayEval[ok] /= ccdGrayEvalNStars[ok]

                    ccdGray[eInd, ok] = -2.5*np.log10(ccdGrayEval[ok])

                    if self.ccdGraySubCCD[bInd]:
                        # Do the sub-ccd fit
                        for cInd in ok:
                            draOff = deltaMapper['delta_ra'][cInd, :] - offsetRA
                            ddecOff = deltaMapper['delta_dec'][cInd, :] - offsetDec

                            try:
                                cField = Cheb2dField.fit(deltaMapper['x_size'][cInd],
                                                         deltaMapper['y_size'][cInd],
                                                         self.ccdGraySubCCDChebyshevOrder,
                                                         deltaMapper['x'][cInd, :],
                                                         deltaMapper['y'][cInd, :],
                                                         field.evaluate(draOff, ddecOff),
                                                         triangular=self.ccdGraySubCCDTriangular)
                                ccdGraySubCCDPars[eInd, cInd, :] = cField.pars.ravel()
                            except (ValueError, RuntimeError, TypeError):
                                self.fgcmLog.warning("Focal plane to ccd mapping fit failed on %d/%d" %
                                                     (self.fgcmPars.expArray[eInd], cInd + self.ccdStartIndex))
                                # Put in a filler here
                                ccdGraySubCCDPars[eInd, cInd, 0] = ccdGrayEval

                    elif np.any(self.ccdGraySubCCD):
                        # Do one number per ccd in the parameters if we need to set any.
                        ccdGraySubCCDPars[eInd, :, 0] = ccdGrayEval
                        ccdGraySubCCDPars[eInd, ~ok, 0] = 1.0

        if not np.any(self.ccdGraySubCCD) and not np.any(self.ccdGrayFocalPlane):
            self.fgcmLog.info("Computing CCDGray constants.")
            # This is when we _only_ have per-ccd gray, no focal plane, and
            # we can do all of this at once.
            np.add.at(ccdGray,
                      (obsExpIndex[goodObs],obsCCDIndex[goodObs]),
                      (EGrayGO/EGrayErr2GO).astype(ccdGray.dtype))
            np.add.at(ccdGrayRMS,
                      (obsExpIndex[goodObs],obsCCDIndex[goodObs]),
                      (EGrayGO**2./EGrayErr2GO).astype(ccdGrayRMS.dtype))

            # need at least 3 or else computation can blow up
            gd = np.where((ccdNGoodStars >= 3) & (ccdGrayWt > 0.0) & (ccdGrayRMS > 0.0))
            ccdGray[gd] /= ccdGrayWt[gd]
            tempRMS2 = np.zeros_like(ccdGrayRMS)
            tempRMS2[gd] = (ccdGrayRMS[gd]/ccdGrayWt[gd]) - (ccdGray[gd]**2.)
            ok = np.where(tempRMS2 > 0.0)
            ccdGrayRMS[ok] = np.sqrt(tempRMS2[ok])

        elif np.any(~np.array(self.ccdGrayFocalPlane)):
            self.fgcmLog.info("Computing CCDGray sub-ccd fits (no focal plane).")
            # We are computing on the sub-ccd scale for some bands, and
            # at least 1 band does not have a focal plane fit

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

            h, rev = histogram_rev_sorted(expCcdHash)

            # Anything with 2 or fewer stars will be marked bad
            use, = np.where(h >= 3)
            for i in use:
                i1a = rev[rev[i]: rev[i + 1]]

                eInd = obsExpIndex[goodObs[i1a[0]]]
                cInd = obsCCDIndex[goodObs[i1a[0]]]
                bInd = obsBandIndex[goodObs[i1a[0]]]

                if self.ccdGrayFocalPlane[bInd]:
                    # We already fit this above, so we can skip to the next.
                    continue

                ccdNGoodStars[eInd, cInd] = i1a.size

                computeMean = False
                skip = False

                if not self.ccdGraySubCCD[bInd]:
                    fit = pars.flatten()
                    computeMean = True
                elif i1a.size < 10 * pars.size:
                    # insufficient stars for chebyshev fit
                    fit = pars.flatten()
                    computeMean = True
                else:
                    try:
                        field = Cheb2dField.fit(self.deltaMapperDefault['x_size'][cInd],
                                                self.deltaMapperDefault['y_size'][cInd],
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
                field = Cheb2dField(self.deltaMapperDefault['x_size'][cInd],
                                    self.deltaMapperDefault['y_size'][cInd],
                                    fit)
                ccdGray[eInd, cInd] = -2.5 * np.log10(field.evaluateCenter())

        self.fgcmLog.debug('Computed CCDGray for %d CCDs' % (gd[0].size))

        # set illegalValue for totally bad CCDs
        bad = np.where((ccdNGoodStars <= 2) | (ccdGrayWt <= 0.0))
        ccdGray[bad] = self.illegalValue
        ccdDeltaStd[bad] = self.illegalValue
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
        expDeltaStd[:] = 0.0
        expGrayRMS[:] = 0.0
        expGrayErr[:] = 0.0
        expNGoodStars[:] = 0
        expNGoodCCDs[:] = 0
        expNGoodTilings[:] = 0.0

        # temporary
        expGrayWt = np.zeros_like(expGray)

        np.add.at(expGrayWt,
                  goodCCD[0],
                  (1./ccdGrayErr[goodCCD]**2.).astype(ccdGrayWt.dtype))
        np.add.at(expGray,
                  goodCCD[0],
                  (ccdGray[goodCCD]/ccdGrayErr[goodCCD]**2.).astype(expGray.dtype))
        np.add.at(expGrayRMS,
                  goodCCD[0],
                  (ccdGray[goodCCD]**2./ccdGrayErr[goodCCD]**2.).astype(expGrayRMS.dtype))
        np.add.at(expNGoodCCDs,
                  goodCCD[0],
                  1)
        np.add.at(expNGoodTilings,
                  goodCCD[0],
                  (ccdNGoodTilings[goodCCD]).astype(expNGoodTilings.dtype))
        np.add.at(expNGoodStars,
                  goodCCD[0],
                  (ccdNGoodStars[goodCCD]).astype(expNGoodStars.dtype))
        np.add.at(expDeltaStd,
                  goodCCD[0],
                  (ccdDeltaStd[goodCCD]/ccdGrayErr[goodCCD]**2.).astype(expDeltaStd.dtype))

        if self.fgcmPars.nCCD >= 3:
            # Regular mode, when we have a multi-detector camera.

            # need at least 3 or else computation can blow up
            gd, = np.where(expNGoodCCDs >= 3)
            expGray[gd] /= expGrayWt[gd]
            expDeltaStd[gd] /= expGrayWt[gd]
            expGrayRMS[gd] = np.sqrt(np.clip((expGrayRMS[gd]/expGrayWt[gd]) - (expGray[gd]**2.),
                                             0.0, None))
            expGrayErr[gd] = np.sqrt(1./expGrayWt[gd])
            expNGoodTilings[gd] /= expNGoodCCDs[gd]

            # set illegal value for non-measurements
            bad, = np.where(expNGoodCCDs <= 2)
            expGray[bad] = self.illegalValue
            expDeltaStd[bad] = self.illegalValue
            expGrayRMS[bad] = self.illegalValue
            expGrayErr[bad] = self.illegalValue
            expNGoodTilings[bad] = self.illegalValue

        else:
            # Special mode for 1/2 detector cameras.
            gd, = np.where(expNGoodCCDs >= 1)
            expGray[gd] /= expGrayWt[gd]
            expDeltaStd[gd] /= expGrayWt[gd]
            expGrayRMS[gd] = 0.0
            expGrayErr[gd] = np.sqrt(1./expGrayWt[gd])
            expNGoodTilings[gd] /= expNGoodCCDs[gd]

            # set illegal value for non-measurements
            bad, = np.where(expNGoodCCDs < 1)
            expGray[bad] = self.illegalValue
            expDeltaStd[bad] = self.illegalValue
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

        for c in range(gmiCutLow.size):
            use, = np.where((gmiGO > gmiCutLow[c]) &
                            (gmiGO < gmiCutHigh[c]))

            np.add.at(expGrayColorSplit[:, c],
                      obsExpIndex[goodObs[use]],
                      (EGrayGO[use] / EGrayErr2GO[use]).astype(expGrayColorSplit.dtype))
            np.add.at(expGrayWtColorSplit[:, c],
                      obsExpIndex[goodObs[use]],
                      (1. / EGrayErr2GO[use]).astype(expGrayWtColorSplit.dtype))
            np.add.at(expGrayRMSColorSplit[:, c],
                      obsExpIndex[goodObs[use]],
                      (EGrayGO[use]**2. / EGrayErr2GO[use]).astype(expGrayRMSColorSplit.dtype))
            np.add.at(expGrayNGoodStarsColorSplit[:, c],
                      obsExpIndex[goodObs[use]],
                      1)

            gd_flag = ((expGrayNGoodStarsColorSplit[:, c] >= self.minStarPerExp / 4) &
                       (expGrayWtColorSplit[:, c] > 0.0) &
                       (expGrayRMSColorSplit[:, c] > 0.0))
            gd, = np.where(gd_flag)
            bd, = np.where(~gd_flag)
            expGrayColorSplit[gd, c] /= expGrayWtColorSplit[gd, c]
            expGrayErrColorSplit[gd, c] = np.sqrt(1. / expGrayWtColorSplit[gd, c])
            expGrayRMSColorSplit[gd, c] = np.sqrt(np.clip((expGrayRMSColorSplit[gd, c] / expGrayWtColorSplit[gd, c]) - expGrayColorSplit[gd, c]**2., 0.0, None))

            expGrayColorSplit[bd, c] = self.illegalValue
            expGrayRMSColorSplit[bd, c] = self.illegalValue
            expGrayErrColorSplit[bd, c] = self.illegalValue

        if self.plotPath is not None:
            # main plots:
            #  per band, plot expGray for red vs blue stars!

            for bandIndex, band in enumerate(self.fgcmPars.bands):
                if not self.fgcmPars.hasExposuresInBand[bandIndex]:
                    continue
                use, = np.where((self.fgcmPars.expBandIndex == bandIndex) &
                                (self.fgcmPars.expFlag == 0) &
                                (expGrayColorSplit[:, 0] > self.illegalValue) &
                                (expGrayColorSplit[:, 2] > self.illegalValue))
                if (use.size == 0):
                    self.fgcmLog.info('Could not find photometric color-split exposures in band %d' % (bandIndex))
                    continue

                fig = makeFigure(figsize=(8, 6))
                fig.clf()

                ax = fig.add_subplot(111)
                ax.hexbin(
                    expGrayColorSplit[use, 0]*1000.0,
                    expGrayColorSplit[use, 2]*1000.0,
                    bins='log',
                    cmap=colormaps.get_cmap("viridis"),
                )
                ax.set_xlabel('EXP_GRAY (%s) (%s) (mmag)' % (self.fgcmPars.bands[bandIndex], gmiCutNames[0]))
                ax.set_ylabel('EXP_GRAY (%s) (%s) (mmag)' % (self.fgcmPars.bands[bandIndex], gmiCutNames[2]))
                ax.plot([-10.0, 10.0], [-10.0, 10.0], 'r--')

                text = r'$(%s)$' % (self.fgcmPars.bands[bandIndex]) + '\n'
                text += "(1-1 reference line)"
                ax.annotate(text, (0.95, 0.93), xycoords='axes fraction', ha='right',
                            va='top', fontsize=16, color='r')

                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "CompareRedblueExpgray",
                                    self.cycleNumber,
                                    fig,
                                    band=self.fgcmPars.bands[bandIndex])
                else:
                    fig.savefig('%s/%s_compare-redblue-expgray_%s.png' % (self.plotPath,
                                                                          self.outfileBaseWithCycle,
                                                                          self.fgcmPars.bands[bandIndex]))

                # And a plot as function of time

                deltaColor = (expGrayColorSplit[use, 2] - expGrayColorSplit[use, 0]) * 1000.
                firstMJD = np.floor(np.min(self.fgcmPars.expMJD))

                st = np.argsort(deltaColor)
                extent = [np.min(self.fgcmPars.expMJD[use] - firstMJD),
                          np.max(self.fgcmPars.expMJD[use] - firstMJD),
                          deltaColor[st[int(0.01*deltaColor.size)]],
                          deltaColor[st[int(0.99*deltaColor.size)]]]

                fig = makeFigure(figsize=(8, 6))
                fig.clf()

                ax = fig.add_subplot(111)
                ax.hexbin(self.fgcmPars.expMJD[use] - firstMJD,
                          (expGrayColorSplit[use, 2] - expGrayColorSplit[use, 0]) * 1000.,
                          bins='log', extent=extent, cmap=colormaps.get_cmap("viridis"))
                ax.set_xlabel('MJD - %.0f' % (firstMJD))
                ax.set_ylabel('EXP_GRAY (%s) (%s) - EXP_GRAY (%s) (%s) (mmag)' %
                              (self.fgcmPars.bands[bandIndex], gmiCutNames[2],
                               self.fgcmPars.bands[bandIndex], gmiCutNames[0]))
                ax.plot([extent[0], extent[1]],
                        [0.0, 0.0], 'r--')

                text = r"$(%s)$" % (self.fgcmPars.bands[bandIndex]) + "\n"
                text += "(0 reference line)"
                ax.annotate(text, (0.95, 0.93), xycoords='axes fraction', ha='right',
                            va='top', fontsize=16, color='r')

                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "ExpgrayCompareMjdRedblue",
                                    self.cycleNumber,
                                    fig,
                                    band=self.fgcmPars.bands[bandIndex])
                else:
                    fig.savefig('%s/%s_compare-mjd-redblue-expgray_%s.png' % (self.plotPath,
                                                                              self.outfileBaseWithCycle,
                                                                              self.fgcmPars.bands[bandIndex]))

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

        expUse, = np.where((self.fgcmPars.expFlag == 0) &
                           (expNGoodStars > self.minStarPerExp) &
                           (expGray > self.illegalValue))

        for i in range(self.fgcmPars.nBands):
            inBand, = np.where(self.fgcmPars.expBandIndex[expUse] == i)

            if (inBand.size == 0) :
                continue

            # plot histograms of EXP^gray
            # Note that we use the histogram fit/plot code to get the
            # fit coefficients even if we are not persisting the plots.
            # Fortunately, the makeFigure code now ensures that this
            # does not have any side effects.

            fig = makeFigure(figsize=(8, 6))
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
                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "Expgray",
                                    self.cycleNumber,
                                    fig,
                                    band=self.fgcmPars.bands[i])
                else:
                    fig.savefig('%s/%s_expgray_%s.png' % (self.plotPath,
                                                          self.outfileBaseWithCycle,
                                                          self.fgcmPars.bands[i]))

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

            fig = makeFigure(figsize=(8, 6))
            fig.clf()

            ax=fig.add_subplot(111)

            ax.hexbin(
                secZenith[ok],
                expGray[expUse[inBand[ok]]]*1000.0,
                rasterized=True,
                cmap=colormaps.get_cmap("viridis"),
            )

            text = r'$(%s)$' % (self.fgcmPars.bands[i])
            ax.annotate(text, (0.95, 0.93), xycoords='axes fraction', ha='right',
                        va='top', fontsize=16, color='r')

            ax.set_xlabel(r'$\mathrm{sec}(\mathrm{zd})$',fontsize=16)
            ax.set_ylabel(r'$\mathrm{EXP}^{\mathrm{gray}}\,(\mathrm{mmag})$',fontsize=16)

            if self.plotPath is not None:
                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "ExpgrayAirmass",
                                    self.cycleNumber,
                                    fig,
                                    band=self.fgcmPars.bands[i])
                else:
                    fig.savefig('%s/%s_airmass_expgray_%s.png' % (self.plotPath,
                                                                  self.outfileBaseWithCycle,
                                                                  self.fgcmPars.bands[i]))

            # plot EXP^gray as a function of UT

            fig = makeFigure(figsize=(8, 6))
            fig.clf()

            ax=fig.add_subplot(111)

            ax.hexbin(self.fgcmPars.expDeltaUT[expUse[inBand]],
                      expGray[expUse[inBand]]*1000.0,
                      rasterized=True, cmap=colormaps.get_cmap("viridis"))
            ax.annotate(text, (0.95, 0.93), xycoords='axes fraction', ha='right',
                        va='top', fontsize=16, color='r')

            ax.set_xlabel(r'$\Delta \mathrm{UT}$',fontsize=16)
            ax.set_ylabel(r'$\mathrm{EXP}^{\mathrm{gray}}\,(\mathrm{mmag})$',fontsize=16)

            if self.plotPath is not None:
                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "ExpgrayUT",
                                    self.cycleNumber,
                                    fig,
                                    band=self.fgcmPars.bands[i])
                else:
                    fig.savefig('%s/%s_UT_expgray_%s.png' % (self.plotPath,
                                                             self.outfileBaseWithCycle,
                                                             self.fgcmPars.bands[i]))

        # and plot EXP^gray vs MJD for all bands for deep fields
        fig = makeFigure(figsize=(8, 6))
        fig.clf()

        ax=fig.add_subplot(111)

        firstMJD = np.floor(np.min(self.fgcmPars.expMJD))

        deepUse,=np.where(self.fgcmPars.expDeepFlag[expUse] == 1)

        if deepUse.size > 0:
            ax.hexbin(self.fgcmPars.expMJD[expUse[deepUse]] - firstMJD,
                      expGray[expUse[deepUse]]*1000.0, bins='log', cmap=colormaps.get_cmap("viridis"))
            ax.set_xlabel(r'$\mathrm{MJD}\ -\ %.0f$' % (firstMJD),fontsize=16)
            ax.set_ylabel(r'$\mathrm{EXP}^{\mathrm{gray}}\,(\mathrm{mmag})$',fontsize=16)

            ax.set_title(r'$\mathrm{Deep Fields}$')

            if self.plotPath is not None:
                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "ExpgrayDeepMjd",
                                    self.cycleNumber,
                                    fig)
                else:
                    fig.savefig('%s/%s_mjd_deep_expgray.png' % (self.plotPath,
                                                                self.outfileBaseWithCycle))

        # And plot correlations of EXP^gray between pairs of bands
        for ind, bandIndex0 in enumerate(self.bandFitIndex[:-2]):
            if not self.fgcmPars.hasExposuresInBand[ind] or \
               not self.fgcmPars.hasExposuresInBand[ind + 1]:
                continue

            bandIndex1 = self.bandFitIndex[ind + 1]

            use0, = np.where((self.fgcmPars.expBandIndex == bandIndex0) &
                             (self.fgcmPars.expFlag == 0) &
                             (expGray > self.illegalValue))
            use1, = np.where((self.fgcmPars.expBandIndex == bandIndex1) &
                             (self.fgcmPars.expFlag == 0) &
                             (expGray > self.illegalValue))

            if use0.size == 0 or use1.size == 0:
                self.fgcmLog.info('Could not find photometric exposures in bands %s or %s' %
                                  (self.fgcmPars.bands[bandIndex0], self.fgcmPars.bands[bandIndex1]))
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

            fig = makeFigure(figsize=(8, 6))
            fig.clf()

            ax=fig.add_subplot(111)

            ax.hexbin(expGray[use0[matchInd[ok]]]*1000.0,
                      expGray[use1[ok]]*1000.0, bins='log', cmap=colormaps.get_cmap("viridis"))
            ax.set_xlabel('EXP_GRAY (%s) (mmag)' % (self.fgcmPars.bands[bandIndex0]))
            ax.set_ylabel('EXP_GRAY (%s) (mmag)' % (self.fgcmPars.bands[bandIndex1]))
            ax.plot([-0.01 * 1000, 0.01 * 1000],[-0.01 * 1000, 0.01 * 1000],'r--')
            text = "(1-1 reference line)"
            ax.annotate(text, (0.95, 0.93), xycoords='axes fraction', ha='right',
                        va='top', fontsize=16, color='r')

            if self.plotPath is not None:
                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    "ExpgrayCompareBands",
                                    self.cycleNumber,
                                    fig,
                                    band=self.fgcmPars.bands[bandIndex0])
                else:
                    fig.savefig('%s/%s_expgray-compare_%s_%s.png' % (self.plotPath,
                                                                     self.outfileBaseWithCycle,
                                                                     self.fgcmPars.bands[bandIndex0],
                                                                     self.fgcmPars.bands[bandIndex1]))

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

        expUse, = np.where((self.fgcmPars.expFlag == 0) &
                           (expNGoodStars > self.minStarPerExp) &
                           (expGray > self.illegalValue))

        expGrayPhotometricCut = np.zeros(self.fgcmPars.nBands)
        expGrayHighCut = np.zeros_like(expGrayPhotometricCut)

        # set defaults to those that were set in the config file
        # These are going to be lists of floats for persistence
        expGrayPhotometricCut[:] = [float(f) for f in self.expGrayPhotometricCut]
        expGrayHighCut[:] = [float(f) for f in self.expGrayHighCut]

        for i in range(self.fgcmPars.nBands):
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

            self.fgcmLog.info("ExpGray cut (%s band): %.4f" % (self.fgcmPars.bands[i],
                                                               expGrayPhotometricCut[i]))
            self.fgcmLog.info("ExpGray high cut (%s band): %.4f" % (self.fgcmPars.bands[i],
                                                                    expGrayHighCut[i]))

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

        for i in range(self.fgcmPars.nBands):
            delta = np.clip(self.autoPhotometricCutNSig * self.fgcmPars.compReservedRawRepeatability[i], 0.001, 1e5)

            cut = -1 * int(np.ceil(delta / self.autoPhotometricCutStep)) * self.autoPhotometricCutStep
            # Clip the cut to a range from 2 times the input to 5 mmag
            expGrayPhotometricCut[i] = max(expGrayPhotometricCut[i]*2,
                                           min(cut, -0.005))

            delta = np.clip(self.autoHighCutNSig * self.fgcmPars.compReservedRawRepeatability[i], 0.001, 1e5)
            cut = int(np.ceil(delta / self.autoPhotometricCutStep)) * self.autoPhotometricCutStep
            expGrayHighCut[i] = max(0.005,
                                    min(cut, expGrayHighCut[i]*2))

            self.fgcmLog.info("ExpGray repeatability cut (%s band): %.4f" % (self.fgcmPars.bands[i],
                                                                             expGrayPhotometricCut[i]))
            self.fgcmLog.info("ExpGray repeatability cut (%s band): %.4f" % (self.fgcmPars.bands[i],
                                                                             expGrayHighCut[i]))

        return (expGrayPhotometricCut, expGrayHighCut)

    def computeCCDAndExpDeltaMagBkg(self):
        """
        Compute CCD and exposure delta-bkg offsets.
        """
        if not self.fgcmStars.hasDeltaMagBkg:
            # There is nothing to compute.  Leave as 0s.
            return

        startTime = time.time()
        self.fgcmLog.debug('Computing ccdDeltaMagBkg and ExpDeltaMagBkg.')

        ccdDeltaMagBkg = snmm.getArray(self.ccdDeltaMagBkgHandle)
        expDeltaMagBkg = snmm.getArray(self.expDeltaMagBkgHandle)

        obsDeltaMagBkg = snmm.getArray(self.fgcmStars.obsDeltaMagBkgHandle)
        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        goodObs, = np.where(obsFlag == 0)

        # Do the exposures first
        h, rev = histogram_rev_sorted(obsExpIndex[goodObs])

        expDeltaMagBkg[:] = self.illegalValue

        use, = np.where(h > int(3./self.deltaMagBkgOffsetPercentile))
        for i in use:
            i1a = rev[rev[i]: rev[i + 1]]

            eInd = obsExpIndex[goodObs[i1a[0]]]

            mag = obsMagADU[goodObs[i1a]]

            st = np.argsort(mag)
            bright, = np.where(mag < mag[st[int(self.deltaMagBkgOffsetPercentile*st.size)]])
            expDeltaMagBkg[eInd] = np.median(obsDeltaMagBkg[goodObs[i1a[bright]]])

        # Set the per-ccd defaults to the exposure numbers
        ccdDeltaMagBkg[:, :] = np.repeat(expDeltaMagBkg, self.fgcmPars.nCCD).reshape(ccdDeltaMagBkg.shape)

        # Do the exp/ccd second
        expCcdHash = obsExpIndex[goodObs]*(self.fgcmPars.nCCD + 1) + obsCCDIndex[goodObs]

        h, rev = histogram_rev_sorted(expCcdHash)

        # We need at least 3 for a median, and of those from the percentile...
        use, = np.where(h > int(3./self.deltaMagBkgOffsetPercentile))
        for i in use:
            i1a = rev[rev[i]: rev[i + 1]]

            eInd = obsExpIndex[goodObs[i1a[0]]]
            cInd = obsCCDIndex[goodObs[i1a[0]]]

            mag = obsMagADU[goodObs[i1a]]

            st = np.argsort(mag)
            bright, = np.where(mag < mag[st[int(self.deltaMagBkgOffsetPercentile*st.size)]])
            ccdDeltaMagBkg[eInd, cInd] = np.median(obsDeltaMagBkg[goodObs[i1a[bright]]])

        self.fgcmPars.compExpDeltaMagBkg[:] = expDeltaMagBkg

        if not self.quietMode:
            self.fgcmLog.info('Computed ccdDeltaBkg and expDeltaBkg in %.2f seconds.' %
                              (time.time() - startTime))

    def computeExposureReferenceOffsets(self):
        """Compute exposure reference offsets.

        This method computes per-exposure offsets between the calibrated stars and the reference
        stars for an end-of-calibration "fixup".
        """
        if not self.fgcmStars.hasRefstars:
            # Nothing to do here
            return

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex

        objRefIDIndex = snmm.getArray(self.fgcmStars.objRefIDIndexHandle)
        refMag = snmm.getArray(self.fgcmStars.refMagHandle)

        goodStars = self.fgcmStars.getGoodStarIndices(checkMinObs=True, removeRefstarOutliers=True, removeRefstarBadcols=True, removeRefstarReserved=True)
        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=self.fgcmPars.expFlag, checkBadMag=True)

        # Add in the gray values
        obsExpIndexGO = obsExpIndex[goodObs]
        obsCCDIndexGO = obsCCDIndex[goodObs]
        obsBandIndexGO = obsBandIndex[goodObs]
        obsMagStdGO = obsMagStd[goodObs]

        ccdGray = snmm.getArray(self.ccdGrayHandle)
        if np.any(self.ccdGraySubCCD):
            ccdGraySubCCDPars = snmm.getArray(self.ccdGraySubCCDParsHandle)

        ok, = np.where(ccdGray[obsExpIndexGO, obsCCDIndexGO] > self.illegalValue)

        if np.any(self.ccdGraySubCCD):
            obsXGO = snmm.getArray(self.fgcmStars.obsXHandle)[goodObs]
            obsYGO = snmm.getArray(self.fgcmStars.obsYHandle)[goodObs]
            expCcdHash = (obsExpIndexGO[ok] * (self.fgcmPars.nCCD + 1) +
                          obsCCDIndexGO[ok])
            h, rev = histogram_rev_sorted(expCcdHash)
            use, = np.where(h > 0)
            for i in use:
                i1a = rev[rev[i]: rev[i + 1]]
                eInd = obsExpIndexGO[ok[i1a[0]]]
                cInd = obsCCDIndexGO[ok[i1a[0]]]
                field = Cheb2dField(self.deltaMapperDefault['x_size'][cInd],
                                    self.deltaMapperDefault['y_size'][cInd],
                                    ccdGraySubCCDPars[eInd, cInd, :])
                fluxScale = field.evaluate(obsXGO[ok[i1a]], obsYGO[ok[i1a]])
                obsMagStdGO[ok[i1a]] += -2.5 * np.log10(np.clip(fluxScale, 0.1, None))
        else:
            # Regular non-sub-ccd
            obsMagStdGO[ok] += ccdGray[obsExpIndexGO[ok], obsCCDIndexGO[ok]]

        goodRefObsGO, = np.where(objRefIDIndex[obsObjIDIndex[goodObs]] >= 0)

        obsUse, = np.where((obsMagStd[goodObs[goodRefObsGO]] < 90.0) &
                           (refMag[objRefIDIndex[obsObjIDIndex[goodObs[goodRefObsGO]]],
                                   obsBandIndex[goodObs[goodRefObsGO]]] < 90.0))

        goodRefObsGO = goodRefObsGO[obsUse]

        EGrayGRO = (refMag[objRefIDIndex[obsObjIDIndex[goodObs[goodRefObsGO]]],
                           obsBandIndex[goodObs[goodRefObsGO]]] -
                    obsMagStdGO[goodRefObsGO])

        if len(goodRefObsGO) == 0:
            self.fgcmLog.warning("No reference objects found to compare to exposures.")
            return

        # And then this can be split per exposure.

        h, rev = histogram_rev_sorted(obsExpIndexGO[goodRefObsGO])

        use, = np.where(h >= self.minStarPerExp)

        self.fgcmPars.compExpRefOffset[:] = self.illegalValue
        for i in use:
            i1a = rev[rev[i]: rev[i + 1]]

            eInd = obsExpIndexGO[goodRefObsGO[i1a[0]]]
            bInd = obsBandIndexGO[goodRefObsGO[i1a[0]]]

            self.fgcmPars.compExpRefOffset[eInd] = np.median(EGrayGRO[i1a])

        # Do plots if necessary.
        if self.plotPath is None:
            return

        rejectMask = (expFlagDict['TOO_FEW_STARS'] |
                      expFlagDict['NO_STARS'] |
                      expFlagDict['BAND_NOT_IN_LUT'])

        expUse, = np.where((self.fgcmPars.expFlag & rejectMask) == 0)

        for i in range(self.fgcmPars.nBands):
            inBand, = np.where((self.fgcmPars.expBandIndex[expUse] == i) &
                               (self.fgcmPars.compExpRefOffset[expUse] > self.illegalValue))

            if inBand.size == 0:
                continue

            fig = makeFigure(figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)

            coeff = histoGauss(ax, self.fgcmPars.compExpRefOffset[expUse[inBand]]*1000.0)
            coeff[1] /= 1000.0
            coeff[2] /= 1000.0

            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.locator_params(axis='x', nbins=5)

            text=r'$(%s)$' % (self.fgcmPars.bands[i]) + '\n' + \
                r'$\mathrm{Cycle\ %d}$' % (self.cycleNumber) + '\n' + \
                r'$\mu = %.2f$' % (coeff[1]*1000.0) + '\n' + \
                r'$\sigma = %.2f$' % (coeff[2]*1000.0)

            ax.annotate(text, (0.95, 0.93), xycoords='axes fraction', ha='right', va='top', fontsize=16)
            ax.set_xlabel(r'$\mathrm{EXP}^{\mathrm{ref}}\,(\mathrm{mmag})$', fontsize=16)
            ax.set_ylabel(r'# of Exposures',fontsize=14)

            if self.butlerQC is not None:
                putButlerFigure(self.fgcmLog,
                                self.butlerQC,
                                self.plotHandleDict,
                                "ExpgrayReference",
                                self.cycleNumber,
                                fig,
                                band=self.fgcmPars.bands[i])
            else:
                fig.savefig('%s/%s_expref_%s.png' % (self.plotPath,
                                                     self.outfileBaseWithCycle,
                                                     self.fgcmPars.bands[i]))

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        del state['focalPlaneProjector']
        del state['butlerQC']
        del state['plotHandleDict']
        return state

    def freeSharedMemory(self):
        """Free shared memory"""
        if not self.arraysPrepared:
            return

        snmm.freeArray(self.expGrayForInitialSelectionHandle)
        snmm.freeArray(self.expGrayRMSForInitialSelectionHandle)
        snmm.freeArray(self.expNGoodStarForInitialSelectionHandle)
        snmm.freeArray(self.ccdGrayHandle)
        snmm.freeArray(self.ccdDeltaStdHandle)
        snmm.freeArray(self.ccdGrayRMSHandle)
        snmm.freeArray(self.ccdGrayErrHandle)
        snmm.freeArray(self.ccdNGoodObsHandle)
        snmm.freeArray(self.ccdNGoodStarsHandle)
        snmm.freeArray(self.ccdNGoodTilingsHandle)
        if np.any(self.ccdGraySubCCD):
            snmm.freeArray(self.ccdGraySubCCDParsHandle)
        snmm.freeArray(self.expGrayHandle)
        snmm.freeArray(self.expDeltaStdHandle)
        snmm.freeArray(self.expGrayRMSHandle)
        snmm.freeArray(self.expGrayErrHandle)
        snmm.freeArray(self.expNGoodStarsHandle)
        snmm.freeArray(self.expNGoodCCDsHandle)
        snmm.freeArray(self.expNGoodTilingsHandle)
        snmm.freeArray(self.expGrayColorSplitHandle)
        snmm.freeArray(self.expGrayRMSColorSplitHandle)
        snmm.freeArray(self.expGrayErrColorSplitHandle)
        snmm.freeArray(self.expGrayNGoodStarsColorSplitHandle)
        snmm.freeArray(self.ccdDeltaMagBkgHandle)
