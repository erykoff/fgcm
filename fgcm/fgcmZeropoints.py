import numpy as np
import os
import sys
import esutil

from .fgcmUtilities import zpFlagDict
from .fgcmUtilities import expFlagDict
from .fgcmUtilities import Cheb2dField
from .fgcmUtilities import dataBinner
from .fgcmUtilities import makeFigure, putButlerFigure

from matplotlib import colormaps

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm


class FgcmZeropoints(object):
    """
    Class to compute final zeropoints

    parameters
    ----------
    fgcmConfig: FgcmConfig
    fgcmPars: FgcmParameters
    fgcmLUT: FgcmLUT
    fgcmGray: FgcmGray
    fgcmRetrieval: FgcmRetrieval

    Config variables
    ----------------
    minCCDPerExp: int
       Minimum number of CCDs on an exposure to recover bad ccds
    minStarPerCCD: int
       Minimum number of stars to recover zeropoint through matching
    maxCCDGrayErr: float
       Maximum CCD Gray error to recover zeropoint through matching
    expGrayRecoverCut: float
       Minimum (negative!) exposure gray to consider recovering via focal-plane average
    expGrayErrRecoverCut: float
       Maximum exposure gray error to consider recovering via focal-plane average
    expVarGrayPhotometricCut: `list`
       Exposure gray variance to consider recovering via focal-plane average
    """

    def __init__(self, fgcmConfig, fgcmPars, fgcmLUT, fgcmGray, fgcmRetrieval, fgcmStars, butlerQC=None, plotHandleDict=None):

        self.fgcmPars = fgcmPars
        self.fgcmLUT = fgcmLUT
        self.fgcmGray = fgcmGray
        self.fgcmRetrieval = fgcmRetrieval
        self.fgcmStars = fgcmStars

        self.butlerQC = butlerQC
        self.plotHandleDict = plotHandleDict

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.debug('Initializing FgcmZeropoints...')

        self.illegalValue = fgcmConfig.illegalValue
        self.outputPath = fgcmConfig.outputPath
        self.cycleNumber = fgcmConfig.cycleNumber
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.plotPath = fgcmConfig.plotPath
        self.zptABNoThroughput = fgcmConfig.zptABNoThroughput
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.focalPlaneProjector = fgcmConfig.focalPlaneProjector
        self.defaultCameraOrientation = fgcmConfig.defaultCameraOrientation
        self.minCCDPerExp = fgcmConfig.minCCDPerExp
        self.minStarPerCCD = fgcmConfig.minStarPerCCD
        self.maxCCDGrayErr = fgcmConfig.maxCCDGrayErr
        self.expField = fgcmConfig.expField
        self.ccdField = fgcmConfig.ccdField
        self.expGrayRecoverCut = fgcmConfig.expGrayRecoverCut
        self.expGrayErrRecoverCut = fgcmConfig.expGrayErrRecoverCut
        self.expVarGrayPhotometricCut = fgcmConfig.expVarGrayPhotometricCut
        self.deltaMagBkgPerCcd = fgcmConfig.deltaMagBkgPerCcd
        self.superStarSubCCD = fgcmConfig.superStarSubCCD
        self.seeingSubExposure = fgcmConfig.seeingSubExposure
        self.ccdGraySubCCD = fgcmConfig.ccdGraySubCCD
        self.colorSplitBands = fgcmConfig.colorSplitBands
        self.colorSplitIndices = fgcmConfig.colorSplitIndices
        self.I10StdBand = fgcmConfig.I10StdBand
        self.I0StdBand = fgcmConfig.I0StdBand
        self.I1StdBand = fgcmConfig.I1StdBand
        self.lambdaStdBand = fgcmConfig.lambdaStdBand
        self.outputFgcmcalZpts = fgcmConfig.outputFgcmcalZpts
        self.useExposureReferenceOffset = fgcmConfig.useExposureReferenceOffset
        self.quietMode = fgcmConfig.quietMode
        self.rng = fgcmConfig.rng

    def computeZeropoints(self, doPlots=False):
        """
        Compute the zeropoints from all the fits that have been performed.

        Parameters
        ----------
        doPlots : `bool`, optional
            Make plots?

        Output attributes
        -----------------
        zpStruct: Zero point recarray (nExp * nCCD)
           expField: Exposure field name
           ccdField: CCD field name
           'FGCM_FLAG': Quality flag value
           'FGCM_ZPT': Zeropoint
           'FGCM_ZPTERR': Error on zeropoint
           'FGCM_FZPT_CHEB': Chebyshev polynomial coefficients for zeropoint (flux units) (if outputFgcmcalZpts)
           'FGCM_FZPT_SSTAR_CHEB': Chebyshev polynomial coefficients for superstar (flux units) (if outputFgcmcalZpts)
           'FGCM_FZPT_XYMAX': Maximum X/Y maximum range (if outputFgcmcalZpts)
           'FGCM_I0': I0 for exp/ccd (throughput)
           'FGCM_I10': I10 for exp/ccd (chromatic)
           'FGCM_R0': Retrieved throughput integral
           'FGCM_R10': Retrieved chromatic integral
           'FGCM_GRY': Delta-zeropoint due to CCD gray
           'FGCM_ZPTVAR': Variance of zeropoint as estimated from gray corrections
           'FGCM_TILINGS': Average number of observations of calib stars on CCD
           'FGCM_FPGRY': Average focal-plane gray (exp_gray)
           'FGCM_FPVAR': Focal-plane variance
           'FGCM_DUST': Delta-zeropoint due to mirror/corrector dust buildup
           'FGCM_FILTER': Delta-zeropoint due to the filter offset
           'FGCM_FLAT': Delta-zeropoint due to superStarFlat
           'FGCM_APERCORR': Delta-zeropoint due to aperture correction
           'FGCM_DELTAMAGBKG': Delta-zeropoint due to background offset correction
           'FGCM_CTRANS': Transmission curve adjustment constant
           'EXPTIME': Exposure time (seconds)
           'FILTERNAME': Filter name
           'BAND': band name
           'MJD': Date of observation
        atmStruct: Atmosphere parameter recarray (nExp)
           'PMB': Barometric pressure (mb)
           'PWV': preciptable water vapor (mm)
           'TAU': aerosol optical index
           'ALPHA': Aerosol slope
           'O3': Ozone (dob)
           'SECZENITH': secant(zenith angle)
        """

        # first, we need to get relevant quantities from shared memory.
        expGray = snmm.getArray(self.fgcmGray.expGrayHandle)
        expDeltaStd = snmm.getArray(self.fgcmGray.expDeltaStdHandle)
        expGrayErr = snmm.getArray(self.fgcmGray.expGrayErrHandle)
        expGrayRMS = snmm.getArray(self.fgcmGray.expGrayRMSHandle)
        expNGoodCCDs = snmm.getArray(self.fgcmGray.expNGoodCCDsHandle)
        expNGoodTilings = snmm.getArray(self.fgcmGray.expNGoodTilingsHandle)
        expGrayColorSplit = snmm.getArray(self.fgcmGray.expGrayColorSplitHandle)
        expGrayErrColorSplit = snmm.getArray(self.fgcmGray.expGrayErrColorSplitHandle)
        expGrayRMSColorSplit = snmm.getArray(self.fgcmGray.expGrayRMSColorSplitHandle)

        ccdGray = snmm.getArray(self.fgcmGray.ccdGrayHandle)
        ccdDeltaStd = snmm.getArray(self.fgcmGray.ccdDeltaStdHandle)
        ccdGrayRMS = snmm.getArray(self.fgcmGray.ccdGrayRMSHandle)
        ccdGrayErr = snmm.getArray(self.fgcmGray.ccdGrayErrHandle)
        ccdNGoodStars = snmm.getArray(self.fgcmGray.ccdNGoodStarsHandle)
        ccdNGoodTilings = snmm.getArray(self.fgcmGray.ccdNGoodTilingsHandle)

        r0 = snmm.getArray(self.fgcmRetrieval.r0Handle)
        r10 = snmm.getArray(self.fgcmRetrieval.r10Handle)

        deltaMapperDefault = self.focalPlaneProjector(int(self.defaultCameraOrientation))

        # and we need to make sure we have the parameters, and
        #  set these to the exposures
        parArray = self.fgcmPars.getParArray(fitterUnits=False)
        # Use *input* values for the retrieved atmosphere parameters (if needed)
        self.fgcmPars.parsToExposures(retrievedInput=True)

        # set up output structures

        if not self.quietMode:
            self.fgcmLog.info('Building zeropoint structure...')

        dtype = [(self.expField, 'i8'),
                 (self.ccdField, 'i2'),
                 ('FGCM_FLAG', 'i2'),
                 ('FGCM_ZPT', 'f8'),
                 ('FGCM_ZPTERR', 'f8')]

        if self.outputFgcmcalZpts:
            self.nChebParGray = 1
            if np.any(self.ccdGraySubCCD):
                self.nChebParGray = self.fgcmGray.ccdGrayNPar

            self.nChebParSstar = 1
            if np.any(self.superStarSubCCD):
                self.nChebParSstar = self.fgcmPars.superStarNPar

            dtype.extend([('FGCM_FZPT_CHEB', 'f8', (self.nChebParGray, )),
                          ('FGCM_FZPT_SSTAR_CHEB', 'f8', (self.nChebParSstar, )),
                          ('FGCM_FZPT_XYMAX', 'f4', 2)])

        maxBandLen = len(max(self.fgcmPars.bands, key=len))
        maxFilterLen = len(max(self.fgcmPars.lutFilterNames, key=len))

        dtype.extend([('FGCM_I0','f8'),
                      ('FGCM_I10','f8'),
                      ('FGCM_R0','f8'),
                      ('FGCM_R10','f8'),
                      ('FGCM_GRY','f8'),
                      ('FGCM_ZPTVAR','f8'),
                      ('FGCM_TILINGS','f8'),
                      ('FGCM_FPGRY','f8'),
                      ('FGCM_FPVAR','f8'),
                      ('FGCM_FPGRY_CSPLIT', 'f8', 3),
                      ('FGCM_FPGRY_CSPLITERR', 'f8', 3),
                      ('FGCM_FPGRY_CSPLITVAR', 'f8', 3),
                      ('FGCM_FPREF', 'f8'),
                      ('FGCM_DUST','f8'),
                      ('FGCM_FILTER','f8'),
                      ('FGCM_FLAT','f8'),
                      ('FGCM_APERCORR','f8'),
                      ('FGCM_DELTAMAGBKG', 'f8'),
                      ('FGCM_DELTACHROM', 'f8'),
                      ('FGCM_CTRANS', 'f4'),
                      ('EXPTIME','f4'),
                      ('FILTERNAME', 'S%d' % (maxFilterLen)),
                      ('BAND', 'S%d' % (maxBandLen)),
                      ('MJD', 'f8')])

        zpStruct = np.zeros(self.fgcmPars.nExp*self.fgcmPars.nCCD,
                            dtype=dtype)

        atmStruct = np.zeros(self.fgcmPars.nExp,
                             dtype=[(self.expField, 'i8'),
                                    ('PMB', 'f8'),
                                    ('PWV', 'f8'),
                                    ('TAU', 'f8'),
                                    ('ALPHA', 'f8'),
                                    ('O3', 'f8'),
                                    ('CTRANS', 'f8'),
                                    ('LAMSTD', 'f8'),
                                    ('SECZENITH', 'f8')])

        ## start with zpStruct

        # fill out exposures and ccds
        zpStruct[self.expField][:] = np.repeat(self.fgcmPars.expArray,
                                          self.fgcmPars.nCCD)
        zpStruct[self.ccdField][:] = np.tile(np.arange(self.fgcmPars.nCCD)+self.ccdStartIndex,
                                        self.fgcmPars.nExp)

        # get the exposure indices and CCD indices
        zpExpIndex = np.searchsorted(self.fgcmPars.expArray,zpStruct[self.expField])
        zpCCDIndex = zpStruct[self.ccdField] - self.ccdStartIndex

        # fill exposure quantities
        lutFilterNameArray = np.array(self.fgcmPars.lutFilterNames)
        zpStruct['FILTERNAME'][:] = lutFilterNameArray[self.fgcmPars.expLUTFilterIndex[zpExpIndex]]
        bandArray = np.array(self.fgcmPars.bands)
        zpStruct['BAND'][:] = bandArray[self.fgcmPars.expBandIndex[zpExpIndex]]
        zpStruct['EXPTIME'][:] = self.fgcmPars.expExptime[zpExpIndex]
        zpStruct['MJD'][:] = self.fgcmPars.expMJD[zpExpIndex]

        # Set the x/y sizes
        if self.outputFgcmcalZpts:
            zpStruct['FGCM_FZPT_XYMAX'][:, 0] = deltaMapperDefault['x_size'][zpCCDIndex]
            zpStruct['FGCM_FZPT_XYMAX'][:, 1] = deltaMapperDefault['y_size'][zpCCDIndex]

        # fill in the superstar flat
        zpStruct['FGCM_FLAT'][:] = self.fgcmPars.expCCDSuperStar[zpExpIndex,
                                                                 zpCCDIndex]

        # fill in the optics dust
        zpStruct['FGCM_DUST'][:] = self.fgcmPars.expQESys[zpExpIndex]

        # And the filter offset
        zpStruct['FGCM_FILTER'][:] = self.fgcmPars.expFilterOffset[zpExpIndex]

        # fill in the aperture correction
        if self.seeingSubExposure:
            zpStruct['FGCM_APERCORR'][:] = self.fgcmPars.ccdApertureCorrection[zpExpIndex, zpCCDIndex]
        else:
            zpStruct['FGCM_APERCORR'][:] = self.fgcmPars.expApertureCorrection[zpExpIndex]

        # Fill in the background correction
        ccdDeltaMagBkg = snmm.getArray(self.fgcmGray.ccdDeltaMagBkgHandle)
        if self.deltaMagBkgPerCcd:
            ccdDeltaMagBkg = snmm.getArray(self.fgcmGray.ccdDeltaMagBkgHandle)
            zpStruct['FGCM_DELTAMAGBKG'][:] = ccdDeltaMagBkg[zpExpIndex, zpCCDIndex]
        else:
            expDeltaMagBkg = snmm.getArray(self.fgcmGray.expDeltaMagBkgHandle)
            zpStruct['FGCM_DELTAMAGBKG'][:] = expDeltaMagBkg[zpExpIndex]

        # Fill in the transmission adjustment constant
        zpStruct['FGCM_CTRANS'][:] = self.fgcmPars.expCTrans[zpExpIndex]

        # fill in the retrieved values
        zpStruct['FGCM_R0'][:] = r0[zpExpIndex, zpCCDIndex]
        zpStruct['FGCM_R10'][:] = r10[zpExpIndex, zpCCDIndex]

        # and the focal-plane gray and var...
        # these are only filled in for those exposures where we have it computed
        zpStruct['FGCM_FPGRY'][:] = self.illegalValue
        zpStruct['FGCM_FPVAR'][:] = self.illegalValue

        zpExpOk, = np.where(expNGoodCCDs[zpExpIndex] >= self.minCCDPerExp)
        zpStruct['FGCM_FPGRY'][zpExpOk] = expGray[zpExpIndex[zpExpOk]]
        zpStruct['FGCM_FPVAR'][zpExpOk] = expGrayRMS[zpExpIndex[zpExpOk]]**2.

        zpStruct['FGCM_FPGRY_CSPLIT'][zpExpOk, :] = expGrayColorSplit[zpExpIndex[zpExpOk], :]
        zpStruct['FGCM_FPGRY_CSPLITVAR'][zpExpOk, :] = expGrayRMSColorSplit[zpExpIndex[zpExpOk], :]**2.
        zpStruct['FGCM_FPGRY_CSPLITERR'][zpExpOk, :] = expGrayErrColorSplit[zpExpIndex[zpExpOk], :]

        self.fgcmLog.info('%d exposure/ccd sets have exposures with >=%d good ccds' %
                         (zpExpOk.size, self.minCCDPerExp))

        # look up the I0 and I10s.  These are defined for everything
        #  (even if only standard bandpass, it'll grab instrumental)

        ## FIXME: will probably need some sort of rotation information at some point

        # need secZenith for each exp/ccd pair
        deltaRA = np.zeros(zpStruct.size)
        deltaDec = np.zeros(zpStruct.size)
        h, rev = esutil.stat.histogram(zpExpIndex, rev=True)
        ok, = np.where(h > 0)
        for i in ok:
            i1a = rev[rev[i]: rev[i + 1]]
            # rot = self.fgcmPars.expTelRot[zpExpIndex[i1a[0]]]
            # ccdIndex = zpCCDIndex[i1a]

            deltaMapper = self.focalPlaneProjector(int(self.fgcmPars.expTelRot[zpExpIndex[i1a[0]]]))
            deltaRA[i1a] = deltaMapper['delta_ra_cent'][zpCCDIndex[i1a]]
            deltaDec[i1a] = deltaMapper['delta_dec_cent'][zpCCDIndex[i1a]]

        ccdHA = (self.fgcmPars.expTelHA[zpExpIndex] -
                 np.radians(deltaRA[zpCCDIndex])*np.cos(self.fgcmPars.expTelDec[zpExpIndex]))
        ccdDec = (self.fgcmPars.expTelDec[zpExpIndex] +
                  np.radians(deltaDec[zpCCDIndex]))
        ccdSecZenith = 1./(np.sin(ccdDec) * self.fgcmPars.sinLatitude +
                           np.cos(ccdDec) * self.fgcmPars.cosLatitude * np.cos(ccdHA))

        # and do the LUT lookups
        lutIndices = self.fgcmLUT.getIndices(self.fgcmPars.expLUTFilterIndex[zpExpIndex],
                                             self.fgcmPars.expLnPwv[zpExpIndex],
                                             self.fgcmPars.expO3[zpExpIndex],
                                             self.fgcmPars.expLnTau[zpExpIndex],
                                             self.fgcmPars.expAlpha[zpExpIndex],
                                             ccdSecZenith,
                                             zpCCDIndex,
                                             self.fgcmPars.expPmb[zpExpIndex])
        zpStruct['FGCM_I0'][:] = self.fgcmLUT.computeI0(self.fgcmPars.expLnPwv[zpExpIndex],
                                                        self.fgcmPars.expO3[zpExpIndex],
                                                        self.fgcmPars.expLnTau[zpExpIndex],
                                                        self.fgcmPars.expAlpha[zpExpIndex],
                                                        ccdSecZenith,
                                                        self.fgcmPars.expPmb[zpExpIndex],
                                                        lutIndices)
        zpStruct['FGCM_I10'][:] = self.fgcmLUT.computeI1(self.fgcmPars.expLnPwv[zpExpIndex],
                                                         self.fgcmPars.expO3[zpExpIndex],
                                                         self.fgcmPars.expLnTau[zpExpIndex],
                                                         self.fgcmPars.expAlpha[zpExpIndex],
                                                         ccdSecZenith,
                                                         self.fgcmPars.expPmb[zpExpIndex],
                                                         lutIndices) / zpStruct['FGCM_I0'][:]

        bad, = np.where((~np.isfinite(zpStruct['FGCM_I0'])) |
                        (~np.isfinite(zpStruct['FGCM_I10'])))
        if bad.size > 0:
            self.fgcmLog.warning("There are %d ccds with bad I0/I10 values" % (bad.size))
            zpStruct['FGCM_I0'][bad] = self.illegalValue
            zpStruct['FGCM_I10'][bad] = self.illegalValue

        # Set the tilings, gray values, and zptvar

        zpStruct['FGCM_TILINGS'][:] = self.illegalValue
        zpStruct['FGCM_GRY'][:] = self.illegalValue
        zpStruct['FGCM_DELTACHROM'][:] = self.illegalValue
        zpStruct['FGCM_ZPTVAR'][:] = self.illegalValue

        goodCCD, = np.where((ccdNGoodStars[zpExpIndex,zpCCDIndex] >=
                             self.minStarPerCCD) &
                            (ccdGrayErr[zpExpIndex,zpCCDIndex] <=
                             self.maxCCDGrayErr))
        zpStruct['FGCM_TILINGS'][goodCCD] = ccdNGoodTilings[zpExpIndex[goodCCD],
                                                            zpCCDIndex[goodCCD]]
        zpStruct['FGCM_GRY'][goodCCD] = ccdGray[zpExpIndex[goodCCD],
                                                zpCCDIndex[goodCCD]]
        zpStruct['FGCM_DELTACHROM'][goodCCD] = ccdDeltaStd[zpExpIndex[goodCCD],
                                                         zpCCDIndex[goodCCD]]
        zpStruct['FGCM_ZPTVAR'][goodCCD] = ccdGrayErr[zpExpIndex[goodCCD],
                                                      zpCCDIndex[goodCCD]]**2.

        self.fgcmLog.info('%d CCDs are Good (>=%d stars; err <= %.3f)' %
                         (goodCCD.size, self.minStarPerCCD, self.maxCCDGrayErr))

        # Record FGCM_FPREF (focal-plane reference offset).
        zpStruct['FGCM_FPREF'] = self.illegalValue

        fpRefOk, = np.where(self.fgcmPars.compExpRefOffset[zpExpIndex] > self.illegalValue)
        zpStruct['FGCM_FPREF'][fpRefOk] = self.fgcmPars.compExpRefOffset[zpExpIndex[fpRefOk]]

        # check: if this has too few stars on the ccd OR the ccd error is too big
        #        AND the exposure has enough ccds
        #        AND the exposure gray error is small enough
        #        AND the exposure gray rms is small enough
        #        AND the exposure gray is not very large (configurable)
        #  then we can use the exposure stats to fill the variables

        badCCDGoodExp, = np.where(((ccdNGoodStars[zpExpIndex,zpCCDIndex] <
                            self.minStarPerCCD) |
                           (ccdGrayErr[zpExpIndex,zpCCDIndex] >
                            self.maxCCDGrayErr)) &
                          (expNGoodCCDs[zpExpIndex] >=
                           self.minCCDPerExp) &
                          (expGrayErr[zpExpIndex] <=
                           self.expGrayErrRecoverCut) &
                          (expGrayRMS[zpExpIndex] <=
                           np.sqrt(self.expVarGrayPhotometricCut[self.fgcmPars.expBandIndex[zpExpIndex]])) &
                          (expGray[zpExpIndex] >=
                           self.expGrayRecoverCut))

        zpStruct['FGCM_TILINGS'][badCCDGoodExp] = expNGoodTilings[zpExpIndex[badCCDGoodExp]]
        zpStruct['FGCM_GRY'][badCCDGoodExp] = expGray[zpExpIndex[badCCDGoodExp]]
        zpStruct['FGCM_DELTACHROM'][badCCDGoodExp] = expDeltaStd[zpExpIndex[badCCDGoodExp]]
        # And fill in the chebyshev parameters if necessary
        if self.outputFgcmcalZpts and np.any(self.ccdGraySubCCD):
            # We need to alter the gray parameters to record the constant (interpolated)
            # gray offset
            ccdGraySubCCDPars = snmm.getArray(self.fgcmGray.ccdGraySubCCDParsHandle)

            ccdGraySubCCDPars[zpExpIndex[badCCDGoodExp], zpCCDIndex[badCCDGoodExp], :] = 0.0
            ccdGraySubCCDPars[zpExpIndex[badCCDGoodExp], zpCCDIndex[badCCDGoodExp], 0] = 10.**(zpStruct['FGCM_GRY'][badCCDGoodExp] / (-2.5))

        zpStruct['FGCM_ZPTVAR'][badCCDGoodExp] = expGrayRMS[zpExpIndex[badCCDGoodExp]]**2.

        self.fgcmLog.info('%d CCDs recovered from good exposures (>=%d good CCDs, etc.)' %
                         (badCCDGoodExp.size, self.minCCDPerExp))

        # flag the photometric (fit) exposures
        photZpIndex, = np.where(self.fgcmPars.expFlag[zpExpIndex] == 0)

        photFitBand, = np.where(~self.fgcmPars.expNotFitBandFlag[zpExpIndex[photZpIndex]])
        zpStruct['FGCM_FLAG'][photZpIndex[photFitBand]] |= (
            zpFlagDict['PHOTOMETRIC_FIT_EXPOSURE'])

        self.fgcmLog.info('%d CCDs marked as photometric, used in fit' %
                         (photFitBand.size))

        photNotFitBand, = np.where(self.fgcmPars.expNotFitBandFlag[zpExpIndex[photZpIndex]])
        zpStruct['FGCM_FLAG'][photZpIndex[photNotFitBand]] |= (
            zpFlagDict['PHOTOMETRIC_NOTFIT_EXPOSURE'])

        self.fgcmLog.info('%d CCDs marked as photometric, not used in fit' %
                         (photNotFitBand.size))

        # flag the non-photometric exposures on calibratable nights
        rejectMask = (expFlagDict['TOO_FEW_EXP_ON_NIGHT'] |
                      expFlagDict['BAND_NOT_IN_LUT'] |
                      expFlagDict['NO_STARS'])
        acceptMask = (expFlagDict['EXP_GRAY_TOO_NEGATIVE'] |
                      expFlagDict['EXP_GRAY_TOO_POSITIVE'] |
                      expFlagDict['VAR_GRAY_TOO_LARGE'] |
                      expFlagDict['TOO_FEW_STARS'] |
                      expFlagDict['BAD_FWHM'])

        nonPhotZpIndex, = np.where(((self.fgcmPars.expFlag[zpExpIndex] & acceptMask) > 0) &
                           ((self.fgcmPars.expFlag[zpExpIndex] & rejectMask) == 0))
        zpStruct['FGCM_FLAG'][nonPhotZpIndex] |= zpFlagDict['NONPHOTOMETRIC_FIT_NIGHT']

        self.fgcmLog.info('%d CCDs marked non-photometric, on a night with a fit' %
                         (nonPhotZpIndex.size))

        # and the exposures on non-calibratable nights (photometric or not, we don't know)
        rejectMask = (expFlagDict['NO_STARS'] |
                      expFlagDict['BAND_NOT_IN_LUT'])
        acceptMask = (expFlagDict['TOO_FEW_EXP_ON_NIGHT'])
        badNightZpIndex, = np.where(((self.fgcmPars.expFlag[zpExpIndex] & acceptMask) > 0) &
                                    ((self.fgcmPars.expFlag[zpExpIndex] & rejectMask) == 0))
        zpStruct['FGCM_FLAG'][badNightZpIndex] |= zpFlagDict['NOFIT_NIGHT']

        self.fgcmLog.info('%d CCDs on nights without a fit (assume standard atmosphere)' %
                         (badNightZpIndex.size))

        # and finally, the hopeless exposures
        acceptMask = (expFlagDict['NO_STARS'] |
                      expFlagDict['BAND_NOT_IN_LUT'])
        hopelessZpIndex, = np.where(((self.fgcmPars.expFlag[zpExpIndex] & acceptMask) > 0))
        zpStruct['FGCM_FLAG'][hopelessZpIndex] |= zpFlagDict['CANNOT_COMPUTE_ZEROPOINT']

        self.fgcmLog.info('%d CCDs marked as hopeless (cannot compute zeropoint)' %
                         (hopelessZpIndex.size))

        # now we can fill the zeropoints

        zpStruct['FGCM_ZPT'][:] = self.illegalValue
        zpStruct['FGCM_ZPTERR'][:] = self.illegalValue

        if self.outputFgcmcalZpts:
            zpStruct['FGCM_FZPT_CHEB'][:, :] = self.illegalValue
            zpStruct['FGCM_FZPT_SSTAR_CHEB'][:, :] = self.illegalValue

        # start with the passable flag 1,2,4 exposures

        acceptMask = (zpFlagDict['PHOTOMETRIC_FIT_EXPOSURE'] |
              zpFlagDict['PHOTOMETRIC_NOTFIT_EXPOSURE'] |
              zpFlagDict['NONPHOTOMETRIC_FIT_NIGHT'])

        okZpIndex, = np.where((zpStruct['FGCM_FLAG'] & acceptMask) > 0)

        okCCDZpIndexFlag = ((zpStruct['FGCM_I0'][okZpIndex] > 0.0) &
                            (zpStruct['FGCM_FLAT'][okZpIndex] > self.illegalValue) &
                            (zpStruct['FGCM_DUST'][okZpIndex] > self.illegalValue) &
                            (zpStruct['FGCM_FILTER'][okZpIndex] > self.illegalValue) &
                            (zpStruct['FGCM_APERCORR'][okZpIndex] > self.illegalValue) &
                            (zpStruct['FGCM_DELTAMAGBKG'][okZpIndex] > self.illegalValue) &
                            (zpStruct['FGCM_GRY'][okZpIndex] > self.illegalValue))

        okCCDZpIndex = okZpIndex[okCCDZpIndexFlag]

        if self.outputFgcmcalZpts:
            zptChebPars, zptChebSstarPars = self._computeZptChebPars(zpStruct, okCCDZpIndex)
            zpStruct['FGCM_FZPT_CHEB'][okCCDZpIndex, :] = zptChebPars
            zpStruct['FGCM_FZPT_SSTAR_CHEB'][okCCDZpIndex, :] = zptChebSstarPars

        """
        # This should be first
        if self.useZptCheb:
            # Note that if we are combining different chebyshev polynomials, then
            # FGCM_GRY will be updated for consistency at the sub-mmag level.
            zpStruct['FGCM_FZPT_CHEB'][okCCDZpIndex, :] = self._computeZptCheb(zpStruct, okCCDZpIndex)
            """
        zpStruct['FGCM_ZPT'][okCCDZpIndex] = self._computeZpt(zpStruct, okCCDZpIndex)
        zpStruct['FGCM_ZPTERR'][okCCDZpIndex] = self._computeZptErr(zpStruct,zpExpIndex,okCCDZpIndex)

        badCCDZpExp = okZpIndex[~okCCDZpIndexFlag]
        zpStruct['FGCM_FLAG'][badCCDZpExp] |=  zpFlagDict['TOO_FEW_STARS_ON_CCD']

        # and the flag 8 not-fit exposures

        acceptMask = zpFlagDict['NOFIT_NIGHT']

        mehZpIndex, = np.where((zpStruct['FGCM_FLAG'] & acceptMask) > 0)

        mehCCDZpIndexFlag = ((zpStruct['FGCM_I0'][mehZpIndex] > 0.0) &
                             (zpStruct['FGCM_FLAT'][mehZpIndex] > self.illegalValue) &
                             (zpStruct['FGCM_DUST'][mehZpIndex] > self.illegalValue) &
                             (zpStruct['FGCM_FILTER'][mehZpIndex] > self.illegalValue) &
                             (zpStruct['FGCM_APERCORR'][mehZpIndex] > self.illegalValue) &
                             (zpStruct['FGCM_DELTAMAGBKG'][mehZpIndex] > self.illegalValue) &
                             (zpStruct['FGCM_GRY'][mehZpIndex] > self.illegalValue) &
                             (ccdNGoodStars[zpExpIndex[mehZpIndex],zpCCDIndex[mehZpIndex]] >=
                              self.minStarPerCCD) &
                             (ccdGrayErr[zpExpIndex[mehZpIndex],zpCCDIndex[mehZpIndex]] <=
                              self.maxCCDGrayErr))

        mehCCDZpIndex = mehZpIndex[mehCCDZpIndexFlag]

        if mehCCDZpIndex.size > 0:

            if self.outputFgcmcalZpts:
                zptChebPars, zptChebSstarPars = self._computeZptChebPars(zpStruct, mehCCDZpIndex)
                zpStruct['FGCM_FZPT_CHEB'][mehCCDZpIndex, :] = zptChebPars
                zpStruct['FGCM_FZPT_SSTAR_CHEB'][mehCCDZpIndex, :] = zptChebSstarPars

            """
            if self.useZptCheb:
                # Note that if we are combining different chebyshev polynomials, then
                # FGCM_GRY will be updated for consistency at the sub-mmag level.
                zpStruct['FGCM_FZPT_CHEB'][mehCCDZpIndex, :] = self._computeZptCheb(zpStruct, mehCCDZpIndex)
                """
            zpStruct['FGCM_ZPT'][mehCCDZpIndex] = self._computeZpt(zpStruct,mehCCDZpIndex)
            zpStruct['FGCM_ZPTERR'][mehCCDZpIndex] = self._computeZptErr(zpStruct,zpExpIndex,mehCCDZpIndex)

        badCCDZpExp = mehZpIndex[~mehCCDZpIndexFlag]
        zpStruct['FGCM_FLAG'][badCCDZpExp] |= zpFlagDict['TOO_FEW_STARS_ON_CCD']
        zpStruct['FGCM_FLAG'][badCCDZpExp] |= zpFlagDict['CANNOT_COMPUTE_ZEROPOINT']

        # record as a class element
        self.zpStruct = zpStruct

        #################################
        # and make the parameter file
        atmStruct[self.expField] = self.fgcmPars.expArray
        atmStruct['PMB'] = self.fgcmPars.expPmb
        atmStruct['PWV'] = np.exp(self.fgcmPars.expLnPwv)
        atmStruct['TAU'] = np.exp(self.fgcmPars.expLnTau)
        atmStruct['ALPHA'] = self.fgcmPars.expAlpha
        atmStruct['O3'] = self.fgcmPars.expO3
        atmStruct['CTRANS'] = self.fgcmPars.expCTrans
        atmStruct['LAMSTD'] = self.lambdaStdBand[self.fgcmPars.expBandIndex]
        atmStruct['SECZENITH'] = 1./(np.sin(self.fgcmPars.expTelDec) *
                                     self.fgcmPars.sinLatitude +
                                     np.cos(self.fgcmPars.expTelDec) *
                                     self.fgcmPars.cosLatitude *
                                     np.cos(self.fgcmPars.expTelHA))

        # record as a class element
        self.atmStruct=atmStruct

        ############
        ## plots
        ############

        if doPlots:
            if not self.quietMode:
                self.fgcmLog.info('Making I1/R1 plots...')

            plotter = FgcmZeropointPlotter(zpStruct, self.fgcmStars, self.fgcmPars,
                                           self.I0StdBand, self.I1StdBand, self.I10StdBand,
                                           self.colorSplitIndices, self.colorSplitBands,
                                           self.plotPath, self.outfileBaseWithCycle,
                                           self.cycleNumber, self.fgcmLog,
                                           butlerQC=self.butlerQC, plotHandleDict=self.plotHandleDict,
                                           rng=self.rng)

            plotter.makeR1I1Plots()
            plotter.makeR1I1Maps(deltaMapperDefault, ccdField=self.ccdField)
            plotter.makeR1I1TemporalResidualPlots()

            if not self.quietMode:
                self.fgcmLog.info('Making zeropoint summary plots...')

            expZpMean = np.zeros(self.fgcmPars.nExp,dtype='f4')
            expZpNCCD = np.zeros(self.fgcmPars.nExp,dtype='i4')

            rejectMask = (zpFlagDict['CANNOT_COMPUTE_ZEROPOINT'] |
                          zpFlagDict['TOO_FEW_STARS_ON_CCD'])

            okCCD,=np.where((zpStruct['FGCM_FLAG'] & rejectMask) == 0)

            np.add.at(expZpMean,
                      zpExpIndex[okCCD],
                      (zpStruct['FGCM_ZPT'][okCCD]).astype(expZpMean.dtype))
            np.add.at(expZpNCCD,
                      zpExpIndex[okCCD],
                      1)

            gd,=np.where(expZpNCCD > 0)
            expZpMean[gd] /= expZpNCCD[gd]

            expZpScaledMean = expZpMean - 2.5*np.log10(self.fgcmPars.expExptime)

            firstMJD = np.floor(np.min(self.fgcmPars.expMJD))

            # FIXME: make configurable
            stdColDict = {
                # These are the LSST color-blind friendly colors.
                "u": "#0c71ff",
                "g": "#49be61",
                "r": "#c61c00",
                "i": "#ffc200",
                "z": "#f341a2",
                "y": "#5d0000",
            }

            extraCols = ['g', 'r', 'b', 'm', 'y']
            markers = ['.', '+', 'o', '*', 'X']

            fig = makeFigure(figsize=(8, 12))
            fig.clf()

            for mode in ["unscaled", "scaled"]:
                if mode == "unscaled":
                    ax = fig.add_subplot(211)
                    ylabel = "Zero Point"
                    valueToPlot = expZpMean
                else:
                    ax = fig.add_subplot(212)
                    ylabel = "Zero Point - 2.5*log10(exptime)"
                    valueToPlot = expZpScaledMean

                for i, band in enumerate(self.fgcmPars.bands):
                    use, = np.where((self.fgcmPars.expBandIndex == i) &
                                    (valueToPlot > 0.0))

                    if (use.size == 0) :
                        continue

                    if band in stdColDict:
                        col = stdColDict[band]
                    else:
                        col = extraCols[i % 5]

                    ax.plot(
                        self.fgcmPars.expMJD[use] - firstMJD,
                        valueToPlot[use],
                        color=col,
                        marker=markers[i % 5],
                        label=r'$(%s)$' % (self.fgcmPars.bands[i]),
                    )

                    ax.legend(loc=3)
                    ax.set_xlabel(f"Days after {firstMJD}")
                    ax.set_ylabel(ylabel)

            if self.butlerQC is not None:
                putButlerFigure(self.fgcmLog,
                                self.butlerQC,
                                self.plotHandleDict,
                                "Zeropoints",
                                self.cycleNumber,
                                fig)
            elif self.plotPath is not None:
                fig.savefig('%s/%s_zeropoints.png' % (self.plotPath,
                                                      self.outfileBaseWithCycle))

    def _computeZpt(self, zpStruct, indices, includeFlat=True, includeGray=True):
        """
        Internal method to compute the zeropoint from constituents

        parameters
        ----------
        zpStruct: recarray
           Zero point structure
        indices: int array
           Indices where to compute
        includeFlat: bool, default=True
           Include superstar flat corrections
        includeGray: bool, default=True
           Include gray corrections

        returns
        -------
        Zeropoint values
        """

        if includeFlat:
            flatValue = zpStruct['FGCM_FLAT'][indices]
        else:
            flatValue = np.zeros_like(zpStruct['FGCM_FLAT'][indices])

        if includeGray:
            grayValue = zpStruct['FGCM_GRY'][indices]
        else:
            grayValue = np.zeros_like(zpStruct['FGCM_GRY'][indices])

        if self.useExposureReferenceOffset:
            fpRefValue = zpStruct['FGCM_FPREF'][indices]
            fpRefValue[fpRefValue <= self.illegalValue] = 0.0
        else:
            fpRefValue = np.zeros_like(zpStruct['FGCM_FPREF'][indices])

        return (2.5*np.log10(zpStruct['FGCM_I0'][indices]) +
                flatValue +
                zpStruct['FGCM_DUST'][indices] +
                zpStruct['FGCM_FILTER'][indices] +  # includes throughput correction
                zpStruct['FGCM_APERCORR'][indices] +
                zpStruct['FGCM_DELTAMAGBKG'][indices] +
                2.5*np.log10(zpStruct['EXPTIME'][indices]) +
                self.zptABNoThroughput +
                grayValue +
                fpRefValue)

    def _computeZptChebPars(self, zpStruct, zpIndex):
        """
        Internal method to compute flux zeropoints, including spatial variation

        parameters
        ----------
        zpStruct: recarray
           Zero point structure
        zpIndex: int array
           Array of indices to compute zeropoints

        returns
        -------
        zptChebPars: float array
           nxm array of flux zeropoint parameters
        zptChebSstarPars: float array
           nxm array of flux superstar parameters
        """

        if np.any(self.fgcmPars.superStarSubCCD):
            zptChebSstarPars = np.zeros((zpIndex.size, self.fgcmPars.superStarNPar))
        else:
            zptChebSstarPars = np.zeros((zpIndex.size, 1))

        zpExpIndex = (np.searchsorted(self.fgcmPars.expArray, zpStruct[self.expField]))[zpIndex]
        zpCCDIndex = (zpStruct[self.ccdField] - self.ccdStartIndex)[zpIndex]
        epochFilterHash = (self.fgcmPars.expEpochIndex[zpExpIndex] *
                           (self.fgcmPars.nLUTFilter + 1)*(self.fgcmPars.nCCD + 1) +
                           self.fgcmPars.expLUTFilterIndex[zpExpIndex] *
                           (self.fgcmPars.nCCD + 1) +
                           zpCCDIndex)

        h, rev = esutil.stat.histogram(epochFilterHash, rev=True)

        for i in range(h.size):
            if h[i] == 0: continue

            i1a = rev[rev[i]: rev[i + 1]]

            epInd = self.fgcmPars.expEpochIndex[zpExpIndex[i1a[0]]]
            fiInd = self.fgcmPars.expLUTFilterIndex[zpExpIndex[i1a[0]]]
            cInd = zpCCDIndex[i1a[0]]

            if np.any(self.fgcmPars.superStarSubCCD):
                zptChebSstarPars[i1a, :] = self.fgcmPars.parSuperStarFlat[epInd, fiInd, cInd, :]
            else:
                zptChebSstarPars[i1a, :] = self.fgcmPars.parSuperStarFlat[epInd, fiInd, cInd, 0]

        if np.any(self.fgcmGray.ccdGraySubCCD):
            ccdGraySubCCDPars = snmm.getArray(self.fgcmGray.ccdGraySubCCDParsHandle)

            zpExpIndex = (np.searchsorted(self.fgcmPars.expArray, zpStruct[self.expField]))[zpIndex]
            zpCCDIndex = (zpStruct[self.ccdField] - self.ccdStartIndex)[zpIndex]

            zptChebPars = ccdGraySubCCDPars[zpExpIndex,
                                            zpCCDIndex,
                                            :]
            # Do not include gray part in the computeZpt call below because this is
            # already encoded in zptChebPars
            includeGray = False
        else:
            zptChebPars = np.ones((zpIndex.size, 1))
            # Include gray part in the computeZpt call below because this is not encoded
            # in zptChebPars
            includeGray = True

        zptChebPars[:, :] = (zptChebPars.T * 10.**(self._computeZpt(zpStruct, zpIndex, includeFlat=False, includeGray=includeGray) / (-2.5))).T

        return zptChebPars, zptChebSstarPars

    def _computeZptErr(self,zpStruct,zpExpIndex,zpIndex):
        """
        Internal method to compute zeropoint error from constituents

        parameters
        ----------
        zpStruct: recarray
           Zero point structure
        zpExpIndex: int array
           Index to go from zeropoint structure to exposures
        zpIndex: int array
           Array of indices to compute zeropoint errors
        """

        # sigFgcm is computed per *band* not per *filter*
        sigFgcm = self.fgcmPars.compSigFgcm[self.fgcmPars.expBandIndex[zpExpIndex[zpIndex]]]
        nTilingsM1 = np.clip(zpStruct['FGCM_TILINGS'][zpIndex]-1.0,1.0,1e10)
        sigmaCal = self.fgcmPars.compSigmaCal[self.fgcmPars.expBandIndex[zpExpIndex[zpIndex]]]

        return np.sqrt((sigFgcm**2./nTilingsM1) +
                       zpStruct['FGCM_ZPTVAR'][zpIndex] +
                       sigmaCal**2.)

    def saveZptFits(self):
        """
        Save zeropoint to fits file
        """

        import fitsio

        outFile = '%s/%s_zpt.fits' % (self.outputPath,self.outfileBaseWithCycle)
        self.fgcmLog.info('Saving zeropoints to %s' % (outFile))
        fitsio.write(outFile,self.zpStruct,clobber=True,extname='ZPTS')

    def saveAtmFits(self):
        """
        Save atmosphere parameters to fits file
        """

        import fitsio

        outFile = '%s/%s_atm.fits' % (self.outputPath,self.outfileBaseWithCycle)
        self.fgcmLog.info('Saving atmosphere parameters to %s' % (outFile))
        fitsio.write(outFile,self.atmStruct,clobber=True,extname='ATMPARS')


class FgcmZeropointPlotter(object):
    """
    Class to make zeropoint plots

    parameters
    ----------
    zpStruct: recarray
       Zero point structure
    fgcmStars: fgcmStars object
    fgcmPars: fgcmParameters object
    colorSplitIndices: list
       2 element list with colors to split on
    plotPath: string
       Directory to make plots
    outfileBase: string
       Output file base string
    """

    def __init__(self, zpStruct, fgcmStars, fgcmPars,
                 I0StdBand, I1StdBand, I10StdBand,
                 colorSplitIndices, colorSplitBands, plotPath, outfileBase,
                 cycleNumber, fgcmLog, butlerQC=None, plotHandleDict=None, rng=None):
        self.zpStruct = zpStruct
        self.bands = fgcmPars.bands
        self.filterNames = fgcmPars.lutFilterNames
        self.plotPath = plotPath
        self.outfileBase = outfileBase
        self.cycleNumber = cycleNumber
        self.fgcmLog = fgcmLog
        self.filterToBand = fgcmPars.filterToBand
        self.colorSplitIndices = colorSplitIndices
        self.colorSplitBands = colorSplitBands
        self.I0StdBand = I0StdBand
        self.I1StdBand = I1StdBand
        self.I10StdBand = I10StdBand

        self.butlerQC = butlerQC
        self.plotHandleDict = plotHandleDict

        self.i1Conversions, self.blueString, self.redString = self.computeI1Conversions(fgcmStars)

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState()

    def computeI1Conversions(self, fgcmStars):
        """
        Compute I1 to mmag conversions for stars from 0.5 to 3.0 in g-i color.
        """

        i1Conversions = np.zeros(fgcmStars.nBands) + 1000.0

        objMagStdMean = snmm.getArray(fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(fgcmStars.objMagStdMeanErrHandle)
        objSEDSlope = snmm.getArray(fgcmStars.objSEDSlopeHandle)

        # Use the reserve stars, it's a reasonable sample
        goodStars = fgcmStars.getGoodStarIndices(onlyReserve=True, checkMinObs=True, checkHasColor=True)
        gmi = (objMagStdMean[goodStars, self.colorSplitIndices[0]] -
               objMagStdMean[goodStars, self.colorSplitIndices[1]])

        cuts = np.percentile(gmi, [5.0, 95.0])
        blueStars, = np.where(gmi < cuts[0])
        redStars, = np.where(gmi > cuts[1])

        blueColor = np.median(gmi[blueStars])
        redColor = np.median(gmi[redStars])

        blueString = "%s-%s ~ %.2f" % (self.colorSplitBands[0], self.colorSplitBands[1], np.median(blueColor))
        redString = "%s-%s ~ %.2f" % (self.colorSplitBands[0], self.colorSplitBands[1], np.median(redColor))

        deltaI1 = 1.0

        for i, band in enumerate(fgcmStars.bands):
            # We need to make sure we only select valid stars if at all
            # possible. This is particularly important for u band!
            # We also want to avoid stars that have "default" SED slopes.
            okBlue, = np.where(
                (objMagStdMean[goodStars[blueStars], i] < 90.0)
                & (objSEDSlope[goodStars[blueStars], i] != np.median(objSEDSlope[goodStars[blueStars], i]))
            )
            okRed, = np.where(
                (objMagStdMean[goodStars[redStars], i] < 90.0)
                & (objSEDSlope[goodStars[redStars], i] != np.median(objSEDSlope[goodStars[redStars], i]))
            )

            if okBlue.size < 3:
                # This will just pick up the overall median SED slope
                # which is better than nothing.
                sedSlopeBlue = np.median(objSEDSlope[goodStars[blueStars], i])
            else:
                sedSlopeBlue = np.median(objSEDSlope[goodStars[blueStars[okBlue]], i])
            if okRed.size < 3:
                sedSlopeRed = np.median(objSEDSlope[goodStars[redStars], i])
            else:
                sedSlopeRed = np.median(objSEDSlope[goodStars[redStars[okRed]], i])

            deltaMagBlue = 2.5 * np.log10((1.0 + sedSlopeBlue * ((self.I1StdBand[i] + deltaI1) / self.I0StdBand[i])) / (1.0 + sedSlopeBlue * self.I10StdBand[i]))
            deltaMagRed = 2.5 * np.log10((1.0 + sedSlopeRed * ((self.I1StdBand[i] + deltaI1) / self.I0StdBand[i])) / (1.0 + sedSlopeRed * self.I10StdBand[i]))

            i1Conversions[i] = 1000.0 * (deltaMagRed - deltaMagBlue) / deltaI1

            self.fgcmLog.info("i1 conversion for %s band is %.2f" % (band, i1Conversions[i]))

        return i1Conversions, blueString, redString

    def makeR1I1Plots(self):
        """
        Make R1 vs I1 plots.
        """
        acceptMask = (zpFlagDict['PHOTOMETRIC_FIT_EXPOSURE'] |
                      zpFlagDict['PHOTOMETRIC_NOTFIT_EXPOSURE'])
        for filterName in self.filterNames:
            use,=np.where((np.char.rstrip(self.zpStruct['FILTERNAME']) == filterName.encode('utf-8')) &
                          ((self.zpStruct['FGCM_FLAG'] & acceptMask) > 0) &
                          (np.abs(self.zpStruct['FGCM_R10']) < 1000.0) &
                          (np.abs(self.zpStruct['FGCM_R0']) < 1000.0))

            if (use.size == 0):
                continue

            i1Conversion = self.i1Conversions[self.bands.index(self.filterToBand[filterName])]

            i1 = self.zpStruct['FGCM_I10'][use] * self.zpStruct['FGCM_I0'][use] * i1Conversion
            r1 = self.zpStruct['FGCM_R10'][use] * self.zpStruct['FGCM_R0'][use] * i1Conversion

            # limit to a reasonable range
            #  note that r1 is much noisier than i1
            ok, = np.where((r1 > (i1.min() - 2.0 * i1Conversion)) &
                           (r1 < (i1.max() + 2.0 * i1Conversion)))

            i1=i1[ok]
            r1=r1[ok]

            fig = makeFigure(figsize=(8, 6))
            fig.clf()

            ax=fig.add_subplot(111)

            ax.hexbin(i1, r1, cmap=colormaps.get_cmap('gray_r'), rasterized=True)
            # and overplot a 1-1 line that best covers the range of the data
            xlim = ax.get_xlim()
            range0 = xlim[0]+0.001
            range1 = xlim[1]-0.001
            ax.plot([range0,range1],[range0,range1],'b--',linewidth=2)

            ax.set_xlabel(r'$I_1$ from FGCM Fit (red-blue mmag)',fontsize=16)
            ax.set_ylabel(r'$R_1$ from Retrieval (red-blue mmag)',fontsize=16)

            text = "(%s)\n" % (filterName)
            text += "Blue: %s\n" % (self.blueString)
            text += "Red: %s\n" % (self.redString)
            text += "(1-1 reference line)"
            ax.annotate(text,(0.1,0.93),xycoords='axes fraction',
                        ha='left',va='top',fontsize=16)

            if self.butlerQC is not None:
                putButlerFigure(self.fgcmLog,
                                self.butlerQC,
                                self.plotHandleDict,
                                "I1R1",
                                self.cycleNumber,
                                fig,
                                filterName=filterName)
            elif self.plotPath is not None:
                fig.savefig('%s/%s_i1r1_%s.png' % (self.plotPath,
                                                   self.outfileBase,
                                                   filterName))

    def makeR1I1Maps(self, deltaMapper, ccdField='CCDNUM'):
        """
        Make R1, I1, and R1-I1 maps.

        parameters
        ----------
        deltaMapper: ccd offset struct
        ccdField: string, default='CCDNUM'
        """
        from .fgcmUtilities import plotCCDMap
        from matplotlib import colormaps

        acceptMask = (zpFlagDict['PHOTOMETRIC_FIT_EXPOSURE'] |
                      zpFlagDict['PHOTOMETRIC_NOTFIT_EXPOSURE'])

        plotTypes = ['I1', 'R1', 'R1 - I1 (Matched scales)', 'R1 - I1']

        ccdMin = np.min(self.zpStruct[ccdField])
        ccdMax = np.max(self.zpStruct[ccdField])
        nCCD = (ccdMax - ccdMin) + 1

        for filterName in self.filterNames:
            use0,=np.where((np.char.rstrip(self.zpStruct['FILTERNAME']) == filterName.encode('utf-8')) &
                           ((self.zpStruct['FGCM_FLAG'] & acceptMask) > 0) &
                           (np.abs(self.zpStruct['FGCM_R10']) < 1000.0) &
                           (np.abs(self.zpStruct['FGCM_R0']) < 1000.0))

            if (use0.size == 0):
                continue

            ccdIndex = np.searchsorted(np.arange(ccdMin,ccdMax+1),
                                       self.zpStruct[ccdField][use0])

            i1Conversion = self.i1Conversions[self.bands.index(self.filterToBand[filterName])]

            i1 = self.zpStruct['FGCM_I10'][use0] * self.zpStruct['FGCM_I0'][use0] * i1Conversion
            r1 = self.zpStruct['FGCM_R10'][use0] * self.zpStruct['FGCM_R0'][use0] * i1Conversion

            meanI1 = np.zeros(nCCD)
            meanR1 = np.zeros(nCCD)
            nPerCCD = np.zeros(nCCD,dtype=np.int32)

            np.add.at(meanI1, ccdIndex, i1.astype(meanI1.dtype))
            np.add.at(meanR1, ccdIndex, r1.astype(meanR1.dtype))
            np.add.at(nPerCCD, ccdIndex, 1)

            use,=np.where(nPerCCD > 0)
            if use.size < 3:
                continue
            meanI1[use] /= nPerCCD[use]
            meanR1[use] /= nPerCCD[use]

            # use the same range scale for all the plots
            stMatchscale = np.argsort(meanR1[use])
            loMatchscale = meanR1[use[stMatchscale[int(0.02*use.size)]]]
            hiMatchscale = meanR1[use[stMatchscale[int(0.98*use.size)]]]

            r1mI1 = meanR1 - meanI1
            st = np.argsort(r1mI1[use])
            lo = r1mI1[use[st[int(0.02*use.size)]]]
            hi = r1mI1[use[st[int(0.98*use.size)]]]

            for plotType in plotTypes:
                fig = makeFigure(figsize=(8, 6))
                fig.clf()

                ax=fig.add_subplot(111)

                try:
                    if (plotType == 'R1'):
                        plotCCDMap(ax, deltaMapper[use], meanR1[use], 'R1 (red-blue mmag)', loHi=[loMatchscale, hiMatchscale])
                    elif (plotType == 'I1'):
                        plotCCDMap(ax, deltaMapper[use], meanI1[use], 'I1 (red-blue mmag)', loHi=[loMatchscale, hiMatchscale])
                    elif "Matched" in plotType:
                        # For the matched scale residuals, center at zero, but use lo/hi
                        amp = np.abs((hi - lo)/2.)
                        plotCCDMap(
                            ax,
                            deltaMapper[use],
                            r1mI1[use],
                            "R1 - I1 (red-blue mmag)",
                            loHi=[-amp, amp],
                            cmap=colormaps.get_cmap("bwr"),
                        )
                    else:
                        plotCCDMap(
                            ax,
                            deltaMapper[use],
                            r1mI1[use],
                            "R1 - I1 (red-blue mmag)",
                            loHi=[lo, hi],
                            cmap=colormaps.get_cmap("bwr"),
                        )
                except ValueError:
                    continue

                text = "(%s)\n" % (filterName)
                text += "%s\n" % (plotType)
                text += "Blue: %s\n" % (self.blueString)
                text += "Red: %s\n" % (self.redString)
                ax.annotate(text,
                            (0.1,0.93),xycoords='axes fraction',
                            ha='left',va='top',fontsize=18)

                if self.butlerQC is not None:
                    if "R1 - I1" in plotType:
                        if "Match" in plotType:
                            name = "R1mI1Matchscale"
                        else:
                            name = "R1mI1"
                    else:
                        name = plotType

                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    name,
                                    self.cycleNumber,
                                    fig,
                                    filterName=filterName)
                elif self.plotPath is not None:
                    fig.savefig('%s/%s_%s_%s.png' % (self.plotPath,
                                                     self.outfileBase,
                                                     plotType.replace(" ",""),
                                                     filterName))

        return None

    def makeR1I1TemporalResidualPlots(self):
        """
        Make R1 - I1 vs time plots.
        """
        acceptMask = (zpFlagDict['PHOTOMETRIC_FIT_EXPOSURE'] |
                      zpFlagDict['PHOTOMETRIC_NOTFIT_EXPOSURE'])

        for filterName in self.filterNames:
            use, = np.where((np.char.rstrip(self.zpStruct['FILTERNAME']) == filterName.encode('utf-8')) &
                            ((self.zpStruct['FGCM_FLAG'] & acceptMask) > 0) &
                            (np.abs(self.zpStruct['FGCM_R10']) < 1000.0) &
                            (np.abs(self.zpStruct['FGCM_R0']) < 1000.0) &
                            (np.abs(self.zpStruct['FGCM_I10']) < 1000.0) &
                            (np.abs(self.zpStruct['FGCM_I0']) < 1000.0))

            if use.size == 0:
                continue

            mjd0 = np.floor(np.min(self.zpStruct['MJD'][use]))

            i1Conversion = self.i1Conversions[self.bands.index(self.filterToBand[filterName])]

            xValues = self.zpStruct['MJD'][use] - mjd0
            yValues = (self.zpStruct['FGCM_R10'][use] * self.zpStruct['FGCM_R0'][use] -
                       self.zpStruct['FGCM_I10'][use] * self.zpStruct['FGCM_I0'][use]) * i1Conversion

            st = np.argsort(yValues)
            u, = np.where((yValues > yValues[st[int(0.01 * st.size)]]) &
                          (yValues < yValues[st[int(0.99 * st.size)]]))
            xValues = xValues[u]
            yValues = yValues[u]
            try:
                xRange = [np.min(xValues), np.max(xValues)]
                yRange = [np.min(yValues), np.max(yValues)]
            except ValueError:
                continue

            # Arbitrarily do 50 days...
            binStruct = dataBinner(xValues, yValues, 50.0, xRange, rng=self.rng)
            gd, = np.where(binStruct['Y_ERR'] > 0.0)

            if gd.size < 2:
                continue

            binStruct = binStruct[gd]

            fig = makeFigure(figsize=(8, 6))
            fig.clf()

            ax = fig.add_subplot(111)

            ax.hexbin(xValues, yValues, bins='log', extent=[xRange[0], xRange[1],
                                                            yRange[0], yRange[1]],
                      cmap=colormaps.get_cmap('viridis'))
            ax.set_xlabel('MJD - %.1f' % (mjd0), fontsize=16)
            ax.set_ylabel('R1 - I1 (red-blue mmag)', fontsize=16)
            ax.plot(xRange, [0.0, 0.0], 'r:')

            ax.errorbar(binStruct['X_BIN'], binStruct['Y'],
                        yerr=binStruct['Y_ERR'], fmt='r.', markersize=10)

            text = "(%s)\n" % (filterName)
            text += "Blue: %s\n" % (self.blueString)
            text += "Red: %s\n" % (self.redString)
            ax.annotate(text,
                        (0.1, 0.93), xycoords='axes fraction',
                        ha='left', va='top', fontsize=18, color='r')

            if self.butlerQC is not None:
                putButlerFigure(self.fgcmLog,
                                self.butlerQC,
                                self.plotHandleDict,
                                "R1mI1_vs_mjd",
                                self.cycleNumber,
                                fig,
                                filterName=filterName)
            elif self.plotPath is not None:
                fig.savefig('%s/%s_r1-i1_vs_mjd_%s.png' % (self.plotPath,
                                                           self.outfileBase,
                                                           filterName))
