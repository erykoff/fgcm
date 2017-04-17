from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmZeropoints(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmLUT,fgcmGray,fgcmRetrieval):

        ## FIXME: I think we need fgcmLUT!        zpStruc

        self.fgcmPars = fgcmPars
        self.fgcmLUT = fgcmLUT
        self.fgcmGray = fgcmGray

        self.illegalValue = fgcmConfig.illegalValue
        self.outputPath = fgcmConfig.outputPath
        self.cycleNumber = fgcmConfig.cycleNumber
        self.outfileBase = fgcmConfig.outfileBase
        self.zptAB = fgcmConfig.zptAB
        self.ccdStartIndex = fgcmConfig.ccdStartIndex

    def computeZeropoints(self):
        """
        """

        # first, we need to get relevant quantities from shared memory.
        expGray = snmm.getArray(self.fgcmGray.expGrayHandle)
        expGrayRMS = snmm.getArray(self.fgcmGray.expGrayRMSHandle)

        # and we need to make sure we have the parameters, and
        #  set these to the exposures
        parArray = self.fgcmPars.getParArray(fitterUnits=False)
        self.fgcmPars.parsToExposures()

        # set up output structures
        zpStruct = np.zeros(self.fgcmPars.nExp*self.fgcmPars.nCCD,
                            dtype=[('EXPNUM','i4'), # done
                                   ('CCDNUM','i2'), # done
                                   ('FGCM_FLAG','i2'),
                                   ('FGCM_ZPT','f8'),
                                   ('FGCM_ZPTERR','f8'),
                                   ('FGCM_I0','f8'),
                                   ('FGCM_I10','f8'),
                                   ('FGCM_R0','f8'),
                                   ('FGCM_R10','f8'),
                                   ('FGCM_GRY','f8'),
                                   ('FGCM_ZPTVAR','f8'),
                                   ('FGCM_TILINGS','f8'),
                                   ('FGCM_FPGRY','f8'), # done
                                   ('FGCM_FPVAR','f8'), # done
                                   ('FGCM_DUST','f8'), # done
                                   ('FGCM_FLAT','f8'), # done
                                   ('FGCM_APERCORR','f8'), # done
                                   ('EXPTIME','f4'), # done
                                   ('BAND','a2')]) # done

        atmStruct = np.zeros(self.fgcmPars.nExp,
                             dtype=[('EXPNUM','i4'),
                                    ('PMB','f8'),
                                    ('PWV','f8'),
                                    ('TAU','f8'),
                                    ('ALPHA','f8'),
                                    ('O3','f8'),
                                    ('ZD','f8')])

        ## start with zpStruct

        # fill out exposures and ccds
        zpStruct['EXPNUM'][:] = np.repeat(self.fgcmPars.expArray,
                                          self.fgcmPars.nCCD)
        ## FIXME: make a config for the starting CCD number
        zpStruct['CCDNUM'][:] = np.tile(np.arange(self.fgcmPars.nCCD)+1,
                                        self.fgcmPars.nExp)

        # get the exposure indices and CCD indices
        zpExpIndex = np.searchsorted(self.fgcmPars.expArray,zpStruct['EXPNUM'])
        zpCCDIndex = zpStruct['CCDNUM'] - self.ccdStartIndex

        # fill exposure quantities
        zpStruct['BAND'][:] = self.fgcmPars.bands[self.fgcmPars.expBandIndex[zpExpIndex]]
        zpStruct['EXPTIME'][:] = self.fgcmPars.expExptime[zpExpIndex]

        # fill in the superstar flat
        zpStruct['FGCM_FLAT'][:] = self.fgcmPars.expCCDSuperStar[zpExpIndex,
                                                                 zpCCDIndex]

        # fill in the optics dust
        zpStruct['FGCM_DUST'][:] = self.fgcmPars.expQESys[zpExpIndex]

        # fill in the aperture correction
        zpStruct['FGCM_APERCORR'][:] = self.fgcmPars.expApertureCorrection[zpExpIndex]

        # and the focal-plane gray and var...
        zpStruct['FGCM_FPGRY'][:] = expGray[zpExpIndex]
        zpStruct['FGCM_FPVAR'][:] = expGrayRMS[zpExpIndex]**2.
        bad,=np.where(expGrayRMS[zpExpIndex] < 0.0)
        zpStruct['FGCM_FPVAR'][bad] = self.illegalValue

        # look up the I0 and I10s.  These are defined for everything
        #  (even if only standard bandpass, it'll grab instrumental)

        ## FIXME: WTH do I do with secZenith?
        # in fgcmChisq, this is computed for each individual object
        # so what we need is the mean secZenith of all the objects on a CCD...
        # or look at the center of the CCD.
        # look at other code ...
        lutIndices = self.fgcmLUT.getIndices(self.fgcmPars.expBandIndex[zpExpIndex],
                                             self.fgcmPars.expPWV[zpExpIndex],
                                             self.fgcmPars.expO3[zpExpIndex],
                                             np.log(self.fgcmPars.expTau[zpExpIndex]),
                                             self.fgcmPars.expAlpha[zpExpIndex],
                                             secZenith,
                                             zpCCDIndex,
                                             self.fgcmPars.expPmb[zpExpIndex])
        zpStruct['FGCM_I0'][:] = self.fgcmLUT.computeI0(self.fgcmPars.expBandIndex[zpExpIndex],
                                                        self.fgcmPars.expPWV[zpExpIndex],
                                                        self.fgcmPars.expO3[zpExpIndex],
                                                        np.log(self.fgcmPars.expTau[zpExpIndex]),
                                                        self.fgcmPars.expAlpha[thisObsExpIndex],
                                                        secZenith,
                                                        zpCCDIndex,
                                                        self.fgcmPars.expPmb[zpExpIndex],
                                                        lutIndices)
        zpStruct['FGCM_I10'][:] = self.fgcmLUT.computeI1(lutIndices) / zpStruct['FGCM_I0'][:]

        # grade the exposures/ccds and compute accordingly
        # flag 1 or 2 (PHOTOMETRIC_FIT_EXPOSURE, PHOTOMETRIC_EXTRA_EXPOSURE)

        goodExpIndex, = np.where(self.fgcmPars.expFlag == 0)
        goodZpExpIndex = np.searchsorted(self.fgcmPars.expArray[goodExpIndex],
                                         zpStruct['EXPNUM'])
        

        # question: do we have enough information for the flag 8s?
        # do we use the I0?  It was computed, but see if we want to ignore it.





