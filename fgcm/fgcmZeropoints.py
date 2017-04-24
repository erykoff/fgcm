from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil

from fgcmUtilities import zpFlagDict
from fgcmUtilities import expFlagDict

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmZeropoints(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmLUT,fgcmGray,fgcmRetrieval):

        self.fgcmPars = fgcmPars
        self.fgcmLUT = fgcmLUT
        self.fgcmGray = fgcmGray
        self.fgcmRetrieval = fgcmRetrieval

        self.illegalValue = fgcmConfig.illegalValue
        self.outputPath = fgcmConfig.outputPath
        self.cycleNumber = fgcmConfig.cycleNumber
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.zptAB = fgcmConfig.zptAB
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.ccdOffsets = fgcmConfig.ccdOffsets
        self.bandRequired = fgcmConfig.bandRequired
        self.bandRequiredIndex = np.where(self.bandRequired)[0]
        self.bandExtra = fgcmConfig.bandExtra
        self.bandExtraIndex = np.where(self.bandExtra)[0]
        self.minCCDPerExp = fgcmConfig.minCCDPerExp
        self.minStarPerCCD = fgcmConfig.minStarPerCCD
        self.maxCCDGrayErr = fgcmConfig.maxCCDGrayErr
        self.sigma0Cal = fgcmConfig.sigma0Cal
        self.expField = fgcmConfig.expField
        self.ccdField = fgcmConfig.ccdField
        self.expGrayRecoverCut = fgcmConfig.expGrayRecoverCut
        self.expGrayErrRecoverCut = fgcmConfig.expGrayErrRecoverCut
        self.expVarGrayPhotometricCut = fgcmConfig.expVarGrayPhotometricCut


    def computeZeropoints(self):
        """
        """

        # first, we need to get relevant quantities from shared memory.
        expGray = snmm.getArray(self.fgcmGray.expGrayHandle)
        expGrayErr = snmm.getArray(self.fgcmGray.expGrayErrHandle)
        expGrayRMS = snmm.getArray(self.fgcmGray.expGrayRMSHandle)
        expGrayRMS = snmm.getArray(self.fgcmGray.expGrayRMSHandle)
        expNGoodCCDs = snmm.getArray(self.fgcmGray.expNGoodCCDsHandle)
        expNGoodTilings = snmm.getArray(self.fgcmGray.expNGoodTilingsHandle)

        ccdGray = snmm.getArray(self.fgcmGray.ccdGrayHandle)
        ccdGrayRMS = snmm.getArray(self.fgcmGray.ccdGrayRMSHandle)
        ccdGrayErr = snmm.getArray(self.fgcmGray.ccdGrayErrHandle)
        ccdNGoodStars = snmm.getArray(self.fgcmGray.ccdNGoodStarsHandle)
        ccdNGoodTilings = snmm.getArray(self.fgcmGray.ccdNGoodTilingsHandle)

        r0 = snmm.getArray(self.fgcmRetrieval.r0Handle)
        r10 = snmm.getArray(self.fgcmRetrieval.r10Handle)


        # and we need to make sure we have the parameters, and
        #  set these to the exposures
        parArray = self.fgcmPars.getParArray(fitterUnits=False)
        self.fgcmPars.parsToExposures()

        # set up output structures

        zpStruct = np.zeros(self.fgcmPars.nExp*self.fgcmPars.nCCD,
                            dtype=[(self.expField,'i4'), # done
                                   (self.ccdField,'i2'), # done
                                   ('FGCM_FLAG','i2'), # done
                                   ('FGCM_ZPT','f8'), # done
                                   ('FGCM_ZPTERR','f8'), # done
                                   ('FGCM_I0','f8'), # done
                                   ('FGCM_I10','f8'), # done
                                   ('FGCM_R0','f8'),
                                   ('FGCM_R10','f8'),
                                   ('FGCM_GRY','f8'), # done
                                   ('FGCM_ZPTVAR','f8'), # done
                                   ('FGCM_TILINGS','f8'), # done
                                   ('FGCM_FPGRY','f8'), # done
                                   ('FGCM_FPVAR','f8'), # done
                                   ('FGCM_DUST','f8'), # done
                                   ('FGCM_FLAT','f8'), # done
                                   ('FGCM_APERCORR','f8'), # done
                                   ('EXPTIME','f4'), # done
                                   ('BAND','a2')]) # done

        atmStruct = np.zeros(self.fgcmPars.nExp,
                             dtype=[(self.expField,'i4'),
                                    ('PMB','f8'),
                                    ('PWV','f8'),
                                    ('TAU','f8'),
                                    ('ALPHA','f8'),
                                    ('O3','f8'),
                                    ('SECZENITH','f8')])

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
        zpStruct['BAND'][:] = self.fgcmPars.bands[self.fgcmPars.expBandIndex[zpExpIndex]]
        zpStruct['EXPTIME'][:] = self.fgcmPars.expExptime[zpExpIndex]

        # fill in the superstar flat
        zpStruct['FGCM_FLAT'][:] = self.fgcmPars.expCCDSuperStar[zpExpIndex,
                                                                 zpCCDIndex]

        # fill in the optics dust
        zpStruct['FGCM_DUST'][:] = self.fgcmPars.expQESys[zpExpIndex]

        # fill in the aperture correction
        zpStruct['FGCM_APERCORR'][:] = self.fgcmPars.expApertureCorrection[zpExpIndex]

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

        # look up the I0 and I10s.  These are defined for everything
        #  (even if only standard bandpass, it'll grab instrumental)

        ## FIXME: will probably need some sort of rotation information at some point

        ## FIXME: check that the signs are correct!
        # need secZenith for each exp/ccd pair
        ccdHA = (self.fgcmPars.expTelHA[zpExpIndex] -
                 np.radians(self.ccdOffsets['DELTA_RA'][zpCCDIndex]))
        ccdDec = (self.fgcmPars.expTelDec[zpExpIndex] +
                  np.radians(self.ccdOffsets['DELTA_DEC'][zpCCDIndex]))
        ccdSecZenith = 1./(np.sin(ccdDec) * self.fgcmPars.sinLatitude +
                           np.cos(ccdDec) * self.fgcmPars.cosLatitude * np.cos(ccdHA))

        # and do the LUT lookups
        lutIndices = self.fgcmLUT.getIndices(self.fgcmPars.expBandIndex[zpExpIndex],
                                             self.fgcmPars.expPWV[zpExpIndex],
                                             self.fgcmPars.expO3[zpExpIndex],
                                             np.log(self.fgcmPars.expTau[zpExpIndex]),
                                             self.fgcmPars.expAlpha[zpExpIndex],
                                             ccdSecZenith,
                                             zpCCDIndex,
                                             self.fgcmPars.expPmb[zpExpIndex])
        zpStruct['FGCM_I0'][:] = self.fgcmLUT.computeI0(self.fgcmPars.expBandIndex[zpExpIndex],
                                                        self.fgcmPars.expPWV[zpExpIndex],
                                                        self.fgcmPars.expO3[zpExpIndex],
                                                        np.log(self.fgcmPars.expTau[zpExpIndex]),
                                                        self.fgcmPars.expAlpha[zpExpIndex],
                                                        ccdSecZenith,
                                                        zpCCDIndex,
                                                        self.fgcmPars.expPmb[zpExpIndex],
                                                        lutIndices)
        zpStruct['FGCM_I10'][:] = self.fgcmLUT.computeI1(lutIndices) / zpStruct['FGCM_I0'][:]

        # Set the tilings, gray values, and zptvar

        zpStruct['FGCM_TILINGS'][:] = self.illegalValue
        zpStruct['FGCM_GRY'][:] = self.illegalValue
        zpStruct['FGCM_ZPTVAR'][:] = self.illegalValue

        goodCCD, = np.where((ccdNGoodStars[zpExpIndex,zpCCDIndex] >=
                             self.minStarPerCCD) &
                            (ccdGrayErr[zpExpIndex,zpCCDIndex] <=
                             self.maxCCDGrayErr))
        zpStruct['FGCM_TILINGS'][goodCCD] = ccdNGoodTilings[zpExpIndex[goodCCD],
                                                            zpCCDIndex[goodCCD]]
        zpStruct['FGCM_GRY'][goodCCD] = ccdGray[zpExpIndex[goodCCD],
                                                zpCCDIndex[goodCCD]]
        zpStruct['FGCM_ZPTVAR'][goodCCD] = ccdGrayErr[zpExpIndex[goodCCD],
                                                      zpCCDIndex[goodCCD]]**2.

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
                           np.sqrt(self.expVarGrayPhotometricCut)) &
                          (expGray[zpExpIndex] >=
                           self.expGrayRecoverCut))

        zpStruct['FGCM_TILINGS'][badCCDGoodExp] = expNGoodTilings[zpExpIndex[badCCDGoodExp]]
        zpStruct['FGCM_GRY'][badCCDGoodExp] = expGray[zpExpIndex[badCCDGoodExp]]
        zpStruct['FGCM_ZPTVAR'][badCCDGoodExp] = expGrayRMS[zpExpIndex[badCCDGoodExp]]**2.

        # flag the photometric (fit) exposures
        photZpIndex, = np.where(self.fgcmPars.expFlag[zpExpIndex] == 0)

        photFitBand, = np.where(~self.fgcmPars.expExtraBandFlag[zpExpIndex[photZpIndex]])
        zpStruct['FGCM_FLAG'][photZpIndex[photFitBand]] |= (
            zpFlagDict['PHOTOMETRIC_FIT_EXPOSURE'])

        photExtraBand, = np.where(self.fgcmPars.expExtraBandFlag[zpExpIndex[photZpIndex]])
        zpStruct['FGCM_FLAG'][photZpIndex[photExtraBand]] |= (
            zpFlagDict['PHOTOMETRIC_EXTRA_EXPOSURE'])

        # flag the non-photometric exposures on calibratable nights
        rejectMask = (expFlagDict['TOO_FEW_EXP_ON_NIGHT'] |
                      expFlagDict['BAND_NOT_IN_LUT'] |
                      expFlagDict['NO_STARS'])
        acceptMask = (expFlagDict['EXP_GRAY_TOO_LARGE'] |
                      expFlagDict['VAR_GRAY_TOO_LARGE'] |
                      expFlagDict['TOO_FEW_STARS'])

        nonPhotZpIndex, = np.where(((self.fgcmPars.expFlag[zpExpIndex] & acceptMask) > 0) &
                           ((self.fgcmPars.expFlag[zpExpIndex] & rejectMask) == 0))
        zpStruct['FGCM_FLAG'][nonPhotZpIndex] |= zpFlagDict['NONPHOTOMETRIC_FIT_NIGHT']

        # and the exposures on non-calibratable nights (photometric or not, we don't know)
        rejectMask = (expFlagDict['NO_STARS'] |
                      expFlagDict['BAND_NOT_IN_LUT'])
        acceptMask = (expFlagDict['TOO_FEW_EXP_ON_NIGHT'])
        badNightZpIndex, = np.where(((self.fgcmPars.expFlag[zpExpIndex] & acceptMask) > 0) &
                                    ((self.fgcmPars.expFlag[zpExpIndex] & rejectMask) == 0))
        zpStruct['FGCM_FLAG'][badNightZpIndex] |= zpFlagDict['NOFIT_NIGHT']

        # and finally, the hopeless exposures
        acceptMask = (expFlagDict['NO_STARS'] |
                      expFlagDict['BAND_NOT_IN_LUT'])
        hopelessZpIndex, = np.where(((self.fgcmPars.expFlag[zpExpIndex] & acceptMask) > 0))
        zpStruct['FGCM_FLAG'][hopelessZpIndex] |= zpFlagDict['CANNOT_COMPUTE_ZEROPOINT']

        # now we can fill the zeropoints

        zpStruct['FGCM_ZPT'][:] = self.illegalValue
        zpStruct['FGCM_ZPTERR'][:] = self.illegalValue

        # start with the passable flag 1,2,4 exposures

        acceptMask = (zpFlagDict['PHOTOMETRIC_FIT_EXPOSURE'] |
              zpFlagDict['PHOTOMETRIC_EXTRA_EXPOSURE'] |
              zpFlagDict['NONPHOTOMETRIC_FIT_NIGHT'])

        okZpIndex, = np.where((zpStruct['FGCM_FLAG'] & acceptMask) > 0)

        okCCDZpIndexFlag = ((zpStruct['FGCM_I0'][okZpIndex] > 0.0) &
                    (zpStruct['FGCM_FLAT'][okZpIndex] > self.illegalValue) &
                    (zpStruct['FGCM_DUST'][okZpIndex] > self.illegalValue) &
                    (zpStruct['FGCM_APERCORR'][okZpIndex] > self.illegalValue) &
                    (zpStruct['FGCM_GRY'][okZpIndex] > self.illegalValue))

        okCCDZpIndex = okZpIndex[okCCDZpIndexFlag]

        self._computeZpt(zpStruct,okCCDZpIndex)
        self._computeZptErr(zpStruct,zpExpIndex,okCCDZpIndex)

        badCCDZpExp = okZpIndex[~okCCDZpIndexFlag]
        zpStruct['FGCM_FLAG'][badCCDZpExp] |=  zpFlagDict['TOO_FEW_STARS_ON_CCD']

        # and the flag 8 extra exposures

        acceptMask = zpFlagDict['NOFIT_NIGHT']

        mehZpIndex, = np.where((zpStruct['FGCM_FLAG'] & acceptMask) > 0)

        mehCCDZpIndexFlag = ((zpStruct['FGCM_I0'][mehZpIndex] > 0.0) &
                             (zpStruct['FGCM_FLAT'][mehZpIndex] > self.illegalValue) &
                             (zpStruct['FGCM_DUST'][mehZpIndex] > self.illegalValue) &
                             (zpStruct['FGCM_APERCORR'][mehZpIndex] > self.illegalValue) &
                             (zpStruct['FGCM_GRY'][mehZpIndex] > self.illegalValue) &
                             (ccdNGoodStars[zpExpIndex[mehZpIndex],zpCCDIndex[mehZpIndex]] >=
                              self.minStarPerCCD) &
                             (ccdGrayErr[zpExpIndex[mehZpIndex],zpCCDIndex[mehZpIndex]] <=
                              self.maxCCDGrayErr))

        mehCCDZpIndex = mehZpIndex[mehCCDZpIndexFlag]

        self._computeZpt(zpStruct,mehCCDZpIndex)
        self._computeZptErr(zpStruct,zpExpIndex,mehCCDZpIndex)

        badCCDZpExp = mehZpIndex[~mehCCDZpIndexFlag]
        zpStruct['FGCM_FLAG'][badCCDZpExp] |= zpFlagDict['TOO_FEW_STARS_ON_CCD']
        zpStruct['FGCM_FLAG'][badCCDZpExp] |= zpFlagDict['CANNOT_COMPUTE_ZEROPOINT']


        # and save...
        outFile = '%s/%s_zpt.fits' % (self.outputPath,self.outfileBaseWithCycle)
        fitsio.write(outFile,zpStruct,clobber=True,extname='ZPTS')

        #################################
        # and make the parameter file
        atmStruct[self.expField] = self.fgcmPars.expArray
        atmStruct['PMB'] = self.fgcmPars.expPmb
        atmStruct['PWV'] = self.fgcmPars.expPWV
        atmStruct['TAU'] = self.fgcmPars.expTau
        atmStruct['ALPHA'] = self.fgcmPars.expAlpha
        atmStruct['O3'] = self.fgcmPars.expO3
        atmStruct['SECZENITH'] = 1./(np.sin(self.fgcmPars.expTelDec) *
                                     self.fgcmPars.sinLatitude +
                                     np.cos(self.fgcmPars.expTelDec) *
                                     self.fgcmPars.cosLatitude *
                                     np.cos(self.fgcmPars.expTelHA))

        outFile = '%s/%s_atm.fits' % (self.outputPath,self.outfileBaseWithCycle)
        fitsio.write(outFile,atmStruct,clobber=True,extname='ATMPARS')


        ############
        ## plots
        ############


        ## compare I0 and R0, I1 and R1
        for i in xrange(self.fgcmPars.nBands):
            use,=np.where((zpStruct['BAND'] == self.fgcmPars.bands[i]) &
                          (zpStruct['FGCM_FLAG'] == 1) &
                          (np.abs(zpStruct['FGCM_R10']) < 1000.0))

            i1 = zpStruct['FGCM_I10'][use]*zpStruct['FGCM_I0'][use]
            r1 = zpStruct['FGCM_R10'][use]*zpStruct['FGCM_R10'][use]

            fig = plt.figure(1,figsize=(8,6))
            fig.clf()

            ax=fig.add_subplot(111)

            ax.hexbin(i1,r1,cmap=plt.get_cmap('gray_r'),rasterized=True)
            ax.plot(ax.get_xlim(),ax.get_ylim(),'b--',linewidth=2)

            ax.set_xlabel(r'$I_1$ from FGCM Fit',fontsize=16)
            ax.set_ylabel(r'$R_1$ from Retrieval',fontsize=16)

            text=r'$(%s)$' % (bands[i])
            ax.annotate(text,(0.1,0.93),xycoords='axes fraction',
                        ha='left',va='top',fontsize=16)

            fig.savefig('%s/%s_i1r1_%s.png' % (self.fgcmConfig.plotPath,
                                               self.outfileBaseWithCycle,
                                               self.fgcmPars.bands[i]))

        ## FIXME: plots?  what plots would be interesting?
        ##  zeropoint as a function of MJD
        ##  colors per band
        ##  focal plane average
        ##  plot symbols for flag

        # need to know the mean zeropoint per exposure
        expZpMean = np.zeros(self.fgcmPars.nExp,dtype='f4')
        expZpNCCD = np.zeros(self.fgcmPars.nExp,dtype='i4')

        rejectMask = (zpFlagDict['CANNOT_COMPUTE_ZEROPOINT'] |
                      zpFlagDict['TOO_FEW_STARS_ON_CCD'])

        okCCD,=np.where((zpStruct['FGCM_FLAG'] & rejectMask) == 0)

        np.add.at(expZpMean,
                  zpExpIndex[okCCD],
                  zpStruct['FGCM_ZPT'][okCCD])
        np.add.at(expZpNCCD,
                  zpExpIndex[okCCD],
                  1)

        gd,=np.where(expZpNCCD > 0)
        expZpMean[gd] /= expZpNCCD[gd]

        fig=plt.figure(1,figsize(8,6))
        fig.clf()

        ax=fig.add_subplot(111)

        firstMJD = np.floor(np.min(self.fgcmPars.mjdNight))

        # FIXME: make configurable
        cols = ['g','r','b','m','y']
        syms = ['.','+','o','*','x']

        for i in xrange(self.fgcmPars.nBands):
            use,=np.where((self.fgcmPars.expBandIndex == i) &
                          (expZpMean > 0.0))

            plt.plot(self.fgcmPars.expMJD[use] - firstMJD,
                     expZpMean[use],cols[i]+syms[j],
                     label=r'$(%s)$' % (self.fgcmPars.bands[i]))

        ax.legend(3)

        fig.savefig('%s/%s_zeropoints.png' % (self.fgcmConfig.plotPath,
                                              self.outfileBaseWithCycle))

    def _computeZpt(self,zpStruct,zpIndex):
        """
        """

        zpStruct['FGCM_ZPT'][zpIndex] = (2.5*np.log10(zpStruct['FGCM_I0'][zpIndex]) +
                                         zpStruct['FGCM_FLAT'][zpIndex] +
                                         zpStruct['FGCM_DUST'][zpIndex] +
                                         zpStruct['FGCM_APERCORR'][zpIndex] +
                                         2.5*np.log10(zpStruct['EXPTIME'][zpIndex]) +
                                         self.zptAB +
                                         zpStruct['FGCM_GRY'][zpIndex])

    def _computeZptErr(self,zpStruct,zpExpIndex,zpIndex):
        """
        """

        sigFgcm = self.fgcmGray.sigFgcm[self.fgcmPars.expBandIndex[zpExpIndex[zpIndex]]]
        nTilingsM1 = np.clip(zpStruct['FGCM_TILINGS'][zpIndex]-1.0,1.0,1e10)

        zpStruct['FGCM_ZPTERR'][zpIndex] = np.sqrt((sigFgcm**2./nTilingsM1) +
                                                   zpStruct['FGCM_ZPTVAR'][zpIndex] +
                                                   self.sigma0Cal**2.)

