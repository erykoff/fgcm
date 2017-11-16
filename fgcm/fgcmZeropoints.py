from __future__ import print_function

import numpy as np
import os
import sys
import esutil

import matplotlib.pyplot as plt

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

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.log('INFO','Initializing FgcmZeropoints...')

        self.illegalValue = fgcmConfig.illegalValue
        self.outputPath = fgcmConfig.outputPath
        self.cycleNumber = fgcmConfig.cycleNumber
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.plotPath = fgcmConfig.plotPath
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
        # Use *input* values for the retrieved atmosphere parameters (if needed)
        self.fgcmPars.parsToExposures(retrievedInput=True)

        # set up output structures

        self.fgcmLog.log('INFO','Building zeropoint structure...')

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

        self.fgcmLog.log('INFO','%d exposure/ccd sets have exposures with >=%d good ccds' %
                         (zpExpOk.size, self.minCCDPerExp))

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
                                             #np.log(self.fgcmPars.expTau[zpExpIndex]),
                                             self.fgcmPars.expLnTau[zpExpIndex],
                                             self.fgcmPars.expAlpha[zpExpIndex],
                                             ccdSecZenith,
                                             zpCCDIndex,
                                             self.fgcmPars.expPmb[zpExpIndex])
        zpStruct['FGCM_I0'][:] = self.fgcmLUT.computeI0(self.fgcmPars.expPWV[zpExpIndex],
                                                        self.fgcmPars.expO3[zpExpIndex],
                                                        #np.log(self.fgcmPars.expTau[zpExpIndex]),
                                                        self.fgcmPars.expLnTau[zpExpIndex],
                                                        self.fgcmPars.expAlpha[zpExpIndex],
                                                        ccdSecZenith,
                                                        self.fgcmPars.expPmb[zpExpIndex],
                                                        lutIndices)
        zpStruct['FGCM_I10'][:] = self.fgcmLUT.computeI1(self.fgcmPars.expPWV[zpExpIndex],
                                                         self.fgcmPars.expO3[zpExpIndex],
                                                         #np.log(self.fgcmPars.expTau[zpExpIndex]),
                                                         self.fgcmPars.expLnTau[zpExpIndex],
                                                         self.fgcmPars.expAlpha[zpExpIndex],
                                                         ccdSecZenith,
                                                         self.fgcmPars.expPmb[zpExpIndex],
                                                         lutIndices) / zpStruct['FGCM_I0'][:]

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

        self.fgcmLog.log('INFO','%d CCDs are Good (>=%d stars; err <= %.3f)' %
                         (goodCCD.size, self.minStarPerCCD, self.maxCCDGrayErr))

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

        self.fgcmLog.log('INFO','%d CCDs recovered from good exposures (>=%d good CCDs, etc.)' %
                         (badCCDGoodExp.size, self.minCCDPerExp))

        # flag the photometric (fit) exposures
        photZpIndex, = np.where(self.fgcmPars.expFlag[zpExpIndex] == 0)

        photFitBand, = np.where(~self.fgcmPars.expExtraBandFlag[zpExpIndex[photZpIndex]])
        zpStruct['FGCM_FLAG'][photZpIndex[photFitBand]] |= (
            zpFlagDict['PHOTOMETRIC_FIT_EXPOSURE'])

        self.fgcmLog.log('INFO','%d CCDs marked as photometric, used in fit' %
                         (photFitBand.size))

        photExtraBand, = np.where(self.fgcmPars.expExtraBandFlag[zpExpIndex[photZpIndex]])
        zpStruct['FGCM_FLAG'][photZpIndex[photExtraBand]] |= (
            zpFlagDict['PHOTOMETRIC_EXTRA_EXPOSURE'])

        self.fgcmLog.log('INFO','%d CCDs marked as photometric, not used in fit' %
                         (photExtraBand.size))

        # flag the non-photometric exposures on calibratable nights
        rejectMask = (expFlagDict['TOO_FEW_EXP_ON_NIGHT'] |
                      expFlagDict['BAND_NOT_IN_LUT'] |
                      expFlagDict['NO_STARS'])
        acceptMask = (expFlagDict['EXP_GRAY_TOO_NEGATIVE'] |
                      expFlagDict['EXP_GRAY_TOO_POSITIVE'] |
                      expFlagDict['VAR_GRAY_TOO_LARGE'] |
                      expFlagDict['TOO_FEW_STARS'])

        nonPhotZpIndex, = np.where(((self.fgcmPars.expFlag[zpExpIndex] & acceptMask) > 0) &
                           ((self.fgcmPars.expFlag[zpExpIndex] & rejectMask) == 0))
        zpStruct['FGCM_FLAG'][nonPhotZpIndex] |= zpFlagDict['NONPHOTOMETRIC_FIT_NIGHT']

        self.fgcmLog.log('INFO','%d CCDs marked non-photometric, on a night with a fit' %
                         (nonPhotZpIndex.size))

        # and the exposures on non-calibratable nights (photometric or not, we don't know)
        rejectMask = (expFlagDict['NO_STARS'] |
                      expFlagDict['BAND_NOT_IN_LUT'])
        acceptMask = (expFlagDict['TOO_FEW_EXP_ON_NIGHT'])
        badNightZpIndex, = np.where(((self.fgcmPars.expFlag[zpExpIndex] & acceptMask) > 0) &
                                    ((self.fgcmPars.expFlag[zpExpIndex] & rejectMask) == 0))
        zpStruct['FGCM_FLAG'][badNightZpIndex] |= zpFlagDict['NOFIT_NIGHT']

        self.fgcmLog.log('INFO','%d CCDs on nights without a fit (assume standard atmosphere)' %
                         (badNightZpIndex.size))

        # and finally, the hopeless exposures
        acceptMask = (expFlagDict['NO_STARS'] |
                      expFlagDict['BAND_NOT_IN_LUT'])
        hopelessZpIndex, = np.where(((self.fgcmPars.expFlag[zpExpIndex] & acceptMask) > 0))
        zpStruct['FGCM_FLAG'][hopelessZpIndex] |= zpFlagDict['CANNOT_COMPUTE_ZEROPOINT']

        self.fgcmLog.log('INFO','%d CCDs marked as hopeless (cannot compute zeropoint)' %
                         (hopelessZpIndex.size))

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

        # record as a class element
        self.zpStruct = zpStruct

        #################################
        # and make the parameter file
        atmStruct[self.expField] = self.fgcmPars.expArray
        atmStruct['PMB'] = self.fgcmPars.expPmb
        atmStruct['PWV'] = self.fgcmPars.expPWV
        #atmStruct['TAU'] = self.fgcmPars.expTau
        atmStruct['TAU'] = np.exp(self.fgcmPars.expLnTau)
        atmStruct['ALPHA'] = self.fgcmPars.expAlpha
        atmStruct['O3'] = self.fgcmPars.expO3
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
        self.fgcmLog.log('INFO','Making I1/R1 plots...')

        plotter = FgcmZeropointPlotter(zpStruct, self.fgcmPars.bands,
                                       self.plotPath, self.outfileBaseWithCycle)

        plotter.makeR1I1Plots()
        plotter.makeR1I1Maps(self.ccdOffsets, ccdField=self.ccdField)

        #self.fgcmLog.log('INFO','Making zeropoint summary plots...')
        #plotter.makeZpPlots()


        ## compare I0 and R0, I1 and R1

        #self.fgcmLog.log('INFO','Making I1/R1 plots...')
        #acceptMask = (zpFlagDict['PHOTOMETRIC_FIT_EXPOSURE'] |
        #              zpFlagDict['PHOTOMETRIC_EXTRA_EXPOSURE'])
        #for i in xrange(self.fgcmPars.nBands):
        #    use,=np.where((np.core.defchararray.rstrip(zpStruct['BAND']) ==
        #                   self.fgcmPars.bands[i]) &
        #                  ((zpStruct['FGCM_FLAG'] & acceptMask) > 0) &
        #                  (np.abs(zpStruct['FGCM_R10']) < 1000.0) &
        #                  (np.abs(zpStruct['FGCM_R0']) < 1000.0))

        #    if (use.size == 0):
        #        continue

        #    i1 = zpStruct['FGCM_I10'][use]*zpStruct['FGCM_I0'][use]
        #    r1 = zpStruct['FGCM_R10'][use]*zpStruct['FGCM_R0'][use]

            # and limit to a reasonable range
        #    ok, = np.where((r1 > (i1.min()-2.0)) &
        #                   (r1 < (i1.max()+2.0)))

        #    i1=i1[ok]
        #    r1=r1[ok]

        #    fig = plt.figure(1,figsize=(8,6))
        #    fig.clf()

        #    ax=fig.add_subplot(111)

        #    ax.hexbin(i1,r1,cmap=plt.get_cmap('gray_r'),rasterized=True)
            # and overplot a 1-1 line that best covers the range of the data
       #     xlim = ax.get_xlim()
            #ylim = ax.get_ylim()
            #range0 = np.min([xlim[0],ylim[0]])+0.0001
            #range1 = np.max([xlim[1],ylim[1]])-0.0001
        #    range0 = xlim[0]+0.001
        #    range1 = xlim[1]-0.001
        #    ax.plot([range0,range1],[range0,range1],'b--',linewidth=2)

        #    ax.set_xlabel(r'$I_1$ from FGCM Fit',fontsize=16)
        #    ax.set_ylabel(r'$R_1$ from Retrieval',fontsize=16)

        #    text=r'$(%s)$' % (self.fgcmPars.bands[i])
        #    ax.annotate(text,(0.1,0.93),xycoords='axes fraction',
        #                ha='left',va='top',fontsize=16)#

        #    fig.savefig('%s/%s_i1r1_%s.png' % (self.plotPath,
        #                                       self.outfileBaseWithCycle,
        #                                       self.fgcmPars.bands[i]))#

        # need to know the mean zeropoint per exposure
        self.fgcmLog.log('INFO','Making zeropoint summary plots...')

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

        fig=plt.figure(1,figsize=(8,6))
        fig.clf()

        ax=fig.add_subplot(111)

        firstMJD = np.floor(np.min(self.fgcmPars.expMJD))

        # FIXME: make configurable
        cols = ['g','r','b','m','y']
        syms = ['.','+','o','*','x']

        for i in xrange(self.fgcmPars.nBands):
            use,=np.where((self.fgcmPars.expBandIndex == i) &
                          (expZpMean > 0.0))

            if (use.size == 0) :
                continue

            plt.plot(self.fgcmPars.expMJD[use] - firstMJD,
                     expZpMean[use],cols[i]+syms[i],
                     label=r'$(%s)$' % (self.fgcmPars.bands[i]))

        ax.legend(loc=3)

        fig.savefig('%s/%s_zeropoints.png' % (self.plotPath,
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

        sigFgcm = self.fgcmPars.compSigFgcm[self.fgcmPars.expBandIndex[zpExpIndex[zpIndex]]]
        nTilingsM1 = np.clip(zpStruct['FGCM_TILINGS'][zpIndex]-1.0,1.0,1e10)

        zpStruct['FGCM_ZPTERR'][zpIndex] = np.sqrt((sigFgcm**2./nTilingsM1) +
                                                   zpStruct['FGCM_ZPTVAR'][zpIndex] +
                                                   self.sigma0Cal**2.)

    def saveZptFits(self):
        """
        """

        import fitsio

        outFile = '%s/%s_zpt.fits' % (self.outputPath,self.outfileBaseWithCycle)
        self.fgcmLog.log('INFO','Saving zeropoints to %s' % (outFile))
        fitsio.write(outFile,self.zpStruct,clobber=True,extname='ZPTS')

    def saveAtmFits(self):
        """
        """

        import fitsio

        outFile = '%s/%s_atm.fits' % (self.outputPath,self.outfileBaseWithCycle)
        self.fgcmLog.log('INFO','Saving atmosphere parameters to %s' % (outFile))
        fitsio.write(outFile,self.atmStruct,clobber=True,extname='ATMPARS')

class FgcmZeropointPlotter(object):
    """
    """
    def __init__(self, zpStruct, bands, plotPath, outfileBase):
        self.zpStruct = zpStruct
        self.bands = bands
        self.plotPath = plotPath
        self.outfileBase = outfileBase

    def makeR1I1Plots(self):
        """
        """
        acceptMask = (zpFlagDict['PHOTOMETRIC_FIT_EXPOSURE'] |
                      zpFlagDict['PHOTOMETRIC_EXTRA_EXPOSURE'])
        for band in self.bands:
            use,=np.where((np.core.defchararray.rstrip(self.zpStruct['BAND']) == band) &
                          ((self.zpStruct['FGCM_FLAG'] & acceptMask) > 0) &
                          (np.abs(self.zpStruct['FGCM_R10']) < 1000.0) &
                          (np.abs(self.zpStruct['FGCM_R0']) < 1000.0))

            if (use.size == 0):
                continue

            i1 = self.zpStruct['FGCM_I10'][use] * self.zpStruct['FGCM_I0'][use]
            r1 = self.zpStruct['FGCM_R10'][use] * self.zpStruct['FGCM_R0'][use]

            # limit to a reasonable range
            #  note that r1 is much noisier than i1
            ok, = np.where((r1 > (i1.min()-2.0)) &
                           (r1 < (i1.max()+2.0)))

            i1=i1[ok]
            r1=r1[ok]

            fig = plt.figure(1,figsize=(8,6))
            fig.clf()

            ax=fig.add_subplot(111)

            ax.hexbin(i1,r1,cmap=plt.get_cmap('gray_r'),rasterized=True)
            # and overplot a 1-1 line that best covers the range of the data
            xlim = ax.get_xlim()
            range0 = xlim[0]+0.001
            range1 = xlim[1]-0.001
            ax.plot([range0,range1],[range0,range1],'b--',linewidth=2)

            ax.set_xlabel(r'$I_1$ from FGCM Fit',fontsize=16)
            ax.set_ylabel(r'$R_1$ from Retrieval',fontsize=16)

            text=r'$(%s)$' % (band)
            ax.annotate(text,(0.1,0.93),xycoords='axes fraction',
                        ha='left',va='top',fontsize=16)

            fig.savefig('%s/%s_i1r1_%s.png' % (self.plotPath,
                                               self.outfileBase,
                                               band))

    def makeR1I1Maps(self, ccdOffsets, ccdField='CCDNUM'):
        """
        """

        from fgcmUtilities import plotCCDMap

        acceptMask = (zpFlagDict['PHOTOMETRIC_FIT_EXPOSURE'] |
                      zpFlagDict['PHOTOMETRIC_EXTRA_EXPOSURE'])

        plotTypes=['I1', 'R1', 'R1 - I1']

        ccdMin = np.min(self.zpStruct[ccdField])
        ccdMax = np.max(self.zpStruct[ccdField])
        nCCD = (ccdMax - ccdMin) + 1

        for band in self.bands:
            use0,=np.where((np.core.defchararray.rstrip(self.zpStruct['BAND']) == band) &
                           ((self.zpStruct['FGCM_FLAG'] & acceptMask) > 0) &
                           (np.abs(self.zpStruct['FGCM_R10']) < 1000.0) &
                           (np.abs(self.zpStruct['FGCM_R0']) < 1000.0))

            ccdIndex = np.searchsorted(np.arange(ccdMin,ccdMax+1),
                                       self.zpStruct[ccdField][use0])

            i1 = self.zpStruct['FGCM_I10'][use0] * self.zpStruct['FGCM_I0'][use0]
            r1 = self.zpStruct['FGCM_R10'][use0] * self.zpStruct['FGCM_R0'][use0]

            meanI1 = np.zeros(nCCD)
            meanR1 = np.zeros(nCCD)
            nPerCCD = np.zeros(nCCD,dtype=np.int32)

            np.add.at(meanI1, ccdIndex, i1)
            np.add.at(meanR1, ccdIndex, r1)
            np.add.at(nPerCCD, ccdIndex, 1)

            use,=np.where(nPerCCD > 0)
            meanI1[use] /= nPerCCD[use]
            meanR1[use] /= nPerCCD[use]

            # use the same scale for all the plots
            st = np.argsort(meanR1[use])
            lo = meanR1[use[st[int(0.02*st.size)]]]
            hi = meanR1[use[st[int(0.98*st.size)]]]

            for plotType in plotTypes:
                fig=plt.figure(1,figsize=(8,6))
                fig.clf()

                ax=fig.add_subplot(111)

                if (plotType == 'R1'):
                    plotCCDMap(ax, ccdOffsets[use], meanR1[use], plotType, loHi=[lo,hi])
                elif (plotType == 'I1'):
                    plotCCDMap(ax, ccdOffsets[use], meanI1[use], plotType, loHi=[lo,hi])
                else:
                    plotCCDMap(ax, ccdOffsets[use], meanR1[use] - meanI1[use], plotType, loHi=[lo,hi])

                text = r'$(%s)$' % (band) + '\n' + \
                    r'%s' % (plotType)
                ax.annotate(text,
                            (0.1,0.93),xycoords='axes fraction',
                            ha='left',va='top',fontsize=18)

                fig.savefig('%s/%s_%s_%s.png' % (self.plotPath,
                                                 self.outfileBase,
                                                 plotType.replace(" ",""),
                                                 band))

        return None



