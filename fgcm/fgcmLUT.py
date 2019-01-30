from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import os
import sys
from pkg_resources import resource_filename


from .modtranGenerator import ModtranGenerator
from .fgcmAtmosphereTable import FgcmAtmosphereTable


from .sharedNumpyMemManager import SharedNumpyMemManager as snmm
from .fgcmLogger import FgcmLogger


class FgcmLUTMaker(object):
    """
    Class to make a look-up table.

    parameters
    ----------
    lutConfig: dict
       Dictionary with LUT config variables
    makeSeds: bool, default=False
       Make a SED-table in the look-up table (experimental)
    """

    def __init__(self,lutConfig,makeSeds=False):
        self._checkLUTConfig(lutConfig)

        self.magConstant = 2.5/np.log(10)
        self._setThroughput = False
        self.makeSeds = makeSeds

        try:
            self.stellarTemplateFile = resource_filename(__name__,'data/templates/stellar_templates_master.fits')
        except:
            raise IOError("Could not find stellar template file")

        if (not os.path.isfile(self.stellarTemplateFile)):
            raise IOError("Could not find stellar template file")

        if 'logger' in lutConfig:
            self.fgcmLog = lutConfig['logger']
        else:
            self.fgcmLog = FgcmLogger('dummy.log', 'INFO', printLogger=True)

    def _checkLUTConfig(self,lutConfig):
        """
        Internal method to check the lutConfig dictionary

        parameters
        ----------
        lutConfig: dict
        """

        self.runModtran = False

        requiredKeys = ['filterNames', 'stdFilterNames', 'nCCD']


        # first: check if there is a tableName here!
        if 'atmosphereTableName' in lutConfig:
            # Can we find this table?

            # load parameters from it and stuff into config dict
            self.atmosphereTable = FgcmAtmosphereTable.initWithTableName(lutConfig['atmosphereTableName'])

            # look for consistency between configs?
            # Note that we're assuming if somebody asked for a table they wanted
            # what was in the table
            for key in self.atmosphereTable.atmConfig:
                if key in lutConfig:
                    if 'Range' in key:
                        if (not np.isclose(lutConfig[key][0], self.atmosphereTable.atmConfig[key][0]) or
                            not np.isclose(lutConfig[key][1], self.atmosphereTable.atmConfig[key][1])):
                            print("Warning: input config %s is %.5f-%.5f but precomputed table is %.5f-%.5f" %
                                  (key, lutConfig[key][0], lutConfig[key][1],
                                   self.atmosphereTable.atmConfig[key][0],
                                   self.atmosphereTable.atmConfig[key][1]))
                    else:
                        if not np.isclose(lutConfig[key], self.atmosphereTable.atmConfig[key]):
                            print("Warning: input config %s is %.5f but precomputed table is %.5f" %
                                  (key, lutConfig[key], self.atmosphereTable.atmConfig[key]))

        else:
            # regular config with parameters
            self.runModtran = True
            self.atmosphereTable = FgcmAtmosphereTable(lutConfig)

        for key in requiredKeys:
            if (key not in lutConfig):
                raise ValueError("required %s not in lutConfig" % (key))
            if ('Range' in key):
                if (len(lutConfig[key]) != 2):
                    raise ValueError("%s must have 2 elements" % (key))

        self.lutConfig = lutConfig

        self.filterNames = self.lutConfig['filterNames']
        self.stdFilterNames = self.lutConfig['stdFilterNames']

        if len(self.filterNames) != len(self.stdFilterNames):
            raise ValueError("Length of filterNames must be same as stdFilterNames")

        for stdFilterName in self.stdFilterNames:
            if stdFilterName not in self.filterNames:
                raise ValueError("stdFilterName %s not in list of filterNames" % (stdFilterName))

        self.nCCD = self.lutConfig['nCCD']
        self.nCCDStep = self.nCCD+1

        # and record the standard values out of the config
        #  (these will also come out of the save file)

        self.pmbStd = self.atmosphereTable.atmConfig['pmbStd']
        self.pwvStd = self.atmosphereTable.atmConfig['pwvStd']
        self.lnPwvStd = np.log(self.atmosphereTable.atmConfig['pwvStd'])
        self.o3Std = self.atmosphereTable.atmConfig['o3Std']
        self.tauStd = self.atmosphereTable.atmConfig['tauStd']
        self.lnTauStd = np.log(self.tauStd)
        self.alphaStd = self.atmosphereTable.atmConfig['alphaStd']
        self.secZenithStd = self.atmosphereTable.atmConfig['airmassStd']
        self.zenithStd = np.arccos(1./self.secZenithStd)*180./np.pi

        if ('lambdaRange' in self.atmosphereTable.atmConfig):
            self.lambdaRange = np.array(self.atmosphereTable.atmConfig['lambdaRange'])
        else:
            self.lambdaRange = np.array([3000.0,11000.0])

        if ('lambdaStep' in self.atmosphereTable.atmConfig):
            self.lambdaStep = self.atmosphereTable.atmConfig['lambdaStep']
        else:
            self.lambdaStep = 0.5

        if ('lambdaNorm' in self.atmosphereTable.atmConfig):
            self.lambdaNorm = self.atmosphereTable.atmConfig['lambdaNorm']
        else:
            self.lambdaNorm = 7750.0

    def setThroughputs(self, throughputDict):
        """
        Set the throughputs per CCD

        parameters
        ----------
        throughputDict: dict
           Dict with throughput information

        The throughput dict should have one entry for each filterName.
        Each of these elements should be a dictionary with the following keys:
           'LAMBDA': numpy float array with wavelength values
           ccd_index: numpy float array with throughput values for the ccd_index
        There should be one entry for each CCD.
        """

        self.inThroughputs=[]
        for filterName in self.filterNames:
            try:
                lam = throughputDict[filterName]['LAMBDA']
            except:
                raise ValueError("Wavelength LAMBDA not found for filter %s in throughputDict!" % (filterName))

            tput = np.zeros(lam.size, dtype=[('LAMBDA','f4'),
                                             ('THROUGHPUT_AVG','f4'),
                                             ('THROUGHPUT_CCD','f4',self.nCCD)])
            tput['LAMBDA'][:] = lam
            for ccdIndex in xrange(self.nCCD):
                try:
                    tput['THROUGHPUT_CCD'][:,ccdIndex] = throughputDict[filterName][ccdIndex]
                except:
                    raise ValueError("CCD Index %d not found for filter %s in throughputDict!" % (ccdIndex,filterName))

            # check if the average is there, if not compute it
            if ('AVG' in throughputDict[filterName]):
                tput['THROUGHPUT_AVG'][:] = throughputDict[filterName]['AVG']
            else:
                self.fgcmLog.info("Average throughput not found in throughputDict for filter %s.  Computing now..." % (filterName))
                for i in xrange(lam.size):
                    use,=np.where(tput['THROUGHPUT_CCD'][i,:] > 0.0)
                    if (use.size > 0):
                        tput['THROUGHPUT_AVG'][i] = np.mean(tput['THROUGHPUT_CCD'][i,use])

            self.inThroughputs.append(tput)

        self._setThroughput = True

    def makeLUT(self):
        """
        Make the look-up table.  This can either be saved with saveLUT or accessed via
         attributes.

        parameters
        ----------
        None

        output attributes
        -----------------
        pmb: float array
           Pressure (millibars)
        pmbFactor: float array
           Pressure factor
        pmbElevation: float
           Standard PMB at elevation
        pwv: float array
           Water vapor array
        o3: float array
           Ozone array
        tau: float array
           Aerosol optical index array
        lambdaNorm: float
           Aerosol normalization wavelength
        alpha: float array
           Aerosol slope array
        zenith: float array
           Zenith angle array
        nccd: int
           Number of CCDs in table
        pmbStd: float
           Standard PMB
        pwvStd: float
           Standard PWV
        o3Std: float
           Standard O3
        tauStd: float
           Standard tau
        alphaStd: float
           Standard alpha
        zenithStd: float
           Standard zenith angle (deg)
        lambdaRange: numpy float array
           Wavelength range (A)
        lambdaStep: float
           Wavelength step (A)
        lambdaStd: numpy float array
           Standard wavelength for each filterName (A) at each standard band
        lambdaStdRaw: numpy float array
           Standard wavelength for each filterName (A)
        I0Std: numpy float array
           Standard value for I0 for each filterName
        I1Std: numpy float array
           Standard value for I1 for each filterName
        I10Std: numpy float array
           Standard value for I10 for each filterName
        lambdaB: numpy float array
           Standard wavelength for each filterName (A) assuming no atmosphere
        atmLambda: numpy float array
           Standard atmosphere wavelength array (A)
        atmStdTrans: numpy float array
           Standard atmosphere transmission array
        lut: Look-up table recarray
           lut['I0']: I0 LUT (multi-dimensional)
           lut['I1']: I1 LUT (multi-dimensional)
        lutDeriv: Derivative recarray
           lutDeriv['D_PMB']: I0 PMB derivative
           lutDeriv['D_LNPWV']: I0 log(PWV) derivative
           lutDeriv['D_O3']: I0 O3 derivative
           lutDeriv['D_LNTAU']: I0 log(tau) derivative
           lutDeriv['D_ALPHA']: I0 alpha derivative
           lutDeriv['D_SECZENITH']: I0 sec(zenith) derivative
           lutDeriv['D_PMB_I1']: I1 PMB derivative
           lutDeriv['D_LNPWV_I1']: I1 log(PWV) derivative
           lutDeriv['D_O3_I1']: I1 O3 derivative
           lutDeriv['D_LNTAU_I1']: I1 log(tau) derivative
           lutDeriv['D_ALPHA_I1']: I1 alpha derivative
           lutDeriv['D_SECZENITH_I1']: I0 sec(zenith) derivative
        """

        if not self._setThroughput:
            raise ValueError("Must set the throughput before running makeLUT")

        if self.runModtran:
            # need to build the table
            self.atmosphereTable.generateTable()
        else:
            # load from data
            self.atmosphereTable.loadTable()

        # and grab from the table
        self.atmLambda = self.atmosphereTable.atmLambda
        self.atmStdTrans = self.atmosphereTable.atmStdTrans

        self.pmbElevation = self.atmosphereTable.pmbElevation

        self.pmb = self.atmosphereTable.pmb
        self.pmbDelta = self.atmosphereTable.pmbDelta
        pmbPlus = np.append(self.pmb, self.pmb[-1] + self.pmbDelta)

        self.lnPwv = self.atmosphereTable.lnPwv
        self.lnPwvDelta = self.atmosphereTable.lnPwvDelta
        self.pwv = np.exp(self.lnPwv)
        lnPwvPlus = np.append(self.lnPwv, self.lnPwv[-1] + self.lnPwvDelta)
        pwvPlus = np.exp(lnPwvPlus)

        self.o3 = self.atmosphereTable.o3
        self.o3Delta = self.atmosphereTable.o3Delta
        o3Plus = np.append(self.o3, self.o3[-1] + self.o3Delta)

        self.lnTau = self.atmosphereTable.lnTau
        self.lnTauDelta = self.atmosphereTable.lnTauDelta
        self.tau = np.exp(self.lnTau)
        lnTauPlus = np.append(self.lnTau, self.lnTau[-1] + self.lnTauDelta)
        tauPlus = np.exp(lnTauPlus)

        self.alpha = self.atmosphereTable.alpha
        self.alphaDelta = self.atmosphereTable.alphaDelta
        alphaPlus = np.append(self.alpha, self.alpha[-1] + self.alphaDelta)

        self.secZenith = self.atmosphereTable.secZenith
        self.secZenithDelta = self.atmosphereTable.secZenithDelta
        self.zenith = np.arccos(1./self.secZenith)*180./np.pi
        secZenithPlus = np.append(self.secZenith, self.secZenith[-1] + self.secZenithDelta)
        zenithPlus = np.arccos(1./secZenithPlus)*180./np.pi

        # and compute the proper airmass...
        self.airmass = self.secZenith - 0.0018167*(self.secZenith-1.0) - 0.002875*(self.secZenith-1.0)**2.0 - 0.0008083*(self.secZenith-1.0)**3.0
        airmassPlus = secZenithPlus - 0.0018167*(secZenithPlus-1.0) - 0.002875*(secZenithPlus-1.0)**2.0 - 0.0008083*(secZenithPlus-1.0)**3.0

        pwvAtmTable = self.atmosphereTable.pwvAtmTable
        o3AtmTable = self.atmosphereTable.o3AtmTable
        o2AtmTable = self.atmosphereTable.o2AtmTable
        rayleighAtmTable = self.atmosphereTable.rayleighAtmTable

        # get the filters over the same lambda ranges...
        self.fgcmLog.info("\nInterpolating filters...")
        self.throughputs = []
        for i in xrange(len(self.filterNames)):
            inLam = self.inThroughputs[i]['LAMBDA']

            tput = np.zeros(self.atmLambda.size, dtype=[('LAMBDA','f4'),
                                                        ('THROUGHPUT_AVG','f4'),
                                                        ('THROUGHPUT_CCD','f4',self.nCCD)])
            tput['LAMBDA'][:] = self.atmLambda

            for ccdIndex in xrange(self.nCCD):
                ifunc = interpolate.interp1d(inLam, self.inThroughputs[i]['THROUGHPUT_CCD'][:,ccdIndex])
                tput['THROUGHPUT_CCD'][:,ccdIndex] = np.clip(ifunc(self.atmLambda),
                                                             0.0,
                                                             1e100)
            ifunc = interpolate.interp1d(inLam, self.inThroughputs[i]['THROUGHPUT_AVG'])
            tput['THROUGHPUT_AVG'][:] = np.clip(ifunc(self.atmLambda), 0.0, 1e100)

            self.throughputs.append(tput)

        # and now we can get the standard atmosphere and lambda_b
        self.fgcmLog.info("Computing lambdaB")
        self.lambdaB = np.zeros(len(self.filterNames))
        for i in xrange(len(self.filterNames)):
            num = integrate.simps(self.atmLambda * self.throughputs[i]['THROUGHPUT_AVG'] / self.atmLambda, self.atmLambda)
            denom = integrate.simps(self.throughputs[i]['THROUGHPUT_AVG'] / self.atmLambda, self.atmLambda)
            self.lambdaB[i] = num / denom
            self.fgcmLog.info("Filter: %s, lambdaB = %.3f" % (self.filterNames[i], self.lambdaB[i]))

        self.fgcmLog.info("Computing lambdaStdFilter")
        self.lambdaStdFilter = np.zeros(len(self.filterNames))
        for i in xrange(len(self.filterNames)):
            num = integrate.simps(self.atmLambda * self.throughputs[i]['THROUGHPUT_AVG'] * self.atmStdTrans / self.atmLambda, self.atmLambda)
            denom = integrate.simps(self.throughputs[i]['THROUGHPUT_AVG'] * self.atmStdTrans / self.atmLambda, self.atmLambda)
            self.lambdaStdFilter[i] = num / denom
            self.fgcmLog.info("Filter: %s, lambdaStdFilter = %.3f" % (self.filterNames[i],self.lambdaStdFilter[i]))

        # now compute lambdaStd based on the desired standards...
        self.fgcmLog.info("Calculating lambdaStd")
        self.lambdaStd = np.zeros(len(self.filterNames))

        for i, filterName in enumerate(self.filterNames):
            ind = self.filterNames.index(self.stdFilterNames[i])
            #ind, = np.where(self.filterNames == self.stdFilterNames[i])
            #self.lambdaStd[i] = self.lambdaStdFilter[ind[0]]
            self.lambdaStd[i] = self.lambdaStdFilter[ind]
            self.fgcmLog.info("Filter: %s (from %s) lambdaStd = %.3f" %
                              (filterName, self.stdFilterNames[i], self.lambdaStd[i]))

        self.fgcmLog.info("Computing I0Std/I1Std")
        self.I0Std = np.zeros(len(self.filterNames))
        self.I1Std = np.zeros(len(self.filterNames))

        for i in xrange(len(self.filterNames)):
            self.I0Std[i] = integrate.simps(self.throughputs[i]['THROUGHPUT_AVG'] * self.atmStdTrans / self.atmLambda, self.atmLambda)
            self.I1Std[i] = integrate.simps(self.throughputs[i]['THROUGHPUT_AVG'] * self.atmStdTrans * (self.atmLambda - self.lambdaStd[i]) / self.atmLambda, self.atmLambda)

        self.I10Std = self.I1Std / self.I0Std

        #################################
        ## Make the I0/I1 LUT
        #################################

        self.fgcmLog.info("Building look-up table...")
        lutPlus = np.zeros((len(self.filterNames),
                            pwvPlus.size,
                            o3Plus.size,
                            tauPlus.size,
                            alphaPlus.size,
                            zenithPlus.size,
                            self.nCCDStep),
                           dtype=[('I0','f4'),
                                  ('I1','f4')])

        # pre-compute pmb factors
        pmbMolecularScattering = np.exp(-(pmbPlus - self.pmbElevation)/self.pmbElevation)
        pmbMolecularAbsorption = pmbMolecularScattering ** 0.6
        pmbFactorPlus = pmbMolecularScattering * pmbMolecularAbsorption
        self.pmbFactor = pmbFactorPlus[:-1]

        # this set of nexted for loops could probably be vectorized in some way
        for i in xrange(len(self.filterNames)):
            self.fgcmLog.info("Working on filter %s" % (self.filterNames[i]))
            for j in xrange(pwvPlus.size):
                self.fgcmLog.info("  and on pwv #%d" % (j))
                for k in xrange(o3Plus.size):
                    self.fgcmLog.info("   and on o3 #%d" % (k))
                    for m in xrange(tauPlus.size):
                        for n in xrange(alphaPlus.size):
                            for o in xrange(zenithPlus.size):
                                aerosolTauLambda = np.exp(-1.0*tauPlus[m]*airmassPlus[o]*(self.atmLambda/self.lambdaNorm)**(-alphaPlus[n]))
                                for p in xrange(self.nCCDStep):
                                    if (p == self.nCCD):
                                        Sb = self.throughputs[i]['THROUGHPUT_AVG'] * o2AtmTable[o,:] * rayleighAtmTable[o,:] * pwvAtmTable[j,o,:] * o3AtmTable[k,o,:] * aerosolTauLambda
                                    else:
                                        Sb = self.throughputs[i]['THROUGHPUT_CCD'][:,p] * o2AtmTable[o,:] * rayleighAtmTable[o,:] * pwvAtmTable[j,o,:] * o3AtmTable[k,o,:] * aerosolTauLambda

                                    lutPlus['I0'][i,j,k,m,n,o,p] = integrate.simps(Sb / self.atmLambda, self.atmLambda)
                                    lutPlus['I1'][i,j,k,m,n,o,p] = integrate.simps(Sb * (self.atmLambda - self.lambdaStd[i]) / self.atmLambda, self.atmLambda)

        # and create the LUT (not plus)
        self.lut = np.zeros((len(self.filterNames),
                             self.pwv.size,
                             self.o3.size,
                             self.tau.size,
                             self.alpha.size,
                             self.zenith.size,
                             self.nCCDStep),
                            dtype=lutPlus.dtype)

        temp = np.delete(lutPlus['I0'], self.pwv.size, axis=1)
        temp = np.delete(temp, self.o3.size, axis=2)
        temp = np.delete(temp, self.tau.size, axis=3)
        temp = np.delete(temp, self.alpha.size, axis=4)
        temp = np.delete(temp, self.zenith.size, axis=5)

        self.lut['I0'] = temp

        temp = np.delete(lutPlus['I1'], self.pwv.size, axis=1)
        temp = np.delete(temp, self.o3.size, axis=2)
        temp = np.delete(temp, self.tau.size, axis=3)
        temp = np.delete(temp, self.alpha.size, axis=4)
        temp = np.delete(temp, self.zenith.size, axis=5)

        self.lut['I1'] = temp

        #################################
        ## Make the I0/I1 derivative LUTs
        #################################

        # This is *not* done plus-size

        self.lutDeriv = np.zeros((len(self.filterNames),
                                  self.pwv.size,
                                  self.o3.size,
                                  self.tau.size,
                                  self.alpha.size,
                                  self.zenith.size,
                                  self.nCCDStep),
                                 dtype=[('D_PMB','f4'),
                                        ('D_LNPWV','f4'),
                                        ('D_O3','f4'),
                                        ('D_LNTAU','f4'),
                                        ('D_ALPHA','f4'),
                                        ('D_SECZENITH','f4'),
                                        ('D_PMB_I1','f4'),
                                        ('D_LNPWV_I1','f4'),
                                        ('D_O3_I1','f4'),
                                        ('D_LNTAU_I1','f4'),
                                        ('D_ALPHA_I1','f4'),
                                        ('D_SECZENITH_I1','f4')])

        self.fgcmLog.info("Computing derivatives...")

        ## FIXME: figure out PMB derivative?

        for i in xrange(len(self.filterNames)):
            self.fgcmLog.info("Working on filter %s" % (self.filterNames[i]))
            for j in xrange(self.pwv.size):
                for k in xrange(self.o3.size):
                    for m in xrange(self.tau.size):
                        for n in xrange(self.alpha.size):
                            for o in xrange(self.zenith.size):
                                for p in xrange(self.nCCDStep):
                                    self.lutDeriv['D_LNPWV'][i,j,k,m,n,o,p] = (
                                        ((lutPlus['I0'][i,j+1,k,m,n,o,p] -
                                          lutPlus['I0'][i,j,k,m,n,o,p]) /
                                         self.lnPwvDelta)
                                        )
                                    self.lutDeriv['D_LNPWV_I1'][i,j,k,m,n,o,p] = (
                                        ((lutPlus['I1'][i,j+1,k,m,n,o,p] -
                                          lutPlus['I1'][i,j,k,m,n,o,p]) /
                                         self.lnPwvDelta)
                                        )
                                    self.lutDeriv['D_O3'][i,j,k,m,n,o,p] = (
                                        ((lutPlus['I0'][i,j,k+1,m,n,o,p] -
                                          lutPlus['I0'][i,j,k,m,n,o,p]) /
                                         self.o3Delta)
                                        )
                                    self.lutDeriv['D_O3_I1'][i,j,k,m,n,o,p] = (
                                        ((lutPlus['I1'][i,j,k+1,m,n,o,p] -
                                          lutPlus['I1'][i,j,k,m,n,o,p]) /
                                         self.o3Delta)
                                        )
                                    self.lutDeriv['D_LNTAU'][i,j,k,m,n,o,p] = (
                                        ((lutPlus['I0'][i,j,k,m+1,n,o,p] -
                                          lutPlus['I0'][i,j,k,m,n,o,p]) /
                                         self.lnTauDelta)
                                        )
                                    self.lutDeriv['D_LNTAU_I1'][i,j,k,m,n,o,p] = (
                                        ((lutPlus['I1'][i,j,k,m+1,n,o,p] -
                                          lutPlus['I1'][i,j,k,m,n,o,p]) /
                                         self.lnTauDelta)
                                        )
                                    self.lutDeriv['D_ALPHA'][i,j,k,m,n,o,p] = (
                                        ((lutPlus['I0'][i,j,k,m,n+1,o,p] -
                                          lutPlus['I0'][i,j,k,m,n,o,p]) /
                                         self.alphaDelta)
                                        )
                                    self.lutDeriv['D_ALPHA_I1'][i,j,k,m,n,o,p] = (
                                        ((lutPlus['I1'][i,j,k,m,n+1,o,p] -
                                          lutPlus['I1'][i,j,k,m,n,o,p]) /
                                         self.alphaDelta)
                                        )
                                    self.lutDeriv['D_SECZENITH'][i,j,k,m,n,o,p] = (
                                        ((lutPlus['I0'][i,j,k,m,n,o+1,p] -
                                          lutPlus['I0'][i,j,k,m,n,o,p]) /
                                         self.secZenithDelta)
                                        )
                                    self.lutDeriv['D_SECZENITH_I1'][i,j,k,m,n,o,p] = (
                                        ((lutPlus['I1'][i,j,k,m,n,o+1,p] -
                                          lutPlus['I1'][i,j,k,m,n,o,p]) /
                                         self.secZenithDelta)
                                        )


        if (self.makeSeds):
            # and the SED LUT
            self.fgcmLog.info("Building SED LUT")

            # arbitrary.  Configure?  Fit?  Seems stable...
            delta = 600.0

            # blah on fits here...
            import fitsio

            # how many extensions?
            fits=fitsio.FITS(self.stellarTemplateFile)
            fits.update_hdu_list()
            extNames = []
            for hdu in fits.hdu_list:
                extName = hdu.get_extname()
                if ('TEMPLATE_' in extName):
                    extNames.append(extName)

            # set up SED look-up table
            nTemplates = len(extNames)

            self.sedLUT = np.zeros(nTemplates, dtype=[('TEMPLATE','i4'),
                                                      ('SYNTHMAG','f4',len(self.filterNames)),
                                                      ('FPRIME','f4',len(self.filterNames))])

            # now do it...looping is no problem since there aren't that many.

            for i in xrange(nTemplates):
                data = fits[extNames[i]].read()

                templateLambda = data['LAMBDA']
                templateFLambda = data['FLUX']
                templateFnu = templateFLambda * templateLambda * templateLambda

                parts=extNames[i].split('_')
                self.sedLUT['TEMPLATE'][i] = int(parts[1])

                # interpolate to atmLambda
                intFunc = interpolate.interp1d(templateLambda, templateFnu)
                fnu = np.zeros(self.atmLambda.size)
                good,=np.where((self.atmLambda >= templateLambda[0]) &
                               (self.atmLambda <= templateLambda[-1]))
                fnu[good] = intFunc(self.atmLambda[good])

                # out of range, let it hit the limit
                lo,=np.where(self.atmLambda < templateLambda[0])
                if (lo.size > 0):
                    fnu[lo] = intFunc(self.atmLambda[good[0]])
                hi,=np.where(self.atmLambda > templateLambda[-1])
                if (hi.size > 0):
                    fnu[hi] = intFunc(self.atmLambda[good[-1]])

                # compute synthetic mags
                for j in xrange(len(self.filterNames)):
                    num = integrate.simps(fnu * self.throughputs[j]['THROUGHPUT_AVG'][:] * self.atmStdTrans / self.atmLambda, self.atmLambda)
                    denom = integrate.simps(self.throughputs[j]['THROUGHPUT_AVG'][:] * self.atmStdTrans / self.atmLambda, self.atmLambda)

                    self.sedLUT['SYNTHMAG'][i,j] = -2.5*np.log10(num/denom)

                # and compute fprimes
                for j in xrange(len(self.filterNames)):
                    use,=np.where((templateLambda >= (self.lambdaStd[j]-delta)) &
                                  (templateLambda <= (self.lambdaStd[j]+delta)))

                    fit = np.polyfit(templateLambda[use] - self.lambdaStd[j],
                                     templateFnu[use],
                                     1)

                    self.sedLUT['FPRIME'][i,j] = fit[0] / fit[1]

            fits.close()

    def saveLUT(self,lutFile,clobber=False):
        """
        """

        import fitsio

        if (os.path.isfile(lutFile) and not clobber):
            self.fgcmLog.info("lutFile %s already exists, and clobber is False." % (lutFile))
            return

        self.fgcmLog.info("Saving LUT to %s" % (lutFile))

        # first, save the LUT itself
        fitsio.write(lutFile,self.lut.flatten(),extname='LUT',clobber=True)

        # and now save the indices
        maxFilterLen = len(max(self.filterNames, key=len))

        indexVals = np.zeros(1,dtype=[('FILTERNAMES', 'a%d' % (maxFilterLen), len(self.filterNames)),
                                      ('STDFILTERNAMES', 'a%d' % (maxFilterLen), len(self.stdFilterNames)),
                                      ('PMB','f8',self.pmb.size),
                                      ('PMBFACTOR','f8',self.pmb.size),
                                      ('PMBELEVATION','f8'),
                                      ('PWV','f8',self.pwv.size),
                                      ('O3','f8',self.o3.size),
                                      ('TAU','f8',self.tau.size),
                                      ('LAMBDANORM','f8'),
                                      ('ALPHA','f8',self.alpha.size),
                                      ('ZENITH','f8',self.zenith.size),
                                      ('NCCD','i4')])
        indexVals['FILTERNAMES'] = self.filterNames
        indexVals['STDFILTERNAMES'] = self.stdFilterNames
        indexVals['PMB'] = self.pmb
        indexVals['PMBFACTOR'] = self.pmbFactor
        indexVals['PMBELEVATION'] = self.pmbElevation
        indexVals['PWV'] = self.pwv
        indexVals['O3'] = self.o3
        indexVals['TAU'] = self.tau
        indexVals['LAMBDANORM'] = self.lambdaNorm
        indexVals['ALPHA'] = self.alpha
        indexVals['ZENITH'] = self.zenith
        indexVals['NCCD'] = self.nCCD

        fitsio.write(lutFile,indexVals,extname='INDEX')

        # and the standard values
        stdVals = np.zeros(1,dtype=[('PMBSTD','f8'),
                                    ('PWVSTD','f8'),
                                    ('O3STD','f8'),
                                    ('TAUSTD','f8'),
                                    ('ALPHASTD','f8'),
                                    ('ZENITHSTD','f8'),
                                    ('LAMBDARANGE','f8',2),
                                    ('LAMBDASTEP','f8'),
                                    ('LAMBDASTD','f8',len(self.filterNames)),
                                    ('LAMBDASTDFILTER','f8',len(self.filterNames)),
                                    ('LAMBDANORM','f8'),
                                    ('I0STD','f8',len(self.filterNames)),
                                    ('I1STD','f8',len(self.filterNames)),
                                    ('I10STD','f8',len(self.filterNames)),
                                    ('LAMBDAB','f8',len(self.filterNames)),
                                    ('ATMLAMBDA','f8',self.atmLambda.size),
                                    ('ATMSTDTRANS','f8',self.atmStdTrans.size)])
        stdVals['PMBSTD'] = self.pmbStd
        stdVals['PWVSTD'] = self.pwvStd
        stdVals['O3STD'] = self.o3Std
        stdVals['TAUSTD'] = self.tauStd
        stdVals['ALPHASTD'] = self.alphaStd
        stdVals['ZENITHSTD'] = self.zenithStd
        stdVals['LAMBDARANGE'] = self.lambdaRange
        stdVals['LAMBDASTEP'] = self.lambdaStep
        stdVals['LAMBDASTD'][:] = self.lambdaStd
        stdVals['LAMBDASTDFILTER'][:] = self.lambdaStdFilter
        stdVals['LAMBDANORM'][:] = self.lambdaNorm
        stdVals['I0STD'][:] = self.I0Std
        stdVals['I1STD'][:] = self.I1Std
        stdVals['I10STD'][:] = self.I10Std
        stdVals['LAMBDAB'][:] = self.lambdaB
        stdVals['ATMLAMBDA'][:] = self.atmLambda
        stdVals['ATMSTDTRANS'][:] = self.atmStdTrans

        fitsio.write(lutFile,stdVals,extname='STD')

        # and the derivatives
        self.fgcmLog.info("Writing Derivative LUT")
        fitsio.write(lutFile,self.lutDeriv.flatten(),extname='DERIV')

        # and the SED LUT
        if (self.makeSeds):
            self.fgcmLog.info("Writing SED LUT")
            fitsio.write(lutFile,self.sedLUT,extname='SED')


class FgcmLUT(object):
    """
    Class to hold the main throughput look-up table and apply it.  If loading from
     a fits table, initialize with initFromFits(lutFile).

    parameters
    ----------
    indexVals: numpy recarray
       With LUT index values
    lutFlat: numpy recarray
       Flattened I0/I1 arrays
    lutDerivFlat: numpy recarray
       Flattened I0/I1 derivative arrays
    stdVals: numpy recarray
       Standard atmosphere and associated values
    sedLUT: bool, default=False
       Use SED look-up table instead of colors (experimental).
    filterToBand: dict, optional
       Dictionary to map filterNames to bands if not unique
    """

    def __init__(self, indexVals, lutFlat, lutDerivFlat, stdVals, sedLUT=None, filterToBand=None):

        #self.filterNames = indexVals['FILTERNAMES'][0]
        #self.stdFilterNames = indexVals['STDFILTERNAMES'][0]
        self.filterNames = [n.decode('utf-8') for n in indexVals['FILTERNAMES'][0]]
        self.stdFilterNames = [n.decode('utf-8') for n in indexVals['STDFILTERNAMES'][0]]
        self.pmb = indexVals['PMB'][0]
        self.pmbFactor = indexVals['PMBFACTOR'][0]
        self.pmbDelta = self.pmb[1] - self.pmb[0]
        self.pmbElevation = indexVals['PMBELEVATION'][0]
        self.lambdaNorm = indexVals['LAMBDANORM'][0]

        self.pwv = indexVals['PWV'][0]
        self.lnPwv = np.log(self.pwv)
        self.lnPwvDelta = self.lnPwv[1] - self.lnPwv[0]
        self.o3 = indexVals['O3'][0]
        self.o3Delta = self.o3[1] - self.o3[0]
        self.tau = indexVals['TAU'][0]
        self.lnTau = np.log(self.tau)
        self.lnTauDelta = self.lnTau[1] - self.lnTau[0]
        self.alpha = indexVals['ALPHA'][0]
        self.alphaDelta = self.alpha[1] - self.alpha[0]
        self.zenith = indexVals['ZENITH'][0]
        self.secZenith = 1./np.cos(self.zenith*np.pi/180.)
        self.secZenithDelta = self.secZenith[1] - self.secZenith[0]
        self.nCCD = indexVals['NCCD'][0]
        self.nCCDStep = self.nCCD+1

        # make shared memory arrays for LUTs
        sizeTuple = (len(self.filterNames),self.pwv.size,self.o3.size,
                     self.tau.size,self.alpha.size,self.zenith.size,self.nCCDStep)

        self.lutI0Handle = snmm.createArray(sizeTuple,dtype='f4')
        snmm.getArray(self.lutI0Handle)[:,:,:,:,:,:,:] = lutFlat['I0'].reshape(sizeTuple)

        self.lutI1Handle = snmm.createArray(sizeTuple,dtype='f4')
        snmm.getArray(self.lutI1Handle)[:,:,:,:,:,:,:] = lutFlat['I1'].reshape(sizeTuple)

        # and read in the derivatives

        # create shared memory
        self.lutDLnPwvHandle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDO3Handle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDLnTauHandle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDAlphaHandle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDSecZenithHandle = snmm.createArray(sizeTuple,dtype='f4')

        self.lutDLnPwvI1Handle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDO3I1Handle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDLnTauI1Handle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDAlphaI1Handle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDSecZenithI1Handle = snmm.createArray(sizeTuple,dtype='f4')

        snmm.getArray(self.lutDLnPwvHandle)[:,:,:,:,:,:,:] = lutDerivFlat['D_LNPWV'].reshape(sizeTuple)
        snmm.getArray(self.lutDO3Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_O3'].reshape(sizeTuple)
        snmm.getArray(self.lutDLnTauHandle)[:,:,:,:,:,:,:] = lutDerivFlat['D_LNTAU'].reshape(sizeTuple)
        snmm.getArray(self.lutDAlphaHandle)[:,:,:,:,:,:,:] = lutDerivFlat['D_ALPHA'].reshape(sizeTuple)
        snmm.getArray(self.lutDSecZenithHandle)[:,:,:,:,:,:,:] = lutDerivFlat['D_SECZENITH'].reshape(sizeTuple)

        self.hasI1Derivatives = False
        try:
            snmm.getArray(self.lutDLnPwvI1Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_LNPWV_I1'].reshape(sizeTuple)
            snmm.getArray(self.lutDO3I1Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_O3_I1'].reshape(sizeTuple)
            snmm.getArray(self.lutDLnTauI1Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_LNTAU_I1'].reshape(sizeTuple)
            snmm.getArray(self.lutDAlphaI1Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_ALPHA_I1'].reshape(sizeTuple)
            snmm.getArray(self.lutDSecZenithI1Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_SECZENITH_I1'].reshape(sizeTuple)
            self.hasI1Derivatives = True
        except:
            # just fill with zeros
            pass
            #print("No I1 derivative information")

        # get the standard values

        self.pmbStd = stdVals['PMBSTD'][0]
        self.pwvStd = stdVals['PWVSTD'][0]
        self.lnPwvStd = np.log(self.pwvStd)
        self.o3Std = stdVals['O3STD'][0]
        self.tauStd = stdVals['TAUSTD'][0]
        self.lnTauStd = np.log(self.tauStd)
        self.alphaStd = stdVals['ALPHASTD'][0]
        self.zenithStd = stdVals['ZENITHSTD'][0]
        self.secZenithStd = 1./np.cos(np.radians(self.zenithStd))
        self.lambdaRange = stdVals['LAMBDARANGE'][0]
        self.lambdaStep = stdVals['LAMBDASTEP'][0]
        self.lambdaStd = stdVals['LAMBDASTD'][0]
        self.lambdaStdFilter = stdVals['LAMBDASTDFILTER'][0]
        self.I0Std = stdVals['I0STD'][0]
        self.I1Std = stdVals['I1STD'][0]
        self.I10Std = stdVals['I10STD'][0]
        self.lambdaB = stdVals['LAMBDAB'][0]
        self.atmLambda = stdVals['ATMLAMBDA'][0]
        self.atmStdTrans = stdVals['ATMSTDTRANS'][0]

        self.magConstant = 2.5/np.log(10)

        if (filterToBand is None):
            # just set up a 1-1 mapping
            self.filterToBand = {}
            for filterName in self.filterNames:
                self.filterToBand[filterName] = filterName
        else:
            self.filterToBand = filterToBand

        # finally, read in the sedLUT
        ## this is experimental
        self.hasSedLUT = False
        if (sedLUT is not None):
            self.sedLUT = sedLUT

            self.hasSedLUT = True

            ## FIXME: make general
            self.sedColor = self.sedLUT['SYNTHMAG'][:,0] - self.sedLUT['SYNTHMAG'][:,2]
            st = np.argsort(self.sedColor)

            self.sedColor = self.sedColor[st]
            self.sedLUT = self.sedLUT[st]

    @classmethod
    def initFromFits(cls, lutFile, filterToBand=None):
        """
        Initials FgcmLUT using fits file.

        parameters
        ----------
        lutFile: string
           Name of the LUT file
        """

        import fitsio

        lutFlat = fitsio.read(lutFile, ext='LUT')
        lutDerivFlat = fitsio.read(lutFile, ext='DERIV')
        indexVals = fitsio.read(lutFile, ext='INDEX')
        stdVals = fitsio.read(lutFile, ext='STD')

        try:
            sedLUT = fitsio.read(lutFile, ext='SED')
        except Exception as inst:
            sedLUT = None

        return cls(indexVals, lutFlat, lutDerivFlat, stdVals,
                   sedLUT=sedLUT, filterToBand=filterToBand)


    def getIndices(self, filterIndex, lnPwv, o3, lnTau, alpha, secZenith, ccdIndex, pmb):
        """
        Compute indices in the look-up table.  These are in regular (non-normalized) units.

        parameters
        ----------
        filterIndex: int array
           Array with values pointing to the filterName index
        lnPwv: float array
        o3: float array
        lnTau: float array
        alpha: float array
        secZenith: float array
        ccdIndex: int array
           Array with values point to the ccd index
        pmb: float array
        """

        return (filterIndex,
                np.clip(((lnPwv - self.lnPwv[0])/self.lnPwvDelta).astype(np.int32), 0,
                        self.lnPwv.size-1),
                np.clip(((o3 - self.o3[0])/self.o3Delta).astype(np.int32), 0,
                        self.o3.size-1),
                np.clip(((lnTau - self.lnTau[0])/self.lnTauDelta).astype(np.int32), 0,
                        self.lnTau.size-1),
                np.clip(((alpha - self.alpha[0])/self.alphaDelta).astype(np.int32), 0,
                        self.alpha.size-1),
                np.clip(((secZenith - self.secZenith[0])/self.secZenithDelta).astype(np.int32), 0,
                        self.secZenith.size-1),
                ccdIndex,
                (np.exp(-(pmb - self.pmbElevation)/self.pmbElevation)) ** 1.6)


    def computeI0(self, lnPwv, o3, lnTau, alpha, secZenith, pmb, indices):
        """
        Compute I0 from the look-up table.

        parameters
        ----------
        lnPwv: float array
        o3: float array
        lnTau: float array
        alpha: float array
        secZenith: float array
        pmb: float array
        indices: tuple, from getIndices()
        """

        # do a simple linear interpolation
        dlnPwv = lnPwv - (self.lnPwv[0] + indices[1] * self.lnPwvDelta)
        dO3 = o3 - (self.o3[0] + indices[2] * self.o3Delta)
        dlnTau = lnTau - (self.lnTau[0] + indices[3] * self.lnTauDelta)
        dAlpha = alpha - (self.alpha[0] + indices[4] * self.alphaDelta)
        dSecZenith = secZenith - (self.secZenith[0] + indices[5] * self.secZenithDelta)

        indicesSecZenithPlus = np.array(indices[:-1])
        indicesSecZenithPlus[5] += 1
        indicesPwvPlus = np.array(indices[:-1])
        indicesPwvPlus[1] = np.clip(indicesPwvPlus[1] + 1, 0, self.lnPwv.size-1)

        # also include cross-terms for tau and pwv
        # and a second-derivative term for pwv
        #  note that indices[-1] is the PMB vactor

        return indices[-1]*(snmm.getArray(self.lutI0Handle)[indices[:-1]] +
                            dlnPwv * snmm.getArray(self.lutDLnPwvHandle)[indices[:-1]] +
                            dO3 * snmm.getArray(self.lutDO3Handle)[indices[:-1]] +
                            dlnTau * snmm.getArray(self.lutDLnTauHandle)[indices[:-1]] +
                            dAlpha * snmm.getArray(self.lutDAlphaHandle)[indices[:-1]] +
                            dSecZenith * snmm.getArray(self.lutDSecZenithHandle)[indices[:-1]] +
                            dlnTau * dSecZenith * (snmm.getArray(self.lutDLnTauHandle)[tuple(indicesSecZenithPlus)] -
                                                   snmm.getArray(self.lutDLnTauHandle)[indices[:-1]])/self.secZenithDelta +
                            dlnPwv * dSecZenith * (snmm.getArray(self.lutDLnPwvHandle)[tuple(indicesSecZenithPlus)] -
                                                 snmm.getArray(self.lutDLnPwvHandle)[indices[:-1]])/self.secZenithDelta +
                            dlnPwv * (dlnPwv - self.lnPwvDelta) * (snmm.getArray(self.lutDLnPwvHandle)[tuple(indicesPwvPlus)] -
                                                             snmm.getArray(self.lutDLnPwvHandle)[indices[:-1]]))


    def computeI1(self, lnPwv, o3, lnTau, alpha, secZenith, pmb, indices):
        """
        Compute I1 from the look-up table.

        parameters
        ----------
        lnPwv: float array
        o3: float array
        lnTau: float array
        alpha: float array
        secZenith: float array
        pmb: float array
        indices: tuple, from getIndices()
        """

        # do a simple linear interpolation
        dlnPwv = lnPwv - (self.lnPwv[0] + indices[1] * self.lnPwvDelta)
        dO3 = o3 - (self.o3[0] + indices[2] * self.o3Delta)
        dlnTau = lnTau - (self.lnTau[0] + indices[3] * self.lnTauDelta)
        dAlpha = alpha - (self.alpha[0] + indices[4] * self.alphaDelta)
        dSecZenith = secZenith - (self.secZenith[0] + indices[5] * self.secZenithDelta)

        indicesSecZenithPlus = np.array(indices[:-1])
        indicesSecZenithPlus[5] += 1
        indicesPwvPlus = np.array(indices[:-1])
        indicesPwvPlus[1] = np.clip(indicesPwvPlus[1] + 1, 0, self.lnPwv.size-1)

        # also include a cross-term for tau
        #  note that indices[-1] is the PMB vactor

        return indices[-1]*(snmm.getArray(self.lutI1Handle)[indices[:-1]] +
                            dlnPwv * snmm.getArray(self.lutDLnPwvI1Handle)[indices[:-1]] +
                            dO3 * snmm.getArray(self.lutDO3I1Handle)[indices[:-1]] +
                            dlnTau * snmm.getArray(self.lutDLnTauI1Handle)[indices[:-1]] +
                            dAlpha * snmm.getArray(self.lutDAlphaI1Handle)[indices[:-1]] +
                            dSecZenith * snmm.getArray(self.lutDSecZenithI1Handle)[indices[:-1]] +
                            dlnTau * dSecZenith * (snmm.getArray(self.lutDLnTauI1Handle)[tuple(indicesSecZenithPlus)] -
                                                   snmm.getArray(self.lutDLnTauI1Handle)[indices[:-1]])/self.secZenithDelta +
                            dlnPwv * dSecZenith * (snmm.getArray(self.lutDLnPwvI1Handle)[tuple(indicesSecZenithPlus)] -
                                                 snmm.getArray(self.lutDLnPwvI1Handle)[indices[:-1]])/self.secZenithDelta +
                            dlnPwv * (dlnPwv - self.lnPwvDelta) * (snmm.getArray(self.lutDLnPwvI1Handle)[tuple(indicesPwvPlus)] -
                                                             snmm.getArray(self.lutDLnPwvI1Handle)[indices[:-1]]))

    def computeI1Old(self, indices):
        """
        Unused
        """
        return indices[-1] * snmm.getArray(self.lutI1Handle)[indices[:-1]]

    def computeLogDerivatives(self, indices, I0):
        """
        Compute log derivatives.  Used in FgcmChisq.

        parameters
        ----------
        indices: tuple, from getIndices()
        I0: float array, from computeI0()
        """

        # dL(i,j|p) = d/dp(2.5*log10(LUT(i,j|p)))
        #           = 1.086*(LUT'(i,j|p)/LUT(i,j|p))
        return (self.magConstant*snmm.getArray(self.lutDLnPwvHandle)[indices[:-1]] / I0,
                self.magConstant*snmm.getArray(self.lutDO3Handle)[indices[:-1]] / I0,
                self.magConstant*snmm.getArray(self.lutDLnTauHandle)[indices[:-1]] / I0,
                self.magConstant*snmm.getArray(self.lutDAlphaHandle)[indices[:-1]] / I0)


    def computeLogDerivativesI1(self, indices, I0, I10, sedSlope):
        """
        Compute log derivatives for I1.  Used in FgcmChisq.

        parameters
        ----------
        indices: tuple, from getIndices()
        I0: float array, from computeI0()
        I10: float array, from computeI1()/computeI0()
        sedSlope: float array, fnuprime
        """

        # dL(i,j|p) += d/dp(2.5*log10((1+F'*I10^obs) / (1+F'*I10^std)))
        #  the std part cancels...
        #            = 1.086*(F'/(1+F'*I10)*((I0*LUT1' - I1*LUT0')/(I0^2))

        preFactor = (self.magConstant * (sedSlope / (1 + sedSlope*I10))) / I0**2.

        return (preFactor * (I0 * snmm.getArray(self.lutDLnPwvHandle)[indices[:-1]] -
                             I10 * I0 * snmm.getArray(self.lutDLnPwvI1Handle)[indices[:-1]]),
                preFactor * (I0 * snmm.getArray(self.lutDO3Handle)[indices[:-1]] -
                             I10 * I0 * snmm.getArray(self.lutDO3I1Handle)[indices[:-1]]),
                preFactor * (I0 * snmm.getArray(self.lutDLnTauHandle)[indices[:-1]] -
                             I10 * I0 * snmm.getArray(self.lutDLnTauI1Handle)[indices[:-1]]),
                preFactor * (I0 * snmm.getArray(self.lutDAlphaHandle)[indices[:-1]] -
                             I10 * I0 * snmm.getArray(self.lutDAlphaI1Handle)[indices[:-1]]))

    def computeSEDSlopes(self, objectSedColor):
        """
        Compute SED slopes using the SED look-up table.  Experimental.

        parameters
        ----------
        objectSedColor: float array
           Color used for SED look-up (typically g-i)
        """

        indices = np.clip(np.searchsorted(self.sedColor, objectSedColor),0,self.sedColor.size-2)
        # right now, a straight matching to the nearest sedColor (g-i)
        #  though I worry about this.
        #  in fact, maybe the noise will make this not work?  Or this is real?
        #  but the noise in g-r is going to cause things to bounce around.  Pout.

        return self.sedLUT['FPRIME'][indices,:]

    def computeStepUnits(self, stepUnitReference, stepGrain, meanNightDuration,
                         meanWashIntervalDuration, fitBands, bands, nCampaignNights):
        """
        Compute normalization factors for fit step units.  Note that this might need
         to be tweaked.

        parameters
        ----------
        stepUnitReference: float
           How much should a typical step move things?  0.001 mag is default.
        stepGrain: float
           Additional fudge factor to apply to all steps.
        meanNightDuration: float
           Mean duration of a night (days).
        meanWashIntervalDuration: float
           Mean duration between washes (days).
        fitBands: string array
           Which bands are used for the fit?
        bands: string array
           What are all the bands?
        nCampaignNights: int
           Total number of nights in observing campaign to be calibrated.
        """

        unitDict = {}

        # bigger unit, smaller step

        # compute tau units

        deltaMagLnTau = (2.5*np.log10(np.exp(-self.secZenithStd*np.exp(self.lnTauStd))) -
                         2.5*np.log10(np.exp(-self.secZenithStd*np.exp(self.lnTauStd+1.0))))

        unitDict['lnTauUnit'] = np.abs(deltaMagLnTau) / stepUnitReference / stepGrain
        unitDict['lnTauUnit'] /= 5.0

        # FIXME?
        unitDict['lnTauSlopeUnit'] = unitDict['lnTauUnit'] * meanNightDuration

        # look for first use of 'g' or 'r' band in filterToBand...
        #  this is the reference filter for tau/alpha

        alphaFilterIndex = -1
        for i,filterName in enumerate(self.filterNames):
            if (self.filterToBand[filterName] == 'g' or
                self.filterToBand[filterName] == 'r'):
                alphaFilterIndex = i
                break

        if alphaFilterIndex == -1:
            # We don't have anything here...
            # Just set this to 1.0, since it's not sensitive?
            unitDict['alphaUnit'] = 1.0 / stepUnitReference / stepGrain
        else:
            deltaMagAlpha = (2.5*np.log10(np.exp(-self.secZenithStd*self.tauStd*(self.lambdaStd[alphaFilterIndex]/self.lambdaNorm)**self.alphaStd)) -
                             2.5*np.log10(np.exp(-self.secZenithStd*self.tauStd*(self.lambdaStd[alphaFilterIndex]/self.lambdaNorm)**(self.alphaStd+1.0))))
            unitDict['alphaUnit'] = np.abs(deltaMagAlpha) / stepUnitReference / stepGrain

            # and scale these by fraction of bands affected...
            alphaNAffectedBands = 0
            for filterName in self.filterNames:
                if ((self.filterToBand[filterName] == 'u' and
                     'u' in fitBands) or
                    (self.filterToBand[filterName] == 'g' and
                     'g' in fitBands) or
                    (self.filterToBand[filterName] == 'r' and
                     'r' in fitBands)):
                    alphaNAffectedBands += 1

            unitDict['alphaUnit'] *= float(alphaNAffectedBands) / float(len(fitBands))

        # pwv units -- reference to z or y or Y
        pwvFilterIndex = -1
        for i,filterName in enumerate(self.filterNames):
            if (self.filterToBand[filterName] == 'z' or
                self.filterToBand[filterName] == 'y' or
                self.filterToBand[filterName] == 'Y'):
                pwvFilterIndex = i
                break

        if pwvFilterIndex == -1:
            unitDict['lnPwvUnit'] = 1.0 / stepUnitReference / stepGrain
        else:
            indicesStd = self.getIndices(pwvFilterIndex,np.log(self.pwvStd),self.o3Std,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.nCCD,self.pmbStd)
            i0Std = self.computeI0(np.log(self.pwvStd),self.o3Std,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.pmbStd,indicesStd)

            # Want the change from one step unit
            indicesMinus = self.getIndices(pwvFilterIndex,np.log(self.pwvStd)-1.0,self.o3Std,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.nCCD,self.pmbStd)
            i0Minus = self.computeI0(np.log(self.pwvStd)-1.0,self.o3Std,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.pmbStd,indicesMinus)
            deltaMagPwv = 2.5*np.log10(i0Minus) - 2.5*np.log10(i0Std)

            unitDict['lnPwvUnit'] = np.abs(deltaMagPwv) / stepUnitReference / stepGrain

            # scale by fraction of bands affected
            pwvNAffectedBands = 0
            for filterName in self.filterNames:
                if ((self.filterToBand[filterName] == 'z' and
                     'z' in fitBands) or
                    (self.filterToBand[filterName] == 'y' and
                     'y' in fitBands) or
                    (self.filterToBand[filterName] == 'Y' and
                     'Y' in fitBands)):
                    pwvNAffectedBands += 1
            unitDict['lnPwvUnit'] *= float(pwvNAffectedBands) / float(len(fitBands))

        # PWV slope units
        unitDict['lnPwvSlopeUnit'] = unitDict['lnPwvUnit'] * meanNightDuration
        unitDict['lnPwvQuadraticUnit'] = unitDict['lnPwvUnit'] * meanNightDuration**2.

        # PWV Global step units
        unitDict['lnPwvGlobalUnit'] = unitDict['lnPwvUnit'] * nCampaignNights

        # O3 units -- reference to r
        o3FilterIndex = -1
        for i,filterName in enumerate(self.filterNames):
            if (self.filterToBand[filterName] == 'u' or
                self.filterToBand[filterName] == 'r'):
                o3FilterIndex = i
                break

        if o3FilterIndex == -1:
            unitDict['o3Unit'] = 1.0 / stepUnitReference / stepGrain
        else:
            indicesStd = self.getIndices(o3FilterIndex,np.log(self.pwvStd),self.o3Std,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.nCCD,self.pmbStd)
            i0Std = self.computeI0(np.log(self.pwvStd),self.o3Std,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.pmbStd,indicesStd)
            indicesPlus = self.getIndices(o3FilterIndex,np.log(self.pwvStd),self.o3Std+1.0,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.nCCD,self.pmbStd)
            i0Plus = self.computeI0(np.log(self.pwvStd),self.o3Std+1.0,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.pmbStd,indicesPlus)
            deltaMagO3 = 2.5*np.log10(i0Std) - 2.5*np.log10(i0Plus)

            unitDict['o3Unit'] = np.abs(deltaMagO3) / stepUnitReference / stepGrain

            # scale by fraction of bands that are affected
            o3NAffectedBands = 0
            for filterName in self.filterNames:
                if (self.filterToBand[filterName] == 'u' or
                    self.filterToBand[filterName] == 'r'):
                    o3NAffectedBands += 1
            unitDict['o3Unit'] *= float(o3NAffectedBands) / float(len(fitBands))

        # wash parameters units...
        unitDict['qeSysUnit'] = 1.0 / stepUnitReference / stepGrain
        unitDict['qeSysSlopeUnit'] = unitDict['qeSysUnit'] * meanWashIntervalDuration

        # And filter offset units...
        # Unsure about this, we might need to get fancy per filter about overlaps
        # But this is going to be roughly in the right direction, I hope.
        unitDict['filterOffsetUnit'] = 1.0 / stepUnitReference / stepGrain

        # Test this out
        unitDict['absOffsetUnit'] = 1.0 / stepUnitReference / stepGrain / 1000.
        unitDict['refOnlyAbsOffsetUnit'] = 1.0 / stepUnitReference / stepGrain

        return unitDict


