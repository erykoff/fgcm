from __future__ import print_function

import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import os
import sys
from pkg_resources import resource_filename


from modtranGenerator import ModtranGenerator

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

## FIXME: add better logging


class FgcmLUTMaker(object):
    """
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

    def _checkLUTConfig(self,lutConfig):
        """
        """

        requiredKeys=['elevation','bands','nCCD',
                      'pmbRange','pmbSteps',
                      'pwvRange','pwvSteps',
                      'o3Range','o3Steps',
                      'tauRange','tauSteps',
                      'alphaRange','alphaSteps',
                      'zenithRange','zenithSteps',
                      'pmbStd','pwvStd','o3Std',
                      'tauStd','alphaStd','airmassStd']

        for key in requiredKeys:
            if (key not in lutConfig):
                raise ValueError("required %s not in lutConfig" % (key))
            if ('Range' in key):
                if (len(lutConfig[key]) != 2):
                    raise ValueError("%s must have 2 elements" % (key))

        self.lutConfig = lutConfig

        # this will generate an exception if things aren't set up properly
        self.modGen = ModtranGenerator(self.lutConfig['elevation'])
        self.pmbElevation = self.modGen.pmbElevation

        self.bands = np.array(self.lutConfig['bands'])

        self.nCCD = self.lutConfig['nCCD']
        self.nCCDStep = self.nCCD+1

        # and record the standard values out of the config
        #  (these will also come out of the save file)
        self.pmbStd = self.lutConfig['pmbStd']
        self.pwvStd = self.lutConfig['pwvStd']
        self.o3Std = self.lutConfig['o3Std']
        self.tauStd = self.lutConfig['tauStd']
        self.alphaStd = self.lutConfig['alphaStd']
        self.secZenithStd = self.lutConfig['airmassStd']
        self.zenithStd = np.arccos(1./self.secZenithStd)*180./np.pi

        if ('lambdaRange' in self.lutConfig):
            self.lambdaRange = np.array(self.lutConfig['lambdaRange'])
        else:
            self.lambdaRange = np.array([3000.0,11000.0])

        if ('lambdaStep' in self.lutConfig):
            self.lambdaStep = self.lutConfig['lambdaStep']
        else:
            self.lambdaStep = 0.5

        if ('lambdaNorm' in self.lutConfig):
            self.lambdaNorm = self.lutConfig['lambdaNorm']
        else:
            self.lambdaNorm = 7750.0

    def setThroughputs(self, throughputDict):
        """
        """

        self.inThroughputs=[]
        for b in self.bands:
            try:
                lam = throughputDict[b]['LAMBDA']
            except:
                raise ValueError("Wavelength LAMBDA not found for band %s in throughputDict!" % (b))

            tput = np.zeros(lam.size, dtype=[('LAMBDA','f4'),
                                             ('THROUGHPUT_AVG','f4'),
                                             ('THROUGHPUT_CCD','f4',self.nCCD)])
            tput['LAMBDA'][:] = lam
            for ccdIndex in xrange(self.nCCD):
                try:
                    tput['THROUGHPUT_CCD'][:,ccdIndex] = throughputDict[b][ccdIndex]
                except:
                    raise ValueError("CCD Index %d not found for band %s in throughputDict!" % (ccdIndex,b))

            # check if the average is there, if not compute it
            if ('AVG' in throughputDict[b]):
                tput['THROUGHPUT_AVG'][:] = throughputDict[b]['AVG']
            else:
                print("Average throughput not found in throughputDict for band %s.  Computing now..." % (b))
                for i in xrange(lam.size):
                    use,=np.where(tput['THROUGHPUT_CCD'][i,:] > 0.0)
                    if (use.size > 0):
                        tput['THROUGHPUT_AVG'][i] = np.mean(tput['THROUGHPUT_CCD'][i,use])

            self.inThroughputs.append(tput)

        self._setThroughput = True

    def makeLUT(self):
        """
        """

        if not self._setThroughput:
            raise ValueError("Must set the throughput before running makeLUT")

        # we need a standard atmosphere and lambdas...
        self.atmStd = self.modGen(pmb=self.pmbStd,pwv=self.pwvStd,
                                  o3=self.o3Std,tau=self.tauStd,
                                  alpha=self.alphaStd,zenith=self.zenithStd,
                                  lambdaRange=self.lambdaRange/10.0,
                                  lambdaStep=self.lambdaStep)
        self.atmLambda = self.atmStd['LAMBDA']
        self.atmStdTrans = self.atmStd['COMBINED']

        # get all the steps
        self.pmb = np.linspace(self.lutConfig['pmbRange'][0],
                               self.lutConfig['pmbRange'][1],
                               num=self.lutConfig['pmbSteps'])
        self.pmbDelta = self.pmb[1] - self.pmb[0]
        pmbPlus = np.append(self.pmb, self.pmb[-1] + self.pmbDelta)

        self.pwv = np.linspace(self.lutConfig['pwvRange'][0],
                               self.lutConfig['pwvRange'][1],
                               num=self.lutConfig['pwvSteps'])
        self.pwvDelta = self.pwv[1] - self.pwv[0]
        pwvPlus = np.append(self.pwv, self.pwv[-1] + self.pwvDelta)

        self.o3 = np.linspace(self.lutConfig['o3Range'][0],
                               self.lutConfig['o3Range'][1],
                               num=self.lutConfig['o3Steps'])
        self.o3Delta = self.o3[1] - self.o3[0]
        o3Plus = np.append(self.o3, self.o3[-1] + self.o3Delta)

        self.lnTau = np.linspace(np.log(self.lutConfig['tauRange'][0]),
                                 np.log(self.lutConfig['tauRange'][1]),
                                 num=self.lutConfig['tauSteps'])
        self.lnTauDelta = self.lnTau[1] - self.lnTau[0]
        self.tau = np.exp(self.lnTau)
        lnTauPlus = np.append(self.lnTau, self.lnTau[-1] + self.lnTauDelta)
        tauPlus = np.exp(lnTauPlus)

        self.alpha = np.linspace(self.lutConfig['alphaRange'][0],
                               self.lutConfig['alphaRange'][1],
                               num=self.lutConfig['alphaSteps'])
        self.alphaDelta = self.alpha[1] - self.alpha[0]
        alphaPlus = np.append(self.alpha, self.alpha[-1] + self.alphaDelta)

        self.secZenith = np.linspace(1./np.cos(self.lutConfig['zenithRange'][0]*np.pi/180.),
                                     1./np.cos(self.lutConfig['zenithRange'][1]*np.pi/180.),
                                     num=self.lutConfig['zenithSteps'])
        self.secZenithDelta = self.secZenith[1]-self.secZenith[0]
        self.zenith = np.arccos(1./self.secZenith)*180./np.pi
        secZenithPlus = np.append(self.secZenith, self.secZenith[-1] + self.secZenithDelta)
        zenithPlus = np.arccos(1./secZenithPlus)*180./np.pi

        # and compute the proper airmass...
        self.airmass = self.secZenith - 0.0018167*(self.secZenith-1.0) - 0.002875*(self.secZenith-1.0)**2.0 - 0.0008083*(self.secZenith-1.0)**3.0
        airmassPlus = secZenithPlus - 0.0018167*(secZenithPlus-1.0) - 0.002875*(secZenithPlus-1.0)**2.0 - 0.0008083*(secZenithPlus-1.0)**3.0

        # run MODTRAN a bunch of times
        # we need for each airmass, to run the array of pwv and o3 and pull these out

        #print("Generating %d*%d=%d PWV atmospheres..." % (self.pwv.size,self.zenith.size,self.pwv.size*self.zenith.size))
        print("Generating %d*%d=%d PWV atmospheres..." % (pwvPlus.size,zenithPlus.size,pwvPlus.size*zenithPlus.size))
        #self.pwvAtmTable = np.zeros((self.pwv.size,self.zenith.size,self.atmLambda.size))
        pwvAtmTable = np.zeros((pwvPlus.size,zenithPlus.size,self.atmLambda.size))

        #for i in xrange(self.pwv.size):
        for i in xrange(pwvPlus.size):
            sys.stdout.write('%d' % (i))
            sys.stdout.flush()
            #for j in xrange(self.zenith.size):
            for j in xrange(zenithPlus.size):
                sys.stdout.write('.')
                sys.stdout.flush()
                atm=self.modGen(pwv=pwvPlus[i],zenith=zenithPlus[j],
                                lambdaRange=self.lambdaRange/10.0,
                                lambdaStep=self.lambdaStep)
                pwvAtmTable[i,j,:] = atm['H2O']

        print("\nGenerating %d*%d=%d O3 atmospheres..." % (o3Plus.size,zenithPlus.size,o3Plus.size*zenithPlus.size))
        #self.o3AtmTable = np.zeros((self.o3.size,self.zenith.size,self.atmLambda.size))
        o3AtmTable = np.zeros((o3Plus.size, zenithPlus.size, self.atmLambda.size))

        for i in xrange(o3Plus.size):
            sys.stdout.write('%d' % (i))
            sys.stdout.flush()
            for j in xrange(zenithPlus.size):
                sys.stdout.write('.')
                sys.stdout.flush()
                atm=self.modGen(o3=o3Plus[i],zenith=zenithPlus[j],
                                lambdaRange=self.lambdaRange/10.0,
                                lambdaStep=self.lambdaStep)
                o3AtmTable[i,j,:] = atm['O3']

        print("\nGenerating %d O2/Rayleigh atmospheres..." % (zenithPlus.size))
        #self.o2AtmTable = np.zeros((self.zenith.size,self.atmLambda.size))
        #self.rayleighAtmTable = np.zeros((self.zenith.size,self.atmLambda.size))
        o2AtmTable = np.zeros((zenithPlus.size, self.atmLambda.size))
        rayleighAtmTable = np.zeros((zenithPlus.size, self.atmLambda.size))

        for j in xrange(zenithPlus.size):
            sys.stdout.write('.')
            sys.stdout.flush()
            atm=self.modGen(zenith=zenithPlus[j],
                            lambdaRange=self.lambdaRange/10.0,
                            lambdaStep=self.lambdaStep)
            o2AtmTable[j,:] = atm['O2']
            rayleighAtmTable[j,:] = atm['RAYLEIGH']

        # get the filters over the same lambda ranges...
        print("\nInterpolating filters...")
        self.throughputs = []
        for i in xrange(len(self.bands)):
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
        print("Computing lambdaB")
        self.lambdaB = np.zeros(self.bands.size)
        for i in xrange(self.bands.size):
            num = integrate.simps(self.atmLambda * self.throughputs[i]['THROUGHPUT_AVG'] / self.atmLambda, self.atmLambda)
            denom = integrate.simps(self.throughputs[i]['THROUGHPUT_AVG'] / self.atmLambda, self.atmLambda)
            self.lambdaB[i] = num / denom
            print("Band: %s, lambdaB = %.3f" % (self.bands[i], self.lambdaB[i]))

        print("Computing lambdaStd")
        self.lambdaStd = np.zeros(self.bands.size)
        for i in xrange(self.bands.size):
            num = integrate.simps(self.atmLambda * self.throughputs[i]['THROUGHPUT_AVG'] * self.atmStdTrans / self.atmLambda, self.atmLambda)
            denom = integrate.simps(self.throughputs[i]['THROUGHPUT_AVG'] * self.atmStdTrans / self.atmLambda, self.atmLambda)
            self.lambdaStd[i] = num / denom
            print("Band: %s, lambdaStd = %.3f" % (self.bands[i],self.lambdaStd[i]))


        print("Computing I0Std/I1Std")
        self.I0Std = np.zeros(self.bands.size)
        self.I1Std = np.zeros(self.bands.size)

        for i in xrange(self.bands.size):
            self.I0Std[i] = integrate.simps(self.throughputs[i]['THROUGHPUT_AVG'] * self.atmStdTrans / self.atmLambda, self.atmLambda)
            self.I1Std[i] = integrate.simps(self.throughputs[i]['THROUGHPUT_AVG'] * self.atmStdTrans * (self.atmLambda - self.lambdaStd[i]) / self.atmLambda, self.atmLambda)

        self.I10Std = self.I1Std / self.I0Std

        #################################
        ## Make the I0/I1 LUT
        #################################

        print("Building look-up table...")
        lutPlus = np.zeros((self.bands.size,
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
        for i in xrange(self.bands.size):
            print("Working on band %s" % (self.bands[i]))
            for j in xrange(pwvPlus.size):
                print("  and on pwv #%d" % (j))
                for k in xrange(o3Plus.size):
                    print("   and on o3 #%d" % (k))
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
        self.lut = np.zeros((self.bands.size,
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

        self.lutDeriv = np.zeros((self.bands.size,
                                  self.pwv.size,
                                  self.o3.size,
                                  self.tau.size,
                                  self.alpha.size,
                                  self.zenith.size,
                                  self.nCCDStep),
                                 dtype=[('D_PMB','f4'),
                                        ('D_PWV','f4'),
                                        ('D_O3','f4'),
                                        ('D_LNTAU','f4'),
                                        ('D_ALPHA','f4'),
                                        ('D_SECZENITH','f4'),
                                        ('D_PMB_I1','f4'),
                                        ('D_PWV_I1','f4'),
                                        ('D_O3_I1','f4'),
                                        ('D_LNTAU_I1','f4'),
                                        ('D_ALPHA_I1','f4'),
                                        ('D_SECZENITH_I1','f4')])

        print("Computing derivatives...")

        ## FIXME: figure out PMB derivative?

        for i in xrange(self.bands.size):
            print("Working on band %s" % (self.bands[i]))
            for j in xrange(self.pwv.size):
                for k in xrange(self.o3.size):
                    for m in xrange(self.tau.size):
                        for n in xrange(self.alpha.size):
                            for o in xrange(self.zenith.size):
                                for p in xrange(self.nCCDStep):
                                    self.lutDeriv['D_PWV'][i,j,k,m,n,o,p] = (
                                        ((lutPlus['I0'][i,j+1,k,m,n,o,p] -
                                          lutPlus['I0'][i,j,k,m,n,o,p]) /
                                         self.pwvDelta)
                                        )
                                    self.lutDeriv['D_PWV_I1'][i,j,k,m,n,o,p] = (
                                        ((lutPlus['I1'][i,j+1,k,m,n,o,p] -
                                          lutPlus['I1'][i,j,k,m,n,o,p]) /
                                         self.pwvDelta)
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
        ## and the SED LUT
            print("Building SED LUT")

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
                                                      ('SYNTHMAG','f4',self.bands.size),
                                                      ('FPRIME','f4',self.bands.size)])

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
                for j in xrange(self.bands.size):
                    num = integrate.simps(fnu * self.throughputs[j]['THROUGHPUT_AVG'][:] * self.atmStdTrans / self.atmLambda, self.atmLambda)
                    denom = integrate.simps(self.throughputs[j]['THROUGHPUT_AVG'][:] * self.atmStdTrans / self.atmLambda, self.atmLambda)

                    self.sedLUT['SYNTHMAG'][i,j] = -2.5*np.log10(num/denom)

                # and compute fprimes
                for j in xrange(self.bands.size):
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
            print("lutFile %s already exists, and clobber is False.")
            return

        print("Saving LUT to %s" % (lutFile))

        # first, save the LUT itself
        fitsio.write(lutFile,self.lut.flatten(),extname='LUT',clobber=True)

        # and now save the indices
        indexVals = np.zeros(1,dtype=[('BANDS','a1',self.bands.size),
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
        indexVals['BANDS'] = self.bands
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
                                    ('LAMBDASTD','f8',self.bands.size),
                                    ('LAMBDANORM','f8'),
                                    ('I0STD','f8',self.bands.size),
                                    ('I1STD','f8',self.bands.size),
                                    ('I10STD','f8',self.bands.size),
                                    ('LAMBDAB','f8',self.bands.size),
                                    ('ATMLAMBDA','f8',self.atmLambda.size),
                                    ('ATMSTDTRANS','f8',self.atmStd.size)])
        stdVals['PMBSTD'] = self.pmbStd
        stdVals['PWVSTD'] = self.pwvStd
        stdVals['O3STD'] = self.o3Std
        stdVals['TAUSTD'] = self.tauStd
        stdVals['ALPHASTD'] = self.alphaStd
        stdVals['ZENITHSTD'] = self.zenithStd
        stdVals['LAMBDARANGE'] = self.lambdaRange
        stdVals['LAMBDASTEP'] = self.lambdaStep
        stdVals['LAMBDASTD'][:] = self.lambdaStd
        stdVals['LAMBDANORM'][:] = self.lambdaNorm
        stdVals['I0STD'][:] = self.I0Std
        stdVals['I1STD'][:] = self.I1Std
        stdVals['I10STD'][:] = self.I10Std
        stdVals['LAMBDAB'][:] = self.lambdaB
        stdVals['ATMLAMBDA'][:] = self.atmLambda
        stdVals['ATMSTDTRANS'][:] = self.atmStdTrans

        fitsio.write(lutFile,stdVals,extname='STD')

        # and the derivatives
        print("Writing Derivative LUT")
        fitsio.write(lutFile,self.lutDeriv.flatten(),extname='DERIV')

        # and the SED LUT
        print("Writing SED LUT")
        fitsio.write(lutFile,self.sedLUT,extname='SED')


class FgcmLUT(object):
    """
    """

    #def __init__(self,lutFile):
    def __init__(self, indexVals, lutFlat, lutDerivFlat, stdVals, sedLUT=None):
        #lutFlat = fitsio.read(self.lutFile,ext='LUT')
        #indexVals = fitsio.read(self.lutFile,ext='INDEX')

        self.bands = indexVals['BANDS'][0]
        self.pmb = indexVals['PMB'][0]
        self.pmbFactor = indexVals['PMBFACTOR'][0]
        self.pmbDelta = self.pmb[1] - self.pmb[0]
        self.pmbElevation = indexVals['PMBELEVATION'][0]
        self.lambdaNorm = indexVals['LAMBDANORM'][0]

        self.pwv = indexVals['PWV'][0]
        self.pwvDelta = self.pwv[1] - self.pwv[0]
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
        sizeTuple = (self.bands.size,self.pwv.size,self.o3.size,
                     self.tau.size,self.alpha.size,self.zenith.size,self.nCCDStep)

        self.lutI0Handle = snmm.createArray(sizeTuple,dtype='f4')
        snmm.getArray(self.lutI0Handle)[:,:,:,:,:,:,:] = lutFlat['I0'].reshape(sizeTuple)

        self.lutI1Handle = snmm.createArray(sizeTuple,dtype='f4')
        snmm.getArray(self.lutI1Handle)[:,:,:,:,:,:,:] = lutFlat['I1'].reshape(sizeTuple)

        # and read in the derivatives
        #lutDerivFlat = fitsio.read(self.lutFile,ext='DERIV')

        # create shared memory
        self.lutDPWVHandle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDO3Handle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDLnTauHandle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDAlphaHandle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDSecZenithHandle = snmm.createArray(sizeTuple,dtype='f4')

        self.lutDPWVI1Handle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDO3I1Handle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDLnTauI1Handle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDAlphaI1Handle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDSecZenithI1Handle = snmm.createArray(sizeTuple,dtype='f4')

        snmm.getArray(self.lutDPWVHandle)[:,:,:,:,:,:,:] = lutDerivFlat['D_PWV'].reshape(sizeTuple)
        snmm.getArray(self.lutDO3Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_O3'].reshape(sizeTuple)
        snmm.getArray(self.lutDLnTauHandle)[:,:,:,:,:,:,:] = lutDerivFlat['D_LNTAU'].reshape(sizeTuple)
        snmm.getArray(self.lutDAlphaHandle)[:,:,:,:,:,:,:] = lutDerivFlat['D_ALPHA'].reshape(sizeTuple)
        snmm.getArray(self.lutDSecZenithHandle)[:,:,:,:,:,:,:] = lutDerivFlat['D_SECZENITH'].reshape(sizeTuple)

        self.hasI1Derivatives = False
        try:
            snmm.getArray(self.lutDPWVI1Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_PWV_I1'].reshape(sizeTuple)
            snmm.getArray(self.lutDO3I1Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_O3_I1'].reshape(sizeTuple)
            snmm.getArray(self.lutDLnTauI1Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_LNTAU_I1'].reshape(sizeTuple)
            snmm.getArray(self.lutDAlphaI1Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_ALPHA_I1'].reshape(sizeTuple)
            snmm.getArray(self.lutDSecZenithI1Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_SECZENITH_I1'].reshape(sizeTuple)
            self.hasI1Derivatives = True
        except:
            # just fill with zeros
            print("No I1 derivative information")

        # get the standard values
        #stdVals = fitsio.read(lutFile,ext='STD')

        self.pmbStd = stdVals['PMBSTD'][0]
        self.pwvStd = stdVals['PWVSTD'][0]
        self.o3Std = stdVals['O3STD'][0]
        self.tauStd = stdVals['TAUSTD'][0]
        self.alphaStd = stdVals['ALPHASTD'][0]
        self.zenithStd = stdVals['ZENITHSTD'][0]
        self.secZenithStd = 1./np.cos(np.radians(self.zenithStd))
        self.lambdaRange = stdVals['LAMBDARANGE'][0]
        self.lambdaStep = stdVals['LAMBDASTEP'][0]
        self.lambdaStd = stdVals['LAMBDASTD'][0]
        self.I0Std = stdVals['I0STD'][0]
        self.I1Std = stdVals['I1STD'][0]
        self.I10Std = stdVals['I10STD'][0]
        self.lambdaB = stdVals['LAMBDAB'][0]
        self.atmLambda = stdVals['ATMLAMBDA'][0]
        self.atmStdTrans = stdVals['ATMSTDTRANS'][0]

        self.magConstant = 2.5/np.log(10)

        # finally, read in the sedLUT
        ## this is experimental
        self.hasSedLUT = False
        if (sedLUT is not None):
            self.sedLUT = sedLUT

            self.hasSedLUT = True
        #try:
        #    self.sedLUT = fitsio.read(lutFile,ext='SED')
        #    self.hasSedLUT = True
        #except:
        #    print("Warning: Could not find SED LUT in lutfile.")

        #if (self.hasSedLUT):
            # this is currently *not* general, but quick
            ## FIXME: make general
            self.sedColor = self.sedLUT['SYNTHMAG'][:,0] - self.sedLUT['SYNTHMAG'][:,2]
            st = np.argsort(self.sedColor)

            self.sedColor = self.sedColor[st]
            self.sedLUT = self.sedLUT[st]

    @classmethod
    def initFromFits(cls, lutFile):
        """
        """

        import fitsio

        lutFlat = fitsio.read(lutFile, ext='LUT')
        lutDerivFlat = fitsio.read(lutFile, ext='DERIV')
        indexVals = fitsio.read(lutFile, ext='INDEX')
        stdVals = fitsio.read(lutFile, ext='STD')

        try:
            sedLUT = fitsio.read(lutFile, ext='SED')
        except:
            sedLUT = None

        return cls(indexVals, lutFlat, lutDerivFlat, stdVals, sedLUT=sedLUT)


    def getIndices(self, bandIndex, pwv, o3, lnTau, alpha, secZenith, ccdIndex, pmb):
        """
        """

        return (bandIndex,
                np.clip(((pwv - self.pwv[0])/self.pwvDelta).astype(np.int32), 0,
                        self.pwv.size-1),
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


    def computeI0(self, pwv, o3, lnTau, alpha, secZenith, pmb, indices):
        """
        """

        # do a simple linear interpolation
        dPWV = pwv - (self.pwv[0] + indices[1] * self.pwvDelta)
        dO3 = o3 - (self.o3[0] + indices[2] * self.o3Delta)
        dlnTau = lnTau - (self.lnTau[0] + indices[3] * self.lnTauDelta)
        dAlpha = alpha - (self.alpha[0] + indices[4] * self.alphaDelta)
        dSecZenith = secZenith - (self.secZenith[0] + indices[5] * self.secZenithDelta)

        indicesSecZenithPlus = np.array(indices[:-1])
        indicesSecZenithPlus[5] += 1
        indicesPWVPlus = np.array(indices[:-1])
        indicesPWVPlus[1] += 1

        # also include cross-terms for tau and pwv
        # and a second-derivative term for pwv
        #  note that indices[-1] is the PMB vactor

        return indices[-1]*(snmm.getArray(self.lutI0Handle)[indices[:-1]] +
                            dPWV * snmm.getArray(self.lutDPWVHandle)[indices[:-1]] +
                            dO3 * snmm.getArray(self.lutDO3Handle)[indices[:-1]] +
                            dlnTau * snmm.getArray(self.lutDLnTauHandle)[indices[:-1]] +
                            dAlpha * snmm.getArray(self.lutDAlphaHandle)[indices[:-1]] +
                            dSecZenith * snmm.getArray(self.lutDSecZenithHandle)[indices[:-1]] +
                            dlnTau * dSecZenith * (snmm.getArray(self.lutDLnTauHandle)[tuple(indicesSecZenithPlus)] -
                                                   snmm.getArray(self.lutDLnTauHandle)[indices[:-1]])/self.secZenithDelta +
                            dPWV * dSecZenith * (snmm.getArray(self.lutDPWVHandle)[tuple(indicesSecZenithPlus)] -
                                                 snmm.getArray(self.lutDPWVHandle)[indices[:-1]])/self.secZenithDelta +
                            dPWV * (dPWV - self.pwvDelta) * (snmm.getArray(self.lutDPWVHandle)[tuple(indicesPWVPlus)] -
                                                             snmm.getArray(self.lutDPWVHandle)[indices[:-1]]))


    def computeI1(self, pwv, o3, lnTau, alpha, secZenith, pmb, indices):
        # do a simple linear interpolation
        dPWV = pwv - (self.pwv[0] + indices[1] * self.pwvDelta)
        dO3 = o3 - (self.o3[0] + indices[2] * self.o3Delta)
        dlnTau = lnTau - (self.lnTau[0] + indices[3] * self.lnTauDelta)
        dAlpha = alpha - (self.alpha[0] + indices[4] * self.alphaDelta)
        dSecZenith = secZenith - (self.secZenith[0] + indices[5] * self.secZenithDelta)

        indicesSecZenithPlus = np.array(indices[:-1])
        indicesSecZenithPlus[5] += 1
        indicesPWVPlus = np.array(indices[:-1])
        indicesPWVPlus[1] += 1

        # also include a cross-term for tau
        #  note that indices[-1] is the PMB vactor

        return indices[-1]*(snmm.getArray(self.lutI1Handle)[indices[:-1]] +
                            dPWV * snmm.getArray(self.lutDPWVI1Handle)[indices[:-1]] +
                            dO3 * snmm.getArray(self.lutDO3I1Handle)[indices[:-1]] +
                            dlnTau * snmm.getArray(self.lutDLnTauI1Handle)[indices[:-1]] +
                            dAlpha * snmm.getArray(self.lutDAlphaI1Handle)[indices[:-1]] +
                            dSecZenith * snmm.getArray(self.lutDSecZenithI1Handle)[indices[:-1]] +
                            dlnTau * dSecZenith * (snmm.getArray(self.lutDLnTauI1Handle)[tuple(indicesSecZenithPlus)] -
                                                   snmm.getArray(self.lutDLnTauI1Handle)[indices[:-1]])/self.secZenithDelta +
                            dPWV * dSecZenith * (snmm.getArray(self.lutDPWVI1Handle)[tuple(indicesSecZenithPlus)] -
                                                 snmm.getArray(self.lutDPWVI1Handle)[indices[:-1]])/self.secZenithDelta +
                            dPWV * (dPWV - self.pwvDelta) * (snmm.getArray(self.lutDPWVI1Handle)[tuple(indicesPWVPlus)] -
                                                             snmm.getArray(self.lutDPWVI1Handle)[indices[:-1]]))

    def computeI1Old(self, indices):
        return indices[-1] * snmm.getArray(self.lutI1Handle)[indices[:-1]]

    def computeLogDerivatives(self, indices, I0, tau):
        # dL(i,j|p) = d/dp(2.5*log10(LUT(i,j|p)))
        #           = 1.086*(LUT'(i,j|p)/LUT(i,j|p))
        return (self.magConstant*snmm.getArray(self.lutDPWVHandle)[indices[:-1]] / I0,
                self.magConstant*snmm.getArray(self.lutDO3Handle)[indices[:-1]] / I0,
                self.magConstant*snmm.getArray(self.lutDLnTauHandle)[indices[:-1]] / (I0*tau), # ln space
                self.magConstant*snmm.getArray(self.lutDAlphaHandle)[indices[:-1]] / I0)


    def computeLogDerivativesI1(self, indices, I0, I10, sedSlope, tau):
        # dL(i,j|p) += d/dp(2.5*log10((1+F'*I10^obs) / (1+F'*I10^std)))
        #  the std part cancels...
        #            = 1.086*(F'/(1+F'*I10)*((I0*LUT1' - I1*LUT0')/(I0^2))

        preFactor = (self.magConstant * (sedSlope / (1 + sedSlope*I10))) / I0**2.

        return (preFactor * (I0 * snmm.getArray(self.lutDPWVHandle)[indices[:-1]] -
                             I10 * I0 * snmm.getArray(self.lutDPWVI1Handle)[indices[:-1]]),
                preFactor * (I0 * snmm.getArray(self.lutDO3Handle)[indices[:-1]] -
                             I10 * I0 * snmm.getArray(self.lutDO3I1Handle)[indices[:-1]]),
                preFactor * (I0 * snmm.getArray(self.lutDLnTauHandle)[indices[:-1]] -
                             I10 * I0 * snmm.getArray(self.lutDLnTauI1Handle)[indices[:-1]]) /
                (tau),
                preFactor * (I0 * snmm.getArray(self.lutDAlphaHandle)[indices[:-1]] -
                             I10 * I0 * snmm.getArray(self.lutDAlphaI1Handle)[indices[:-1]]))

    def computeSEDSlopes(self, objectSedColor):
        """
        """

        indices = np.clip(np.searchsorted(self.sedColor, objectSedColor),0,self.sedColor.size-2)
        # right now, a straight matching to the nearest sedColor (g-i)
        #  though I worry about this.
        #  in fact, maybe the noise will make this not work?  Or this is real?
        #  but the noise in g-r is going to cause things to bounce around.  Pout.

        return self.sedLUT['FPRIME'][indices,:]

    def computeStepUnits(self, stepUnitReference, stepGrain, meanNightDuration,
                         meanWashIntervalDuration, fitBands):
        """
        """

        unitDict = {}


        # compute tau units
        deltaMagTau = (2.5*np.log10(np.exp(-self.secZenithStd*self.tauStd)) -
                       2.5*np.log10(np.exp(-self.secZenithStd*(self.tauStd+1.0))))

        unitDict['tauUnit'] = np.abs(deltaMagTau) / stepUnitReference / stepGrain

        # tau percent slope units
        ## FIXME: check these
        unitDict['tauPerSlopeUnit'] = (unitDict['tauUnit'] * meanNightDuration /
                                            self.tauStd)

        # alpha units -- reference to g, or r if not available
        bandIndex, = np.where(self.bands == 'g')
        if bandIndex.size == 0:
            bandIndex, = np.where(self.bands == 'r')
            if bandIndex.size == 0:
                raise ValueError("Must have either g or r band...")

        deltaMagAlpha = (2.5*np.log10(np.exp(-self.secZenithStd*self.tauStd*(self.lambdaStd[bandIndex]/self.lambdaNorm)**self.alphaStd)) -
                         2.5*np.log10(np.exp(-self.secZenithStd*self.tauStd*(self.lambdaStd[bandIndex]/self.lambdaNorm)**(self.alphaStd+1.0))))
        unitDict['alphaUnit'] = np.abs(deltaMagAlpha[0]) / stepUnitReference / stepGrain

        # and scale these by fraction of bands affected...
        use, = np.where((fitBands == 'u') |
                        (fitBands == 'g') |
                        (fitBands == 'r'))
        unitDict['alphaUnit'] *= float(use.size) / float(fitBands.size)

        # pwv units -- reference to z
        bandIndex, = np.where(self.bands == 'z')
        if bandIndex.size == 0:
            raise ValueError("Require z band for PWV...")

        indicesStd = self.getIndices(bandIndex,self.pwvStd,self.o3Std,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.nCCD,self.pmbStd)
        i0Std = self.computeI0(self.pwvStd,self.o3Std,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.pmbStd,indicesStd)
        indicesPlus = self.getIndices(bandIndex,self.pwvStd+1.0,self.o3Std,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.nCCD,self.pmbStd)
        i0Plus = self.computeI0(self.pwvStd+1.0,self.o3Std,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.pmbStd,indicesPlus)
        deltaMagPWV = 2.5*np.log10(i0Std) - 2.5*np.log10(i0Plus)

        unitDict['pwvUnit'] = np.abs(deltaMagPWV[0]) / stepUnitReference / stepGrain

        # scale by fraction of bands affected
        use,=np.where((fitBands == 'z') |
                      (fitBands == 'Y'))
        unitDict['pwvUnit'] *= float(use.size) / float(fitBands.size)

        # PWV percent slope units
        ## FIXME: check these
        unitDict['pwvPerSlopeUnit'] = unitDict['pwvUnit'] * meanNightDuration / self.pwvStd

        # O3 units -- reference to r
        bandIndex,=np.where(self.bands == 'r')
        if bandIndex.size == 0:
            raise ValueError("Require r band for O3...")

        indicesStd = self.getIndices(bandIndex,self.pwvStd,self.o3Std,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.nCCD,self.pmbStd)
        i0Std = self.computeI0(self.pwvStd,self.o3Std,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.pmbStd,indicesStd)
        indicesPlus = self.getIndices(bandIndex,self.pwvStd,self.o3Std+1.0,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.nCCD,self.pmbStd)
        i0Plus = self.computeI0(self.pwvStd,self.o3Std+1.0,np.log(self.tauStd),self.alphaStd,self.secZenithStd,self.pmbStd,indicesPlus)
        deltaMagO3 = 2.5*np.log10(i0Std) - 2.5*np.log10(i0Plus)

        unitDict['o3Unit'] = np.abs(deltaMagO3[0]) / stepUnitReference / stepGrain

        # scale by fraction of bands that are affected
        use,=np.where((fitBands == 'r'))
        unitDict['o3Unit'] *= float(use.size) / float(fitBands.size)

        # wash parameters units...
        unitDict['qeSysUnit'] = 1.0 / stepUnitReference / stepGrain
        unitDict['qeSysSlopeUnit'] = unitDict['qeSysUnit'] * meanWashIntervalDuration

        return unitDict


