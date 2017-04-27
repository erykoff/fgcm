from __future__ import print_function

import numpy as np
import fitsio
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import fgcm_y3a1_tools
import os
import sys

from fgcmUtilities import _pickle_method

import types
import copy_reg
#import sharedmem as shm

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

copy_reg.pickle(types.MethodType, _pickle_method)


class FgcmLUT(object):
    """
    """
    def __init__(self,lutFile=None,
                 lutConfig=None):
        if (lutFile is None and lutConfig is None):
            raise ValueError("Must specify at least one of lutFile or lutConfig")

        if (lutFile is not None):
            if (not os.path.isfile(lutFile)):
                raise IOError("Could not find lutFile %s" % (lutFile))

            self._readLUTFile(lutFile)
        else :
            self._checkLUTConfig(lutConfig)

        self.magConstant = 2.5/np.log(10)

    def _readLUTFile(self,lutFile):
        """
        """
        lutFlat = fitsio.read(lutFile,ext='LUT')
        indexVals = fitsio.read(lutFile,ext='INDEX')

        self.bands = indexVals['BANDS'][0]
        self.pmb = indexVals['PMB'][0]
        self.pmbFactor = indexVals['PMBFACTOR'][0]
        self.pmbDelta = self.pmb[1] - self.pmb[0]
        ## temporary hacks
        try:
            self.pmbElevation = indexVals['PMBELEVATION'][0]
        except:
            self.pmbElevation = 2220.0
        try:
            self.lambdaNorm = indexVals['LAMBDANORM'][0]
        except:
            self.lambdaNorm = 7750.0
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

        self.lut = lutFlat.reshape((self.bands.size,
                                    self.pwv.size,
                                    self.o3.size,
                                    self.tau.size,
                                    self.alpha.size,
                                    self.zenith.size,
                                    self.nCCDStep))

        ccdDeltaFlat = fitsio.read(lutFile,ext='CCD')
        self.ccdDelta = ccdDeltaFlat.reshape((indexVals['BANDS'].size,
                                              indexVals['NCCD'][0]))


        stdVals = fitsio.read(lutFile,ext='STD')

        self.pmbStd = stdVals['PMBSTD'][0]
        self.pwvStd = stdVals['PWVSTD'][0]
        self.o3Std = stdVals['O3STD'][0]
        self.tauStd = stdVals['TAUSTD'][0]
        self.alphaStd = stdVals['ALPHASTD'][0]
        self.zenithStd = stdVals['ZENITHSTD'][0]
        self.lambdaRange = stdVals['LAMBDARANGE'][0]
        self.lambdaStep = stdVals['LAMBDASTEP'][0]
        self.lambdaStd = stdVals['LAMBDASTD'][0]
        self.I0Std = stdVals['I0STD'][0]
        self.I1Std = stdVals['I1STD'][0]
        self.I10Std = stdVals['I10STD'][0]
        self.lambdaB = stdVals['LAMBDAB'][0]
        self.atmLambda = stdVals['ATMLAMBDA'][0]
        self.atmStdTrans = stdVals['ATMSTDTRANS'][0]


    def _checkLUTConfig(self,lutConfig):
        """
        """

        requiredKeys=['elevation','bands',
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
        self.modGen = fgcm_y3a1_tools.ModtranGenerator(self.lutConfig['elevation'])
        self.pmbElevation = self.modGen.pmbElevation

        self.bands = np.array(self.lutConfig['bands'])

        self.filters = fgcm_y3a1_tools.DESFilters()
        self.nCCD = self.filters.nCCD
        self.nCCDStep = self.nCCD+1
        # and match up the band indices
        self.bInd = np.zeros(self.bands.size,dtype='i2')-1
        for i in xrange(self.bands.size):
            if (self.bands[i] not in self.filters.bands):
                raise ValueError("Requested band %s not in list of filters!" % (self.bands[i]))
            self.bInd[i], = np.where(self.bands[i] == self.filters.bands)

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

        if ('nproc' in self.lutConfig):
            self.nproc = self.lutConfig['nproc']
        else:
            self.nproc = 1

    def makeLUT(self,lutFile,clobber=False):
        """
        """

        if (os.path.isfile(lutFile) and not clobber) :
            print("lutFile %s already exists, and clobber is False.")
            return

        # we need a standard atmosphere and lambdas...
        self.atmStd = self.modGen(pmb=self.pmbStd,pwv=self.pwvStd,
                                  o3=self.o3Std,tau=self.tauStd,
                                  alpha=self.alphaStd,zenith=self.zenithStd,
                                  lambdaRange=self.lambdaRange/10.0,
                                  lambdaStep=self.lambdaStep)
        self.atmLambda = self.atmStd['LAMBDA']
        self.atmStdTrans = self.atmStd['COMBINED']


        self.pmb = np.linspace(self.lutConfig['pmbRange'][0],
                               self.lutConfig['pmbRange'][1],
                               num=self.lutConfig['pmbSteps'])
        self.pmbDelta = self.pmb[1] - self.pmb[0]
        self.pwv = np.linspace(self.lutConfig['pwvRange'][0],
                               self.lutConfig['pwvRange'][1],
                               num=self.lutConfig['pwvSteps'])
        self.pwvDelta = self.pwv[1] - self.pwv[0]
        self.o3 = np.linspace(self.lutConfig['o3Range'][0],
                               self.lutConfig['o3Range'][1],
                               num=self.lutConfig['o3Steps'])
        self.o3Delta = self.o3[1] - self.o3[0]
        self.lnTau = np.linspace(np.log(self.lutConfig['tauRange'][0]),
                                 np.log(self.lutConfig['tauRange'][1]),
                                 num=self.lutConfig['tauSteps'])
        self.lnTauDelta = self.lnTau[1] - self.lnTau[0]
        self.tau = np.exp(self.lnTau)
        self.alpha = np.linspace(self.lutConfig['alphaRange'][0],
                               self.lutConfig['alphaRange'][1],
                               num=self.lutConfig['alphaSteps'])
        self.alphaDelta = self.alpha[1] - self.alpha[0]
        self.secZenith = np.linspace(1./np.cos(self.lutConfig['zenithRange'][0]*np.pi/180.),
                                     1./np.cos(self.lutConfig['zenithRange'][1]*np.pi/180.),
                                     num=self.lutConfig['zenithSteps'])
        self.secZenithDelta = self.secZenith[1]-self.secZenith[0]
        self.zenith = np.arccos(1./self.secZenith)*180./np.pi

        # and compute the proper airmass...
        self.airmass = self.secZenith - 0.0018167*(self.secZenith-1.0) - 0.002875*(self.secZenith-1.0)**2.0 - 0.0008083*(self.secZenith-1.0)**3.0

        # run MODTRAN a bunch of times
        # we need for each airmass, to run the array of pwv and o3 and pull these out

        print("Generating %d*%d=%d PWV atmospheres..." % (self.pwv.size,self.zenith.size,self.pwv.size*self.zenith.size))

        self.pwvAtmTable = np.zeros((self.pwv.size,self.zenith.size,self.atmLambda.size))

        for i in xrange(self.pwv.size):
            sys.stdout.write('%d' % (i))
            sys.stdout.flush()
            for j in xrange(self.zenith.size):
                sys.stdout.write('.')
                sys.stdout.flush()
                atm=self.modGen(pwv=self.pwv[i],zenith=self.zenith[j],
                                lambdaRange=self.lambdaRange/10.0,
                                lambdaStep=self.lambdaStep)
                self.pwvAtmTable[i,j,:] = atm['H2O']

        print("\nGenerating %d*%d=%d O3 atmospheres..." % (self.o3.size,self.zenith.size,self.o3.size*self.zenith.size))
        self.o3AtmTable = np.zeros((self.o3.size,self.zenith.size,self.atmLambda.size))

        for i in xrange(self.o3.size):
            sys.stdout.write('%d' % (i))
            sys.stdout.flush()
            for j in xrange(self.zenith.size):
                sys.stdout.write('.')
                sys.stdout.flush()
                atm=self.modGen(o3=self.o3[i],zenith=self.zenith[j],
                                lambdaRange=self.lambdaRange/10.0,
                                lambdaStep=self.lambdaStep)
                self.o3AtmTable[i,j,:] = atm['O3']

        print("\nGenerating %d O2/Rayleigh atmospheres..." % (self.zenith.size))
        self.o2AtmTable = np.zeros((self.zenith.size,self.atmLambda.size))
        self.rayleighAtmTable = np.zeros((self.zenith.size,self.atmLambda.size))

        for j in xrange(self.zenith.size):
            sys.stdout.write('.')
            sys.stdout.flush()
            atm=self.modGen(zenith=self.zenith[j],
                            lambdaRange=self.lambdaRange/10.0,
                            lambdaStep=self.lambdaStep)
            self.o2AtmTable[j,:] = atm['O2']
            self.rayleighAtmTable[j,:] = atm['RAYLEIGH']

        # get the filters over the same lambda ranges...
        print("\nInterpolating filters...")
        self.filters.interpolateFilters(self.atmLambda)

        # and now we can get the standard atmosphere and lambda_b

        print("Computing lambdaB")
        self.lambdaB = np.zeros(self.bands.size)
        for i in xrange(self.bands.size):
            num = integrate.simps(self.atmLambda * self.filters.interpolatedFilters[self.bInd[i]]['THROUGHPUT_AVG'] / self.atmLambda, self.atmLambda)
            denom = integrate.simps(self.filters.interpolatedFilters[self.bInd[i]]['THROUGHPUT_AVG'] / self.atmLambda, self.atmLambda)
            self.lambdaB[i] = num / denom
            print("Band: %s, lambdaB = %.3f" % (self.bands[i], self.lambdaB[i]))

        print("Computing lambdaStd")
        self.lambdaStd = np.zeros(self.bands.size)
        for i in xrange(self.bands.size):
            num = integrate.simps(self.atmLambda * self.filters.interpolatedFilters[self.bInd[i]]['THROUGHPUT_AVG'] * self.atmStdTrans / self.atmLambda, self.atmLambda)
            denom = integrate.simps(self.filters.interpolatedFilters[self.bInd[i]]['THROUGHPUT_AVG'] * self.atmStdTrans / self.atmLambda, self.atmLambda)
            self.lambdaStd[i] = num / denom
            print("Band: %s, lambdaStd = %.3f" % (self.bands[i],self.lambdaStd[i]))


        # now make the LUT!
        print("Building look-up table...")
        self.lut = np.zeros((self.bands.size,
                             self.pwv.size,
                             self.o3.size,
                             self.tau.size,
                             self.alpha.size,
                             self.zenith.size,
                             self.nCCDStep),
                            dtype=[('I0','f4'),
                                   ('I1','f4')])

        pmbMolecularScattering = np.exp(-(self.pmb - self.pmbElevation)/self.pmbElevation)
        pmbMolecularAbsorption = pmbMolecularScattering ** 0.6
        self.pmbFactor = pmbMolecularScattering * pmbMolecularAbsorption

        for i in xrange(self.bands.size):
            print("Working on band %s" % (self.bands[i]))
            for j in xrange(self.pwv.size):
                print("  and on pwv #%d" % (j))
                for k in xrange(self.o3.size):
                    print("   and on o3 #%d" % (k))
                    for m in xrange(self.tau.size):
                        for n in xrange(self.alpha.size):
                            for o in xrange(self.zenith.size):
                                aerosolTauLambda = np.exp(-1.0*self.tau[m]*self.airmass[o]*(self.atmLambda/self.lambdaNorm)**(-self.alpha[n]))
                                for p in xrange(self.nCCDStep):
                                    if (p == self.nCCD):
                                        Sb = self.filters.interpolatedFilters[self.bInd[i]]['THROUGHPUT_AVG'] * self.o2AtmTable[o,:] * self.rayleighAtmTable[o,:] * self.pwvAtmTable[j,o,:] * self.o3AtmTable[k,o,:] * aerosolTauLambda
                                    else:
                                        Sb = self.filters.interpolatedFilters[self.bInd[i]]['THROUGHPUT_CCD'][:,p] * self.o2AtmTable[o,:] * self.rayleighAtmTable[o,:] * self.pwvAtmTable[j,o,:] * self.o3AtmTable[k,o,:] * aerosolTauLambda

                                    self.lut['I0'][i,j,k,m,n,o,p] = integrate.simps(Sb / self.atmLambda, self.atmLambda)
                                    self.lut['I1'][i,j,k,m,n,o,p] = integrate.simps(Sb * (self.atmLambda - self.lambdaStd[i]) / self.atmLambda, self.atmLambda)


        # and now we need the CCD deltas
        # integrate deltaSb for the standard passband
        # this is just a check currently...
        print("Computing CCD deltas...")
        self.ccdDelta = np.zeros((self.bands.size,self.nCCD),
                                 dtype=[('DELTAI0','f8'),
                                        ('DELTAI1','f8')])

        self.I0Std = np.zeros(self.bands.size)
        self.I1Std = np.zeros(self.bands.size)

        for i in xrange(self.bands.size):
            self.I0Std[i] = integrate.simps(self.filters.interpolatedFilters[self.bInd[i]]['THROUGHPUT_AVG'] * self.atmStdTrans / self.atmLambda, self.atmLambda)
            self.I1Std[i] = integrate.simps(self.filters.interpolatedFilters[self.bInd[i]]['THROUGHPUT_AVG'] * self.atmStdTrans * (self.atmLambda - self.lambdaStd[i]) / self.atmLambda, self.atmLambda)
            for j in xrange(self.nCCD):
                deltaSb = self.filters.interpolatedFilters[self.bInd[i]]['THROUGHPUT_CCD'][:,j] - self.filters.interpolatedFilters[self.bInd[i]]['THROUGHPUT_AVG']
                self.ccdDelta['DELTAI0'][i,j] = integrate.simps(deltaSb * self.atmStdTrans / self.atmLambda, self.atmLambda)
                self.ccdDelta['DELTAI1'][i,j] = integrate.simps(deltaSb * self.atmStdTrans * (self.atmLambda - self.lambdaStd[i]) / self.atmLambda, self.atmLambda)


        self.I10Std = self.I1Std / self.I0Std
        # and we can write it out.

        # start by writing the LUT.  And clobber, though we checked at start
        print("Saving LUT to %s" % (lutFile))

        fitsio.write(lutFile,self.lut.flatten(),extname='LUT',clobber=True)

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

        fitsio.write(lutFile,self.ccdDelta.flatten(),extname='CCD')

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

    def makeLUTDerivatives(self, lutFile):

        # need setup
        # NOTE: will need to add pmb
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
                                        ('D_SECZENITH','f4')])

        print("Computing derivatives...")

        for i in xrange(self.bands.size):
            print("Working on band %s" % (self.bands[i]))
            for j in xrange(self.pwv.size-1):
                for k in xrange(self.o3.size-1):
                    for m in xrange(self.tau.size-1):
                        for n in xrange(self.alpha.size-1):
                            for o in xrange(self.zenith.size-1):
                                for p in xrange(self.nCCDStep):
                                    #self.lutDeriv['D_PMB'][i,j,k,m,n,o,p,q] = (
                                    #        ((self.lutDeriv['I0'][i,j+1,k,m,n,o,p,q] -
                                    #          self.lutDeriv['I0'][i,j,k,m,n,o,p,q]) /
                                    #         self.pmbDelta)
                                    #        )
                                    self.lutDeriv['D_PWV'][i,j,k,m,n,o,p] = (
                                        ((self.lut['I0'][i,j+1,k,m,n,o,p] -
                                          self.lut['I0'][i,j,k,m,n,o,p]) /
                                         self.pwvDelta)
                                        )
                                    self.lutDeriv['D_O3'][i,j,k,m,n,o,p] = (
                                        ((self.lut['I0'][i,j,k+1,m,n,o,p] -
                                          self.lut['I0'][i,j,k,m,n,o,p]) /
                                         self.o3Delta)
                                        )
                                    self.lutDeriv['D_LNTAU'][i,j,k,m,n,o,p] = (
                                        ((self.lut['I0'][i,j,k,m+1,n,o,p] -
                                          self.lut['I0'][i,j,k,m,n,o,p]) /
                                         self.lnTauDelta)
                                        )
                                    self.lutDeriv['D_ALPHA'][i,j,k,m,n,o,p] = (
                                            ((self.lut['I0'][i,j,k,m,n+1,o,p] -
                                              self.lut['I0'][i,j,k,m,n,o,p]) /
                                             self.alphaDelta)
                                            )
                                    self.lutDeriv['D_SECZENITH'][i,j,k,m,n,o,p] = (
                                            ((self.lut['I0'][i,j,k,m,n,o+1,p] -
                                              self.lut['I0'][i,j,k,m,n,o,p]) /
                                             self.secZenithDelta)
                                            )

        print("Saving DERIV extension to %s" % (lutFile))
        fitsio.write(lutFile,self.lutDeriv.flatten(),extname='DERIV')

#class FgcmLUTSHM(object):
#    """
#    """
#    def __init__(self,lutFile):
#        self.lutFile = lutFile

#        lutFlat = fitsio.read(self.lutFile,ext='LUT')
#        indexVals = fitsio.read(self.lutFile,ext='INDEX')

#        self.bands = indexVals['BANDS'][0]
#        self.pmb = indexVals['PMB'][0]
#        self.pmbFactor = indexVals['PMBFACTOR'][0]
#        self.pmbDelta = self.pmb[1] - self.pmb[0]
        ## temporary hacks
#        try:
#            self.pmbElevation = indexVals['PMBELEVATION'][0]
#        except:
#            self.pmbElevation = 2220.0
#        try:
#            self.lambdaNorm = indexVals['LAMBDANORM'][0]
#        except:
#            self.lambdaNorm = 7775.0
#        self.pwv = indexVals['PWV'][0]
#        self.pwvDelta = self.pwv[1] - self.pwv[0]
#        self.o3 = indexVals['O3'][0]
#        self.o3Delta = self.o3[1] - self.o3[0]
#        self.tau = indexVals['TAU'][0]
#        self.lnTau = np.log(self.tau)
#        self.lnTauDelta = self.lnTau[1] - self.lnTau[0]
#        self.alpha = indexVals['ALPHA'][0]
#        self.alphaDelta = self.alpha[1] - self.alpha[0]
#        self.zenith = indexVals['ZENITH'][0]
#        self.secZenith = 1./np.cos(self.zenith*np.pi/180.)
#        self.secZenithDelta = self.secZenith[1] - self.secZenith[0]
#        self.nCCD = indexVals['NCCD'][0]
#        self.nCCDStep = self.nCCD+1

#        sizeTuple = (self.bands.size,self.pwv.size,self.o3.size,
#                     self.tau.size,self.alpha.size,self.zenith.size,self.nCCDStep)


#        self.lutI0SHM = shm.zeros(sizeTuple,dtype='f4')
#        self.lutI1SHM = shm.zeros(sizeTuple,dtype='f4')

#        self.lutI0SHM[:,:,:,:,:,:,:] = lutFlat['I0'].reshape(sizeTuple)
#        self.lutI1SHM[:,:,:,:,:,:,:] = lutFlat['I1'].reshape(sizeTuple)

        # clear memory
#        lutFlat = 0

#        lutDerivFlat = fitsio.read(self.lutFile,ext='DERIV')
#        self.lutDPWVSHM = shm.zeros(sizeTuple,dtype='f4')
#        self.lutDO3SHM = shm.zeros(sizeTuple,dtype='f4')
#        self.lutDLnTauSHM = shm.zeros(sizeTuple,dtype='f4')
#        self.lutDAlphaSHM = shm.zeros(sizeTuple,dtype='f4')
#        self.lutDSecZenithSHM = shm.zeros(sizeTuple,dtype='f4')

#        self.lutDPWVSHM[:,:,:,:,:,:,:] = lutDerivFlat['D_PWV'].reshape(sizeTuple)
#        self.lutDO3SHM[:,:,:,:,:,:,:] = lutDerivFlat['D_O3'].reshape(sizeTuple)
#        self.lutDLnTauSHM[:,:,:,:,:,:,:] = lutDerivFlat['D_LNTAU'].reshape(sizeTuple)
#        self.lutDAlphaSHM[:,:,:,:,:,:,:] = lutDerivFlat['D_ALPHA'].reshape(sizeTuple)
#        self.lutDSecZenithSHM[:,:,:,:,:,:,:] = lutDerivFlat['D_SECZENITH'].reshape(sizeTuple)

        # get the standard values
#        stdVals = fitsio.read(lutFile,ext='STD')

#        self.pmbStd = stdVals['PMBSTD'][0]
#        self.pwvStd = stdVals['PWVSTD'][0]
#        self.o3Std = stdVals['O3STD'][0]
#        self.tauStd = stdVals['TAUSTD'][0]
#        self.alphaStd = stdVals['ALPHASTD'][0]
#        self.zenithStd = stdVals['ZENITHSTD'][0]
#        self.lambdaRange = stdVals['LAMBDARANGE'][0]
#        self.lambdaStep = stdVals['LAMBDASTEP'][0]
#        self.lambdaStd = stdVals['LAMBDASTD'][0]
#        self.I0Std = stdVals['I0STD'][0]
#        self.I1Std = stdVals['I1STD'][0]
#        self.I10Std = stdVals['I10STD'][0]
#        self.lambdaB = stdVals['LAMBDAB'][0]
#        self.atmLambda = stdVals['ATMLAMBDA'][0]
#        self.atmStdTrans = stdVals['ATMSTDTRANS'][0]


#    def getIndices(self, bandIndex, pwv, o3, lnTau, alpha, secZenith, ccdIndex, pmb):
        # need to make sure we have the right ccd indices...
        # this will happen externally.

#        return (bandIndex,
#                np.clip(((pwv - self.pwv[0])/self.pwvDelta).astype(np.int32), 0,
#                        self.pwv.size-1),
#                np.clip(((o3 - self.o3[0])/self.o3Delta).astype(np.int32), 0,
#                        self.o3.size-1),
#                np.clip(((lnTau - self.lnTau[0])/self.lnTauDelta).astype(np.int32), 0,
#                        self.lnTau.size-1),
#                np.clip(((alpha - self.alpha[0])/self.alphaDelta).astype(np.int32), 0,
#                        self.alpha.size-1),
#                np.clip(((secZenith - self.secZenith[0])/self.secZenithDelta).astype(np.int32), 0,
#                        self.secZenith.size-1),
#                ccdIndex,
#                (np.exp(-(pmb - self.pmbElevation)/self.pmbElevation)) ** 1.6)

#    def computeI0(self, bandIndex, pwv, o3, lnTau, alpha, secZenith, ccdIndex, pmb, indices):
#        dPWV = pwv - (self.pwv[0] + indices[1] * self.pwvDelta)
#        dO3 = o3 - (self.o3[0] + indices[2] * self.o3Delta)
#        dlnTau = lnTau - (self.lnTau[0] + indices[3] * self.lnTauDelta)
#        dAlpha = alpha - (self.alpha[0] + indices[4] * self.alphaDelta)
#        dSecZenith = secZenith - (self.secZenith[0] + indices[5] * self.secZenithDelta)

#        indicesPlus = np.array(indices[:-1])
#        indicesPlus[5] += 1

#        return indices[-1]*(self.lutI0SHM[indices[:-1]] +
#                            dPWV * self.lutDPWVSHM[indices[:-1]] +
#                            dO3 * self.lutDO3SHM[indices[:-1]] +
#                            dlnTau * self.lutDLnTauSHM[indices[:-1]] +
#                            dAlpha * self.lutDAlphaSHM[indices[:-1]] +
#                            dSecZenith * self.lutDSecZenithSHM[indices[:-1]] +
#                            dlnTau * dSecZenith * (self.lutDLnTauSHM[tuple(indicesPlus)] - self.lutDLnTauSHM[indices[:-1]])/self.secZenithDelta)


#    def computeI1(self, indices):
#        return indices[-1] * self.lutI1SHM[indices[:-1]]

#    def computeDerivatives(self, indices, I0):
#        # need to worry about lnTau
#        return (self.lutDPWVSHM[indices[:-1]] / I0,
#                self.lutDO3SHM[indices[:-1]] / I0,
#                self.lutDlnTauSHM[indices[:-1]] / I0,
#                self.lutDAlphaSHM[indices[:-1]] / I0)

class FgcmLUTSHM(object):
    """
    """
    def __init__(self,lutFile):
        self.lutFile = lutFile

        lutFlat = fitsio.read(self.lutFile,ext='LUT')
        indexVals = fitsio.read(self.lutFile,ext='INDEX')

        self.bands = indexVals['BANDS'][0]
        self.pmb = indexVals['PMB'][0]
        self.pmbFactor = indexVals['PMBFACTOR'][0]
        self.pmbDelta = self.pmb[1] - self.pmb[0]
        ## temporary hacks
        try:
            self.pmbElevation = indexVals['PMBELEVATION'][0]
        except:
            self.pmbElevation = 773.4763121915056
        try:
            self.lambdaNorm = indexVals['LAMBDANORM'][0]
        except:
            self.lambdaNorm = 7775.0
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

        sizeTuple = (self.bands.size,self.pwv.size,self.o3.size,
                     self.tau.size,self.alpha.size,self.zenith.size,self.nCCDStep)

        self.lutI0Handle = snmm.createArray(sizeTuple,dtype='f4')
        snmm.getArray(self.lutI0Handle)[:,:,:,:,:,:,:] = lutFlat['I0'].reshape(sizeTuple)
        #lutI0 = snmm.getArray(self.lutI0Handle)
        #lutI0[:,:,:,:,:,:,:] = lutFlat['I0'].reshape(sizeTuple)
        # clear reference to memory
        #lutI0 = None

        self.lutI1Handle = snmm.createArray(sizeTuple,dtype='f4')
        snmm.getArray(self.lutI1Handle)[:,:,:,:,:,:,:] = lutFlat['I1'].reshape(sizeTuple)
        #lutI1 = snmm.getArray(self.lutI1Handle)
        #lutI1[:,:,:,:,:,:,:] = lutFlat['I1'].reshape(sizeTuple)
        #lutI1 = None

        # clear all memory
        lutFlag = 0

        # now read in derivative tables...
        lutDerivFlat = fitsio.read(self.lutFile,ext='DERIV')

        self.lutDPWVHandle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDO3Handle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDLnTauHandle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDAlphaHandle = snmm.createArray(sizeTuple,dtype='f4')
        self.lutDSecZenithHandle = snmm.createArray(sizeTuple,dtype='f4')

        snmm.getArray(self.lutDPWVHandle)[:,:,:,:,:,:,:] = lutDerivFlat['D_PWV'].reshape(sizeTuple)
        snmm.getArray(self.lutDO3Handle)[:,:,:,:,:,:,:] = lutDerivFlat['D_O3'].reshape(sizeTuple)
        snmm.getArray(self.lutDLnTauHandle)[:,:,:,:,:,:,:] = lutDerivFlat['D_LNTAU'].reshape(sizeTuple)
        snmm.getArray(self.lutDAlphaHandle)[:,:,:,:,:,:,:] = lutDerivFlat['D_ALPHA'].reshape(sizeTuple)
        snmm.getArray(self.lutDSecZenithHandle)[:,:,:,:,:,:,:] = lutDerivFlat['D_SECZENITH'].reshape(sizeTuple)

        lutDerivFlat = None

        # get the standard values
        stdVals = fitsio.read(lutFile,ext='STD')

        self.pmbStd = stdVals['PMBSTD'][0]
        self.pwvStd = stdVals['PWVSTD'][0]
        self.o3Std = stdVals['O3STD'][0]
        self.tauStd = stdVals['TAUSTD'][0]
        self.alphaStd = stdVals['ALPHASTD'][0]
        self.zenithStd = stdVals['ZENITHSTD'][0]
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


    def getIndices(self, bandIndex, pwv, o3, lnTau, alpha, secZenith, ccdIndex, pmb):
        """
        """
        
        # need to make sure we have the right ccd indices...
        # this will happen externally.

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

    def computeI0(self, bandIndex, pwv, o3, lnTau, alpha, secZenith, ccdIndex, pmb, indices):
        """
        """
        
        dPWV = pwv - (self.pwv[0] + indices[1] * self.pwvDelta)
        dO3 = o3 - (self.o3[0] + indices[2] * self.o3Delta)
        dlnTau = lnTau - (self.lnTau[0] + indices[3] * self.lnTauDelta)
        dAlpha = alpha - (self.alpha[0] + indices[4] * self.alphaDelta)
        dSecZenith = secZenith - (self.secZenith[0] + indices[5] * self.secZenithDelta)

        indicesPlus = np.array(indices[:-1])
        indicesPlus[5] += 1

        return indices[-1]*(snmm.getArray(self.lutI0Handle)[indices[:-1]] +
                            dPWV * snmm.getArray(self.lutDPWVHandle)[indices[:-1]] +
                            dO3 * snmm.getArray(self.lutDO3Handle)[indices[:-1]] +
                            dlnTau * snmm.getArray(self.lutDLnTauHandle)[indices[:-1]] +
                            dAlpha * snmm.getArray(self.lutDAlphaHandle)[indices[:-1]] +
                            dSecZenith * snmm.getArray(self.lutDSecZenithHandle)[indices[:-1]] +
                            dlnTau * dSecZenith * (snmm.getArray(self.lutDLnTauHandle)[tuple(indicesPlus)] -
                                                   snmm.getArray(self.lutDLnTauHandle)[indices[:-1]])/self.secZenithDelta)

    def computeI1(self, indices):
        """
        """
        
        return indices[-1] * snmm.getArray(self.lutI1Handle)[indices[:-1]]

    def computeLogDerivatives(self, indices, I0, tau):
        # dL(i,j|p) = d/dp(2.5*log10(LUT(i,j|p)))
        #           = 1.086*(LUT'(i,j|p)/LUT(i,j|p))
        return (self.magConstant*snmm.getArray(self.lutDPWVHandle)[indices[:-1]] / I0,
                self.magConstant*snmm.getArray(self.lutDO3Handle)[indices[:-1]] / I0,
                self.magConstant*snmm.getArray(self.lutDLnTauHandle)[indices[:-1]] / (I0*tau), # ln space
                self.magConstant*snmm.getArray(self.lutDAlphaHandle)[indices[:-1]] / I0)





