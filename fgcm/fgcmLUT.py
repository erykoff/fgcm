from __future__ import print_function

import numpy as np
import fitsio
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import fgcm_y3a1_tools
import os

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

    def _readLUTFile(self,lutFile):
        """
        """
        lutFlat = fitsio.read(lutFile,ext='LUT')
        indexVals = fitsio.read(lutFile,ext='INDEX')

        self.bands = indexVals['BANDS'][0]
        self.pmb = indexVals['PMB'][0]
        self.pwv = indexVals['PWV'][0]
        self.o3 = indexVals['O3'][0]
        self.tau = indexVals['TAU'][0]
        self.alpha = indexVals['ALPHA'][0]
        self.zenith = indexVals['ZENITH'][0]
        self.nCCD = indexVals['NCCD'][0]

        self.lut = lutFlat.reshape((self.bands.size,
                                    self.pmb.size,
                                    self.pwv.size,
                                    self.o3.size,
                                    self.tau.size,
                                    self.alpha.size,
                                    self.zenith.size))
        ccdDeltaFlat = fitsio.read(lutFile,ext='CCD')

        self.ccdDelta = ccdDeltaFlat.reshape((indexVals['BANDS'].size,
                                              indexVals['NCCD']))

        stdVals = fitsio.read(lutFile,ext='STD')

        self.pmbStd = stdVals['PMBSTD']
        self.pwvStd = stdVals['PWVSTD']
        self.o3Std = stdVals['O3STD']
        self.tauStd = stdVals['TAUSTD']
        self.alphaStd = stdVals['ALPHASTD']
        self.zenithStd = stdVals['ZENITHSTD']
        self.lambdaRange = stdVals['LAMBDARANGE']
        self.lambdaStep = stdVals['LAMBDASTEP']
        self.lambdaStd = stdVals['LAMBDASTD']
        self.I0Std = stdVals['I0STD']
        self.I1Std = stdVals['I1STD']
        self.I10Std = stdVals['I10STD']
        self.lambdaB = stdVals['LAMBDAB']
        self.atmLambda = stdVals['ATMLAMBDA']
        self.atmStdTrans = stdVals['ATMSTDTRANS']


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

        self.bands = np.array(self.lutConfig['bands'])

        self.filters = fgcm_y3a1_tools.DESFilters()
        self.nCCD = self.filters.nCCD
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

        print("Generating PWV atmospheres...")

        pwvAtmTable = np.zeros((self.pwv.size,self.zenith.size,self.atmLambda.size))

        for i in xrange(self.pwv.size):
            for j in xrange(self.zenith.size):
                atm=self.modGen(pwv=self.pwv[i],zenith=self.zenith[j],
                                lambdaRange=self.lambdaRange/10.0,
                                lambdaStep=self.lambdaStep)
                pwvAtmTable[i,j,:] = atm['H2O']

        print("Generating O3 atmospheres...")
        o3AtmTable = np.zeros((self.o3.size,self.zenith.size,self.atmLambda.size))

        for i in xrange(self.o3.size):
            for j in xrange(self.zenith.size):
                atm=self.modGen(o3=self.o3[i],zenith=self.zenith[j],
                                lambdaRange=self.lambdaRange/10.0,
                                lambdaStep=self.lambdaStep)
                o3AtmTable[i,j,:] = atm['O3']

        print("Generating O2/Rayleigh atmospheres...")
        o2AtmTable = np.zeros((self.zenith.size,self.atmLambda.size))
        rayleighAtmTable = np.zeros((self.zenith.size,self.atmLambda.size))

        for j in xrange(self.zenith.size):
            atm=self.modGen(zenith=self.zenith[j],
                            lambdaRange=self.lambdaRange/10.0,
                            lambdaStep=self.lambdaStep)
            o2AtmTable[j,:] = atm['O2']
            rayleighAtmTable[j,:] = atm['RAYLEIGH']

        # get the filters over the same lambda ranges...
        print("Interpolating filters...")
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
                             self.pmb.size,
                             self.pwv.size,
                             self.o3.size,
                             self.tau.size,
                             self.alpha.size,
                             self.zenith.size),
                            dtype=[('I0','f8'),
                                   ('I1','f8'),
                                   ('D_PMB','f8'),
                                   ('D_PWV','f8'),
                                   ('D_O3','f8'),
                                   ('D_LNTAU','f8'),
                                   ('D_ALPHA','f8'),
                                   ('D_SECZENITH','f8')])

        for i in xrange(self.bands.size):
            print("Working on band %s" % (self.bands[i]))
            for j in xrange(self.pmb.size):
                pmbMolecularScattering = np.exp(-(self.pmb[j] - self.modGen.pmbElevation)/self.modGen.pmbElevation)
                pmbMolecularAbsorption = pmbMolecularScattering ** 0.6
                pmbFactor = pmbMolecularScattering * pmbMolecularAbsorption
                for k in xrange(self.pwv.size):
                    for m in xrange(self.o3.size):
                        for n in xrange(self.tau.size):
                            for o in xrange(self.alpha.size):
                                for p in xrange(self.zenith.size):
                                    aerosolTauLambda = np.exp(-1.0*self.tau[n]*self.airmass[p]*(self.atmLambda/self.lambdaNorm)**(-self.alpha[o]))

                                    Sb = self.filters.interpolatedFilters[self.bInd[i]]['THROUGHPUT_AVG'] * pmbFactor * o2AtmTable[p,:] * rayleighAtmTable[p,:] * pwvAtmTable[k,p,:] * o3AtmTable[m,p,:] * aerosolTauLambda
                                    self.lut['I0'][i,j,k,m,n,o,p] = integrate.simps(Sb / self.atmLambda, self.atmLambda)
                                    self.lut['I1'][i,j,k,m,n,o,p] = integrate.simps(Sb * (self.atmLambda - self.lambdaStd[i]) / self.atmLambda, self.atmLambda)

        # and now the derivative tables...
        # last boundary is set to zero.

        print("Computing derivatives...")

        for i in xrange(self.bands.size-1):
            print("Working on band %s" % (self.bands[i]))
            for j in xrange(self.pmb.size-1):
                for k in xrange(self.pwv.size-1):
                    for m in xrange(self.o3.size-1):
                        for n in xrange(self.tau.size-1):
                            for o in xrange(self.alpha.size-1):
                                for p in xrange(self.zenith.size-1):
                                    self.lut['D_PMB'][i,j,k,m,n,o,p] = (
                                        ((self.lut['I0'][i,j+1,k,m,n,o,p] -
                                          self.lut['I0'][i,j,k,m,n,o,p]) /
                                         self.pmbDelta)
                                        )
                                    self.lut['D_PWV'][i,j,k,m,n,o,p] = (
                                        ((self.lut['I0'][i,j,k+1,m,n,o,p] -
                                          self.lut['I0'][i,j,k,m,n,o,p]) /
                                         self.pwvDelta)
                                        )
                                    self.lut['D_O3'][i,j,k,m,n,o,p] = (
                                        ((self.lut['I0'][i,j,k,m+1,n,o,p] -
                                          self.lut['I0'][i,j,k,m,n,o,p]) /
                                         self.o3Delta)
                                        )
                                    self.lut['D_LNTAU'][i,j,k,m,n,o,p] = (
                                        ((self.lut['I0'][i,j,k,m,n+1,o,p] -
                                          self.lut['I0'][i,j,k,m,n,o,p]) /
                                         self.lnTauDelta)
                                        )
                                    self.lut['D_ALPHA'][i,j,k,m,n,o,p] = (
                                        ((self.lut['I0'][i,j,k,m,n,o+1,p] -
                                          self.lut['I0'][i,j,k,m,n,o,p]) /
                                         self.alphaDelta)
                                        )
                                    self.lut['D_SECZENITH'][i,j,k,m,n,o,p] = (
                                        ((self.lut['I0'][i,j,k,m,n,o,p+1] -
                                          self.lut['I0'][i,j,k,m,n,o,p]) /
                                         self.secZenithDelta)
                                        )

        # and now we need the CCD deltas
        # integrate deltaSb for the standard passband, not for each individual passband
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
                                      ('PWV','f8',self.pwv.size),
                                      ('O3','f8',self.o3.size),
                                      ('TAU','f8',self.tau.size),
                                      ('ALPHA','f8',self.alpha.size),
                                      ('ZENITH','f8',self.zenith.size),
                                      ('NCCD','i4')])
        indexVals['BANDS'] = self.bands
        indexVals['PMB'] = self.pmb
        indexVals['PWV'] = self.pwv
        indexVals['O3'] = self.o3
        indexVals['TAU'] = self.tau
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
        stdVals['I0STD'][:] = self.I0Std
        stdVals['I1STD'][:] = self.I1Std
        stdVals['I10STD'][:] = self.I10Std
        stdVals['LAMBDAB'][:] = self.lambdaB
        stdVals['ATMLAMBDA'][:] = self.atmLambda
        stdVals['ATMSTDTRANS'][:] = self.atmStdTrans

        fitsio.write(lutFile,stdVals,extname='STD')


