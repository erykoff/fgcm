from __future__ import print_function

import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import os
import sys
from pkg_resources import resource_filename


from modtranGenerator import ModtranGenerator

from sharedNumpyMemManager import SharedNumpyMemManager as snmm
from fgcmLogger import FgcmLogger

class FgcmAtmosphereTable(object):
    """
    """

    def __init__(self):
        # this is a dummy
        pass

    @classmethod
    def initWithTableName(cls, atmosphereTableName):
        # will set self.lutConfig

        # check for consistency between input/output and log warnings?

        # here...
        self.atmosphereTableFile = atmosphereTableFile

        parStruct = fitsio.read(self.atmosphereTableFile, ext='PARS')
        

    def loadTable(self):
        """
        """

        parStruct = fitsio.read(self.atmosphereTableFile, ext='PARS')

        self.elevation = parStruct['ELEVATION'][0]
        self.pmbElevation = parStruct['PMBELEVATION'][0]
        self.pmbStd = parStruct['PMBSTD'][0]
        self.pwvStd = parStruct['PWVSTD'][0]
        self.o3Std = parStruct['O3STD'][0]
        self.tauStd = parStruct['TAUSTD'][0]
        self.alphaStd = parStruct['ALPHASTD'][0]
        self.secZenithStd = parStruct['AIRMASSSTD'][0]
        self.lambdaRange = parStruct['LAMBDARANGE'][0]
        self.lambdaStep = parStruct['LAMBDASTEP'][0]
        self.lambdaNorm = parStruct['LAMBDANORM'][0]

        self.atmLambda = parStruct['ATMLAMBDA'][0]
        self.atmStdTrans = parStruct['ATMSTDTRANS'][0]

        self.pmb = parStruct['PMB'][0]
        self.pmbDelta = parStruct['PMBDELTA'][0]
        self.pwv = parStruct['PWV'][0]
        self.pwvDelta = parStruct['PWVDELTA'][0]
        self.o3 = parStruct['O3'][0]
        self.o3Delta = parStruct['O3DELTA'][0]
        self.lnTau = parStruct['LNTAU'][0]
        self.lnTauDelta = parStruct['LNTAUDELTA'][0]
        self.alpha = parStruct['ALPHA'][0]
        self.alphaDelta = parStruct['ALPHADELTA'][0]
        self.secZenith = parStruct['SECZENITH'][0]
        self.secZenithDelta = parStruct['SECZENITHDELTA'][0]

        self.pwvAtmTable = fitsio.read(self.atmosphereTableFile, ext='PWVATM')
        self.o3AtmTable = fitsio.read(self.atmosphereTableFile, ext='O3ATM')
        self.o2AtmTable = fitsio.read(self.atmosphereTableFile, ext='O2ATM')
        self.rayleighAtmTable = fitsio.read(self.atmosphereTableFile, ext='RAYATM')

    @classmethod
    def initWithConfig(cls, lutConfig, fgcmLog):
        requiredKeys = ['elevation', 'filterNames',
                        'stdFilterNames', 'nCCD',
                        'pmbRange','pmbSteps',
                        'pwvRange','pwvSteps',
                        'o3Range','o3Steps',
                        'tauRange','tauSteps',
                        'alphaRange','alphaSteps',
                        'zenithRange','zenithSteps',
                        'pmbStd','pwvStd','o3Std',
                        'tauStd','alphaStd','airmassStd']

        for key in requiredKeys:
            if key not in lutConfig:
                raise ValueError("Required %s not in lutConfig" % (key))
            if 'Range' in key:
                if len(lutConfig[key]) != 2:
                    raise ValueError("%s must have 2 elements" % (key))

        self.lutConfig = lutConfig

        self.modGen = ModtranGenerator(self.lutConfig['elevation'])
        self.pmbElevation = self.modGen.pmbElevation

        self.pmbStd = self.lutConfig['pmbStd']
        self.pwvStd = self.lutConfig['pwvStd']
        self.o3Std = self.lutConfig['o3Std']
        self.tauStd = self.lutConfig['tauStd']
        self.lnTauStd = np.log(self.tauStd)
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

        self.fgcmLog = fgcmLog


    def generateTable(self):
        """
        """
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

        # run MODTRAN a bunch of times

        self.fgcmLog.info("Generating %d*%d=%d PWV atmospheres..." % (pwvPlus.size,zenithPlus.size,pwvPlus.size*zenithPlus.size))
        self.pwvAtmTable = np.zeros((pwvPlus.size,zenithPlus.size,self.atmLambda.size))

        for i in xrange(pwvPlus.size):
            sys.stdout.write('%d' % (i))
            sys.stdout.flush()
            for j in xrange(zenithPlus.size):
                sys.stdout.write('.')
                sys.stdout.flush()
                atm=self.modGen(pwv=pwvPlus[i],zenith=zenithPlus[j],
                                lambdaRange=self.lambdaRange/10.0,
                                lambdaStep=self.lambdaStep)
                self.pwvAtmTable[i,j,:] = atm['H2O']

        self.fgcmLog.info("\nGenerating %d*%d=%d O3 atmospheres..." % (o3Plus.size,zenithPlus.size,o3Plus.size*zenithPlus.size))
        self.o3AtmTable = np.zeros((o3Plus.size, zenithPlus.size, self.atmLambda.size))

        for i in xrange(o3Plus.size):
            sys.stdout.write('%d' % (i))
            sys.stdout.flush()
            for j in xrange(zenithPlus.size):
                sys.stdout.write('.')
                sys.stdout.flush()
                atm=self.modGen(o3=o3Plus[i],zenith=zenithPlus[j],
                                lambdaRange=self.lambdaRange/10.0,
                                lambdaStep=self.lambdaStep)
                self.o3AtmTable[i,j,:] = atm['O3']

        self.fgcmLog.info("\nGenerating %d O2/Rayleigh atmospheres..." % (zenithPlus.size))

        self.o2AtmTable = np.zeros((zenithPlus.size, self.atmLambda.size))
        self.rayleighAtmTable = np.zeros((zenithPlus.size, self.atmLambda.size))

        for j in xrange(zenithPlus.size):
            sys.stdout.write('.')
            sys.stdout.flush()
            atm=self.modGen(zenith=zenithPlus[j],
                            lambdaRange=self.lambdaRange/10.0,
                            lambdaStep=self.lambdaStep)
            self.o2AtmTable[j,:] = atm['O2']
            self.rayleighAtmTable[j,:] = atm['RAYLEIGH']

    def saveTable(self, fileName, clobber=False):
        """
        """
        import fitsio

        if os.path.isfile(fileName) and not clobber:
            raise IOError("File %s already exists and clobber is set to False" % (fileName))

        fits = fitsio.FITS(fileName, 'rw')

        parStruct = np.zeros(1, dtype=[('ELEVATION','f8'),
                                       ('PMBELEVATION','f8'),
                                       ('PMBSTD','f8'),
                                       ('PWVSTD','f8'),
                                       ('O3STD','f8'),
                                       ('TAUSTD','f8'),
                                       ('ALPHASTD','f8'),
                                       ('AIRMASSSTD','f8'),
                                       ('LAMBDARANGE','f8',2),
                                       ('LAMBDASTEP','f8'),
                                       ('LAMBDANORM','f8'),
                                       ('ATMLAMBDA','f8',self.atmLambda.size),
                                       ('ATMSTDTRANS','f8',self.atmStdTrans.size),
                                       ('PMB','f8',self.pmb.size),
                                       ('PMBDELTA','f8'),
                                       ('PWV','f8',self.pwv.size),
                                       ('PWVDELTA','f8'),
                                       ('O3','f8',self.o3.size),
                                       ('O3DELTA','f8'),
                                       ('LNTAU','f8',self.lnTau.size),
                                       ('LNTAUDELTA','f8'),
                                       ('ALPHA','f8',self.alpha,size),
                                       ('ALPHADELTA','f8'),
                                       ('SECZENITH','f8',self.secZenith.size),
                                       ('SECZENITHDELTA','f8')])

        parStruct['ELEVATION'] = self.elevation
        parStruct['PMBELEVATION'] = self.pmbElevation
        parStruct['PMBSTD'] = self.pmbStd
        parStruct['PWVSTD'] = self.pwvStd
        parStruct['O3STD'] = self.o3Std
        parStruct['TAUSTD'] = self.tauStd
        parStruct['ALPHASTD'] = self.alphaStd
        parStruct['AIRMASSSTD'] = self.secZenithStd
        parStruct['LAMBDARANGE'][:] = self.lambdaRange
        parStruct['LAMBDASTEP'][:] = self.lambdaStep
        parStruct['LAMBDANORM'][:] = self.lambdaNorm

        parStruct['ATMLAMBDA'][:] = self.atmLambda
        parStruct['ATMSTDTRANS'][:] = self.atmStdTrans

        parStruct['PMB'][:] = self.pmb
        parStruct['PMBDELTA'] = self.pmbDelta
        parStruct['PWV'][:] = self.pwv
        parStruct['PWVDELTA'] = self.pwvDelta
        parStruct['O3'][:] = self.o3
        parStruct['O3DELTA'] = self.o3Delta
        parStruct['LNTAU'][:] = self.lnTau
        parStruct['LNTAUDELTA'] = self.lnTauDelta
        parStruct['ALPHA'][:] = self.alpha
        parStruct['ALPHADELTA'] = self.alphaDelta
        parStruct['SECZENITH'][:] = self.secZenith
        parStruct['SECZENITHDELTA'] = self.secZenithDelta

        fits.write_table(parStruct, extname='PARS')

        fits.write_image(self.pwvAtmTable, extname='PWVATM')
        fits.write_image(self.o3AtmTable, extname='O3ATM')
        fits.write_image(self.o2AtmTable, extname='O2ATM')
        fits.write_image(self.rayleighAtmTable, extname='RAYATM')

        fits.close()
