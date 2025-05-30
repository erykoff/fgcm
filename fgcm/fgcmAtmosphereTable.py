import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import os
import sys

hasLsstResources = True
try:
    from lsst.resources.packageresource import PackageResourcePath
except ImportError:
    hasLsstResources = False
    from pkg_resources import resource_exists
    from pkg_resources import resource_filename
    from pkg_resources import resource_listdir

try:
    import fitsio
    fits_package = 'fitsio'
except ImportError:
    try:
        import astropy.io.fits as pyfits
        fits_package = 'pyfits'
    except ImportError:
        try:
            import pyfits
            fits_package = 'pyfits'
        except ImportError:
            pass

from .modtranGenerator import ModtranGenerator

from .fgcmLogger import FgcmLogger

class FgcmAtmosphereTable(object):
    """
    Class to generate, store, and interpolate pre-computed atmospheres.

    parameters
    ----------
    atmConfig: dict
       Dictionary with FgcmAtmosphereTable config variables
    atmosphereTableFile: str, optional
       Name of the table file
    fgcmLog: FgcmLog, optional
       Logger
    """

    def __init__(self, atmConfig, atmosphereTableFile=None, fgcmLog=None):

        if fgcmLog is None:
            self.fgcmLog = FgcmLogger('dummy.log', 'INFO', printLogger=True)
        else:
            self.fgcmLog = fgcmLog

        if atmosphereTableFile is not None:
            # We have an atmosphereTableFile, so read that
            self.atmosphereTableFile = atmosphereTableFile
            self.atmConfig = atmConfig
            self.fgcmLog.info("Using atmosphere table %s" % (atmosphereTableFile))
        else:
            # get modTran stuff ready since we're going to be generating
            # also check the config
            requiredKeys = ['elevation',
                            'pmbRange','pmbSteps',
                            'pwvRange','pwvSteps',
                            'o3Range','o3Steps',
                            'tauRange','tauSteps',
                            'alphaRange','alphaSteps',
                            'zenithRange','zenithSteps',
                            'pmbStd','pwvStd','o3Std',
                            'tauStd','alphaStd','airmassStd']

            for key in requiredKeys:
                if key not in atmConfig:
                    raise ValueError("Required %s not in atmConfig" % (key))
                if 'Range' in key:
                    if len(atmConfig[key]) != 2:
                        raise ValueError("%s must have 2 elements" % (key))

            self.atmConfig = atmConfig

            self.elevation = self.atmConfig['elevation']

            self.modGen = ModtranGenerator(self.elevation)
            self.pmbElevation = self.modGen.pmbElevation

            self.pmbStd = self.atmConfig['pmbStd']
            self.pwvStd = self.atmConfig['pwvStd']
            self.lnPwvStd = np.log(self.pwvStd)
            self.o3Std = self.atmConfig['o3Std']
            self.tauStd = self.atmConfig['tauStd']
            self.lnTauStd = np.log(self.tauStd)
            self.alphaStd = self.atmConfig['alphaStd']
            self.secZenithStd = self.atmConfig['airmassStd']
            self.zenithStd = np.arccos(1./self.secZenithStd)*180./np.pi

            if ('lambdaRange' in self.atmConfig):
                self.lambdaRange = np.array(self.atmConfig['lambdaRange'])
            else:
                self.lambdaRange = np.array([3000.0,11000.0])

            if ('lambdaStep' in self.atmConfig):
                self.lambdaStep = self.atmConfig['lambdaStep']
            else:
                self.lambdaStep = 0.5

            if ('lambdaNorm' in self.atmConfig):
                self.lambdaNorm = self.atmConfig['lambdaNorm']
            else:
                self.lambdaNorm = 7750.0

        self.o2Interpolator = None
        self.o3Interpolator = None
        self.rayleighInterpolator = None
        self.lnPwvInterpolator = None


    @staticmethod
    def getAvailableTables():
        """
        Get list of installed tables

        parameters
        ----------
        None

        returns
        -------
        List of installed table names
        """

        try:
            files = resource_listdir(__name__,'data/tables/')
        except:
            raise IOError("Could not find associated data/tables path!")

        # build a dictionary and put in the key details of each...

        availableTables = {}

        for f in files:
            availableTables[os.path.basename(f)] = FgcmAtmosphereTable.getInfoDict(f)

        return availableTables

    @staticmethod
    def getInfoDict(atmosphereTableFile):
        """
        Get information dictionary on a table file

        parameters
        ----------
        atmosphereTableFile: str
           Name of the table file

        returns
        -------
        infoDict: dict
           Dictionary with key parameters describing the table
        """

        if fits_package == 'fitsio':
            parStruct = fitsio.read(atmosphereTableFile, ext='PARS')
        elif fits_package == 'pyfits':
            parStruct = pyfits.getdata(atmosphereTableFile, ext=('PARS', 1))
        else:
            raise IOError("Reading atmosphere tables not supported without fitsio/astropy.io.fits/pyfits")

        infoDict = {'elevation':parStruct['ELEVATION'][0],
                    'pmbRange':[parStruct['PMB'][0][0], parStruct['PMB'][0][-1]],
                    'pmbSteps':parStruct['PMB'][0].size,
                    'pwvRange':[np.exp(parStruct['LNPWV'][0][0]),
                                np.exp(parStruct['LNPWV'][0][-1])],
                    'pwvSteps':parStruct['LNPWV'][0].size,
                    'o3Range':[parStruct['O3'][0][0], parStruct['O3'][0][-1]],
                    'o3Steps':parStruct['O3'][0].size,
                    'tauRange':[np.exp(parStruct['LNTAU'][0][0]), np.exp(parStruct['LNTAU'][0][-1])],
                    'tauSteps':parStruct['LNTAU'][0].size,
                    'alphaRange':[parStruct['ALPHA'][0][0], parStruct['ALPHA'][0][-1]],
                    'alphaSteps':parStruct['ALPHA'][0].size,
                    'zenithRange':[np.rad2deg(np.arccos(1./parStruct['SECZENITH'][0][0])),
                                   np.rad2deg(np.arccos(1./parStruct['SECZENITH'][0][-1]))],
                    'zenithSteps':parStruct['SECZENITH'][0].size,
                    'pmbStd':parStruct['PMBSTD'][0],
                    'pwvStd':parStruct['PWVSTD'][0],
                    'o3Std':parStruct['O3STD'][0],
                    'tauStd':parStruct['TAUSTD'][0],
                    'alphaStd':parStruct['ALPHASTD'][0],
                    'airmassStd':parStruct['AIRMASSSTD'][0]}

        return infoDict

    @classmethod
    def initWithTableName(cls, atmosphereTableName):
        """
        Initialize FgcmAtmosphereTable with a table name

        parameters
        ----------
        atmosphereTableName: str
           Name of atmosphere table

        returns
        -------
        FgcmAtmosphereTable: FgcmAtmosphereTable object

        notes
        -----
        Raises IOError if atmosphereTableName couldn't be found
        """
        # first, check if we have something in the path...
        if os.path.isfile(atmosphereTableName):
            atmosphereTableFile = os.path.abspath(atmosphereTableName)
        else:
            if hasLsstResources:
                rootResource = PackageResourcePath("resource://fgcm/data/tables", forceDirectory=True)

                if rootResource.join(atmosphereTableName).exists():
                    resource = rootResource.join(atmosphereTableName)
                elif rootResource.join(atmosphereTableName + ".fits").exists():
                    resource = rootResource.join(atmosphereTableName + ".fits")
                else:
                    raise IOError(
                        "Could not find atmosphereTableName (%s) in the path or in data/tables/" %
                        (atmosphereTableName)
                    )

                with resource.as_local() as loc:
                    atmosphereTableFile = loc.ospath

            else:
                # allow for a name with or without .fits extension
                if resource_exists(__name__, 'data/tables/%s' % (atmosphereTableName)):
                    testFile = 'data/tables/%s' % (atmosphereTableName)
                elif resource_exists(__name__, 'data/tables/%s.fits' % (atmosphereTableName)):
                    testFile = 'data/tables/%s.fits' % (atmosphereTableName)
                else:
                    raise IOError("Could not find atmosphereTableName (%s) in the path or in data/tables/" % (atmosphereTableName))
                try:
                    atmosphereTableFile = resource_filename(__name__, testFile)
                except:
                    raise IOError("Error finding atmosphereTableName (%s)" % (atmosphereTableName))

        # will set self.atmConfig
        atmConfig = FgcmAtmosphereTable.getInfoDict(atmosphereTableFile)

        return cls(atmConfig, atmosphereTableFile=atmosphereTableFile)

    def loadTable(self):
        """
        Load atmosphere table

        parameters
        ----------
        None

        returns
        -------
        None
        """

        # note that at this point these are the only options
        if fits_package == 'fitsio':
            parStruct = fitsio.read(self.atmosphereTableFile, ext='PARS')
        else:
            parStruct = pyfits.getdata(self.atmosphereTableFile, ext=('PARS', 1))


        self.elevation = parStruct['ELEVATION'][0].astype(np.float64)
        self.pmbElevation = parStruct['PMBELEVATION'][0].astype(np.float64)
        self.pmbStd = parStruct['PMBSTD'][0].astype(np.float64)
        self.pwvStd = parStruct['PWVSTD'][0].astype(np.float64)
        self.o3Std = parStruct['O3STD'][0].astype(np.float64)
        self.tauStd = parStruct['TAUSTD'][0].astype(np.float64)
        self.alphaStd = parStruct['ALPHASTD'][0].astype(np.float64)
        self.secZenithStd = parStruct['AIRMASSSTD'][0].astype(np.float64)
        self.lambdaRange = parStruct['LAMBDARANGE'][0].astype(np.float64)
        self.lambdaStep = parStruct['LAMBDASTEP'][0].astype(np.float64)
        self.lambdaNorm = parStruct['LAMBDANORM'][0].astype(np.float64)

        self.atmLambda = parStruct['ATMLAMBDA'][0].astype(np.float64)
        self.atmStdTrans = parStruct['ATMSTDTRANS'][0].astype(np.float64)

        self.pmb = parStruct['PMB'][0].astype(np.float64)
        self.pmbDelta = parStruct['PMBDELTA'][0].astype(np.float64)
        self.lnPwv = parStruct['LNPWV'][0].astype(np.float64)
        self.lnPwvDelta = parStruct['LNPWVDELTA'][0].astype(np.float64)
        self.o3 = parStruct['O3'][0].astype(np.float64)
        self.o3Delta = parStruct['O3DELTA'][0].astype(np.float64)
        self.lnTau = parStruct['LNTAU'][0].astype(np.float64)
        self.lnTauDelta = parStruct['LNTAUDELTA'][0].astype(np.float64)
        self.alpha = parStruct['ALPHA'][0].astype(np.float64)
        self.alphaDelta = parStruct['ALPHADELTA'][0].astype(np.float64)
        self.secZenith = parStruct['SECZENITH'][0].astype(np.float64)
        self.secZenithDelta = parStruct['SECZENITHDELTA'][0].astype(np.float64)

        if fits_package == 'fitsio':
            self.pwvAtmTable = fitsio.read(self.atmosphereTableFile, ext='PWVATM')
            self.o3AtmTable = fitsio.read(self.atmosphereTableFile, ext='O3ATM')
            self.o2AtmTable = fitsio.read(self.atmosphereTableFile, ext='O2ATM')
            self.rayleighAtmTable = fitsio.read(self.atmosphereTableFile, ext='RAYATM')
        else:
            self.pwvAtmTable = pyfits.getdata(self.atmosphereTableFile, ext=('PWVATM', 1))
            self.o3AtmTable = pyfits.getdata(self.atmosphereTableFile, ext=('O3ATM', 1))
            self.o2AtmTable = pyfits.getdata(self.atmosphereTableFile, ext=('O2ATM', 1))
            self.rayleighAtmTable = pyfits.getdata(self.atmosphereTableFile, ext=('RAYATM', 1))

    def generateTable(self):
        """
        Generate atmosphere table using MODTRAN

        parameters
        ----------
        None

        returns
        -------
        None
        """
        self.atmStd = self.modGen(pmb=self.pmbStd,pwv=self.pwvStd,
                                  o3=self.o3Std,tau=self.tauStd,
                                  alpha=self.alphaStd,zenith=self.zenithStd,
                                  lambdaRange=self.lambdaRange/10.0,
                                  lambdaStep=self.lambdaStep)
        self.atmLambda = self.atmStd['LAMBDA']
        self.atmStdTrans = self.atmStd['COMBINED']

        # get all the steps
        self.pmb = np.linspace(self.atmConfig['pmbRange'][0],
                               self.atmConfig['pmbRange'][1],
                               num=self.atmConfig['pmbSteps'])
        self.pmbDelta = self.pmb[1] - self.pmb[0]
        pmbPlus = np.append(self.pmb, self.pmb[-1] + self.pmbDelta)

        self.lnPwv = np.linspace(np.log(self.atmConfig['pwvRange'][0]),
                                 np.log(self.atmConfig['pwvRange'][1]),
                                 num=self.atmConfig['pwvSteps'])
        self.lnPwvDelta = self.lnPwv[1] - self.lnPwv[0]
        # We never want this to go above 20 mm.
        lnPwvPlus = np.clip(
            np.append(self.lnPwv, self.lnPwv[-1] + self.lnPwvDelta),
            0.0,
            np.log(20.0)
        )
        pwvPlus = np.exp(lnPwvPlus)

        self.o3 = np.linspace(self.atmConfig['o3Range'][0],
                               self.atmConfig['o3Range'][1],
                               num=self.atmConfig['o3Steps'])
        self.o3Delta = self.o3[1] - self.o3[0]
        o3Plus = np.append(self.o3, self.o3[-1] + self.o3Delta)

        self.lnTau = np.linspace(np.log(self.atmConfig['tauRange'][0]),
                                 np.log(self.atmConfig['tauRange'][1]),
                                 num=self.atmConfig['tauSteps'])
        self.lnTauDelta = self.lnTau[1] - self.lnTau[0]
        self.tau = np.exp(self.lnTau)
        lnTauPlus = np.append(self.lnTau, self.lnTau[-1] + self.lnTauDelta)
        tauPlus = np.exp(lnTauPlus)

        self.alpha = np.linspace(self.atmConfig['alphaRange'][0],
                               self.atmConfig['alphaRange'][1],
                               num=self.atmConfig['alphaSteps'])
        self.alphaDelta = self.alpha[1] - self.alpha[0]
        alphaPlus = np.append(self.alpha, self.alpha[-1] + self.alphaDelta)

        self.secZenith = np.linspace(1./np.cos(self.atmConfig['zenithRange'][0]*np.pi/180.),
                                     1./np.cos(self.atmConfig['zenithRange'][1]*np.pi/180.),
                                     num=self.atmConfig['zenithSteps'])
        self.secZenithDelta = self.secZenith[1]-self.secZenith[0]
        self.zenith = np.arccos(1./self.secZenith)*180./np.pi
        secZenithPlus = np.append(self.secZenith, self.secZenith[-1] + self.secZenithDelta)
        zenithPlus = np.arccos(1./secZenithPlus)*180./np.pi

        # run MODTRAN a bunch of times

        self.fgcmLog.info("Generating %d*%d=%d PWV atmospheres..." % (pwvPlus.size,zenithPlus.size,pwvPlus.size*zenithPlus.size))
        self.pwvAtmTable = np.zeros((pwvPlus.size,zenithPlus.size,self.atmLambda.size))

        for i in range(pwvPlus.size):
            sys.stdout.write('%d' % (i))
            sys.stdout.flush()
            for j in range(zenithPlus.size):
                sys.stdout.write('.')
                sys.stdout.flush()
                atm=self.modGen(pwv=pwvPlus[i],zenith=zenithPlus[j],
                                lambdaRange=self.lambdaRange/10.0,
                                lambdaStep=self.lambdaStep)
                self.pwvAtmTable[i,j,:] = atm['H2O']

        self.fgcmLog.info("\nGenerating %d*%d=%d O3 atmospheres..." % (o3Plus.size,zenithPlus.size,o3Plus.size*zenithPlus.size))
        self.o3AtmTable = np.zeros((o3Plus.size, zenithPlus.size, self.atmLambda.size))

        for i in range(o3Plus.size):
            sys.stdout.write('%d' % (i))
            sys.stdout.flush()
            for j in range(zenithPlus.size):
                sys.stdout.write('.')
                sys.stdout.flush()
                atm=self.modGen(o3=o3Plus[i],zenith=zenithPlus[j],
                                lambdaRange=self.lambdaRange/10.0,
                                lambdaStep=self.lambdaStep)
                self.o3AtmTable[i,j,:] = atm['O3']

        self.fgcmLog.info("\nGenerating %d O2/Rayleigh atmospheres..." % (zenithPlus.size))

        self.o2AtmTable = np.zeros((zenithPlus.size, self.atmLambda.size))
        self.rayleighAtmTable = np.zeros((zenithPlus.size, self.atmLambda.size))

        for j in range(zenithPlus.size):
            sys.stdout.write('.')
            sys.stdout.flush()
            atm=self.modGen(zenith=zenithPlus[j],
                            lambdaRange=self.lambdaRange/10.0,
                            lambdaStep=self.lambdaStep)
            self.o2AtmTable[j,:] = atm['O2']
            self.rayleighAtmTable[j,:] = atm['RAYLEIGH']

        self.fgcmLog.info("\nDone.")

    def saveTable(self, fileName, clobber=False):
        """
        Save atmosphere table to a file

        parameters
        ----------
        fileName: str
           File name to save file
        clobber: bool, default=False
           Clobber fileName if it exists
        """
        # at the moment, only work with fitsio...

        import fitsio

        if os.path.isfile(fileName) and not clobber:
            raise IOError("File %s already exists and clobber is set to False" % (fileName))

        fits = fitsio.FITS(fileName, 'rw')

        parStruct = np.zeros(1, dtype=[('ELEVATION', 'f4'),
                                       ('PMBELEVATION', 'f4'),
                                       ('PMBSTD', 'f4'),
                                       ('PWVSTD', 'f4'),
                                       ('O3STD', 'f4'),
                                       ('TAUSTD', 'f4'),
                                       ('ALPHASTD', 'f4'),
                                       ('AIRMASSSTD', 'f4'),
                                       ('LAMBDARANGE', 'f4',2),
                                       ('LAMBDASTEP', 'f4'),
                                       ('LAMBDANORM', 'f4'),
                                       ('ATMLAMBDA', 'f4', (self.atmLambda.size, )),
                                       ('ATMSTDTRANS', 'f4', (self.atmStdTrans.size, )),
                                       ('PMB', 'f4', (self.pmb.size, )),
                                       ('PMBDELTA', 'f4'),
                                       ('LNPWV', 'f4', (self.lnPwv.size, )),
                                       ('LNPWVDELTA', 'f4'),
                                       ('O3', 'f4', (self.o3.size, )),
                                       ('O3DELTA', 'f4'),
                                       ('LNTAU', 'f4', (self.lnTau.size, )),
                                       ('LNTAUDELTA', 'f4'),
                                       ('ALPHA', 'f4', (self.alpha.size, )),
                                       ('ALPHADELTA', 'f4'),
                                       ('SECZENITH', 'f4', (self.secZenith.size,)),
                                       ('SECZENITHDELTA', 'f4')])

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
        parStruct['LNPWV'][:] = self.lnPwv
        parStruct['LNPWVDELTA'] = self.lnPwvDelta
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


    def interpolateAtmosphere(self, lam=None, pmb=None, pwv=None, o3=None, tau=None,
                              alpha=None, zenith=None, ctranslamstd=None):
        """
        Interpolate the atmosphere table to generate an atmosphere without
        requiring MODTRAN be installed.

        parameters
        ----------
        lam: float array, optional
           wavelengths.  Default is self.atmLambda
        pmb: float, optional
           pressure in millibar.  Default to pmbStd
        pwv: float, optional
           pwv in mm.  Default to pwvStd
        o3: float, optional
           ozone in Dobson.  Default to o3Std
        tau: float, optional
           Aerosol optical depth.  Default to tauStd
        alpha: float, optional
           Aerosol slope.  Default to alphaStd
        zenith: float, optional
           Zenith angle (degrees).  Default to zenithStd
        ctranslamstd: [ctrans, lamstd], optional
           Transmission adjustment constant and lambdastd.
           Default to [0.0, lamnorm]

        returns
        -------
        atmInterpolated: float array
           Interpolated atmosphere at self.atmLambda wavelengths
        """

        if lam is None:
            lam = self.atmLambda
        if pmb is None:
            pmb = self.pmbStd
        if pwv is None:
            pwv = self.pwvStd
        if o3 is None:
            o3 = self.o3Std
        if tau is None:
            tau = self.tauStd
        if alpha is None:
            alpha = self.alphaStd
        if zenith is None:
            zenith = self.zenithStd
        if ctranslamstd is None:
            ctranslamstd = [0.0, 7750.0]

        if self.o2Interpolator is None:
            secZenithPlus = np.append(self.secZenith, self.secZenith[-1] + self.secZenithDelta)
            self.o2Interpolator = interpolate.RegularGridInterpolator((secZenithPlus, self.atmLambda), self.o2AtmTable)

        if self.rayleighInterpolator is None:
            secZenithPlus = np.append(self.secZenith, self.secZenith[-1] + self.secZenithDelta)
            self.rayleighInterpolator = interpolate.RegularGridInterpolator((secZenithPlus, self.atmLambda), self.rayleighAtmTable)

        if self.lnPwvInterpolator is None:
            lnPwvPlus = np.append(self.lnPwv, self.lnPwv[-1] + self.lnPwvDelta)
            secZenithPlus = np.append(self.secZenith, self.secZenith[-1] + self.secZenithDelta)
            self.lnPwvInterpolator = interpolate.RegularGridInterpolator((lnPwvPlus, secZenithPlus, self.atmLambda), self.pwvAtmTable)

        if self.o3Interpolator is None:
            o3Plus = np.append(self.o3, self.o3[-1] + self.o3Delta)
            secZenithPlus = np.append(self.secZenith, self.secZenith[-1] + self.secZenithDelta)
            self.o3Interpolator = interpolate.RegularGridInterpolator((o3Plus, secZenithPlus, self.atmLambda), self.o3AtmTable)

        # And finally the pressure correction
        pmbMolecularScattering = np.exp(-(pmb - self.pmbElevation) / self.pmbElevation)
        pmbMolecularAbsorption = pmbMolecularScattering ** 0.6
        pmbFactor = pmbMolecularScattering * pmbMolecularAbsorption

        _secZenith = np.clip(1./np.cos(np.radians(zenith)), self.secZenith[0], self.secZenith[-1])
        _lnPwv = np.clip(np.log(pwv), self.lnPwv[0], self.lnPwv[-1])
        _o3 = np.clip(o3, self.o3[0], self.o3[-1])
        atmInterpolated = (pmbFactor *
                           self.o2Interpolator((_secZenith, lam)) *
                           self.rayleighInterpolator((_secZenith, lam)) *
                           self.lnPwvInterpolator((_lnPwv, _secZenith, lam)) *
                           self.o3Interpolator((_o3, _secZenith, lam)) *
                           (1.0 + ctranslamstd[0] * (lam - ctranslamstd[1]) / ctranslamstd[1]) *
                           np.exp(-1.0 * tau * _secZenith * (lam / self.lambdaNorm)**(-alpha)))

        return atmInterpolated
