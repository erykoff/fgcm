#!/usr/bin/env python


from __future__ import print_function

import os
import numpy as np
import subprocess
import tempfile
import shutil
import re

GMCM2_ATMCM = 8.03e-4
CO2MX_DEFAULT = 360.0


class ModtranGenerator(object):
    """
    A class to generate a single MODTRAN atmosphere

    parameters
    ----------
    elevation: float
        observatory elevation

    requires
    --------
    environment variable MODTRAN_PATH to be set with modtran executable
       and data files
    """
    def __init__(self, elevation):

        if (elevation < 0.0 or elevation > 11000.0):
            raise ValueError("Elevation out of range.")

        # store the elevation
        self.elevation = elevation
        self.tempRunPath = None
        self.atm = None

        # where is MODTRAN?
        try:
            self.modtranPath = os.environ['MODTRAN_PATH']
        except:
            raise ValueError("Could not find environment variable MODTRAN_PATH")

        # make a temporary path for running...
        try:
            self.tempRunPath = tempfile.mkdtemp()
        except:
            raise IOError("Could not make temporary directory for MODTRAN")

        # and make the modtran.in file
        self.modtranRoot = 'modtranGenerator'
        try:
            with open("%s/modroot.in" % (self.tempRunPath),'w') as f:
                f.write("%s\n" % (self.modtranRoot))
        except:
            raise IOError("Could not write modtran.in")

        # and link in the modtran exe and DATA
        try:
            os.symlink('%s/runmodt4.exe' % (self.modtranPath), self.tempRunPath+'/runmodt4.exe')
            os.symlink('%s/DATA' % (self.modtranPath), self.tempRunPath+'/DATA')
        except:
            raise IOError("Could not link in modtran and data")

        # now do a test run to get conversions
        self._defaultConversionRun()

    def _defaultConversionRun(self):
        """
        internal method to compute default conversion factors
        """
        # set to default values
        self.o3Scaled = 1.0
        self.pwvScaled = 1.0
        self.zenith = 0.0
        self.co2MX = CO2MX_DEFAULT
        # these are unimportant for the test...
        self.lambdaRange = [500.0,600.0]
        self.lambdaStep = 1.0

        # run modtran
        self._runModtran()

        # and process the tp6 output file to get conversions
        with open("%s/%s.tp6" % (self.tempRunPath, self.modtranRoot), 'r') as f:
            lines=f.readlines()

        # for OZONE, find line with "OZONE DENSITIES".
        #  we want the next line
        for i,line in enumerate(lines):
            m=re.search("OZONE DENSITIES", line)
            if m is not None:
                o3Line = lines[i+1]
                break
        m=re.search("CONTAINED\ +(\d+\.\d+)\ +", o3Line)
        self._o3DefaultSealevel = float(m.groups()[0])

        # for PWV, find line with "THE WATER PROFILE".
        #  we want the next line
        for i,line in enumerate(lines):
            m=re.search("THE WATER PROFILE", line)
            if m is not None:
                pwvLine = lines[i+1]
                break
        m=re.search("INITIAL:\ +(\d+\.\d+)\ +", pwvLine)
        self._h2oDefaultSealevel = float(m.groups()[0])

        # for the H2O, O3 values at elevation, find line "H2O         O3" (9 spaces)
        #  we want line+=2
        for i,line in enumerate(lines):
            m=re.search("H2O         O3", line)
            if m is not None:
                valueLine = lines[i+2]
                break
        parts=valueLine.split()
        pwvElevation = float(parts[0])
        o3Elevation = float(parts[1])

        # now compute conversion factors
        #  pwvScaleValue = (gm/cm^2 / atm-cm) * (atm-cm) * 10 (mm / gm/cm^2)
        self.pwvScaleValue = GMCM2_ATMCM * pwvElevation * 10  # in mm of H2O

        #  o3ScaleValue = (atm-cm) * (1000 Dobson / atm-cm)
        self.o3ScaleValue = o3Elevation * 1000  # in Dobson

        # And Pressure, from lapse formula
        Pmb = 1013.25    # sea level, standard, millibars (mb)
        Tb = 288.15      # sea level, standard, Kelvin
        Lb = -0.0065     # lapse rate, standard, K/m
        g0 = 9.80665     # gravitational acceleration, m/s^2
        Matm = 0.0289644 # molar mass of air, kg/mol
        Rstar = 8.31432  # universal gas constant, N-m / mol-K

        self.pmbElevation = Pmb*(Tb/(Tb + Lb*self.elevation))** \
                             ((g0*Matm)/(Rstar*Lb))

    def _runModtran(self):
        """
        internal method to actually run modtran in a temporary directory
        """
        with open('%s/%s.tp5' % (self.tempRunPath, self.modtranRoot), 'w') as f:
            # Card 1:    template model
            f.write('MM  6    3    0    0    0    0    0    0    0    0    0    0   -1\n')
            # Card 1a:   H20 and O3 scalings
            f.write('    1    0   %7.3f  %8.5f  %8.5f   T\n' % (self.co2MX, self.pwvScaled, self.o3Scaled))

            # Card 1a2:  template computational resolution
            f.write('DATA/B2001_05.BIN\n')
            #f.write('DATA/B2001_01.BIN\n')

            # Card 2:    template mix and miscellaneous
            f.write('    0    0    0    0    0    0     0.000      .000      .000      .000\n')

            # Card 3:    line of sight for this run
            f.write('    %6.3f     0.000    %6.3f     0.000     0.000     0.000    0          0.000\n' % (self.elevation/1000.0, self.zenith) )

            # Card 4:    template wavelength range and resolution
            # 1.00TN:
            #  1.00 is slit FGCM
            #  T is for transmittances output (also R: Radiances)
            #  N for nanometers for output (also W: wavenumbers, M: microns)

            # NGAA:
            #  N is for nanometers (also W: wavenumber, M micron)
            #  G is for Gaussian (also T: Triangular, R: rectangular, S: sinc, C: sinc^2, H: Hamming)
            #  A is for absolute FGCM (also R: relative)
            #  A is for degrade all radiance and transmittance components
            f.write('   %7.1f   %7.1f     %5.2f      1.00TN       $NGAA\n' % (self.lambdaRange[0], self.lambdaRange[1], self.lambdaStep))
            #f.write('   %7.1f   %7.1f     %5.2f      0.1TN       $NGAA\n' % (self.lambdaRange[0], self.lambdaRange[1], self.lambdaStep))

            # Line of sight cards
            f.write('    0\n')

        subprocess.call('cd %s; ./runmodt4.exe' % (self.tempRunPath),shell=True)

    def __call__(self, pmb=778.0, pwv=3.0, o3=263.0, tau=0.03, lambdaNorm=7750.0,
                 alpha=1.0, zenith=33.55731, co2MX=CO2MX_DEFAULT,
                 lambdaRange=[300.0,1100.0], lambdaStep=0.5):
        """
        parameters
        ----------
        pmb: float, optional
            barometric pressure.  Default is 778 mb
        pwv: float, optional
            precipitable water vapor.  Default is 3.0 mm
        o3: float, optional
            ozone.  Default is 263 Dobson
        tau: float, optional
            aerosol optical index at lambdaNorm.  Default is 0.03
        lambdaNorm: float, optional
            wavelength of tau.  Default is 7750 Angstrom
        alpha: float, optional
            aerosol power-law index.  Default is 1.0
        zenith: float, optional
            zenith distance.  Default is 33.55731 degrees
        co2MX: float, optional
            CO2 mixing parameter.  Default is 360.0
        lambdaRange: [float,float], optional
            wavelength range to generate atmosphere.  Default is [380,1100] nm
        lambdaStep: float, optional
            wavelength step size.  Default is 0.5 nm

        returns
        -------
        atmosphere numpy record array
        """

        if (pmb < 0.0):
            raise ValueError("Pressure pmb out of range.")
        if (pwv <= 0.0):
            raise ValueError("PWV value pwv cannot be zero or negative.")
        if (o3 <= 0.0):
            raise ValueError("O3 value cannot be zero or negative.")
        if (tau < 0.0):
            raise ValueError("tau value cannot be negative.")
        if (lambdaNorm < 100.0):
            raise ValueError("lambdaNorm must be >= 100 Angstrom.")
        if (zenith < 0.0 or zenith > 90.0):
            raise ValueError("zenith distance must be 0-90.0")
        if (lambdaRange[0] < 50.0 or lambdaRange[1] < 50.0 or lambdaRange[1] <= lambdaRange[0]):
            raise ValueError("lambdaRange must be >=50.0 and low to high")

        # record values
        self.pmb = pmb
        self.pwv = pwv
        self.o3 = o3
        self.tau = tau
        self.lambdaNorm = lambdaNorm
        self.alpha = alpha
        self.zenith = zenith
        self.co2MX = co2MX
        self.lambdaRange = lambdaRange
        self.lambdaStep = lambdaStep

        # scale the values...
        self._elevationKm = self.elevation*1000.0
        self.pwvScaled = (self.pwv / self.pwvScaleValue)   # MODTRAN units
        self.o3Scaled = (self.o3 / self.o3ScaleValue)      # MODTRAN units

        # and run...
        self._runModtran()

        # read in 7sc
        with open("%s/%s.7sc" % (self.tempRunPath, self.modtranRoot), 'r') as f:
            lines=f.readlines()

        nHeaderLines = 12
        nSteps = len(lines)-nHeaderLines-1
        self.atm=np.zeros(nSteps,dtype=[('LAMBDA','f4'),
                                         ('COMBINED','f4'),
                                         ('H2O','f4'),
                                         ('O2','f4'),
                                         ('O3','f4'),
                                         ('RAYLEIGH','f4'),
                                         ('AEROSOL','f4'),
                                         ('PMBFACTOR','f4')])
        i=0
        for line in lines[12:-1]:
            parts=line.split()
            self.atm['LAMBDA'][i] = float(parts[0])*10.0  # convert to Angstrom
            self.atm['COMBINED'][i] = float(parts[1])
            self.atm['H2O'][i] = float(parts[2])*float(parts[7])
            self.atm['O2'][i] = float(parts[3])*float(parts[5])
            self.atm['O3'][i] = float(parts[4])
            self.atm['RAYLEIGH'][i] = float(parts[8])
            i=i+1

        # set default
        self.atm['AEROSOL'][:] = 1.0
        self.atm['PMBFACTOR'][:] = 1.0

        # apply aerosol
        secz = 1./np.cos(self.zenith*np.pi/180.)
        airmass = secz - 0.0018167*(secz-1.0) -  0.002875*(secz-1.0)**2.0 - 0.0008083*(secz-1.0)**3.0

        self.atm['AEROSOL'][:] = np.exp(-1.0*self.tau*airmass*(self.atm['LAMBDA'][:]/self.lambdaNorm)**(-self.alpha))

        # and finally apply the pressure correction
        pmbMolecularScattering = np.exp(-(self.pmb - self.pmbElevation)/self.pmbElevation)
        pmbMolecularAbsorption = pmbMolecularScattering**0.6

        self.atm['PMBFACTOR'][:] = pmbMolecularScattering * pmbMolecularAbsorption

        self.atm['COMBINED'][:] = self.atm['PMBFACTOR'][:] * self.atm['O2'][:] * \
            self.atm['RAYLEIGH'][:] * self.atm['H2O'] * self.atm['O3'] * \
            self.atm['AEROSOL']

        return self.atm


    def saveAtm(self, filename, clobber=False):
        """
        Save the atmosphere to a fits file

        parameters
        ----------
        filename: string
            output filename
        clobber: bool, optional
            clobber existing output file.  Default is False
        """

        import fitsio

        hdr = fitsio.FITSHDR()
        hdr['PMB'] = self.pmb
        hdr['PWV'] = self.pwv
        hdr['O3'] = self.o3
        hdr['TAU'] = self.tau
        hdr['LAMNORM'] = self.lambdaNorm
        hdr['ALPHA'] = self.alpha
        hdr['ZENITH'] = self.zenith
        hdr['CO2MX'] = self.co2MX
        hdr['ELEV'] = self.elevation

        fitsio.write(filename, self.atm, header=hdr, clobber=clobber)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tempRunPath is not None:
            if (os.path.isdir(self.tempRunPath)):
                shutil.rmtree(self.tempRunPath)
            self.tempRunPath = None

    def __del__(self):
        if self.tempRunPath is not None:
            if (os.path.isdir(self.tempRunPath)):
                shutil.rmtree(self.tempRunPath)
            self.tempRunPath = None
