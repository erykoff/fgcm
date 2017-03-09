from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import yaml
import esutil

class DESGPSFormatter(object):
    """
    """

    def __init__(self,gpsFile):
        self.gpsFile = gpsFile

        if not os.path.isfile(self.gpsFile):
            raise ValueError("Could not find gpsFile: %s" % (self.gpsFile))

    def __call__(self,outfile,clobber=False):
        """
        """
        gpsMJD = []
        gpsPWV = []

        with open(self.gpsFile,'r') as f:
            header = f.readline()
            line = f.readline().split()
            while line[0] != 'EOF':
                gpsMJD.append(float(line[0]))
                gpsPWV.append(float(line[1]))
                line = f.readline().split()

        gpsTable = np.zeros(len(gpsMJD),dtype=[('MJD','f8'),
                                               ('PWV','f4')])
        gpsTable['MJD'] = gpsMJD
        gpsTable['PWV'] = gpsPWV

        fitsio.write(outfile,gpsTable,clobber=clobber)
