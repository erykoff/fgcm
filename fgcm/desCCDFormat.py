from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import yaml
import esutil

class DESCCDFormatter(object):
    """
    """

    def __init__(self,ccdOffsetFile):
        self.ccdOffsetFile = ccdOffsetFile

        if (not os.path.isfile(self.ccdOffsetFile)):
            raise ValueError("Could not find ccdOffsetFile: %s" % (self.ccdOffsetFile))

    def __call__(self,outFile,clobber=False):

        offsetIn = np.loadtxt(self.ccdOffsetFile,skiprows=1)

        ccdOffsets = np.zeros(offsetIn.shape[0],dtype=[('CCDNUM','i4'),
                                                       ('DELTA_RA','f8'),
                                                       ('DELTA_DEC','f8'),
                                                       ('RA_SIZE','f8'),
                                                       ('DEC_SIZE','f8')])

        ccdOffsets['CCDNUM'] = offsetIn[:,0]
        ccdOffsets['DELTA_RA'] = offsetIn[:,1]
        ccdOffsets['DELTA_DEC'] = offsetIn[:,2]

        ccdOffsets['RA_SIZE'][:] = 4096*0.263 / 3600.0
        ccdOffsets['DEC_SIZE'][:] = 2048*0.263 / 3600.0

        fitsio.write(outFile,ccdOffsets,clobber=clobber)

