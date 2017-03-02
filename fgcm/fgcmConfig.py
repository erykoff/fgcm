from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import yaml

from fgcmUtilities import _pickle_method

import types
import copy_reg
import sharedmem as shm

copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmConfig(object):
    """
    """
    def __init__(self,configFile):
        self.configFile = configFile

        with open(self.configFile) as f:
            configDict = yaml.load(f)

        requiredKeys=['exposureFile','UTBoundary',
                      'washDates','lutFile','expField',
                      'ccdField','exptimeField','pmbField','mjdField',
                      'hourField','telRAField','telDecField','latitude']

        for key in requiredKeys:
            if (key not in configDict):
                raise ValueError("required %s not in configFile" % (key))

        self.exposureFile = configDict['exposureFile']
        self.UTBoundary = configDict['UTBoundary']
        self.washDates = configDict['washDates']
        self.lutFile = configDict['lutFile']
        self.expField = configDict['expField']
        self.ccdField = configDict['ccdField']
        self.exptimeField = configDict['exptimeField']
        self.pmbField = configDict['pmbField']
        self.mjdField = configDict['mjdField']
        self.hourField = configDict['hourField']
        self.telRAField = configDict['telRAField']
        self.telDecField = configDict['telDecField']
        self.latitude = configDict['latitude']
        self.cosLatitude = np.cos(np.radians(self.latitude))
        self.sinLatitude = np.sin(np.radians(self.latitude))

        if 'pwvFile' in configDict:
            self.pwvFile = configDict['pwvFile']
        else:
            self.pwvFile = None

        if 'tauFile' in configDict:
            self.tauFile = configDict['tauFile']
        else:
            self.tauFile = None

        # and look at the lutFile
        lutStats=fitsio.read(self.lutFile,ext='INDEX')

        self.nCCD = lutStats['NCCD']
        self.bands = lutStats['BANDS']
        self.pmbRange = np.array([np.min(lutStats['PMB']),np.max(lutStats['PMB'])])
        self.pwvRange = np.array([np.min(lutStats['PWV']),np.max(lutStats['PWV'])])
        self.O3Range = np.array([np.min(lutStats['O3']),np.max(lutStats['O3'])])
        self.tauRange = np.array([np.min(lutStats['TAU']),np.max(lutStats['TAU'])])
        self.alphaRange = np.array([np.min(lutStats['ALPHA']),np.max(lutStats['ALPHA'])])
        self.zenithRange = np.array([np.min(lutStats['ZENITH']),np.max(lutStats['ZENITH'])])

        # and look at the exposure file and grab some stats
        expInfo = fitsio.read(self.exposureFile,ext=1)
        self.expRange = np.array([np.min(expInfo[self.expField]),np.max(expInfo[self.expField])])
        self.mjdRange = np.array([np.min(expInfo[self.mjdField]),np.max(expInfo[self.mjdField])])
        self.nExp = expInfo.size

