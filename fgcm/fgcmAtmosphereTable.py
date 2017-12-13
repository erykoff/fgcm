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
    #def __init__(self,atmTableConfig,fgcmLog):

    #    self.fgcmLog = fgcmLog

    def __init__(self, atmosphereTableName):
        pass


class FgcmAtmosphereTableGenerator(object):
    """
    """

    def __init__(self, lutConfig, fgcmLog):
        pass

    def generateTable(self):
        """
        """
        pass

    def saveTable(self):
        """
        """
        # if we can...
        pass
