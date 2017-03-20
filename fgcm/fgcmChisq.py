from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import time

from fgcmUtilities import _pickle_method
from fgcmUtilities import resourceUsage

import types
import copy_reg
import sharedmem as shm
import multiprocessing
from multiprocessing import Pool

#from fgcmLUT import FgcmLUTSHM

copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmChisq(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmStars,fgcmLUT):

        resourceUsage('Start of chisq init')

        # does this need to be shm'd?
        self.fgcmPars = fgcmPars

        # this is shm'd
        self.fgcmLUT = fgcmLUT

        self.fgcmStars = fgcmStars

        # not sure what we need the config for

        # get values out of the fgcmPars

        resourceUsage('End of chisq init')

    def __call__(self,fitParams,computeDerivatives=False,computeSEDSlope=False,fitterUnits=False):
        """
        """

        # computeDerivatives: do we want to compute the derivatives?
        # computeSEDSlope: compute SED Slope and recompute mean mags?
        # fitterUnits: units of the fitter or "true" units?

        # for things that need to be changed, we need to create an array *here*
        # I think.  And copy it back out.  Sigh.

        resourceUsage('Start of call')

        # this is the function that will be called by the fitter, I believe.

        # unpack the parameters...

        # will want to make a pool

        pool = Pool(processes=4)
        resourceUsage('premap')
        pool.map(self._worker,xrange(20))
        resourceUsage('midmap')
        pool.map(self._worker,xrange(20))
        resourceUsage('end')

    def _worker(self,objIndex):
        """
        """

        # this runs on an individual object index...
        print("%d: %d" % (objIndex, self.fgcmStars.objID[objIndex]))
        time.sleep(0.5)
