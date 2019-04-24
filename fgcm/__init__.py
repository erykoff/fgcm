from __future__ import division, absolute_import, print_function

import os

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from ._version import __version__, __version_info__

version = __version__

from .fgcmLUT import FgcmLUTMaker
from .fgcmLUT import FgcmLUT
from .modtranGenerator import ModtranGenerator
from .fgcmMakeStars import FgcmMakeStars
from . import fgcmUtilities
from .fgcmConfig import FgcmConfig
from .fgcmParameters import FgcmParameters
from .fgcmStars import FgcmStars
from .fgcmChisq import FgcmChisq
from .fgcmBrightObs import FgcmBrightObs
from .fgcmGray import FgcmGray
from .fgcmSuperStarFlat import FgcmSuperStarFlat
from .fgcmRetrieval import FgcmRetrieval
from .fgcmApertureCorrection import FgcmApertureCorrection
from .fgcmExposureSelector import FgcmExposureSelector
from .fgcmFitCycle import FgcmFitCycle
from .fgcmZeropoints import FgcmZeropoints
from .fgcmZeropoints import FgcmZeropointPlotter
from .fgcmLogger import FgcmLogger
from .fgcmSigFgcm import FgcmSigFgcm
from .fgcmFlagVariables import FgcmFlagVariables
from .fgcmRetrieveAtmosphere import FgcmRetrieveAtmosphere
from .fgcmAtmosphereTable import FgcmAtmosphereTable
from .fgcmModelMagErrors import FgcmModelMagErrors
from .fgcmConnectivity import FgcmConnectivity
from .fgcmSigmaCal import FgcmSigmaCal
from .fgcmSigmaRef import FgcmSigmaRef
from .fgcmQeSysSlope import FgcmQeSysSlope
from .fgcmComputeStepUnits import FgcmComputeStepUnits
# from .fgcmComputeStepUnits2 import FgcmComputeStepUnits2
from .fgcmComputeStepUnits3 import FgcmComputeStepUnits3
