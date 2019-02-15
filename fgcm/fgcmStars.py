from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import esutil
import time

import matplotlib.pyplot as plt

from .fgcmUtilities import objFlagDict
from .fgcmUtilities import obsFlagDict

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmStars(object):
    """
    Class to describe the stars and observations of the stars.  Note that
     after initialization you must call loadStarsFromFits() or loadStars()
     to load the star information.  This allows an external caller to clear
     out memory after it has been copied to the shared memory buffers.

    parameters
    ----------
    fgcmConfig: FgcmConfig

    Config variables
    ----------------
    minObsPerBand: int
       Minumum number of observations per band to be "good"
    sedFudgeFactors: float array
       Fudge factors for computing fnuprime
    starColorCuts: list
       List that contains lists of [bandIndex0, bandIndex1, minColor, maxColor]
    sigma0Phot: float
       Floor on photometric error to add to every observation
    reserveFraction: float
       Fraction of stars to hold in reserve
    mapLongitudeRef: float
       Reference longitude for plotting maps of stars
    mapNSide: int
       Healpix nside of map plotting.
    superStarSubCCD: bool
       Use sub-ccd info to make superstar flats?
    obsFile: string, only if using fits mode
       Star observation file
    indexFile: string, only if using fits mode
       Star index file
    """

    def __init__(self,fgcmConfig):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.info('Initializing stars.')

        self.obsFile = fgcmConfig.obsFile
        self.indexFile = fgcmConfig.indexFile
        self.refstarFile = fgcmConfig.refstarFile

        self.bands = fgcmConfig.bands
        self.nBands = len(fgcmConfig.bands)
        self.nCCD = fgcmConfig.nCCD
        self.minObsPerBand = fgcmConfig.minObsPerBand
        self.fitBands = fgcmConfig.fitBands
        self.nFitBands = len(fgcmConfig.fitBands)
        self.notFitBands = fgcmConfig.notFitBands
        self.nNotFitBands = len(fgcmConfig.notFitBands)
        self.sedFudgeFactors = fgcmConfig.sedFudgeFactors
        self.starColorCuts = fgcmConfig.starColorCuts
        self.sigma0Phot = fgcmConfig.sigma0Phot
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.expField = fgcmConfig.expField
        self.ccdField = fgcmConfig.ccdField
        self.reserveFraction = fgcmConfig.reserveFraction
        self.modelMagErrors = fgcmConfig.modelMagErrors

        self.inFlagStarFile = fgcmConfig.inFlagStarFile

        self.mapLongitudeRef = fgcmConfig.mapLongitudeRef
        self.mapNSide = fgcmConfig.mapNSide

        self.lambdaStdBand = fgcmConfig.lambdaStdBand

        self.bandRequiredIndex = fgcmConfig.bandRequiredIndex
        self.bandNotRequiredIndex = fgcmConfig.bandNotRequiredIndex
        self.allFitBandsAreRequired = fgcmConfig.allFitBandsAreRequired
        self.bandFitIndex = fgcmConfig.bandFitIndex
        self.bandNotFitIndex = fgcmConfig.bandNotFitIndex
        self.lutFilterNames = fgcmConfig.lutFilterNames
        self.filterToBand = fgcmConfig.filterToBand
        self.colorSplitIndices = fgcmConfig.colorSplitIndices

        self.superStarSubCCD = fgcmConfig.superStarSubCCD
        self.superStarSigmaClip = fgcmConfig.superStarSigmaClip

        self.magStdComputed = False
        self.allMagStdComputed = False
        self.sedSlopeComputed = False

        self.magConstant = 2.5/np.log(10)
        self.zptABNoThroughput = fgcmConfig.zptABNoThroughput
        self.approxThroughput = fgcmConfig.approxThroughput

        self.refStarSnMin = fgcmConfig.refStarSnMin
        self.applyRefStarColorCuts = fgcmConfig.applyRefStarColorCuts

        self.hasXY = False
        self.hasRefstars = False
        self.nRefStars = 0
        self.ccdOffsets = fgcmConfig.ccdOffsets

        self.seeingSubExposure = fgcmConfig.seeingSubExposure

    def loadStarsFromFits(self,fgcmPars,computeNobs=True):
        """
        Load stars from fits files.

        parameters
        ----------
        fgcmPars: FgcmParameters
        computeNobs: bool, default=True
           Compute number of observations of each star/band

        Config variables
        ----------------
        indexFile: string
           Star index file
        obsFile: string
           Star observation file
        inFlagStarFile: string, optional
           Flagged star file
        """

        import fitsio

        # read in the observation indices...
        startTime = time.time()
        self.fgcmLog.info('Reading in observation indices...')
        index = fitsio.read(self.indexFile, ext='INDEX', upper=True)
        self.fgcmLog.info('Done reading in %d observation indices in %.1f seconds.' %
                         (index.size, time.time() - startTime))

        # read in obsfile and cut
        startTime = time.time()
        self.fgcmLog.info('Reading in star observations...')
        obs = fitsio.read(self.obsFile, ext=1, upper=True)
        # cut down to those that are indexed
        obs = obs[index['OBSINDEX']]
        self.fgcmLog.info('Done reading in %d observations in %.1f seconds.' %
                         (obs.size, time.time() - startTime))

        # and positions...
        startTime = time.time()
        self.fgcmLog.info('Reading in star positions...')
        pos = fitsio.read(self.indexFile, ext='POS', upper=True)
        self.fgcmLog.info('Done reading in %d unique star positions in %.1f seconds.' %
                         (pos.size, time.time() - startTime))

        obsFilterName = np.core.defchararray.strip(obs['FILTERNAME'][:])

        # And refstars if available
        if self.refstarFile is not None:
            startTime = time.time()
            self.fgcmLog.info('Reading in reference stars...')
            ref = fitsio.read(self.refstarFile, ext=1, lower=True)
            self.fgcmLog.info('Done reading %d reference starss in %.1f seconds.' %
                              (ref.size, time.time() - startTime))
            refID = ref['fgcm_id']
            refMag = ref['mag']
            refMagErr = ref['mag_err']
        else:
            refID = None
            refMag = None
            refMagErr = None

        if (self.inFlagStarFile is not None):
            self.fgcmLog.info('Reading in list of previous flagged stars from %s' %
                             (self.inFlagStarFile))

            inFlagStars = fitsio.read(self.inFlagStarFile, ext=1, upper=True)

            flagID = inFlagStars['OBJID']
            flagFlag = inFlagStars['OBJFLAG']
        else:
            flagID = None
            flagFlag = None

        if ('X' in obs.dtype.names and 'Y' in obs.dtype.names):
            self.fgcmLog.info('Found X/Y in input observations')
            obsX = obs['X']
            obsY = obs['Y']
        else:
            obsX = None
            obsY = None

        # process
        self.loadStars(fgcmPars,
                       obs[self.expField],
                       obs[self.ccdField],
                       obs['RA'],
                       obs['DEC'],
                       obs['MAG'],
                       obs['MAGERR'],
                       obsFilterName,
                       pos['FGCM_ID'],
                       pos['RA'],
                       pos['DEC'],
                       pos['OBSARRINDEX'],
                       pos['NOBS'],
                       obsX=obsX,
                       obsY=obsY,
                       refID=refID,
                       refMag=refMag,
                       refMagErr=refMagErr,
                       flagID=flagID,
                       flagFlag=flagFlag,
                       computeNobs=computeNobs)

        # and clear memory
        index = None
        obs = None
        pos = None
        ref = None

    def loadStars(self, fgcmPars,
                  obsExp, obsCCD, obsRA, obsDec, obsMag, obsMagErr, obsFilterName,
                  objID, objRA, objDec, objObsIndex, objNobs, obsX=None, obsY=None,
                  refID=None, refMag=None, refMagErr=None,
                  flagID=None, flagFlag=None, computeNobs=True):
        """
        Load stars from arrays

        parameters
        ----------
        fgcmPars: fgcmParameters
        obsExp: int array
           Exposure number (or equivalent) for each observation
        obsCCD: int array
           CCD number (or equivalent) for each observation
        obsRA: double array
           RA for each observation (degrees)
        obsDec: double array
           Dec for each observation (degrees)
        obsMag: float array
           Raw ADU magnitude for each observation
        obsMagErr: float array
           Raw ADU magnitude error for each observation
        obsFilterName: string array
           Filter name for each observation
        objID: int array
           Unique ID number for each object
        objRA: double array
           RA for each object (degrees)
        objDec: double array
           Dec for each object (degrees)
        objObsIndex: int array
           For each object, where in the obs table to look
        objNobs: int array
           number of observations of this object (all bands)
        refID: int array, optional
           ID of each object that is an absolute reference
        refMag: float array, optional
           Absolute mag for each reference object, (nref, nmag).
           Set to >90 for no magnitude.
        refMagErr: float array, optional
           Absolute magnitude error for each reference object (nref, nmag).
           Set to >90 for no magnitude.
        obsX: float array, optional
           x position for each observation
        obsY: float array, optional
           y position for each observation
        flagID: int array, optional
           ID of each object that is flagged from previous cycle
        flagFlag: int array, optional
           Flag value from previous cycle
        computeNobs: bool, default=True
           Compute number of good observations of each object?
        """

        # FIXME: check that these are all the same length!

        self.obsIndexHandle = snmm.createArray(obsRA.size, dtype='i4')
        snmm.getArray(self.obsIndexHandle)[:] = np.arange(obsRA.size)


        # need to stuff into shared memory objects.
        #  nStarObs: total number of observations of all stars
        self.nStarObs = obsRA.size

        #  obsExp: exposure number of individual observation (pointed by obsIndex)
        self.obsExpHandle = snmm.createArray(self.nStarObs,dtype='i4')
        #  obsExpIndex: exposure index
        self.obsExpIndexHandle = snmm.createArray(self.nStarObs,dtype='i4')
        #  obsCCD: ccd number of individual observation
        self.obsCCDHandle = snmm.createArray(self.nStarObs,dtype='i2')
        #  obsBandIndex: band index of individual observation
        self.obsBandIndexHandle = snmm.createArray(self.nStarObs,dtype='i2')
        #  obsLUTFilterIndex: filter index in LUT of individual observation
        self.obsLUTFilterIndexHandle = snmm.createArray(self.nStarObs,dtype='i2')
        #  obsFlag: individual bad observation
        self.obsFlagHandle = snmm.createArray(self.nStarObs,dtype='i2')
        #  obsRA: RA of individual observation
        self.obsRAHandle = snmm.createArray(self.nStarObs,dtype='f8')
        #  obsDec: Declination of individual observation
        self.obsDecHandle = snmm.createArray(self.nStarObs,dtype='f8')
        #  obsSecZenith: secant(zenith) of individual observation
        self.obsSecZenithHandle = snmm.createArray(self.nStarObs,dtype='f8')
        #  obsMagADU: log raw ADU counts of individual observation
        ## FIXME: need to know default zeropoint?
        self.obsMagADUHandle = snmm.createArray(self.nStarObs,dtype='f4')
        #  obsMagADUErr: raw ADU counts error of individual observation
        self.obsMagADUErrHandle = snmm.createArray(self.nStarObs,dtype='f4')
        #  obsMagADUModelErr: modeled ADU counts error of individual observation
        self.obsMagADUModelErrHandle = snmm.createArray(self.nStarObs,dtype='f4')
        #  obsSuperStarApplied: SuperStar correction that was applied
        self.obsSuperStarAppliedHandle = snmm.createArray(self.nStarObs,dtype='f4')
        #  obsMagStd: corrected (to standard passband) mag of individual observation
        self.obsMagStdHandle = snmm.createArray(self.nStarObs,dtype='f4',syncAccess=True)
        if (obsX is not None and obsY is not None):
            self.hasXY = True

            #  obsX: x position on the CCD of the given observation
            self.obsXHandle = snmm.createArray(self.nStarObs,dtype='f4')
            #  obsY: y position on the CCD of the given observation
            self.obsYHandle = snmm.createArray(self.nStarObs,dtype='f4')
        else:
            # hasXY = False
            if self.superStarSubCCD:
                raise ValueError("Input stars do not have x/y but superStarSubCCD is set.")

        if (refID is not None and refMag is not None and refMagErr is not None):
            self.hasRefstars = True

            # Remove any duplicates...
            _, refUInd = np.unique(refID, return_index=True)

            if refUInd.size < refID.size:
                self.fgcmLog.info("Removing %d duplicate reference stars." %
                                  (refID.size - refUInd.size))
                refID = refID[refUInd]
                refMag = refMag[refUInd, :]
                refMagErr = refMagErr[refUInd, :]

            self.nRefstars = refID.size

            # refID: matched ID of reference stars
            self.refIDHandle = snmm.createArray(self.nRefstars, dtype='i4')
            # refMag: absolute magnitudes of reference stars
            self.refMagHandle = snmm.createArray((self.nRefstars, self.nBands), dtype='f4')
            # refMagErr: absolute magnitude errors of reference stars
            self.refMagErrHandle = snmm.createArray((self.nRefstars, self.nBands), dtype='f4')

        snmm.getArray(self.obsExpHandle)[:] = obsExp
        snmm.getArray(self.obsCCDHandle)[:] = obsCCD
        snmm.getArray(self.obsRAHandle)[:] = obsRA
        snmm.getArray(self.obsDecHandle)[:] = obsDec
        # We will apply the approximate AB scaling here, it will make
        # any plots we make have sensible units; will make 99 a sensible sentinal value;
        # and is arbitrary anyway and doesn't enter the fits.
        snmm.getArray(self.obsMagADUHandle)[:] = obsMag + self.zptABNoThroughput
        snmm.getArray(self.obsMagADUErrHandle)[:] = obsMagErr
        snmm.getArray(self.obsMagStdHandle)[:] = obsMag + self.zptABNoThroughput  # same as raw at first
        snmm.getArray(self.obsSuperStarAppliedHandle)[:] = 0.0
        if self.hasXY:
            snmm.getArray(self.obsXHandle)[:] = obsX
            snmm.getArray(self.obsYHandle)[:] = obsY

        if self.hasRefstars:
            # And filter out bad signal to noise, per band, if desired,
            # before filling the arrays
            if self.refStarSnMin > 0.0:
                for i in range(self.nBands):
                    maxErr = (2.5 / np.log(10.)) * (1. / self.refStarSnMin)
                    bad, = np.where(refMagErr[:, i] > maxErr)
                    refMag[bad, i] = 99.0
                    refMagErr[bad, i] = 99.0

            snmm.getArray(self.refIDHandle)[:] = refID
            snmm.getArray(self.refMagHandle)[:, :] = refMag
            snmm.getArray(self.refMagErrHandle)[:, :] = refMagErr


        self.fgcmLog.info('Applying sigma0Phot = %.4f to mag errs' %
                         (self.sigma0Phot))

        obsMagADUErr = snmm.getArray(self.obsMagADUErrHandle)

        obsFlag = snmm.getArray(self.obsFlagHandle)
        bad, = np.where(obsMagADUErr <= 0.0)
        obsFlag[bad] |= obsFlagDict['BAD_ERROR']
        if (bad.size > 0):
            self.fgcmLog.info('Flagging %d observations with bad errors.' %
                             (bad.size))

        obsMagADUErr[:] = np.sqrt(obsMagADUErr[:]**2. + self.sigma0Phot**2.)

        # Initially, we set the model error to the observed error
        obsMagADUModelErr = snmm.getArray(self.obsMagADUModelErrHandle)
        obsMagADUModelErr[:] = obsMagADUErr[:]

        startTime = time.time()
        self.fgcmLog.info('Matching observations to exposure table.')
        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsExpIndex[:] = -1
        a,b=esutil.numpy_util.match(fgcmPars.expArray,
                                    snmm.getArray(self.obsExpHandle)[:])
        obsExpIndex[b] = a
        self.fgcmLog.info('Observations matched in %.1f seconds.' %
                         (time.time() - startTime))

        bad, = np.where(obsExpIndex < 0)
        obsFlag[bad] |= obsFlagDict['NO_EXPOSURE']

        if (bad.size > 0):
            self.fgcmLog.info('Flagging %d observations with no associated exposure.' %
                             (bad.size))

        # match bands and filters to indices
        startTime = time.time()
        self.fgcmLog.info('Matching observations to bands.')

        # new version for multifilter support
        # First, we have the filterNames
        for filterIndex,filterName in enumerate(self.lutFilterNames):
            try:
                bandIndex = self.bands.index(self.filterToBand[filterName])
            except KeyError:
                self.fgcmLog.info('WARNING: observations with filter %s not in config' % (filterName))
                bandIndex = -1

            # obsFilterName is an array from fits/numpy.  filterName needs to be encoded to match
            use, = np.where(obsFilterName == filterName.encode('utf-8'))
            if use.size == 0:
                self.fgcmLog.info('WARNING: no observations in filter %s' % (filterName))
            else:
                snmm.getArray(self.obsLUTFilterIndexHandle)[use] = filterIndex
                snmm.getArray(self.obsBandIndexHandle)[use] = bandIndex

        self.fgcmLog.info('Observations matched in %.1f seconds.' %
                         (time.time() - startTime))

        #  nStars: total number of unique stars
        self.nStars = objID.size

        #  objID: unique object ID
        self.objIDHandle = snmm.createArray(self.nStars,dtype='i4')
        #  objRA: mean RA for object
        self.objRAHandle = snmm.createArray(self.nStars,dtype='f8')
        #  objDec: mean Declination for object
        self.objDecHandle = snmm.createArray(self.nStars,dtype='f8')
        #  objObsIndex: for each object, the first
        self.objObsIndexHandle = snmm.createArray(self.nStars,dtype='i4')
        #  objNobs: number of observations of this object (all bands)
        self.objNobsHandle = snmm.createArray(self.nStars,dtype='i4')
        #  objNGoodObsHandle: number of good observations, per band
        self.objNGoodObsHandle = snmm.createArray((self.nStars,self.nBands),dtype='i4')

        snmm.getArray(self.objIDHandle)[:] = objID
        snmm.getArray(self.objRAHandle)[:] = objRA
        snmm.getArray(self.objDecHandle)[:] = objDec

        snmm.getArray(self.objObsIndexHandle)[:] = objObsIndex
        snmm.getArray(self.objNobsHandle)[:] = objNobs


        #  minObjID: minimum object ID
        self.minObjID = np.min(snmm.getArray(self.objIDHandle))
        #  maxObjID: maximum object ID
        self.maxObjID = np.max(snmm.getArray(self.objIDHandle))

        #  obsObjIDIndex: object ID Index of each observation
        #    (to get objID, then objID[obsObjIDIndex]

        startTime = time.time()
        self.fgcmLog.info('Indexing star observations...')
        self.obsObjIDIndexHandle = snmm.createArray(self.nStarObs,dtype='i4')
        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        objID = snmm.getArray(self.objIDHandle)
        obsIndex = snmm.getArray(self.obsIndexHandle)
        objObsIndex = snmm.getArray(self.objObsIndexHandle)
        objNobs = snmm.getArray(self.objNobsHandle)
        ## FIXME: check if this extra obsIndex reference is necessary or not.
        ##   probably extraneous.
        for i in xrange(self.nStars):
            obsObjIDIndex[obsIndex[objObsIndex[i]:objObsIndex[i]+objNobs[i]]] = i
        self.fgcmLog.info('Done indexing in %.1f seconds.' %
                         (time.time() - startTime))

        # And we need to match the reference stars if necessary
        if self.hasRefstars:
            # self.refIdHandle
            startTime = time.time()
            self.fgcmLog.info('Matching reference star IDs')
            self.objRefIDIndexHandle = snmm.createArray(self.nStars, dtype='i4')
            objRefIDIndex = snmm.getArray(self.objRefIDIndexHandle)

            # Set the default to -1 (no match)
            objRefIDIndex[:] = -1

            objID = snmm.getArray(self.objIDHandle)
            refID = snmm.getArray(self.refIDHandle)

            a, b = esutil.numpy_util.match(refID, objID)
            objRefIDIndex[b] = a

            # Compute the fraction of stars that are reference stars
            self.fracStarsWithRef = (float(len(snmm.getArray(self.refIDHandle))) /
                                     float(len(snmm.getArray(self.objIDHandle))))

            self.fgcmLog.info('%.5f%% of stars have a reference match.' % (self.fracStarsWithRef * 100.0))

            self.fgcmLog.info('Done matching reference stars in %.1f seconds.' %
                              (time.time() - startTime))
        else:
            self.fracStarsWithRef = 0.0

        obsObjIDIndex = None
        objID = None
        obsIndex = None
        objObsIndex = None
        objNobs = None
        refID = None
        refMag = None
        refMagErr = None

        # and create a objFlag which flags bad stars as they fall out...

        self.objFlagHandle = snmm.createArray(self.nStars,dtype='i2')

        # and read in the previous bad stars if available
        if (flagID is not None):
            # the objFlag contains information on RESERVED stars
            objID = snmm.getArray(self.objIDHandle)
            objFlag = snmm.getArray(self.objFlagHandle)

            a,b=esutil.numpy_util.match(flagID, objID)

            test,=np.where((flagFlag[a] & objFlagDict['VARIABLE']) > 0)
            self.fgcmLog.info('Flagging %d stars as variable from previous cycles.' %
                             (test.size))
            test,=np.where((flagFlag[a] & objFlagDict['RESERVED']) > 0)
            self.fgcmLog.info('Flagging %d stars as reserved from previous cycles.' %
                             (test.size))
            test, = np.where((flagFlag[a] & objFlagDict['REFSTAR_OUTLIER']) > 0)
            if test.size > 0:
                self.fgcmLog.info('Flagging %d stars as reference star outliers from previous cycles.' %
                                  (test.size))

            objFlag[b] = flagFlag[a]
        else:
            # we want to reserve stars, if necessary
            if self.reserveFraction > 0.0:
                objFlag = snmm.getArray(self.objFlagHandle)

                nReserve = int(self.reserveFraction * objFlag.size)
                reserve = np.random.choice(objFlag.size,
                                           size=nReserve,
                                           replace=False)

                self.fgcmLog.info('Reserving %d stars from the fit.' % (nReserve))
                objFlag[reserve] |= objFlagDict['RESERVED']

                # If we have a "small" number of reference stars,
                # these should not be held in reserve
                if self.hasRefstars:
                    if self.nRefstars < 100:
                        objRefIDIndex = snmm.getArray(self.objRefIDIndexHandle)
                        cancel, = np.where(((objFlag & objFlagDict['RESERVED']) > 0) &
                                           (objRefIDIndex >= 0))
                        if cancel.size > 0:
                            objFlag[cancel] &= ~objFlagDict['RESERVED']
                            self.fgcmLog.info('Cancelling RESERVED flag on %d reference stars' % (cancel.size))

        # And we need to record the mean mag, error, SED slopes...

        #  objMagStdMean: mean standard magnitude of each object, per band
        self.objMagStdMeanHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4',
                                                    syncAccess=True)
        #  objMagStdMeanErr: error on the mean standard mag of each object, per band
        self.objMagStdMeanErrHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4')
        #  objSEDSlope: linearized approx. of SED slope of each object, per band
        self.objSEDSlopeHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4',
                                                  syncAccess=True)
        #  objMagStdMeanNoChrom: mean std mag of each object, no chromatic correction, per band
        self.objMagStdMeanNoChromHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4')

        # note: if this takes too long it can be moved to the star computation,
        #       but it seems pretty damn fast (which may raise the question of
        #       why it needs to be precomputed...)
        # compute secZenith for every observation

        startTime=time.time()
        self.fgcmLog.info('Computing secZenith for each star observation...')
        objRARad = np.radians(snmm.getArray(self.objRAHandle))
        objDecRad = np.radians(snmm.getArray(self.objDecHandle))
        ## FIXME: deal with this at some point...
        hi,=np.where(objRARad > np.pi)
        objRARad[hi] -= 2*np.pi
        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        obsIndex = snmm.getArray(self.obsIndexHandle)

        obsHARad = (fgcmPars.expTelHA[obsExpIndex] +
                    fgcmPars.expTelRA[obsExpIndex] -
                    objRARad[obsObjIDIndex])
        tempSecZenith = 1./(np.sin(objDecRad[obsObjIDIndex]) * fgcmPars.sinLatitude +
                            np.cos(objDecRad[obsObjIDIndex]) * fgcmPars.cosLatitude *
                            np.cos(obsHARad))

        bad,=np.where(obsFlag != 0)
        tempSecZenith[bad] = 1.0  # filler here, but these stars aren't used
        snmm.getArray(self.obsSecZenithHandle)[:] = tempSecZenith
        self.fgcmLog.info('Computed secZenith in %.1f seconds.' %
                         (time.time() - startTime))

        if (computeNobs):
            self.fgcmLog.info('Checking stars with all exposure numbers')
            allExpsIndex = np.arange(fgcmPars.expArray.size)
            self.selectStarsMinObsExpIndex(allExpsIndex)




    def selectStarsMinObsExpIndex(self, goodExpsIndex, temporary=False,
                                  minObsPerBand=None, reset=True):
        """
        Select stars that have at least the minimum number of observations per band,
         using a list of good exposures

        parameters
        ----------
        goodExpsIndex: int array
           Array of good (photometric) exposure indices
        temporary: bool, default=False
           Only flag bad objects temporarily
        minObsPerBand: int
           Specify the min obs per band, or use self.minObsPerBand
        reset: bool, default=False
           Reset the bad stars
        """

        if (minObsPerBand is None):
            minObsPerBand = self.minObsPerBand

        # Given a list of good exposures, which stars have at least minObs observations
        #  in each required band?

        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        objNGoodObs = snmm.getArray(self.objNGoodObsHandle)
        obsFlag = snmm.getArray(self.obsFlagHandle)
        objFlag = snmm.getArray(self.objFlagHandle)

        self.fgcmLog.info('Selecting good stars from %d exposures.' %
                         (goodExpsIndex.size))
        _,goodObs=esutil.numpy_util.match(goodExpsIndex,obsExpIndex)

        # Filter out bad (previously flagged) individual observations
        gd, = np.where(obsFlag[goodObs] == 0)
        goodObs = goodObs[gd]

        # count all the good observations
        objNGoodObs[:,:] = 0
        np.add.at(objNGoodObs,
                  (obsObjIDIndex[goodObs],
                   obsBandIndex[goodObs]),
                  1)

        if self.bandRequiredIndex.size == 0:
            # we have no *required* bands, but we will insist that there be
            # at least minObsPerBand in *one* band

            maxObs = objNGoodObs[:, :].max(axis=1)

            bad, = np.where(maxObs < minObsPerBand)
        else:
            # We need to ensure we have minObsPerBand observations in each
            # of the required bands

            minObs = objNGoodObs[:, self.bandRequiredIndex].min(axis=1)

            bad, = np.where(minObs < minObsPerBand)

        # reset too few obs flag if it's already set
        if reset:
            objFlag &= ~objFlagDict['TOO_FEW_OBS']

        if (not temporary) :
            objFlag[bad] |= objFlagDict['TOO_FEW_OBS']

            self.fgcmLog.info('Flagging %d of %d stars with TOO_FEW_OBS' % (bad.size,self.nStars))
        else:
            objFlag[bad] |= objFlagDict['TEMPORARY_BAD_STAR']

            self.fgcmLog.info('Flagging %d of %d stars with TEMPORARY_BAD_STAR' % (bad.size,self.nStars))


    def selectStarsMinObsExpAndCCD(self, goodExps, goodCCDs, minObsPerBand=None):
        """
        Select stars that have at least the minimum number of observations per band,
         using a list of good exposures and ccds.

        parameters
        ----------
        goodExps: int array
           Array of good (photometric) exposure numbers
        goodCCDs: int array
           Array of good (photometric) ccd numbers
        minObsPerBand: int
           Specify the min obs per band, or use self.minObsPerBand
        """

        if (minObsPerBand is None):
            minObsPerBand = self.minObsPerBand

        if (goodExps.size != goodCCDs.size) :
            raise ValueError("Length of goodExps and goodCCDs must be the same")

        obsExp = snmm.getArray(self.obsExpHandle)
        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsCCD = snmm.getArray(self.obsCCDHandle)
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        objNGoodObs = snmm.getArray(self.objNGoodObsHandle)
        obsFlag = snmm.getArray(self.obsFlagHandle)
        objFlag = snmm.getArray(self.objFlagHandle)

        self.fgcmLog.info( 'Selecting good stars from %d exposure/ccd pairs.' %
                         (goodExps.size))

        # hash together exposure and ccd and match this
        obsHash = obsExp * (self.nCCD + self.ccdStartIndex) + obsCCD
        goodHash = goodExps * (self.nCCD + self.ccdStartIndex) + goodCCDs

        _,goodObs = esutil.numpy_util.match(goodHash, obsHash)

        # Filter out bad (previously flagged) individual observations
        gd, = np.where(obsFlag[goodObs] == 0)
        goodObs = goodObs[gd]

        # count all the good observations
        objNGoodObs[:,:] = 0
        np.add.at(objNGoodObs,
                  (obsObjIDIndex[goodObs],
                   obsBandIndex[goodObs]),
                  1)

        if self.bandRequiredIndex.size == 0:
            # We have no *required* bands but we will insist that there be
            # at least minObsPerBand in *one* band

            maxObs = objNGoodObs[:, :].max(axis=1)

            bad, = np.where(maxObs < minObsPerBand)
        else:
            # and find the minimum of all the required bands
            minObs = objNGoodObs[:,self.bandRequiredIndex].min(axis=1)

            # choose the bad objects with too few observations
            bad,=np.where(minObs < minObsPerBand)

        # reset too few obs flag if it's already set
        objFlag &= ~objFlagDict['TOO_FEW_OBS']

        objFlag[bad] |= objFlagDict['TOO_FEW_OBS']
        self.fgcmLog.info('Flagging %d of %d stars with TOO_FEW_OBS' % (bad.size,self.nStars))

    def getGoodStarIndices(self, includeReserve=False, onlyReserve=False, checkMinObs=False,
                           checkHasColor=False):
        """
        Get the good star indices.

        parameters
        ----------
        includeReserve: bool, default=False
           optional to include reserved stars
        onlyReserve: bool, default=False
           optional to only include reserved stars
        checkMinObs: bool, default=False
           Extra(?) check for minimum number of observations
        checkHasColor: bool, default=False
           Check that the stars have the g-i or equivalent color

        returns
        -------
        goodStars: np.array of good star indices
        """

        #mask = 255
        mask = (objFlagDict['TOO_FEW_OBS'] |
                objFlagDict['BAD_COLOR'] |
                objFlagDict['VARIABLE'] |
                objFlagDict['TEMPORARY_BAD_STAR'] |
                objFlagDict['RESERVED'])

        if includeReserve or onlyReserve:
            mask &= ~objFlagDict['RESERVED']

        if onlyReserve:
            resMask = objFlagDict['RESERVED']
            goodFlag = (((snmm.getArray(self.objFlagHandle) & resMask) > 0) &
                        ((snmm.getArray(self.objFlagHandle) & mask) == 0))
        else:
            goodFlag = ((snmm.getArray(self.objFlagHandle) & mask) == 0)

        if checkMinObs:
            objNGoodObs = snmm.getArray(self.objNGoodObsHandle)

            if self.bandRequiredIndex.size == 0:
                maxObs = objNGoodObs[:, :].max(axis=1)
                goodFlag &= (maxObs >= self.minObsPerBand)
            else:
                minObs = objNGoodObs[:, self.bandRequiredIndex].min(axis=1)
                goodFlag &= (minObs >= self.minObsPerBand)

        if checkHasColor:
            objNGoodObs = snmm.getArray(self.objNGoodObsHandle)

            goodFlag &= (objNGoodObs[:, self.colorSplitIndices[0]] >= self.minObsPerBand)
            goodFlag &= (objNGoodObs[:, self.colorSplitIndices[1]] >= self.minObsPerBand)

        return np.where(goodFlag)[0]

    def getGoodObsIndices(self, goodStars, expFlag=None, requireSED=False, checkBadMag=False):
        """
        Get the good observation indices.

        parameters
        ----------
        goodStars: np.array
           Indices of the good stars
        expFlag: np.array
           expFlag from fgcmPars, will be selected on if provided
        requireSED: bool, default=False
           Should the good observations require SED measurement?
        checkBadMag: bool, default=False
           Check specifically for bad magnitudes

        returns
        -------
        goodStarsSub: np.array of good star sub-indices
           Sub-indices of the good stars matched to good observations.
        goodObs: np.array of good observation indices
           Indices of good observations, matched to goodStarsSub

        """

        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        obsFlag = snmm.getArray(self.obsFlagHandle)

        goodStarsSub, goodObs = esutil.numpy_util.match(goodStars,
                                                        obsObjIDIndex,
                                                        presorted=True)

        if goodStarsSub[0] != 0:
            raise ValueError("Very strange error that goodStarsSub first element is non-zero.")

        # Always need to filter on good individual observations
        okFlag = (obsFlag[goodObs] == 0)

        if expFlag is not None:
            obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
            okFlag &= (expFlag[obsExpIndex[goodObs]] == 0)

        # Make sure we don't have any 99s or related due to stars going bad
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)

        if checkBadMag:
            objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)
            okFlag &= (objMagStdMean[goodStars[goodStarsSub], obsBandIndex[goodObs]] < 99.0)

        if not self.allFitBandsAreRequired or self.nNotFitBands > 0:
            # We need to do some extra checks since not all fit bands are required
            # Or we have some extra bands.
            objNGoodObs = snmm.getArray(self.objNGoodObsHandle)

            okFlag &= (objNGoodObs[goodStars[goodStarsSub], obsBandIndex[goodObs]] >= self.minObsPerBand)

        if requireSED:
            # We need to ensure that we have an SED
            obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
            objSEDSlope = snmm.getArray(self.objSEDSlopeHandle)

            okFlag &= (objSEDSlope[goodStars[goodStarsSub], obsBandIndex[goodObs]] != 0.0)

        return goodStarsSub[okFlag], goodObs[okFlag]

    def plotStarMap(self,mapType='initial'):
        """
        Plot star map.

        parameters
        ----------
        mapType: string, default='initial'
           A key for labeling the map.
        """

        import healpy as hp
        try:
            from .fgcmPlotmaps import plot_hpxmap
        except ImportError:
            self.fgcmLog.info("Map plotting not available.  Sorry!")
            return

        mask = (objFlagDict['TOO_FEW_OBS'] |
                objFlagDict['BAD_COLOR'] |
                objFlagDict['VARIABLE'] |
                objFlagDict['TEMPORARY_BAD_STAR'])

        goodStars, = np.where((snmm.getArray(self.objFlagHandle) & mask) == 0)

        theta = (90.0-snmm.getArray(self.objDecHandle)[goodStars])*np.pi/180.
        phi = snmm.getArray(self.objRAHandle)[goodStars]*np.pi/180.

        ipring = hp.ang2pix(self.mapNSide,theta,phi)

        densMap = esutil.stat.histogram(ipring,min=0,max=12*self.mapNSide*self.mapNSide-1)
        densMap = densMap.astype(np.float32)

        bad,=np.where(densMap == 0)
        densMap[bad] = hp.UNSEEN

        raStarRot = snmm.getArray(self.objRAHandle)[goodStars]
        hi,=np.where(raStarRot > 180.0)
        raStarRot[hi] -= 360.0

        decStar = snmm.getArray(self.objDecHandle)[goodStars]

        fig,ax = plot_hpxmap(densMap,
                             raRange=[np.min(raStarRot),np.max(raStarRot)],
                             decRange=[np.min(decStar),np.max(decStar)],
                             lonRef = self.mapLongitudeRef)

        fig.savefig('%s/%s_%sGoodStars.png' % (self.plotPath, self.outfileBaseWithCycle,
                                               mapType))
        plt.close(fig)

    def computeObjectSEDSlopes(self,objIndicesIn):
        """
        Compute fnuprime (object SED slopes) for a list of objects.
        Output is saved in objSEDSlope.

        parameters
        ----------
        objIndicesIn: int array
           Array of object indices to do computation
        """

        if self.nBands < 3:
            # cannot compute SED slopes ... just leave at 0
            return

        # work on multiple indices

        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)
        objSEDSlope = snmm.getArray(self.objSEDSlopeHandle)
        objNGoodObs = snmm.getArray(self.objNGoodObsHandle)

        objMagStdMeanLock = snmm.getArrayBase(self.objMagStdMeanHandle).get_lock()
        objSEDSlopeLock = snmm.getArrayBase(self.objSEDSlopeHandle).get_lock()

        # protect access when copying to local
        objMagStdMeanLock.acquire()
        objMagStdMeanOI = objMagStdMean[objIndicesIn, :]
        objMagStdMeanLock.release()

        # and make a temporary local copy of the SED
        objSEDSlopeOI = np.zeros((objIndicesIn.size, self.nBands), dtype='f4')
        objNGoodObsOI = objNGoodObs[objIndicesIn, :]

        # NOTE: There is still an assumption here that the fit bands are sequential.

        # New plan:
        # Compute values for everything, and cut out bad ones below...

        S = np.zeros((objIndicesIn.size, self.nBands - 1), dtype='f8')
        for i in range(self.nBands - 1):
            S[:, i] = (-1. / self.magConstant) * (objMagStdMeanOI[:, i + 1] -
                                                  objMagStdMeanOI[:, i]) / (
                (self.lambdaStdBand[i + 1] - self.lambdaStdBand[i]))

        # FIXME: will need to handle u (or blueward) non-fit bands

        # The bluest one of the fit bands is an extrapolation
        # Note that this is different from the original Burke++17 paper
        tempIndex = self.bandFitIndex[0]
        # Only use stars that are measured in bands 0/1/2 (e.g. g/r)
        use, = np.where((objMagStdMeanOI[:, tempIndex] < 90.0) &
                        (objMagStdMeanOI[:, tempIndex + 1] < 90.0) &
                        (objMagStdMeanOI[:, tempIndex + 2] < 90.0))
        objSEDSlopeOI[use, tempIndex] = (
            S[use, tempIndex] + self.sedFudgeFactors[tempIndex] * (
                S[use, tempIndex + 1] + S[use, tempIndex]))

        # The middle ones are straight averages
        for tempIndex in self.bandFitIndex[1: -1]:
            # Only use stars that are measured in bands 0/1/2, 1/2/3 (e.g. g/r/i, r/i/z)
            use, = np.where((objMagStdMeanOI[:, tempIndex - 1] < 90.0) &
                            (objMagStdMeanOI[:, tempIndex] < 90.0) &
                            (objMagStdMeanOI[:, tempIndex + 1] < 90.0))
            objSEDSlopeOI[use, tempIndex] = (
                self.sedFudgeFactors[tempIndex] * (
                    S[use, tempIndex - 1] + S[use, tempIndex]) / 2.0)

        # The reddest one is another extrapolation
        tempIndex = self.bandFitIndex[-1]
        # Only use stars that are measured in bands e.g. 1/2/3 (e.g. r/i/z)
        use, = np.where((objMagStdMeanOI[:, tempIndex - 2] < 90.0) &
                        (objMagStdMeanOI[:, tempIndex - 1] < 90.0) &
                        (objMagStdMeanOI[:, tempIndex] < 90.0))
        objSEDSlopeOI[use, tempIndex] = (
            S[use, tempIndex - 1] + self.sedFudgeFactors[tempIndex] * (
                (self.lambdaStdBand[tempIndex] - self.lambdaStdBand[tempIndex - 1]) /
                (self.lambdaStdBand[tempIndex] - self.lambdaStdBand[tempIndex - 2])) *
            (S[use, tempIndex - 1] - S[use, tempIndex - 2]))

        # And for the redward non-fit band(s):
        for notFitIndex in self.bandNotFitIndex:
            # Only use stars that are measured in bands e.g. 2/3/4 (e.g. i/z/Y)
            use, = np.where((objMagStdMeanOI[:, notFitIndex - 2] < 90.0) &
                            (objMagStdMeanOI[:, notFitIndex - 1] < 90.0) &
                            (objMagStdMeanOI[:, notFitIndex] < 90.0))
            objSEDSlopeOI[use, notFitIndex] = (
                S[use, notFitIndex-1] + self.sedFudgeFactors[notFitIndex] * (
                    (self.lambdaStdBand[notFitIndex] - self.lambdaStdBand[notFitIndex - 1]) /
                    (self.lambdaStdBand[notFitIndex] - self.lambdaStdBand[notFitIndex - 2])) *
                (S[use, notFitIndex - 1] - S[use, notFitIndex - 2]))


        # Save the values, protected
        objSEDSlopeLock.acquire()
        objSEDSlope[objIndicesIn,:] = objSEDSlopeOI
        objSEDSlopeLock.release()

    def computeObjectSEDSlopesLUT(self, objIndicesIn, fgcmLUT):
        """
        Compute fnuprime (object SED slopes) for a list of objects, from the SED fit
          in the look-up table.  Experimental.
        Output is saved in objSEDSlope.

        parameters
        ----------
        objIndicesIn: int array
           Array of object indices to do computation
        fgcmLUT: FgcmLUT
        """

        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)
        objSEDSlope = snmm.getArray(self.objSEDSlopeHandle)

        objMagStdMeanLock = snmm.getArrayBase(self.objMagStdMeanHandle).get_lock()
        objSEDSlopeLock = snmm.getArrayBase(self.objSEDSlopeHandle).get_lock()

        # protect access to copy to local
        objMagStdMeanLock.acquire()

        objMagStdMeanOI = objMagStdMean[objIndicesIn,:]

        # release access
        objMagStdMeanLock.release()

        # and make a temporary local copy of the SED
        #objSEDSlopeOI = np.zeros((objIndicesIn.size,self.nBands),dtype='f4')

        # compute SED color...
        ## FIXME: make this configurable
        objSEDColorOI = objMagStdMeanOI[:,0] - objMagStdMeanOI[:,2]

        # do the look-up
        objSEDSlopeOI = fgcmLUT.computeSEDSlopes(objSEDColorOI)

        # and save the values, protected
        objSEDSlopeLock.acquire()

        objSEDSlope[objIndicesIn,:] = objSEDSlopeOI

        objSEDSlopeLock.release()

    def computeAbsOffset(self):
        """
        Compute the absolute offset

        Returns
        ------
        deltaOffsetRef: `np.array`
           Float array (nBands) that is the delta offset in abs mag
        """

        if not self.hasRefstars:
            # should this Raise because it's programmer error, or just pass because
            # it's harmless?
            self.fgcmLog.info("Warning: cannot compute abs offset without reference stars.")
            return np.zeros(self.nBands)

        # Set things up
        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.objMagStdMeanErrHandle)
        objRefIDIndex = snmm.getArray(self.objRefIDIndexHandle)
        refMag = snmm.getArray(self.refMagHandle)
        refMagErr = snmm.getArray(self.refMagErrHandle)

        goodStars = self.getGoodStarIndices(includeReserve=False, checkMinObs=True)

        use, = np.where(objRefIDIndex[goodStars] >= 0)
        goodRefStars = goodStars[use]

        deltaOffsetRef = np.zeros(self.nBands)
        deltaOffsetWtRef = np.zeros(self.nBands)

        gdStarInd, gdBandInd = np.where((objMagStdMean[goodRefStars, :] < 90.0) &
                                        (refMag[objRefIDIndex[goodRefStars], :] < 90.0))
        delta = objMagStdMean[goodRefStars, :] - refMag[objRefIDIndex[goodRefStars], :]
        wt = 1. / (objMagStdMeanErr[goodRefStars, :]**2. +
                   refMagErr[objRefIDIndex[goodRefStars], :]**2.)

        np.add.at(deltaOffsetRef, gdBandInd, delta[gdStarInd, gdBandInd] * wt[gdStarInd, gdBandInd])
        np.add.at(deltaOffsetWtRef, gdBandInd, wt[gdStarInd, gdBandInd])

        # Make sure we have a measurement in the band
        ok, = np.where(deltaOffsetWtRef > 0.0)
        deltaOffsetRef[ok] /= deltaOffsetWtRef[ok]

        # And any bands that we do not have a measurement will be a weighted mean
        # of the other bands...
        noRef, = np.where(deltaOffsetWtRef == 0)
        if noRef.size > 0:
            # there are bands to fill in...
            deltaOffsetRef[noRef] = (np.sum(deltaOffsetRef[ok] * deltaOffsetWtRef[ok]) /
                                     np.sum(deltaOffsetWtRef[ok]))

        return deltaOffsetRef

    def applyAbsOffset(self, deltaAbsOffset):
        """
        Apply the absolute offsets.  Used for initial fit cycle.

        Parameters
        ----------
        deltaAbsOffset: `np.array`
           Float array with nbands offsets
        """

        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)

        obsMagStd = snmm.getArray(self.obsMagStdHandle)
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)

        goodStars = self.getGoodStarIndices(includeReserve=True)
        _, goodObs = self.getGoodObsIndices(goodStars, expFlag=None)

        # need goodObs
        obsMagStd[goodObs] -= deltaAbsOffset[obsBandIndex[goodObs]]

        gdMeanStar, gdMeanBand = np.where(objMagStdMean[goodStars, :] < 90.0)
        objMagStdMean[goodStars[gdMeanStar], gdMeanBand] -= deltaAbsOffset[gdMeanBand]

    def computeEGray(self, goodObs, ignoreRef=False, onlyObsErr=False):
        """
        Compute the delta-mag between the observed and true value (EGray) for a set of
        observations (goodObs).

        EGray == <mstd> - mstd
        or
        EGray == mref - mstd

        Parameters
        ----------
        goodObs: `np.array`
           Array of indices of good observations
        ignoreRef: `bool`, default=False
           Ignore reference stars.
        onlyObsErr: `bool`, default=False
           Only use the observational error (for non-ref stars)

        Returns
        -------
        EGrayGO: `np.array`
           Array of gray residuals for goodObs observations
        EGrayErr2GO: `np.array`
           Array of gray residual error squared
        """

        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.objMagStdMeanErrHandle)

        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        obsMagStd = snmm.getArray(self.obsMagStdHandle)
        obsMagErr = snmm.getArray(self.obsMagADUModelErrHandle)
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)

        # First compute EGray for all the observations
        EGrayGO = (objMagStdMean[obsObjIDIndex[goodObs], obsBandIndex[goodObs]] -
                   obsMagStd[goodObs])

        if onlyObsErr:
            EGrayErr2GO = obsMagErr[goodObs]**2.
        else:
            EGrayErr2GO = (obsMagErr[goodObs]**2. -
                           objMagStdMeanErr[obsObjIDIndex[goodObs], obsBandIndex[goodObs]]**2.)

        # And if we need reference stars, replace these
        if self.hasRefstars and not ignoreRef:
            objRefIDIndex = snmm.getArray(self.objRefIDIndexHandle)
            refMag = snmm.getArray(self.refMagHandle)
            refMagErr = snmm.getArray(self.refMagErrHandle)

            goodRefObsGO, = np.where(objRefIDIndex[obsObjIDIndex[goodObs]] >= 0)

            if goodRefObsGO.size > 0:
                obsUse, = np.where((obsMagStd[goodObs[goodRefObsGO]] < 90.0) &
                                   (refMag[objRefIDIndex[obsObjIDIndex[goodObs[goodRefObsGO]]],
                                           obsBandIndex[goodObs[goodRefObsGO]]] < 90.0))

                if obsUse.size > 0:
                    goodRefObsGO = goodRefObsGO[obsUse]

                    EGrayGO[goodRefObsGO] = (refMag[objRefIDIndex[obsObjIDIndex[goodObs[goodRefObsGO]]],
                                                   obsBandIndex[goodObs[goodRefObsGO]]] -
                                             obsMagStd[goodObs[goodRefObsGO]])

                    EGrayErr2GO[goodRefObsGO] = (obsMagErr[goodObs[goodRefObsGO]]**2. +
                                                 refMagErr[objRefIDIndex[obsObjIDIndex[goodObs[goodRefObsGO]]],
                                                           obsBandIndex[goodObs[goodRefObsGO]]]**2.)

        return EGrayGO, EGrayErr2GO

    def performColorCuts(self):
        """
        Make the color cuts that are specified in the config.
        """

        if (not self.magStdComputed):
            raise ValueError("Must compute magStd before performing color cuts")

        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)
        objFlag = snmm.getArray(self.objFlagHandle)

        # Only cut stars where we have a color, and are *not* reference stars

        for cCut in self.starColorCuts:
            ok, = np.where((objMagStdMean[:, cCut[0]] < 90.0) &
                           (objMagStdMean[:, cCut[1]] < 90.0))

            thisColor = objMagStdMean[ok, cCut[0]] - objMagStdMean[ok, cCut[1]]
            bad, = np.where((thisColor < cCut[2]) |
                            (thisColor > cCut[3]))
            objFlag[ok[bad]] |= objFlagDict['BAD_COLOR']

            self.fgcmLog.info('Flag %d stars of %d with BAD_COLOR' % (bad.size,self.nStars))

        if self.hasRefstars and not self.applyRefStarColorCuts:
            objRefIDIndex = snmm.getArray(self.objRefIDIndexHandle)
            cancel, = np.where(((objFlag & objFlagDict['BAD_COLOR']) > 0) &
                               (objRefIDIndex >= 0))
            if cancel.size > 0:
                objFlag[cancel] &= ~objFlagDict['BAD_COLOR']
                self.fgcmLog.info('Cancelling BAD_COLOR flag on %d reference stars' % (cancel.size))

    def performSuperStarOutlierCuts(self, fgcmPars, reset=False):
        """
        Do outlier cuts from common ccd/filter/epochs
        """

        self.fgcmLog.info('Computing superstar outliers')

        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)

        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
        obsMagStd = snmm.getArray(self.obsMagStdHandle)
        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsCCDIndex = snmm.getArray(self.obsCCDHandle) - self.ccdStartIndex
        obsFlag = snmm.getArray(self.obsFlagHandle)

        if reset:
            # reset the obsFlag...
            obsFlag &= ~obsFlagDict['SUPERSTAR_OUTLIER']

        goodStars = self.getGoodStarIndices(checkMinObs=True)
        _, goodObs = self.getGoodObsIndices(goodStars, expFlag=fgcmPars.expFlag)

        # we need to compute E_gray == <mstd> - mstd for each observation
        # compute EGray, GO for Good Obs
        EGrayGO = (objMagStdMean[obsObjIDIndex[goodObs],obsBandIndex[goodObs]] -
                   obsMagStd[goodObs])

        epochFilterHash = (fgcmPars.expEpochIndex[obsExpIndex[goodObs]]*
                           (fgcmPars.nLUTFilter+1)*(fgcmPars.nCCD+1) +
                           fgcmPars.expLUTFilterIndex[obsExpIndex[goodObs]]*
                           (fgcmPars.nCCD+1) +
                           obsCCDIndex[goodObs])

        h, rev = esutil.stat.histogram(epochFilterHash, rev=True)

        nbad = 0

        use, = np.where(h > 0)
        for i in use:
            i1a = rev[rev[i]: rev[i + 1]]

            med = np.median(EGrayGO[i1a])
            sig = 1.4826 * np.median(np.abs(EGrayGO[i1a] - med))
            bad, = np.where(np.abs(EGrayGO[i1a] - med) > self.superStarSigmaClip * sig)

            obsFlag[goodObs[i1a[bad]]] |= obsFlagDict['SUPERSTAR_OUTLIER']

            nbad += bad.size

        self.fgcmLog.info("Marked %d observations (%.4f%%) as SUPERSTAR_OUTLIER" %
                          (nbad, 100. * float(nbad)/float(goodObs.size)))

        # Now we need to flag stars that might have fallen below our threshold
        # when we flagged these outliers
        goodExpsIndex, = np.where(fgcmPars.expFlag == 0)
        self.selectStarsMinObsExpIndex(goodExpsIndex, reset=reset)

        # I had considered it might be necessary to flag bad exposures
        # at this point, but I don't think that's the case.

    def applySuperStarFlat(self,fgcmPars):
        """
        Apply superStarFlat to raw magnitudes.

        parameters
        ----------
        fgcmPars: FgcmParameters
        """

        self.fgcmLog.info('Applying SuperStarFlat to raw magnitudes')

        obsMagADU = snmm.getArray(self.obsMagADUHandle)
        obsSuperStarApplied = snmm.getArray(self.obsSuperStarAppliedHandle)
        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsCCDIndex = snmm.getArray(self.obsCCDHandle) - self.ccdStartIndex

        # two different tracks, if x/y available or not.

        if self.hasXY:
            # With x/y information

            from .fgcmUtilities import Cheb2dField

            # Scale X and Y
            obsX = snmm.getArray(self.obsXHandle)
            obsY = snmm.getArray(self.obsYHandle)

            epochFilterHash = (fgcmPars.expEpochIndex[obsExpIndex]*
                               (fgcmPars.nLUTFilter+1)*(fgcmPars.nCCD+1) +
                               fgcmPars.expLUTFilterIndex[obsExpIndex]*
                               (fgcmPars.nCCD+1) +
                               obsCCDIndex)

            h, rev = esutil.stat.histogram(epochFilterHash, rev=True)

            for i in xrange(h.size):
                if h[i] == 0: continue

                i1a = rev[rev[i]:rev[i+1]]

                # get the indices for this epoch/filter/ccd
                epInd = fgcmPars.expEpochIndex[obsExpIndex[i1a[0]]]
                fiInd = fgcmPars.expLUTFilterIndex[obsExpIndex[i1a[0]]]
                cInd = obsCCDIndex[i1a[0]]

                field = Cheb2dField(self.ccdOffsets['X_SIZE'][cInd],
                                    self.ccdOffsets['Y_SIZE'][cInd],
                                    fgcmPars.parSuperStarFlat[epInd, fiInd, cInd, :])
                fluxScale = field.evaluate(obsX[i1a], obsY[i1a])
                obsSuperStarApplied[i1a] = -2.5 * np.log10(np.clip(fluxScale, 0.1, None))
        else:
            # No x/y available

            obsSuperStarApplied[:] = fgcmPars.expCCDSuperStar[obsExpIndex,
                                                              obsCCDIndex]

        # And finally apply the superstar correction
        obsMagADU[:] += obsSuperStarApplied[:]

    def applyApertureCorrection(self,fgcmPars):
        """
        Apply aperture corrections to raw magnitudes.

        parameters
        ----------
        fgcmPars: FgcmParameters
        """

        self.fgcmLog.info('Applying ApertureCorrections to raw magnitudes')

        if self.seeingSubExposure:
            self.fgcmLog.info('Aperture correction has sub-exposure information')
        else:
            self.fgcmLog.info('Aperture correction is per-exposure')

        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsCCDIndex = snmm.getArray(self.obsCCDHandle) - self.ccdStartIndex

        obsMagADU = snmm.getArray(self.obsMagADUHandle)

        # Note that EXP^gray = < <mstd>_j - mstd_ij >
        #  when we have seeing loss, that makes mstd_ij larger and EXP^gray smaller
        #  So the slope of aperCorr is negative.
        #  If we add aperCorr to each of mstd_ij, then we get a smaller (brighter)
        #  magnitude.  And this will bring mstd_ij closer to <mstd>_j

        if self.seeingSubExposure:
            obsMagADU[:] += fgcmPars.ccdApertureCorrection[obsExpIndex, obsCCDIndex]
        else:
            # Per exposure
            obsMagADU[:] += fgcmPars.expApertureCorrection[obsExpIndex]

    def computeModelMagErrors(self, fgcmPars):
        """
        Compute model magnitude errors.

        parameters
        ----------
        fgcmPars: FgcmParameters
        """

        if (fgcmPars.compModelErrFwhmPivot[0] <= 0.0) :
            self.fgcmLog.info('No model for mag errors, so mag errors are unchanged.')
            return

        if not self.modelMagErrors:
            self.fgcmLog.info('Model magnitude errors are turned off.')
            return

        if not self.magStdComputed:
            raise RuntimeError("Must run FgcmChisq to compute magStd before computeModelMagErrors")

        self.fgcmLog.info('Computing model magnitude errors for photometric observations')

        objFlag = snmm.getArray(self.objFlagHandle)
        objNGoodObs = snmm.getArray(self.objNGoodObsHandle)
        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)

        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        obsFlag = snmm.getArray(self.obsFlagHandle)
        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
        obsMagADU = snmm.getArray(self.obsMagADUHandle)
        obsMagADUErr = snmm.getArray(self.obsMagADUErrHandle)
        obsMagADUModelErr = snmm.getArray(self.obsMagADUModelErrHandle)
        obsMagStd = snmm.getArray(self.obsMagStdHandle)

        obsExptime = fgcmPars.expExptime[obsExpIndex]
        obsFwhm = fgcmPars.expFwhm[obsExpIndex]
        obsSkyBrightness = fgcmPars.expSkyBrightness[obsExpIndex]

        # we will compute all stars that are possibly good, including reserved
        resMask = (objFlagDict['TOO_FEW_OBS'] |
                   objFlagDict['BAD_COLOR'] |
                   objFlagDict['VARIABLE'] |
                   objFlagDict['TEMPORARY_BAD_STAR'])

        #resMask = 255 & ~objFlagDict['RESERVED']
        goodStars, = np.where((objFlag & resMask) == 0)

        goodStarsSub, goodObs = esutil.numpy_util.match(goodStars,
                                                        obsObjIDIndex,
                                                        presorted=True)

        # Do we want to allow more selection of exposures here?
        gd, = np.where((obsFlag[goodObs] == 0) &
                       (fgcmPars.expFlag[obsExpIndex[goodObs]] == 0))
        goodObs = goodObs[gd]
        goodStarsSub = goodStarsSub[gd]

        # loop over bands
        for bandIndex in xrange(fgcmPars.nBands):
            use, = np.where((obsBandIndex[goodObs] == bandIndex) &
                            (objNGoodObs[obsObjIDIndex[goodObs], bandIndex] >= self.minObsPerBand))
            pars = fgcmPars.compModelErrPars[:, bandIndex]
            fwhmPivot = fgcmPars.compModelErrFwhmPivot[bandIndex]
            skyPivot = fgcmPars.compModelErrSkyPivot[bandIndex]
            exptimePivot = fgcmPars.compModelErrExptimePivot[bandIndex]

            obsMagADUMeanGOu = (objMagStdMean[obsObjIDIndex[goodObs[use]], bandIndex] -
                                (obsMagStd[goodObs[use]] - obsMagADU[goodObs[use]]) -
                                2.5 * np.log10(obsExptime[goodObs[use]] / exptimePivot))

            modErr = 10.**(pars[0] + pars[1] * obsMagADUMeanGOu + pars[2] * obsMagADUMeanGOu**2. +
                           pars[3] * np.log10(obsFwhm[goodObs[use]] / fwhmPivot) +
                           pars[4] * np.log10(obsSkyBrightness[goodObs[use]] / skyPivot) +
                           pars[5] * obsMagADUMeanGOu * np.log10(obsFwhm[goodObs[use]] / fwhmPivot) +
                           pars[6] * obsMagADUMeanGOu * np.log10(obsSkyBrightness[goodObs[use]] / skyPivot))

            obsMagADUModelErr[goodObs[use]] = np.sqrt(modErr**2. + self.sigma0Phot**2.)

    def saveFlagStarIndices(self,flagStarFile):
        """
        Save flagged stars to fits.

        parameters
        ----------
        flagStarFile: string
           Filename to output.
        """

        import fitsio

        flagObjStruct = self.getFlagStarIndices()

        self.fgcmLog.info('Saving %d flagged star indices to %s' %
                         (flagObjStruct.size,flagStarFile))

        # set clobber == True?
        fitsio.write(flagStarFile,flagObjStruct,clobber=True)

    def getFlagStarIndices(self):
        """
        Retrieve flagged star indices.
        """

        objID = snmm.getArray(self.objIDHandle)
        objFlag = snmm.getArray(self.objFlagHandle)

        # we only store VARIABLE and RESERVED stars
        # everything else should be recomputed based on the good exposures, calibrations, etc
        flagMask = (objFlagDict['VARIABLE'] |
                    objFlagDict['RESERVED'] |
                    objFlagDict['REFSTAR_OUTLIER'])

        flagged,=np.where((objFlag & flagMask) > 0)

        flagObjStruct = np.zeros(flagged.size,dtype=[('OBJID',objID.dtype),
                                                     ('OBJFLAG',objFlag.dtype)])
        flagObjStruct['OBJID'] = objID[flagged]
        flagObjStruct['OBJFLAG'] = objFlag[flagged]

        return flagObjStruct

    def saveStdStars(self, starFile, fgcmPars):
        """
        Save standard stars.  Note that this does not fill in holes.

        parameters
        ----------
        starFile: string
           Output star file
        fgcmPars: FgcmParameters
        """

        import fitsio

        self.fgcmLog.info('Saving standard stars to %s' % (starFile))

        fitsio.write(starFile, self.retrieveStdStarCatalog(fgcmPars), clobber=True)

    def retrieveStdStarCatalog(self, fgcmPars):
        """
        Retrieve standard star catalog.  Note that this does not fill in holes (yet).

        parameters
        ----------
        fgcmPars: FgcmParameters
        """

        objID = snmm.getArray(self.objIDHandle)
        objFlag = snmm.getArray(self.objFlagHandle)
        objRA = snmm.getArray(self.objRAHandle)
        objDec = snmm.getArray(self.objDecHandle)
        objNGoodObs = snmm.getArray(self.objNGoodObsHandle)
        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.objMagStdMeanErrHandle)

        rejectMask = (objFlagDict['BAD_COLOR'] | objFlagDict['VARIABLE'] |
                      objFlagDict['TOO_FEW_OBS'])

        goodStars, = np.where((objFlag & rejectMask) == 0)

        outCat = np.zeros(goodStars.size, dtype=[('FGCM_ID', 'i8'),
                                                 ('RA', 'f8'),
                                                 ('DEC', 'f8'),
                                                 ('NGOOD', 'i4', len(self.bands)),
                                                 ('MAG_STD', 'f4', len(self.bands)),
                                                 ('MAGERR_STD', 'f4', len(self.bands))])

        outCat['FGCM_ID'] = objID[goodStars]
        outCat['RA'] = objRA[goodStars]
        outCat['DEC'] = objDec[goodStars]
        outCat['NGOOD'] = objNGoodObs[goodStars, :]
        outCat['MAG_STD'][:, :] = objMagStdMean[goodStars, :]
        outCat['MAGERR_STD'][:, :] = objMagStdMeanErr[goodStars, :]

        return outCat

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        return state
