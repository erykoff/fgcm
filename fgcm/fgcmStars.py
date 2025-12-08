import numpy as np
import esutil
import time
import warnings

from .fgcmUtilities import objFlagDict
from .fgcmUtilities import obsFlagDict
from .fgcmUtilities import getMemoryString
from .fgcmUtilities import makeFigure, putButlerFigure
from matplotlib import colormaps

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

    def __init__(self, fgcmConfig, butlerQC=None, plotHandleDict=None):

        self.reloadConfig(fgcmConfig, butlerQC=butlerQC, plotHandleDict=plotHandleDict)

        self.magStdComputed = False
        self.allMagStdComputed = False
        self.sedSlopeComputed = False

        self.hasXY = False
        self.hasDeltaAper = False
        self.hasRefstars = False
        self.nRefStars = 0
        self.hasPsfCandidate = False
        self.hasDeltaMagBkg = False

        self.deltaMapperDefault = None

        self.rng = fgcmConfig.rng

        self.missingSedValues = np.zeros(self.nBands)

        self.starsLoaded = False
        self.starsPrepped = False

    def reloadConfig(self, fgcmConfig, butlerQC=None, plotHandleDict=None):
        """Reload config + friends.

        Parameters
        ----------
        fgcmConfig : `fgcm.FgcmConfig`
        butlerQC : `lsst.daf.butler.ButlerQuantumContext`
        plotHandleDict : `dict`
        """
        self.fgcmLog = fgcmConfig.fgcmLog

        self.butlerQC = butlerQC
        self.plotHandleDict = plotHandleDict

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
        self.starColorCuts = fgcmConfig.starColorCuts
        self.refStarColorCuts = fgcmConfig.refStarColorCuts
        self.quantityCuts = fgcmConfig.quantityCuts
        self.sigma0Phot = fgcmConfig.sigma0Phot
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber
        self.expField = fgcmConfig.expField
        self.ccdField = fgcmConfig.ccdField
        self.reserveFraction = fgcmConfig.reserveFraction
        self.modelMagErrors = fgcmConfig.modelMagErrors
        self.sedBoundaryTermDict = fgcmConfig.sedBoundaryTermDict
        self.sedTermDict = fgcmConfig.sedTermDict
        self.quietMode = fgcmConfig.quietMode

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
        self.colorSplitBands = fgcmConfig.colorSplitBands
        self.colorSplitIndices = fgcmConfig.colorSplitIndices

        self.superStarSubCCD = fgcmConfig.superStarSubCCD
        self.superStarSigmaClip = fgcmConfig.superStarSigmaClip

        self.focalPlaneSigmaClip = fgcmConfig.focalPlaneSigmaClip

        self.magConstant = 2.5/np.log(10)
        self.zptABNoThroughput = fgcmConfig.zptABNoThroughput
        self.approxThroughput = fgcmConfig.approxThroughput

        self.refStarSnMin = fgcmConfig.refStarSnMin
        self.refStarMaxFracUse = fgcmConfig.refStarMaxFracUse
        self.applyRefStarColorCuts = fgcmConfig.applyRefStarColorCuts

        self.seeingSubExposure = fgcmConfig.seeingSubExposure

        self.secZenithRange = 1. / np.cos(np.radians(fgcmConfig.zenithRange))

        self.rng = fgcmConfig.rng

    def updateFlags(self, flagID, flagFlag):
        """
        Update flagged stars.

        Parameters
        ----------
        flagID : `np.ndarray`
            Array of flagged object IDs
        flagFlag : `np.ndarray`
            Array of flagged star flag values.
        """
        objID = snmm.getArray(self.objIDHandle)
        objFlag = snmm.getArray(self.objFlagHandle)

        a, b = esutil.numpy_util.match(flagID, objID)

        test = ((flagFlag[a] & objFlagDict['VARIABLE']) > 0)
        self.fgcmLog.info('Flagging %d stars as variable from previous cycles.' %
                          (test.sum()))
        test = ((flagFlag[a] & objFlagDict['RESERVED']) > 0)
        self.fgcmLog.info('Flagging %d stars as reserved from previous cycles.' %
                          (test.sum()))
        test = ((flagFlag[a] & objFlagDict['REFSTAR_OUTLIER']) > 0)
        if test.sum() > 0:
            self.fgcmLog.info('Flagging %d stars as reference star outliers from previous cycles.' %
                              (test.sum()))
        test = ((flagFlag[a] & objFlagDict['REFSTAR_RESERVED']) > 0)
        if test.sum() > 0:
            self.fgcmLog.info('Flagging %d stars as reference star reserved from previous cycles.' %
                              (test.sum()))

        objFlag[b] = flagFlag[a]

    def setDeltaMapperDefault(self, deltaMapperDefault):
        """
        Set the deltaMapperDefault array.

        Parameters
        ----------
        deltaMapperDefault : `np.recarray`
        """
        self.deltaMapperDefault = deltaMapperDefault

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
        self.fgcmLog.debug('Reading in observation indices...')
        index = fitsio.read(self.indexFile, ext='INDEX', upper=True)
        if not self.quietMode:
            self.fgcmLog.info('Done reading in %d observation indices in %.1f seconds.' %
                              (index.size, time.time() - startTime))

        # read in obsfile and cut
        startTime = time.time()
        self.fgcmLog.debug('Reading in star observations...')
        obs = fitsio.read(self.obsFile, ext=1, upper=True)
        # cut down to those that are indexed
        obs = obs[index['OBSINDEX']]
        if not self.quietMode:
            self.fgcmLog.info('Done reading in %d observations in %.1f seconds.' %
                              (obs.size, time.time() - startTime))

        # and positions...
        startTime = time.time()
        self.fgcmLog.debug('Reading in star positions...')
        pos = fitsio.read(self.indexFile, ext='POS', upper=True)
        if not self.quietMode:
            self.fgcmLog.info('Done reading in %d unique star positions in %.1f seconds.' %
                              (pos.size, time.time() - startTime))

        # Cut down the stars here if desired.
        if len(self.quantityCuts) > 0:
            cut = None
            for qcut in self.quantityCuts:
                quant = qcut[0].upper()
                qmin = qcut[1]
                qmax = qcut[2]

                if quant not in pos.dtype.names:
                    raise ValueError("Could not find cut quantity %s in indexfile %s[POS]" %
                                     (quant, self.indexFile))

                if cut is None:
                    cut = ((pos[quant] >= qmin) & (pos[quant] <= qmax))
                else:
                    cut &= ((pos[quant] >= qmin) & (pos[quant] <= qmax))

            nCut = np.sum(~cut)
            self.fgcmLog.info('Cutting %d objects from extra quantities' % (nCut))

            pos = pos[cut]

        obsFilterName = np.core.defchararray.strip(obs['FILTERNAME'][:])

        # And refstars if available
        if self.refstarFile is not None:
            startTime = time.time()
            self.fgcmLog.debug('Reading in reference stars...')
            ref = fitsio.read(self.refstarFile, ext=1, lower=True)
            if not self.quietMode:
                self.fgcmLog.info('Done reading %d reference stars in %.1f seconds.' %
                                  (ref.size, time.time() - startTime))
            refID = ref['fgcm_id']
            refMag = ref['mag']
            refMagErr = ref['mag_err']
        else:
            refID = None
            refMag = None
            refMagErr = None

        if (self.inFlagStarFile is not None):
            self.fgcmLog.debug('Reading in list of previous flagged stars from %s' %
                               (self.inFlagStarFile))

            inFlagStars = fitsio.read(self.inFlagStarFile, ext=1, upper=True)

            flagID = inFlagStars['OBJID']
            flagFlag = inFlagStars['OBJFLAG']
        else:
            flagID = None
            flagFlag = None

        if ('X' in obs.dtype.names and 'Y' in obs.dtype.names):
            self.fgcmLog.debug('Found X/Y in input observations')
            obsX = obs['X']
            obsY = obs['Y']
        else:
            obsX = None
            obsY = None

        if ('DELTA_APER' in obs.dtype.names):
            self.fgcmLog.info('Found DELTA_APER in input observations')
            obsDeltaAper = obs['DELTA_APER']
        else:
            obsDeltaAper = None

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
                       obsDeltaAper=obsDeltaAper,
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
                  psfCandidate=None, refID=None, refMag=None, refMagErr=None,
                  obsDeltaMagBkg=None, obsDeltaAper=None,
                  flagID=None, flagFlag=None, computeNobs=True, objIDAlternate=None):
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
        obsDeltaMagBkg: float array, optional
           Delta-mag per observation due to local background offset.
        obsDeltaAper: float array, optional
           delta_aper for each observation
        obsX: float array, optional
           x position for each observation
        obsY: float array, optional
           y position for each observation
        psfCandidate: bool array, optional
           Flag if this star was a psf candidate in single-epoch images
        flagID: int array, optional
           ID of each object that is flagged from previous cycle
        flagFlag: int array, optional
           Flag value from previous cycle
        computeNobs: bool, default=True
           Compute number of good observations of each object?
        objIDAlternate : int array, optional
            Alternate (non-consecutive) star id.
        """

        # FIXME: check that these are all the same length!

        self.obsIndexHandle = snmm.createArray(obsRA.size, dtype='i8')
        snmm.getArray(self.obsIndexHandle)[:] = np.arange(obsRA.size)

        # need to stuff into shared memory objects.
        #  nStarObs: total number of observations of all stars
        self.nStarObs = obsRA.size

        #  obsExp: exposure number of individual observation (pointed by obsIndex)
        self.obsExpHandle = snmm.createArray(self.nStarObs, dtype='i8')
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
        self.obsMagADUHandle = snmm.createArray(self.nStarObs,dtype='f4')
        #  obsMagADUErr: raw ADU counts error of individual observation
        self.obsMagADUErrHandle = snmm.createArray(self.nStarObs,dtype='f4')
        # We also store the original values which helps with repeat runs.
        self.obsMagADUOrigHandle = snmm.createArray(self.nStarObs, dtype='f4')
        self.obsMagADUErrOrigHandle = snmm.createArray(self.nStarObs, dtype='f4')
        #  obsMagADUModelErr: modeled ADU counts error of individual observation
        self.obsMagADUModelErrHandle = snmm.createArray(self.nStarObs,dtype='f4')
        #  obsSuperStarApplied: SuperStar correction that was applied
        self.obsSuperStarAppliedHandle = snmm.createArray(self.nStarObs,dtype='f4')
        #  obsMagStd: corrected (to standard passband) mag of individual observation
        self.obsMagStdHandle = snmm.createArray(self.nStarObs,dtype='f8',syncAccess=True)
        #  obsDeltaStd: chromatic correction of individual observation
        self.obsDeltaStdHandle = snmm.createArray(self.nStarObs, dtype='f8')
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
        if psfCandidate is not None:
            self.hasPsfCandidate = True

            self.fgcmLog.info('PSF Candidate Flags found')

            # psfCandidate: bool flag if this is a single-epoch psf candidate
            self.psfCandidateHandle = snmm.createArray(self.nStarObs, dtype=bool)
        if obsDeltaMagBkg is not None:
            # Do not use if all 0s
            if np.min(obsDeltaMagBkg) != 0.0 and np.max(obsDeltaMagBkg) != 0.0:
                self.hasDeltaMagBkg = True
                self.fgcmLog.info('Delta-mag from local background found')
                self.obsDeltaMagBkgHandle = snmm.createArray(self.nStarObs, dtype='f4')
        if obsDeltaAper is not None:
            self.hasDeltaAper = True

            #  obsDeltaAper: delta mag for smaller - larger aperture
            self.obsDeltaAperHandle = snmm.createArray(self.nStarObs, dtype='f4')

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
            self.refIDHandle = snmm.createArray(self.nRefstars, dtype='i8')
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
        snmm.getArray(self.obsMagADUOrigHandle)[:] = obsMag
        snmm.getArray(self.obsMagADUErrOrigHandle)[:] = obsMagErr
        snmm.getArray(self.obsMagStdHandle)[:] = obsMag + self.zptABNoThroughput  # same as raw at first
        snmm.getArray(self.obsDeltaStdHandle)[:] = 0.0
        snmm.getArray(self.obsSuperStarAppliedHandle)[:] = 0.0
        if self.hasXY:
            snmm.getArray(self.obsXHandle)[:] = obsX
            snmm.getArray(self.obsYHandle)[:] = obsY
        if self.hasPsfCandidate:
            snmm.getArray(self.psfCandidateHandle)[:] = psfCandidate
        if self.hasDeltaMagBkg:
            snmm.getArray(self.obsDeltaMagBkgHandle)[:] = obsDeltaMagBkg
        if self.hasDeltaAper:
            snmm.getArray(self.obsDeltaAperHandle)[:] = obsDeltaAper

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

        # new version for multifilter support
        startTime = time.time()

        obsFilterNameIsEncoded = False
        try:
            test = obsFilterName[0].decode('utf-8')
            obsFilterNameIsEncoded = True
        except AttributeError:
            pass

        # First, we have the filterNames
        for filterIndex,filterName in enumerate(self.lutFilterNames):
            if self.filterToBand[filterName] not in self.bands:
                # This LUT filter is not in the list of bands; we can skip it.
                continue

            try:
                bandIndex = self.bands.index(self.filterToBand[filterName])
            except KeyError:
                self.fgcmLog.warning('Observations with filter %s not in config' % (filterName))
                bandIndex = -1

            # obsFilterName is an array from fits/numpy.  filterName needs to be encoded to match
            if obsFilterNameIsEncoded:
                use, = np.where(obsFilterName == filterName.encode('utf-8'))
            else:
                use, = np.where(obsFilterName == filterName)
            if use.size == 0:
                self.fgcmLog.info('No observations in filter %s' % (filterName))
            else:
                snmm.getArray(self.obsLUTFilterIndexHandle)[use] = filterIndex
                snmm.getArray(self.obsBandIndexHandle)[use] = bandIndex

        # Check for minimum number of bands
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
        ok, = np.where(obsBandIndex >= 0)
        if (len(np.unique(obsBandIndex[ok])) < 2):
            raise NotImplementedError("Cannot run fgcmcal with fewer than 2 bands of data.")

        if not self.quietMode:
            self.fgcmLog.info('Observations matched in %.1f seconds.' %
                              (time.time() - startTime))

        #  nStars: total number of unique stars
        self.nStars = objID.size

        #  objID: unique object ID
        self.objIDHandle = snmm.createArray(self.nStars,dtype='i8')
        #  objIDAlternate: alternate, non-consecutive, object ID
        self.objIDAlternateHandle = snmm.createArray(self.nStars, dtype='i8')
        #  objRA: mean RA for object
        self.objRAHandle = snmm.createArray(self.nStars,dtype='f8')
        #  objDec: mean Declination for object
        self.objDecHandle = snmm.createArray(self.nStars,dtype='f8')
        #  objObsIndex: for each object, the first
        self.objObsIndexHandle = snmm.createArray(self.nStars,dtype='i8')
        #  objNobs: number of observations of this object (all bands)
        self.objNobsHandle = snmm.createArray(self.nStars,dtype='i4')
        #  objNGoodObsHandle: number of good observations, per band
        self.objNGoodObsHandle = snmm.createArray((self.nStars,self.nBands),dtype='i4')
        #  objNTotalObsHandle: number of all observations, per band
        self.objNTotalObsHandle = snmm.createArray((self.nStars, self.nBands), dtype='i4')
        if self.hasPsfCandidate:
            #  objNPsfCandidateHandle: number of observations that are a psf candidate, per band
            self.objNPsfCandidateHandle = snmm.createArray((self.nStars, self.nBands), dtype='i4')

        snmm.getArray(self.objIDHandle)[:] = objID
        if objIDAlternate is not None:
            snmm.getArray(self.objIDAlternateHandle)[:] = objIDAlternate
        else:
            snmm.getArray(self.objIDHandle)[:] = objID
        snmm.getArray(self.objRAHandle)[:] = objRA
        snmm.getArray(self.objDecHandle)[:] = objDec

        snmm.getArray(self.objObsIndexHandle)[:] = objObsIndex
        snmm.getArray(self.objNobsHandle)[:] = objNobs

        obsObjIDIndex = None
        objID = None
        objIDAlternate = None
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
            self.updateFlags(flagID, flagFlag)
        else:
            # we want to reserve stars, if necessary
            if self.reserveFraction > 0.0:
                objFlag = snmm.getArray(self.objFlagHandle)

                nReserve = int(self.reserveFraction * objFlag.size)
                reserve = self.rng.choice(objFlag.size,
                                          size=nReserve,
                                          replace=False)

                self.fgcmLog.info('Reserving %d stars from the fit.' % (nReserve))
                objFlag[reserve] |= objFlagDict['RESERVED']

        self._needToComputeNobs = computeNobs
        self.starsLoaded = True

    def prepStars(self, fgcmPars):
        """
        Prepare additional star quantities.  Must be called after loadStars, and is
        separate to reduce memory pressure.

        Parameters
        ----------
        fgcmPars : `fgcm.FgcmParameters`
        allocate : `bool`, optional
            Allocate shared memory?
        """
        self.fgcmLog.debug('Applying sigma0Phot = %.4f to mag errs' %
                           (self.sigma0Phot))

        obsMagADU = snmm.getArray(self.obsMagADUHandle)
        obsMagADUErr = snmm.getArray(self.obsMagADUErrHandle)

        obsFlag = snmm.getArray(self.obsFlagHandle)

        # Apply the delta-bkg if necessary
        if self.hasDeltaMagBkg:
            obsDeltaMagBkg = snmm.getArray(self.obsDeltaMagBkgHandle)
            obsMagADU[:] = obsMagADU[:] + obsDeltaMagBkg[:]

        bad, = np.where(~np.isfinite(obsMagADU))
        obsFlag[bad] |= obsFlagDict['BAD_MAG']
        if bad.size > 0:
            self.fgcmLog.info('Flagging %d observations with bad magnitudes.' %
                              (bad.size))

        bad, = np.where((np.nan_to_num(obsMagADUErr) <= 0.0) | ~np.isfinite(obsMagADUErr))
        obsFlag[bad] |= obsFlagDict['BAD_ERROR']
        if (bad.size > 0):
            self.fgcmLog.info('Flagging %d observations with bad errors.' %
                             (bad.size))

        obsMagADUErr[:] = np.sqrt(obsMagADUErr[:]**2. + self.sigma0Phot**2.)

        # Initially, we set the model error to the observed error
        obsMagADUModelErr = snmm.getArray(self.obsMagADUModelErrHandle)
        obsMagADUModelErr[:] = obsMagADUErr[:]

        startTime = time.time()
        self.fgcmLog.debug('Matching observations to exposure table.')
        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsExpIndex[:] = -1
        a,b=esutil.numpy_util.match(fgcmPars.expArray,
                                    snmm.getArray(self.obsExpHandle)[:])
        obsExpIndex[b] = a
        if not self.quietMode:
            self.fgcmLog.info('Observations matched in %.1f seconds.' %
                              (time.time() - startTime))

        bad, = np.where(obsExpIndex < 0)
        obsFlag[bad] |= obsFlagDict['NO_EXPOSURE']

        if (bad.size > 0):
            self.fgcmLog.info('Flagging %d observations with no associated exposure.' %
                             (bad.size))

        #  minObjID: minimum object ID
        self.minObjID = np.min(snmm.getArray(self.objIDHandle))
        #  maxObjID: maximum object ID
        self.maxObjID = np.max(snmm.getArray(self.objIDHandle))

        #  obsObjIDIndex: object ID Index of each observation
        #    (to get objID, then objID[obsObjIDIndex]

        startTime = time.time()
        self.fgcmLog.debug('Indexing star observations...')
        self.obsObjIDIndexHandle = snmm.createArray(self.nStarObs, dtype='i8')

        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        objID = snmm.getArray(self.objIDHandle)
        obsIndex = snmm.getArray(self.obsIndexHandle)
        objObsIndex = snmm.getArray(self.objObsIndexHandle)
        objNobs = snmm.getArray(self.objNobsHandle)
        ## FIXME: check if this extra obsIndex reference is necessary or not.
        ##   probably extraneous.
        for i in range(self.nStars):
            obsObjIDIndex[obsIndex[objObsIndex[i]:objObsIndex[i]+objNobs[i]]] = i
        if not self.quietMode:
            self.fgcmLog.info('Done indexing in %.1f seconds.' %
                              (time.time() - startTime))

        # And we need to match the reference stars if necessary
        if self.hasRefstars:
            # self.refIdHandle
            startTime = time.time()
            self.fgcmLog.debug('Matching reference star IDs')
            self.objRefIDIndexHandle = snmm.createArray(self.nStars, dtype='i8')
            objRefIDIndex = snmm.getArray(self.objRefIDIndexHandle)

            # Set the default to -1 (no match)
            objRefIDIndex[:] = -1

            objID = snmm.getArray(self.objIDHandle)
            refID = snmm.getArray(self.refIDHandle)
            refMag = snmm.getArray(self.refMagHandle)

            a, b = esutil.numpy_util.match(refID, objID)
            objRefIDIndex[b] = a

            # Check for fraction of reference stars, and downsample if necessary.
            refMatches, = np.where(objRefIDIndex >= 0)
            if (refMatches.size/objRefIDIndex.size > self.refStarMaxFracUse):
                self.fgcmLog.info("Fraction of reference star matches is greater than "
                                  "refStarMaxFracUse (%.3f); down-sampling." % (self.refStarMaxFracUse))
                objFlag = snmm.getArray(self.objFlagHandle)

                nTarget = int(self.refStarMaxFracUse*objRefIDIndex.size)
                nMatch = refMatches.size
                nToRemove = nMatch - nTarget

                remove = self.rng.choice(refMatches.size,
                                         size=nToRemove,
                                         replace=False)
                # Flag these as REFSTAR_RESERVED
                objFlag[refMatches[remove]] |= objFlagDict['REFSTAR_RESERVED']

            # Compute the fraction of stars that are reference stars
            for i, band in enumerate(self.bands):
                if not fgcmPars.hasExposuresInBand[i]:
                    continue

                gd, = np.where(refMag[:, i] < 90.0)
                fracRef = float(gd.size) / float(len(snmm.getArray(self.objIDHandle)))

                self.fgcmLog.info("%.5f%% stars have a reference match in the %s band."
                                  % (fracRef * 100.0, band))

            if not self.quietMode:
                self.fgcmLog.info('Done matching reference stars in %.1f seconds.' %
                                  (time.time() - startTime))

            # If we have a "small" number of reference stars,
            # these should not be held in reserve
            if self.reserveFraction > 0.0 and self.nRefStars == 1:
                objFlag = snmm.getArray(self.objFlagHandle)
                objRefIDIndex = snmm.getArray(self.objRefIDIndexHandle)
                cancel, = np.where(((objFlag & objFlagDict['RESERVED']) > 0) &
                                   (objRefIDIndex >= 0))
                if cancel.size > 0:
                    objFlag[cancel] &= ~objFlagDict['RESERVED']
                    self.fgcmLog.info('Cancelling RESERVED flag on %d reference stars' % (cancel.size))


        # And we need to record the mean mag, error, SED slopes...

        #  objMagStdMean: mean standard magnitude of each object, per band
        self.objMagStdMeanHandle = snmm.createArray((self.nStars, self.nBands), dtype='f8',
                                                    syncAccess=True)
        #  objMagStdMeanErr: error on the mean standard mag of each object, per band
        self.objMagStdMeanErrHandle = snmm.createArray((self.nStars, self.nBands), dtype='f8')
        #  objSEDSlope: linearized approx. of SED slope of each object, per band
        self.objSEDSlopeHandle = snmm.createArray((self.nStars, self.nBands), dtype='f8',
                                                  syncAccess=True)
        #  objMagStdMeanNoChrom: mean std mag of each object, no chromatic correction, per band
        self.objMagStdMeanNoChromHandle = snmm.createArray((self.nStars, self.nBands), dtype='f8')
        if self.hasDeltaAper:
            self.objDeltaAperMeanHandle = snmm.createArray((self.nStars, self.nBands), dtype='f4', syncAccess=True)

        startTime=time.time()
        self.fgcmLog.debug('Computing secZenith for each star observation...')
        objRARad = np.radians(snmm.getArray(self.objRAHandle))
        objDecRad = np.radians(snmm.getArray(self.objDecHandle))
        ## FIXME: deal with this at some point...
        # FIXME: Will need a more elegant solution to how to deal with the ra 0/360
        # discontinuity, especially when we add rotation.
        # Now all the deltas are 0/360 aware.
        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        obsIndex = snmm.getArray(self.obsIndexHandle)

        obsHARad = (fgcmPars.expTelHA[obsExpIndex] + fgcmPars.expTelRA[obsExpIndex] -
                    objRARad[obsObjIDIndex])
        tempSecZenith = 1./(np.sin(objDecRad[obsObjIDIndex]) * fgcmPars.sinLatitude +
                            np.cos(objDecRad[obsObjIDIndex]) * fgcmPars.cosLatitude *
                            np.cos(obsHARad))

        bad, = np.where(obsFlag != 0)
        tempSecZenith[bad] = 1.0  # filler here, but these stars aren't used
        obsSecZenith = snmm.getArray(self.obsSecZenithHandle)
        obsSecZenith[:] = tempSecZenith

        # Check the airmass range
        if ((np.min(obsSecZenith) < self.secZenithRange[0]) |
            (np.max(obsSecZenith) > self.secZenithRange[1])):
            self.fgcmLog.warning("Input stars have a secZenith that is out of range of LUT. "
                                 "Observed range is %.2f to %.2f, and LUT goes from %.2f to %.2f" %
                                 (np.min(obsSecZenith), np.max(obsSecZenith),
                                  self.secZenithRange[0], self.secZenithRange[1]))
            bad, = np.where((obsSecZenith < self.secZenithRange[0]) |
                            (obsSecZenith >= self.secZenithRange[1]))
            self.fgcmLog.warning("Marking %d observations out of airmass range as BAD_AIRMASS" % (bad.size))
            obsSecZenith[bad] = np.clip(obsSecZenith[bad],
                                        self.secZenithRange[0],
                                        self.secZenithRange[1])
            obsFlag[bad] |= obsFlagDict['BAD_AIRMASS']

        if not self.quietMode:
            self.fgcmLog.info('Computed secZenith in %.1f seconds.' %
                              (time.time() - startTime))

        self.starsPrepped = True

        if (self._needToComputeNobs):
            self.computeAllNobs(fgcmPars)

        self._needToComputeNobs = False

    def reloadStarMagnitudes(self, obsMag=None, obsMagErr=None):
        """
        Reload star magnitudes, used when automating multiple fit cycles in memory.

        Parameters
        ----------
        obsMag : `np.ndarray`, optional
            Raw ADU magnitude for each observation
        obsMagErr : `np.ndarray`, optional
            Raw ADU magnitude error for each observation
        """

        if not self.starsLoaded or not self.starsPrepped:
            raise RuntimeError("Cannot call reloadStarMagnitudes until stars have been loaded and prepped.")

        if obsMag is None:
            obsMag = snmm.getArray(self.obsMagADUOrigHandle)
        if obsMagErr is None:
            obsMagErr = snmm.getArray(self.obsMagADUErrOrigHandle)

        if len(obsMag) != self.nStarObs:
            raise RuntimeError("Replacement star magnitude has wrong length.")
        if len(obsMagErr) != self.nStarObs:
            raise RuntimeError("Replacement star magnitude error has wrong length.")

        obsMagADU = snmm.getArray(self.obsMagADUHandle)
        obsMagADUModelErr = snmm.getArray(self.obsMagADUModelErrHandle)
        obsMagADUErr = snmm.getArray(self.obsMagADUErrHandle)
        obsMagStd = snmm.getArray(self.obsMagStdHandle)
        obsDeltaStd = snmm.getArray(self.obsDeltaStdHandle)
        obsSuperStarApplied = snmm.getArray(self.obsSuperStarAppliedHandle)

        obsMagADU[:] = obsMag + self.zptABNoThroughput
        obsMagStd[:] = obsMag + self.zptABNoThroughput
        obsDeltaStd[:] = 0.0
        obsSuperStarApplied[:] = 0.0

        if self.hasDeltaMagBkg:
            obsDeltaMagBkg = snmm.getArray(self.obsDeltaMagBkgHandle)
            obsMagADU[:] += obsDeltaMagBkg[:]

        obsMagADUErr[:] = np.sqrt(obsMagErr**2. + self.sigma0Phot**2.)
        obsMagADUModelErr[:] = obsMagADUErr[:]

        snmm.getArray(self.objNGoodObsHandle)[:, :] = 0
        snmm.getArray(self.objMagStdMeanHandle)[:, :] = 0.0
        snmm.getArray(self.objMagStdMeanErrHandle)[:, :] = 0.0
        snmm.getArray(self.objSEDSlopeHandle)[:, :] = 0.0
        snmm.getArray(self.objMagStdMeanNoChromHandle)[:, :] = 0.0

        # Reset the observation flags.
        obsFlag = snmm.getArray(self.obsFlagHandle)
        obsFlag[:] = 0

        bad, = np.where(~np.isfinite(obsMagADU))
        obsFlag[bad] |= obsFlagDict["BAD_MAG"]
        if bad.size > 0:
            self.fgcmLog.debug("Flagging %d observations with bad magnitudes." %
                               (bad.size))

        bad, = np.where((np.nan_to_num(obsMagADUErr) <= 0.0) | ~np.isfinite(obsMagADUErr))
        obsFlag[bad] |= obsFlagDict["BAD_ERROR"]
        if (bad.size > 0):
            self.fgcmLog.debug("Flagging %d observations with bad errors." %
                               (bad.size))

        # Reset the object flags.
        # We only keep VARIABLE and RESERVED stars.
        objFlag = snmm.getArray(self.objFlagHandle)

        flagMask = (objFlagDict['VARIABLE'] |
                    objFlagDict['RESERVED'] |
                    objFlagDict['REFSTAR_OUTLIER'] |
                    objFlagDict['REFSTAR_RESERVED'])

        objFlag[:] &= flagMask

    def computeAllNobs(self, fgcmPars):
        """Compute all the Nobs arrays.

        Parameters
        ----------
        fgcmPars : `fgcm.fgcmParameters`
        """
        self.fgcmLog.debug("Checking stars with all exposure numbers")

        allExpsIndex = np.arange(fgcmPars.expArray.size)
        self.selectStarsMinObsExpIndex(allExpsIndex)

        self.computeNTotalStats(fgcmPars)

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
        if not self.starsLoaded or not self.starsPrepped:
            raise RuntimeError("Cannot call selectStarsMinObsExpIndex until stars have been loaded and prepped.")

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

        self.fgcmLog.debug('Selecting good stars from %d exposures.' %
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
        if not self.starsLoaded or not self.starsPrepped:
            raise RuntimeError("Cannot call selectStarsMinObsExpAndCCD until stars have been loaded and prepped.")

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

        self.fgcmLog.debug('Selecting good stars from %d exposure/ccd pairs.' %
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

    def computeNTotalStats(self, fgcmPars):
        """
        Compute ntotal statistics and psf candidate statistics if available.

        Parameters
        ----------
        fgcmPars: FgcmParameters
        """
        if not self.starsLoaded or not self.starsPrepped:
            raise RuntimeError("Cannot call computeNTotalStats until stars have been loaded and prepped.")

        goodExpsIndex, = np.where(fgcmPars.expFlag >= 0)

        minObsPerBand = 0

        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        objNTotalObs = snmm.getArray(self.objNTotalObsHandle)

        _, goodObs = esutil.numpy_util.match(goodExpsIndex, obsExpIndex)

        objNTotalObs[:, :] = 0
        np.add.at(objNTotalObs,
                  (obsObjIDIndex[goodObs],
                   obsBandIndex[goodObs]),
                  1)

        # Do the psf candidate computation if available
        if self.hasPsfCandidate:
            psfCandidate = snmm.getArray(self.psfCandidateHandle)
            objNPsfCandidate = snmm.getArray(self.objNPsfCandidateHandle)

            ispsf, = np.where(psfCandidate[goodObs])
            psfObs = goodObs[ispsf]

            objNPsfCandidate[:, :] = 0
            np.add.at(objNPsfCandidate,
                      (obsObjIDIndex[psfObs],
                       obsBandIndex[psfObs]),
                      1)

    def getGoodStarIndices(self, includeReserve=False, onlyReserve=False, checkMinObs=False,
                           checkHasColor=False, removeRefstarOutliers=False, removeRefstarBadcols=False,
                           removeRefstarReserved=False):
        """
        Get the good star indices.

        parameters
        ----------
        includeReserve : `bool`, optional
            Include reserved stars?
        onlyReserve : `bool`, optional
            Only include reserved stars?
        checkMinObs : `bool`, optional
            Extra(?) check for minimum number of observations
        checkHasColor : `bool`, optional
            Check that the stars have the g-i or equivalent color?
        removeRefstarOutliers : `bool`, optional
            Remove reference star outliers.
        removeRefstarBadcols : `bool`, optional
            Remove reference stars with "bad colors".
        removeRefstarReserved : `bool`, optional
            Remove reference stars that are "reserved"?

        returns
        -------
        goodStars: np.array of good star indices
        """
        if not self.starsLoaded or not self.starsPrepped:
            raise RuntimeError("Cannot call getGoodStarIndices until stars have been loaded and prepped.")

        mask = (objFlagDict['TOO_FEW_OBS'] |
                objFlagDict['BAD_COLOR'] |
                objFlagDict['VARIABLE'] |
                objFlagDict['TEMPORARY_BAD_STAR'] |
                objFlagDict['RESERVED'])

        if removeRefstarOutliers:
            mask |= objFlagDict['REFSTAR_OUTLIER']
        if removeRefstarBadcols:
            mask |= objFlagDict['REFSTAR_BAD_COLOR']
        if removeRefstarReserved:
            mask |= objFlagDict['REFSTAR_RESERVED']

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
        if not self.starsLoaded or not self.starsPrepped:
            raise RuntimeError("Cannot call getGoodObsIndices until stars have been loaded and prepped.")

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
            objMagStdMeanErr = snmm.getArray(self.objMagStdMeanErrHandle)
            okFlag &= ((objMagStdMean[goodStars[goodStarsSub], obsBandIndex[goodObs]] < 99.0) &
                       (objMagStdMeanErr[goodStars[goodStarsSub], obsBandIndex[goodObs]] < 99.0))

        if not self.allFitBandsAreRequired or self.nNotFitBands > 0:
            # We need to do some extra checks since not all fit bands are required
            # Or we have some extra bands.
            objNGoodObs = snmm.getArray(self.objNGoodObsHandle)

            okFlag &= (objNGoodObs[goodStars[goodStarsSub], obsBandIndex[goodObs]] >= self.minObsPerBand)

        if requireSED:
            # We need to ensure that we have an SED
            obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
            objSEDSlope = snmm.getArray(self.objSEDSlopeHandle)

            okFlag &= ((objSEDSlope[goodStars[goodStarsSub], obsBandIndex[goodObs]] != 0.0) &
                       (objSEDSlope[goodStars[goodStarsSub], obsBandIndex[goodObs]] != self.missingSedValues[obsBandIndex[goodObs]]))

        return goodStarsSub[okFlag], goodObs[okFlag]

    def plotStarMap(self,mapType='initial'):
        """
        Plot star map.

        parameters
        ----------
        mapType: string, default='initial'
           A key for labeling the map.
        """

        # This is not currently used.
        # FIXME: add skyproj plotting.
        return

    def computeObjectSEDSlopes(self,objIndicesIn):
        """
        Compute fnuprime (object SED slopes) for a list of objects.
        Output is saved in objSEDSlope.

        parameters
        ----------
        objIndicesIn: int array
           Array of object indices to do computation
        """

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
        objSEDSlopeOI = np.zeros((objIndicesIn.size, self.nBands), dtype='f8')
        objNGoodObsOI = objNGoodObs[objIndicesIn, :]

        # New mapping, nothing needs to be sequential, it's all configured

        # First compute the terms
        S = {}
        for boundaryTermName, boundaryTerm in self.sedBoundaryTermDict.items():
            try:
                index0 = self.bands.index(boundaryTerm['primary'])
                index1 = self.bands.index(boundaryTerm['secondary'])
            except ValueError:
                # Not in the list; set to nan
                S[boundaryTermName] = np.full(len(objMagStdMeanOI[:, 0]), np.nan)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                S[boundaryTermName] = (-1. / self.magConstant) * (objMagStdMeanOI[:, index0] - objMagStdMeanOI[:, index1]) / ((self.lambdaStdBand[index0] - self.lambdaStdBand[index1]))

            # And flag the ones that are bad
            bad, = np.where((objMagStdMeanOI[:, index0] >= 90.0) |
                            (objMagStdMeanOI[:, index1] >= 90.0))
            S[boundaryTermName][bad] = np.nan

        # Now for each band
        for bandIndex, band in enumerate(self.bands):
            sedTerm = self.sedTermDict[band]

            if sedTerm['secondaryTerm'] is not None:
                use, = np.where((np.isfinite(S[sedTerm['primaryTerm']])) &
                                (np.isfinite(S[sedTerm['secondaryTerm']])))
            else:
                use, = np.where(np.isfinite(S[sedTerm['primaryTerm']]))

            if sedTerm['extrapolated']:
                # Extrapolated
                primaryIndex = self.bands.index(sedTerm['primaryBand'])
                secondaryIndex = self.bands.index(sedTerm['secondaryBand'])
                tertiaryIndex = self.bands.index(sedTerm['tertiaryBand'])

                objSEDSlopeOI[use, bandIndex] = (
                    S[sedTerm['primaryTerm']][use] + sedTerm['constant'] * (
                        (self.lambdaStdBand[primaryIndex] - self.lambdaStdBand[secondaryIndex]) /
                        (self.lambdaStdBand[primaryIndex] - self.lambdaStdBand[tertiaryIndex])) *
                    (S[sedTerm['primaryTerm']][use] - S[sedTerm['secondaryTerm']][use]))
            else:
                # Interpolated
                if sedTerm['secondaryTerm'] is not None:
                    objSEDSlopeOI[use, bandIndex] = (
                        sedTerm['constant'] * ((S[sedTerm['primaryTerm']][use] +
                                                S[sedTerm['secondaryTerm']][use]) / 2.0))
                else:
                    objSEDSlopeOI[use, bandIndex] = (
                        sedTerm['constant'] * S[sedTerm['primaryTerm']][use])

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
        #objSEDSlopeOI = np.zeros((objIndicesIn.size,self.nBands),dtype='f8')

        # compute SED color...
        ## FIXME: make this configurable
        objSEDColorOI = objMagStdMeanOI[:,0] - objMagStdMeanOI[:,2]

        # do the look-up
        objSEDSlopeOI = fgcmLUT.computeSEDSlopes(objSEDColorOI)

        # and save the values, protected
        objSEDSlopeLock.acquire()

        objSEDSlope[objIndicesIn,:] = objSEDSlopeOI

        objSEDSlopeLock.release()

    def fillMissingSedSlopes(self, fgcmPars):
        """
        Fill missing SED slopes with median values

        Parameters
        ----------
        fgcmPars : `fgcmParameters`
        """
        objFlag = snmm.getArray(self.objFlagHandle)
        objSEDSlope = snmm.getArray(self.objSEDSlopeHandle)
        objNGoodObs = snmm.getArray(self.objNGoodObsHandle)

        for bandIndex, band in enumerate(self.bands):
            # The exact 0.0 is a special value that it wasn't measured.  This isn't
            # a different sentinel value to keep things from blowing up with -9999s
            # We assume here that all stars that have SED measurements are ok to use
            sedNonZero = (objSEDSlope[:, bandIndex] != 0.0)

            if sedNonZero.sum() < 3:
                fgcmPars.compMedianSedSlope[bandIndex] = 0.0
            else:
                fgcmPars.compMedianSedSlope[bandIndex] = np.median(objSEDSlope[sedNonZero, bandIndex])

            self.fgcmLog.info('Median SED slope in %s band (%.8f)' % (band, fgcmPars.compMedianSedSlope[bandIndex]))

            # And replace 0s with the median value.
            objSEDSlope[~sedNonZero, bandIndex] = fgcmPars.compMedianSedSlope[bandIndex]

            # And save this locally
            self.missingSedValues[bandIndex] = fgcmPars.compMedianSedSlope[bandIndex]

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
            self.fgcmLog.warning("Cannot compute abs offset without reference stars.")
            return np.zeros(self.nBands)

        # Set things up
        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.objMagStdMeanErrHandle)
        objRefIDIndex = snmm.getArray(self.objRefIDIndexHandle)
        objFlag = snmm.getArray(self.objFlagHandle)
        refMag = snmm.getArray(self.refMagHandle)
        refMagErr = snmm.getArray(self.refMagErrHandle)

        goodStars = self.getGoodStarIndices(includeReserve=False, checkMinObs=True)

        mask = objFlagDict['REFSTAR_OUTLIER'] | objFlagDict['REFSTAR_BAD_COLOR'] | objFlagDict['REFSTAR_RESERVED']
        use, = np.where((objRefIDIndex[goodStars] >= 0) &
                        ((objFlag[goodStars] & mask) == 0))
        goodRefStars = goodStars[use]

        deltaOffsetRef = np.zeros(self.nBands)
        deltaOffsetWtRef = np.zeros(self.nBands)

        gdStarInd, gdBandInd = np.where((objMagStdMean[goodRefStars, :] < 90.0) &
                                        (refMag[objRefIDIndex[goodRefStars], :] < 90.0))
        delta = objMagStdMean[goodRefStars, :] - refMag[objRefIDIndex[goodRefStars], :]
        wt = 1. / (objMagStdMeanErr[goodRefStars, :]**2. +
                   refMagErr[objRefIDIndex[goodRefStars], :]**2.)

        np.add.at(
            deltaOffsetRef,
            gdBandInd,
            (delta[gdStarInd, gdBandInd] * wt[gdStarInd, gdBandInd]).astype(deltaOffsetRef.dtype),
        )
        np.add.at(
            deltaOffsetWtRef,
            gdBandInd,
            (wt[gdStarInd, gdBandInd]).astype(deltaOffsetWtRef.dtype),
        )

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

    def computeEGray(self, goodObs, ignoreRef=True, onlyObsErr=False):
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
        ignoreRef: `bool`, default=True
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
        objFlag = snmm.getArray(self.objFlagHandle)

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

            # Only use _good_ (non-outlier) reference stars in here.
            mask = objFlagDict['REFSTAR_OUTLIER'] | objFlagDict['REFSTAR_BAD_COLOR'] | objFlagDict['REFSTAR_RESERVED']
            goodRefObsGO, = np.where((objRefIDIndex[obsObjIDIndex[goodObs]] >= 0) &
                                     ((objFlag[obsObjIDIndex[goodObs]] & mask) == 0))

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
            self.fgcmLog.info('Applying color cut: %.3f < %s-%s < %.3f' % (cCut[2],
                                                                           self.bands[cCut[0]],
                                                                           self.bands[cCut[1]],
                                                                           cCut[3]))
            ok, = np.where((objMagStdMean[:, cCut[0]] < 90.0) &
                           (objMagStdMean[:, cCut[1]] < 90.0))

            thisColor = objMagStdMean[ok, cCut[0]] - objMagStdMean[ok, cCut[1]]
            bad, = np.where((thisColor < cCut[2]) |
                            (thisColor > cCut[3]))
            objFlag[ok[bad]] |= objFlagDict['BAD_COLOR']

            self.fgcmLog.info('Flag %d stars of %d with BAD_COLOR' % (bad.size,self.nStars))

        # This config says "apply starColorCuts to reference stars"
        if self.hasRefstars and not self.applyRefStarColorCuts:
            objRefIDIndex = snmm.getArray(self.objRefIDIndexHandle)
            cancel, = np.where(((objFlag & objFlagDict['BAD_COLOR']) > 0) &
                               (objRefIDIndex >= 0))
            if cancel.size > 0:
                objFlag[cancel] &= ~objFlagDict['BAD_COLOR']
                self.fgcmLog.info('Cancelling BAD_COLOR flag on %d reference stars' % (cancel.size))

        if self.hasRefstars:
            objRefIDIndex = snmm.getArray(self.objRefIDIndexHandle)
            refMag = snmm.getArray(self.refMagHandle)

            for cCut in self.refStarColorCuts:
                self.fgcmLog.info('Applying reference star color cut: %.3f < %s-%s < %.3f' % (cCut[2],
                                                                                              self.bands[cCut[0]],
                                                                                              self.bands[cCut[1]],
                                                                                              cCut[3]))
                # First cut based on refstar colors
                ok, = np.where((refMag[objRefIDIndex, cCut[0]] < 90.0) &
                               (refMag[objRefIDIndex, cCut[1]] < 90.0))

                thisColor = refMag[objRefIDIndex[ok], cCut[0]] - refMag[objRefIDIndex[ok], cCut[1]]
                bad, = np.where((thisColor < cCut[2]) | (thisColor > cCut[3]))
                objFlag[ok[bad]] |= objFlagDict['REFSTAR_BAD_COLOR']
                self.fgcmLog.info('Flag %d stars with REFSTAR_BAD_COLOR (ref colors)' % (bad.size))

                # Next cut based on std colors
                ok, = np.where((objMagStdMean[:, cCut[0]] < 90.0) &
                               (objMagStdMean[:, cCut[1]] < 90.0))

                thisColor = objMagStdMean[ok, cCut[0]] - objMagStdMean[ok, cCut[1]]
                bad, = np.where((thisColor < cCut[2]) | (thisColor > cCut[3]))
                objFlag[ok[bad]] |= objFlagDict['REFSTAR_BAD_COLOR']
                self.fgcmLog.info('Flag %d stars with REFSTAR_BAD_COLOR (std colors)' % (bad.size))

    def performSuperStarOutlierCuts(self, fgcmPars, reset=False):
        """
        Do outlier cuts from common ccd/filter/epochs
        """

        self.fgcmLog.debug('Computing superstar outliers')

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

    def performFocalPlaneOutlierCuts(self, fgcmPars, reset=False, ignoreRef=False):
        """Do focal plane outlier cuts per exposure.

        Parameters
        ----------
        fgcmPars : `fgcmParameters`
        reset : `bool`, optional
           Reset the outlier flag
        ignoreRef : `bool`, optional
           Ignore reference stars
        """

        self.fgcmLog.debug('Computing focal plane outliers')

        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)

        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
        obsMagStd = snmm.getArray(self.obsMagStdHandle)
        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.obsFlagHandle)

        if ignoreRef:
            # Without reference stars, these are simply FOCALPLANE_OUTLIERs
            flagName = 'FOCALPLANE_OUTLIER'
        else:
            # With reference stars, we use a separate flag so that we
            # can mark these bad observations specifically when we
            # are using reference star magnitudes.
            flagName = 'FOCALPLANE_OUTLIER_REF'

        if reset:
            # Reset the obsFlag...
            if ignoreRef:
                obsFlag &= ~obsFlagDict[flagName]
            else:
                obsFlag &= ~obsFlagDict[flagName]

        goodStars = self.getGoodStarIndices(checkMinObs=True)
        _, goodObs = self.getGoodObsIndices(goodStars)

        # we need to compute E_gray == <mstd> - mstd for each observation
        # compute EGray, GO for Good Obs
        EGrayGO, EGrayErr2GO = self.computeEGray(goodObs, onlyObsErr=True, ignoreRef=ignoreRef)

        h, rev = esutil.stat.histogram(obsExpIndex[goodObs], rev=True)

        nbad = 0

        use, = np.where(h > 0)
        for i in use:
            i1a = rev[rev[i]: rev[i + 1]]

            med = np.median(EGrayGO[i1a])
            sig = 1.4826*np.median(np.abs(EGrayGO[i1a] - med))
            bad, = np.where(np.abs(EGrayGO[i1a] - med) > self.focalPlaneSigmaClip*sig)

            obsFlag[goodObs[i1a[bad]]] |= obsFlagDict[flagName]

            nbad += bad.size

        self.fgcmLog.info("Marked %d observations (%.4f%%) as %s" %
                          (nbad, 100.*float(nbad)/float(goodObs.size), flagName))

        # Now we need to flag stars that might have fallen below our threshold
        # when we flagged these outliers
        goodExpsIndex, = np.where(fgcmPars.expFlag == 0)
        self.selectStarsMinObsExpIndex(goodExpsIndex, reset=reset)

    def applySuperStarFlat(self,fgcmPars):
        """
        Apply superStarFlat to raw magnitudes.

        Note that this modifies obsMagADU.

        parameters
        ----------
        fgcmPars: FgcmParameters
        """
        if not self.starsLoaded or not self.starsPrepped:
            raise RuntimeError("Cannot call applySuperStarFlat until stars have been loaded and prepped.")

        self.fgcmLog.debug('Applying SuperStarFlat to raw magnitudes')

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

            for i in range(h.size):
                if h[i] == 0: continue

                i1a = rev[rev[i]:rev[i+1]]

                # get the indices for this epoch/filter/ccd
                epInd = fgcmPars.expEpochIndex[obsExpIndex[i1a[0]]]
                fiInd = fgcmPars.expLUTFilterIndex[obsExpIndex[i1a[0]]]
                cInd = obsCCDIndex[i1a[0]]

                field = Cheb2dField(self.deltaMapperDefault['x_size'][cInd],
                                    self.deltaMapperDefault['y_size'][cInd],
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

        Note that this modifies obsMagADU.

        parameters
        ----------
        fgcmPars: FgcmParameters
        """
        if not self.starsLoaded or not self.starsPrepped:
            raise RuntimeError("Cannot call applyApertureCorrection until stars have been loaded and prepped.")

        self.fgcmLog.debug('Applying ApertureCorrections to raw magnitudes')

        if self.seeingSubExposure:
            self.fgcmLog.debug('Aperture correction has sub-exposure information')
        else:
            self.fgcmLog.debug('Aperture correction is per-exposure')

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

    def applyModelMagErrorModel(self, fgcmPars):
        """
        Apply magnitude error model.

        parameters
        ----------
        fgcmPars: FgcmParameters
        """
        if not self.starsLoaded or not self.starsPrepped:
            raise RuntimeError("Cannot call applyModelMagErrorModel until stars have been loaded and prepped.")

        if (np.max(fgcmPars.compModelErrFwhmPivot) <= 0.0) :
            self.fgcmLog.debug('No model for mag errors, so mag errors are unchanged.')
            return

        if not self.modelMagErrors:
            self.fgcmLog.debug('Model magnitude errors are turned off.')
            return

        if not self.magStdComputed:
            raise RuntimeError("Must run FgcmChisq to compute magStd before computeModelMagErrors")

        self.fgcmLog.debug('Computing model magnitude errors for photometric observations')

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
        for bandIndex in range(fgcmPars.nBands):
            if not fgcmPars.hasExposuresInBand[bandIndex]:
                continue
            if fgcmPars.compModelErrFwhmPivot[bandIndex] <= 0.0:
                self.fgcmLog.info('No error model for band %s' % (self.bands[bandIndex]))
                continue

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

    def applyMirrorChromaticityCorrection(self, fgcmPars, fgcmLUT, returnCorrections=False):
        """
        Apply mirror chromaticity model

        Parameters
        ----------
        fgcmPars: `fgcmParameters`
        returnCorrections: `bool`, optional
           Just return the corrections, don't apply them.  Default is False.
        """
        if not self.starsLoaded or not self.starsPrepped:
            raise RuntimeError("Cannot call applyMirrorChromaticityCorrection until stars have been loaded and prepped.")

        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
        obsLUTFilterIndex = snmm.getArray(self.obsLUTFilterIndexHandle)
        obsMagADU = snmm.getArray(self.obsMagADUHandle)

        objSEDSlope = snmm.getArray(self.objSEDSlopeHandle)
        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)

        cAl = fgcmPars.expCTrans[obsExpIndex]

        termOne = 1.0 + (cAl / fgcmLUT.lambdaStd[obsBandIndex]) * fgcmLUT.I10Std[obsBandIndex]
        obsSEDSlope = objSEDSlope[obsObjIDIndex, obsBandIndex]
        termTwo = 1.0 + (((cAl / fgcmLUT.lambdaStd[obsBandIndex]) * (fgcmLUT.I1Std[obsBandIndex] + obsSEDSlope * fgcmLUT.I2Std[obsBandIndex])) /
                         (fgcmLUT.I0Std[obsBandIndex] + obsSEDSlope * fgcmLUT.I1Std[obsBandIndex]))
        deltaMag = -2.5 * np.log10(termOne) + 2.5 * np.log10(termTwo)

        if returnCorrections:
            return deltaMag

        obsMagADU += deltaMag

    def applyCCDChromaticityCorrection(self, fgcmPars, fgcmLUT, returnCorrections=False):
        """
        Apply CCD chromaticity model.

        Parameters
        ----------
        fgcmPars : `fgcm.fgcmParameters`
        returnCorrections : `bool`, optional
            Just return the corrections, don't apply them.
        """
        if not self.starsLoaded or not self.starsPrepped:
            raise RuntimeError("Cannot call applyCCDChromaticityCorrection until stars have been loaded and prepped.")

        obsCCDIndex = snmm.getArray(self.obsCCDHandle) - self.ccdStartIndex
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
        obsLUTFilterIndex = snmm.getArray(self.obsLUTFilterIndexHandle)
        obsMagADU = snmm.getArray(self.obsMagADUHandle)

        objSEDSlope = snmm.getArray(self.objSEDSlopeHandle)
        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)

        c = fgcmPars.compCCDChromaticity[obsCCDIndex, obsLUTFilterIndex]

        termOne = 1.0 + (c / fgcmLUT.lambdaStd[obsLUTFilterIndex]) * fgcmLUT.I10Std[obsLUTFilterIndex]
        obsSEDSlope = objSEDSlope[obsObjIDIndex, obsBandIndex]
        termTwo = 1.0 + (((c / fgcmLUT.lambdaStd[obsLUTFilterIndex]) * (fgcmLUT.I1Std[obsLUTFilterIndex] + obsSEDSlope * fgcmLUT.I2Std[obsLUTFilterIndex])) /
                         (fgcmLUT.I0Std[obsLUTFilterIndex] + obsSEDSlope * fgcmLUT.I1Std[obsLUTFilterIndex]))
        deltaMag = -2.5 * np.log10(termOne) + 2.5 * np.log10(termTwo)

        if returnCorrections:
            return deltaMag

        obsMagADU += deltaMag

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
                    objFlagDict['REFSTAR_OUTLIER'] |
                    objFlagDict['REFSTAR_RESERVED'])

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

        stdStars, goodBands = self.retrieveStdStarCatalog(fgcmPars)

        hdr = fitsio.FITSHDR()
        for i, goodBand in enumerate(goodBands):
            hdr['BAND%d' % (i)] = goodBand

        fitsio.write(starFile, stdStars, clobber=True, header=hdr)

    def retrieveStdStarCatalog(self, fgcmPars):
        """
        Retrieve standard star catalog.  Note that this does not fill in holes (yet).

        parameters
        ----------
        fgcmPars: FgcmParameters
        """

        objID = snmm.getArray(self.objIDHandle)
        objIDAlternate = snmm.getArray(self.objIDAlternateHandle)
        objFlag = snmm.getArray(self.objFlagHandle)
        objRA = snmm.getArray(self.objRAHandle)
        objDec = snmm.getArray(self.objDecHandle)
        objNGoodObs = snmm.getArray(self.objNGoodObsHandle)
        objNTotalObs = snmm.getArray(self.objNTotalObsHandle)
        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.objMagStdMeanErrHandle)
        if self.hasPsfCandidate:
            objNPsfCandidate = snmm.getArray(self.objNPsfCandidateHandle)

        rejectMask = (objFlagDict['BAD_COLOR'] | objFlagDict['VARIABLE'] |
                      objFlagDict['TOO_FEW_OBS'])

        goodStars, = np.where((objFlag & rejectMask) == 0)

        goodBands, = np.where(fgcmPars.hasExposuresInBand)
        goodBandNames = [fgcmPars.bands[i] for i in goodBands]

        dtype=[('FGCM_ID', 'i8'),
               ('ALTERNATE_ID', 'i8'),
               ('RA', 'f8'),
               ('DEC', 'f8'),
               ('FLAG', 'i4'),
               ('NGOOD', 'i4', goodBands.size),
               ('NTOTAL', 'i4', goodBands.size),
               ('MAG_STD', 'f4', goodBands.size),
               ('MAGERR_STD', 'f4', goodBands.size)]
        if self.hasPsfCandidate:
            dtype.append(('NPSFCAND', 'i4', goodBands.size))
        if self.hasDeltaAper:
            objDeltaAperMean = snmm.getArray(self.objDeltaAperMeanHandle)
            dtype.append(('DELTA_APER', 'f4', goodBands.size))

        outCat = np.zeros(goodStars.size, dtype=dtype)

        outCat['FGCM_ID'] = objID[goodStars]
        outCat['ALTERNATE_ID'] = objIDAlternate[goodStars]
        outCat['RA'] = objRA[goodStars]
        outCat['DEC'] = objDec[goodStars]
        outCat['FLAG'] = objFlag[goodStars]
        for i, goodBand in enumerate(goodBands):
            outCat['NGOOD'][:, i] = objNGoodObs[goodStars, goodBand]
            outCat['NTOTAL'][:, i] = objNTotalObs[goodStars, goodBand]
            outCat['MAG_STD'][:, i] = objMagStdMean[goodStars, goodBand]
            outCat['MAGERR_STD'][:, i] = objMagStdMeanErr[goodStars, goodBand]
            if self.hasPsfCandidate:
                outCat['NPSFCAND'][:, i] = objNPsfCandidate[goodStars, goodBand]
            if self.hasDeltaAper:
                outCat['DELTA_APER'][:, i] = objDeltaAperMean[goodStars, goodBand]

        return outCat, goodBandNames

    def plotRefStarColorTermResiduals(self, fgcmPars):
        """
        Plot reference star color-term residuals.

        Parameters
        ----------
        fgcmPars : `fgcm.FgcmParameters`
        """
        from .fgcmUtilities import dataBinner

        if not self.hasRefstars:
            self.fgcmLog.info("No reference stars for color term residual plots.")
            return

        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)
        objNGoodObs = snmm.getArray(self.objNGoodObsHandle)
        objFlag = snmm.getArray(self.objFlagHandle)

        objRefIDIndex = snmm.getArray(self.objRefIDIndexHandle)
        refMag = snmm.getArray(self.refMagHandle)

        for mode in ['all', 'cut']:
            # We leave in the "RESERVED" stars for these plots.
            if (mode == 'all'):
                goodStars = self.getGoodStarIndices(includeReserve=True,
                                                    checkMinObs=True,
                                                    removeRefstarOutliers=True)
            else:
                goodStars = self.getGoodStarIndices(includeReserve=True,
                                                    checkMinObs=True,
                                                    removeRefstarOutliers=True,
                                                    removeRefstarBadcols=True)

            # Select only stars that have reference magnitudes
            use, = np.where(objRefIDIndex[goodStars] >= 0)
            goodRefStars = goodStars[use]

            # Compute "g-i" based on the configured colorSplitIndices
            gmiGRS = (objMagStdMean[goodRefStars, self.colorSplitIndices[0]] -
                      objMagStdMean[goodRefStars, self.colorSplitIndices[1]])

            okColor, = np.where((objMagStdMean[goodRefStars, self.colorSplitIndices[0]] < 90.0) &
                                (objMagStdMean[goodRefStars, self.colorSplitIndices[1]] < 90.0))

            for bandIndex, band in enumerate(self.bands):
                if not fgcmPars.hasExposuresInBand[bandIndex]:
                    continue

                fig = makeFigure(figsize=(8, 6))
                fig.clf()
                ax = fig.add_subplot(111)

                refUse, = np.where((refMag[objRefIDIndex[goodRefStars[okColor]], bandIndex] < 90.0) &
                                   (objMagStdMean[goodRefStars[okColor], bandIndex] < 90.0))
                if refUse.size == 0:
                    continue
                refUse = okColor[refUse]

                delta = (objMagStdMean[goodRefStars[refUse], bandIndex] -
                         refMag[objRefIDIndex[goodRefStars[refUse]], bandIndex])

                st = np.argsort(delta)
                ylow = delta[st[int(0.02*refUse.size)]]
                yhigh = delta[st[int(0.98*refUse.size)]]
                st = np.argsort(gmiGRS[refUse])
                xlow = gmiGRS[refUse[st[int(0.02*refUse.size)]]]
                xhigh = gmiGRS[refUse[st[int(0.98*refUse.size)]]]

                if refUse.size >= 1000:
                    ax.hexbin(gmiGRS[refUse], delta, bins='log', extent=[xlow, xhigh, ylow, yhigh], cmap=colormaps.get_cmap("viridis"))
                else:
                    ax.plot(gmiGRS[refUse], delta, 'k.')
                    ax.set_xlim(xlow, xhigh)
                    ax.set_ylim(ylow, yhigh)

                # Only do the binning if we have data
                if xhigh > xlow and refUse.size > 10:
                    binstruct = dataBinner(gmiGRS[refUse], delta, 0.1, [xlow, xhigh], nTrial=10)
                    ok, = np.where(binstruct['N'] > 0)
                    ax.plot(binstruct['X'][ok], binstruct['Y'][ok], 'r--')

                if mode == 'all':
                    title = '%s band: Ref stars, full color range' % (band)
                else:
                    title = '%s band: Ref stars, cut color range (used in calibration)' % (band)
                ax.set_title(title)
                ax.set_xlabel('%s - %s (mag)' % (self.colorSplitBands[0], self.colorSplitBands[1]))
                ax.set_ylabel('%s_std - %s_ref (mag)' % (band, band))

                fig.tight_layout()
                if self.butlerQC is not None:
                    putButlerFigure(self.fgcmLog,
                                    self.butlerQC,
                                    self.plotHandleDict,
                                    f"RefResidVsColor{mode.title()}",
                                    self.cycleNumber,
                                    fig,
                                    band=band)
                elif self.plotPath is not None:
                    fig.savefig('%s/%s_refresidvscol_%s_%s.png' % (self.plotPath,
                                                                   self.outfileBaseWithCycle,
                                                                   band,
                                                                   mode))

    def __getstate__(self):
        # Don't try to pickle the logger.

        state = self.__dict__.copy()
        del state['fgcmLog']
        del state['butlerQC']
        del state['plotHandleDict']
        return state

    def freeSharedMemory(self):
        """Free shared memory"""
        snmm.freeArray(self.obsIndexHandle)
        snmm.freeArray(self.obsExpHandle)
        snmm.freeArray(self.obsExpIndexHandle)
        snmm.freeArray(self.obsCCDHandle)
        snmm.freeArray(self.obsBandIndexHandle)
        snmm.freeArray(self.obsLUTFilterIndexHandle)
        snmm.freeArray(self.obsFlagHandle)
        snmm.freeArray(self.obsRAHandle)
        snmm.freeArray(self.obsDecHandle)
        snmm.freeArray(self.obsSecZenithHandle)
        snmm.freeArray(self.obsMagADUHandle)
        snmm.freeArray(self.obsMagADUErrHandle)
        snmm.freeArray(self.obsMagADUModelErrHandle)
        snmm.freeArray(self.obsSuperStarAppliedHandle)
        snmm.freeArray(self.obsMagStdHandle)
        snmm.freeArray(self.obsDeltaStdHandle)
        if self.hasXY:
            snmm.freeArray(self.obsXHandle)
            snmm.freeArray(self.obsYHandle)
        if self.hasPsfCandidate:
            snmm.freeArray(self.psfCandidateHandle)
        if self.hasDeltaMagBkg:
            snmm.freeArray(self.obsDeltaMagBkgHandle)
        if self.hasRefstars:
            snmm.freeArray(self.refIDHandle)
            snmm.freeArray(self.refMagHandle)
            snmm.freeArray(self.refMagErrHandle)
        snmm.freeArray(self.objIDHandle)
        snmm.freeArray(self.objRAHandle)
        snmm.freeArray(self.objDecHandle)
        snmm.freeArray(self.objObsIndexHandle)
        snmm.freeArray(self.objNobsHandle)
        snmm.freeArray(self.objNGoodObsHandle)
        snmm.freeArray(self.objNTotalObsHandle)
        if self.hasPsfCandidate:
            snmm.freeArray(self.objNPsfCandidateHandle)
        snmm.freeArray(self.objFlagHandle)
        snmm.freeArray(self.obsObjIDIndexHandle)
        if self.hasRefstars:
            snmm.freeArray(self.objRefIDIndexHandle)
        snmm.freeArray(self.objMagStdMeanHandle)
        snmm.freeArray(self.objMagStdMeanErrHandle)
        snmm.freeArray(self.objSEDSlopeHandle)
        snmm.freeArray(self.objMagStdMeanNoChromHandle)
