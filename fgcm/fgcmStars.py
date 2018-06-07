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
    sedFitBandFudgeFactors: float array
       Fudge factors for computing fnuprime for the fit bands
    sedExtraBandFudgeFactors: float array
       Fudge factors for computing fnuprime for the extra bands
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

        self.bands = fgcmConfig.bands
        self.nBands = len(fgcmConfig.bands)
        self.nCCD = fgcmConfig.nCCD
        self.minObsPerBand = fgcmConfig.minObsPerBand
        self.fitBands = fgcmConfig.fitBands
        self.nFitBands = len(fgcmConfig.fitBands)
        self.extraBands = fgcmConfig.extraBands
        self.sedFitBandFudgeFactors = fgcmConfig.sedFitBandFudgeFactors
        self.sedExtraBandFudgeFactors = fgcmConfig.sedExtraBandFudgeFactors
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

        self.bandRequiredFlag = fgcmConfig.bandRequiredFlag
        self.bandRequiredIndex = np.where(self.bandRequiredFlag)[0]
        self.bandExtraFlag = fgcmConfig.bandExtraFlag
        self.bandExtraIndex = np.where(self.bandExtraFlag)[0]

        self.lutFilterNames = fgcmConfig.lutFilterNames
        self.filterToBand = fgcmConfig.filterToBand

        self.superStarSubCCD = fgcmConfig.superStarSubCCD

        #self.expArray = fgcmPars.expArray

        #self._loadStars(fgcmPars)

        self.magStdComputed = False
        self.allMagStdComputed = False
        self.sedSlopeComputed = False

        #if (computeNobs):
        #    allExps = np.arange(fgcmConfig.expRange[0],fgcmConfig.expRange[1],dtype='i4')
        #    self.fgcmLog.info('Checking stars with full possible range of exp numbers')
            #self.selectStarsMinObs(goodExps=allExps,doPlots=False)
        #    allExpsIndex = np.arange(fgcmPars.expArray.size)
        #    self.selectStarsMinObsExpIndex(allExpsIndex)

        self.magConstant = 2.5/np.log(10)

        self.hasXY = False

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
        index = fitsio.read(self.indexFile, ext='INDEX')
        self.fgcmLog.info('Done reading in %d observation indices in %.1f seconds.' %
                         (index.size, time.time() - startTime))

        # read in obsfile and cut
        startTime = time.time()
        self.fgcmLog.info('Reading in star observations...')
        obs = fitsio.read(self.obsFile, ext=1)
        # cut down to those that are indexed
        obs = obs[index['OBSINDEX']]
        self.fgcmLog.info('Done reading in %d observations in %.1f seconds.' %
                         (obs.size, time.time() - startTime))

        # and positions...
        startTime = time.time()
        self.fgcmLog.info('Reading in star positions...')
        pos = fitsio.read(self.indexFile, ext='POS')
        self.fgcmLog.info('Done reading in %d unique star positions in %.1f secondds.' %
                         (pos.size, time.time() - startTime))

        #obsBand = np.core.defchararray.strip(obs['BAND'][:])
        obsFilterName = np.core.defchararray.strip(obs['FILTERNAME'][:])

        if (self.inFlagStarFile is not None):
            self.fgcmLog.info('Reading in list of previous flagged stars from %s' %
                             (self.inFlagStarFile))

            inFlagStars = fitsio.read(self.inFlagStarFile, ext=1)

            flagID = inFlagStars['OBJID']
            flagFlag = inFlagStars['OBJFLAG']
        else:
            flagID = None
            flagFlag = None

        # FIXME: add support to x/y from fits files
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
                       flagID=flagID,
                       flagFlag=flagFlag,
                       computeNobs=computeNobs)

        # and clear memory
        index = None
        obs = None
        pos = None

    def loadStars(self, fgcmPars,
                  obsExp, obsCCD, obsRA, obsDec, obsMag, obsMagErr, obsFilterName,
                  objID, objRA, objDec, objObsIndex, objNobs, obsX=None, obsY=None,
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
        #  nStarObs: total number of observations of all starus
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

        snmm.getArray(self.obsExpHandle)[:] = obsExp
        snmm.getArray(self.obsCCDHandle)[:] = obsCCD
        snmm.getArray(self.obsRAHandle)[:] = obsRA
        snmm.getArray(self.obsDecHandle)[:] = obsDec
        snmm.getArray(self.obsMagADUHandle)[:] = obsMag
        snmm.getArray(self.obsMagADUErrHandle)[:] = obsMagErr
        snmm.getArray(self.obsMagStdHandle)[:] = obsMag   # same as raw at first
        snmm.getArray(self.obsSuperStarAppliedHandle)[:] = 0.0
        if self.hasXY:
            snmm.getArray(self.obsXHandle)[:] = obsX
            snmm.getArray(self.obsYHandle)[:] = obsY

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

        #for i in xrange(self.nBands):
        #    use, = np.where(obsBand == self.bands[i])
        #    if (use.size == 0):
        #        raise ValueError("No observations in band %s!" % (self.bands[i]))
        #    snmm.getArray(self.obsBandIndexHandle)[use] = i

        # new version for multifilter support
        # First, we have the filterNames
        for filterIndex,filterName in enumerate(self.lutFilterNames):
            #try:
            #    bandIndex, = np.where(self.filterToBand[filterName] == self.bands)
            #except:
            #    self.fgcmLog.info('WARNING: observations with filter %s not in config' % (filterName))
            #    bandIndex = -1
            try:
                bandIndex = self.bands.index(self.filterToBand[filterName])
            except:
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


        #obs=None

        #startTime=time.time()
        #self.fgcmLog.info('Reading in star positions...')
        #pos=fitsio.read(self.indexFile,ext='POS')
        #self.fgcmLog.info('Done reading in %d unique star positions in %.1f secondds.' %
        #                 (pos.size,time.time()-startTime))

        #  nStars: total number of unique stars
        #self.nStars = pos.size
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

        #snmm.getArray(self.objIDHandle)[:] = pos['FGCM_ID'][:]
        #snmm.getArray(self.objRAHandle)[:] = pos['RA'][:]
        #snmm.getArray(self.objDecHandle)[:] = pos['DEC'][:]
        snmm.getArray(self.objIDHandle)[:] = objID
        snmm.getArray(self.objRAHandle)[:] = objRA
        snmm.getArray(self.objDecHandle)[:] = objDec

        #try:
            # new field name
        #    snmm.getArray(self.objObsIndexHandle)[:] = pos['OBSARRINDEX'][:]
        #except:
            # old field name
        #    snmm.getArray(self.objObsIndexHandle)[:] = pos['OBSINDEX'][:]
        #snmm.getArray(self.objNobsHandle)[:] = pos['NOBS'][:]
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

        #pos=None
        obsObjIDIndex = None
        objID = None
        obsIndex = None
        objObsIndex = None
        objNobs = None

        # and create a objFlag which flags bad stars as they fall out...

        self.objFlagHandle = snmm.createArray(self.nStars,dtype='i2')

        # and read in the previous bad stars if available
        #if (self.inBadStarFile is not None):
        #   self.fgcmLog.info('Reading in list of previous bad stars from %s' %
        #                     (self.inBadStarFile))

        #    objID = snmm.getArray(self.objIDHandle)
        #    objFlag = snmm.getArray(self.objFlagHandle)

        #    inBadStars = fitsio.read(self.inBadStarFile,ext=1)

        #    a,b=esutil.numpy_util.match(inBadStars['OBJID'],
        #                                objID)

        #    self.fgcmLog.info('Flagging %d stars as bad.' %
        #                     (a.size))

        #    objFlag[b] = inBadStars['OBJFLAG'][a]
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
                                  minObsPerBand=None):
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

        # and find the minimum of all the required bands
        minObs = objNGoodObs[:,self.bandRequiredIndex].min(axis=1)

        # reset too few obs flag if it's already set
        if not temporary:
            objFlag &= ~objFlagDict['TOO_FEW_OBS']

        # choose the bad objects with too few observations
        bad,=np.where(minObs < minObsPerBand)

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

                # and find the minimum of all the required bands
        minObs = objNGoodObs[:,self.bandRequiredIndex].min(axis=1)

        # reset too few obs flag if it's already set
        objFlag &= ~objFlagDict['TOO_FEW_OBS']

        # choose the bad objects with too few observations
        bad,=np.where(minObs < minObsPerBand)

        objFlag[bad] |= objFlagDict['TOO_FEW_OBS']
        self.fgcmLog.info('Flagging %d of %d stars with TOO_FEW_OBS' % (bad.size,self.nStars))


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
        except:
            self.fgcmLog.info("Map plotting not available.  Sorry!")
            return

        goodStars,=np.where(snmm.getArray(self.objFlagHandle)[:] == 0.0)

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

        objMagStdMeanLock = snmm.getArrayBase(self.objMagStdMeanHandle).get_lock()
        objSEDSlopeLock = snmm.getArrayBase(self.objSEDSlopeHandle).get_lock()

        # select out good ones
        # NOTE: assumes that the required bands are sequential.
        #  in fact, this whole thing does.
        ## FIXME: require required bands to be explicitly sequential

        ## NOTE: this check is probably redundant, since we already have
        #   a list of good stars in most cases.

        # protect access to copy to local
        objMagStdMeanLock.acquire()

        objMagStdMeanOI = objMagStdMean[objIndicesIn,:]

        # release access
        objMagStdMeanLock.release()

        # and make a temporary local copy of the SED
        objSEDSlopeOI = np.zeros((objIndicesIn.size,self.nBands),dtype='f4')

        maxMag = np.max(objMagStdMeanOI[:,self.bandRequiredIndex.min():
                                              self.bandRequiredIndex.max()+1],axis=1)

        goodIndicesOI,=np.where(maxMag < 90.0)


        # can this be non-looped?
        S=np.zeros((goodIndicesOI.size,self.nBands-1),dtype='f8')
        for i in xrange(self.nBands-1):
            S[:,i] = (-1/self.magConstant) * (objMagStdMeanOI[goodIndicesOI,i+1] -
                                              objMagStdMeanOI[goodIndicesOI,i]) / (
                (self.lambdaStdBand[i+1] - self.lambdaStdBand[i]))

        ## FIXME: will have to handle u band "extra"

        tempIndex=self.bandRequiredIndex[0]
        objSEDSlopeOI[goodIndicesOI, tempIndex] = (
            S[:, tempIndex] + self.sedFitBandFudgeFactors[0] * (
                S[:, tempIndex+1] + S[:, tempIndex]))

        # and the middle ones...
        #  these are straight averages
        for tempIndex in self.bandRequiredIndex[1:-1]:
            objSEDSlopeOI[goodIndicesOI,tempIndex] = (
                self.sedFitBandFudgeFactors[tempIndex] * (
                    S[:,tempIndex-1] + S[:,tempIndex]) / 2.0)

        # and the last one
        tempIndex = self.bandRequiredIndex[-1]
        objSEDSlopeOI[goodIndicesOI,tempIndex] = (
            S[:,tempIndex-1] + self.sedFitBandFudgeFactors[-1] * (
                (self.lambdaStdBand[tempIndex] - self.lambdaStdBand[tempIndex-1]) /
                (self.lambdaStdBand[tempIndex] - self.lambdaStdBand[tempIndex-2])) *
            (S[:,tempIndex-1] - S[:,tempIndex-2]))

        # and the extra bands, only redward now
        #tempIndex = self.bandRequiredIndex[-1]
        #for i in xrange(len(self.bandExtraIndex)):
        #    extraIndex=self.bandExtraIndex[i]
        #    use,=np.where(objMagStdMeanOI[goodIndicesOI,extraIndex] < 90.0)
        #    objSEDSlopeOI[goodIndicesOI[use],extraIndex] = (
        #        S[use,tempIndex-1] + self.sedExtraBandFudgeFactors[i] * (
        #            (self.lambdaStd[tempIndex] - self.lambdaStd[tempIndex-1]) /
        #            (self.lambdaStd[tempIndex] - self.lambdaStd[tempIndex-2])) *
        #        (S[use,tempIndex-1] - S[use,tempIndex-2]))
        for i in xrange(len(self.bandExtraIndex)):
            extraIndex=self.bandExtraIndex[i]
            use,=np.where(objMagStdMeanOI[goodIndicesOI,extraIndex] < 90.0)
            objSEDSlopeOI[goodIndicesOI[use],extraIndex] = (
                S[use,extraIndex-1] + self.sedExtraBandFudgeFactors[i] * (
                    (self.lambdaStdBand[extraIndex] - self.lambdaStdBand[extraIndex-1]) /
                    (self.lambdaStdBand[extraIndex] - self.lambdaStdBand[extraIndex-2])) *
                (S[use,extraIndex-1] - S[use,extraIndex-2]))

        # and save the values, protected
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



    def performColorCuts(self):
        """
        Make the color cuts that are specified in the config.
        """

        if (not self.magStdComputed):
            raise ValueError("Must compute magStd before performing color cuts")

        objMagStdMean = snmm.getArray(self.objMagStdMeanHandle)
        objFlag = snmm.getArray(self.objFlagHandle)

        for cCut in self.starColorCuts:
            thisColor = objMagStdMean[:,cCut[0]] - objMagStdMean[:,cCut[1]]
            bad,=np.where((thisColor < cCut[2]) |
                          (thisColor > cCut[3]))
            objFlag[bad] |= objFlagDict['BAD_COLOR']

            self.fgcmLog.info('Flag %d stars of %d with BAD_COLOR' % (bad.size,self.nStars))

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
            # new style

            from .fgcmUtilities import poly2dFunc

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

                obsSuperStarApplied[i1a] = poly2dFunc(np.vstack((obsX[i1a],
                                                        obsY[i1a])),
                                                       *fgcmPars.parSuperStarFlat[epInd, fiInd, cInd, :])
        else:
            # old style

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

        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)

        obsMagADU = snmm.getArray(self.obsMagADUHandle)

        # Note that EXP^gray = < <mstd>_j - mstd_ij >
        #  when we have seeing loss, that makes mstd_ij larger and EXP^gray smaller
        #  So the slope of aperCorr is negative.
        #  If we add aperCorr to each of mstd_ij, then we get a smaller (brighter)
        #  magnitude.  And this will bring mstd_ij closer to <mstd>_j

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
        resMask = 255 & ~objFlagDict['RESERVED']
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
                            (objNGoodObs[obsObjIDIndex[goodObs], bandIndex] > self.minObsPerBand))
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

            # debug bit...
            """
            plt.set_cmap('viridis')
            fig = plt.figure(1, figsize=(8,6))
            fig.clf()
            ax = fig.add_subplot(111)

            ax.hexbin(obsMagADUErr[goodObs[use]], obsMagADUModelErr[goodObs[use]], bins='log')
            ax.plot([0., 0.08], [0., 0.08], 'r--')
            ax.set_title('band = %s, %d' % (self.bands[bandIndex], use.size))
            ax.set_xlabel('Observed error')
            ax.set_ylabel('Model Error')

            fig.savefig('temp_%s.png' % (self.bands[bandIndex]))
            plt.close(fig)
            """
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
                    objFlagDict['RESERVED'])

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
