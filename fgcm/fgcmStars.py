from __future__ import print_function

import numpy as np
import fitsio
import esutil

from fgcmUtilities import _pickle_method
from fgcmUtilities import objFlagDict

import types
import copy_reg
#import sharedmem as shm

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmStars(object):
    """
    """

    def __init__(self,fgcmConfig,fgcmPars,computeNobs=True):
        # need fgcmPars for the exposures

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.log('INFO','Initializing stars.')

        self.obsFile = fgcmConfig.obsFile
        self.indexFile = fgcmConfig.indexFile

        self.bands = fgcmConfig.bands
        self.nBands = fgcmConfig.bands.size
        self.minPerBand = fgcmConfig.minObsPerBand
        self.fitBands = fgcmConfig.fitBands
        self.nFitBands = fgcmConfig.fitBands.size
        self.extraBands = fgcmConfig.extraBands
        self.sedFitBandFudgeFactors = fgcmConfig.sedFitBandFudgeFactors
        self.sedExtraBandFudgeFactors = fgcmConfig.sedExtraBandFudgeFactors
        self.starColorCuts = fgcmConfig.starColorCuts
        self.sigma0Phot = fgcmConfig.sigma0Phot
        self.ccdStartIndex = fgcmConfig.ccdStartIndex

        self.lambdaStd = fgcmConfig.lambdaStd

        #self.bandRequired = np.zeros(self.nBands,dtype=np.bool)
        #for i in xrange(self.nBands):
        #    if (self.bands[i] in self.fitBands):
        #        self.bandRequired[i] = True
        self.bandRequired = fgcmConfig.bandRequired
        self.bandRequiredIndex = np.where(self.bandRequired)[0]
        self.bandExtra = fgcmConfig.bandExtra
        self.bandExtraIndex = np.where(self.bandExtra)[0]

        self.expArray = fgcmPars.expArray

        self._loadStars(fgcmPars)

        self.magStdComputed = False
        self.allMagStdComputed = False
        self.sedSlopeComputed = False

        if (computeNobs):
            allExps = np.arange(fgcmConfig.expRange[0],fgcmConfig.expRange[1],dtype='i4')
            self.selectStarsMinObs(goodExps=allExps)

        self.magConstant = 2.5/np.log(10)

    def _loadStars(self,fgcmPars):

        # read in the observational indices
        index=fitsio.read(self.indexFile,ext='INDEX')

        # sort them for reference
        indexSort = np.argsort(index['OBSINDEX'])

        # and only read these entries from the obs table
        obs=fitsio.read(self.obsFile,ext=1,rows=index['OBSINDEX'][indexSort])

        self.fgcmLog.log('INFO','Loaded %d observations' % (obs.size))

        # and fill in new, cut indices
        #  obsIndex: pointer to a particular row in the obs table
        #            this is used with objObsIndex to get all the observations
        #            of an individual object
        self.obsIndexHandle = snmm.createArray(index.size,dtype='i4')
        snmm.getArray(self.obsIndexHandle)[:] = np.searchsorted(index['OBSINDEX'][indexSort],index['OBSINDEX'])


        # need to stuff into shared memory objects.
        #  nStarObs: total number of observations of all starus
        self.nStarObs = obs.size

        #  obsExp: exposure number of individual observation (pointed by obsIndex)
        self.obsExpHandle = snmm.createArray(self.nStarObs,dtype='i4')
        #  obsExpIndex: exposure index
        self.obsExpIndexHandle = snmm.createArray(self.nStarObs,dtype='i4')
        #  obsCCD: ccd number of individual observation
        self.obsCCDHandle = snmm.createArray(self.nStarObs,dtype='i2')
        #  obsBandIndex: band index of individual observation
        self.obsBandIndexHandle = snmm.createArray(self.nStarObs,dtype='i2')
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
        #  obsMagStd: corrected (to standard passband) mag of individual observation
        self.obsMagStdHandle = snmm.createArray(self.nStarObs,dtype='f4')

        snmm.getArray(self.obsExpHandle)[:] = obs['EXPNUM'][:]
        snmm.getArray(self.obsCCDHandle)[:] = obs['CCDNUM'][:]
        snmm.getArray(self.obsRAHandle)[:] = obs['RA'][:]
        snmm.getArray(self.obsDecHandle)[:] = obs['DEC'][:]
        snmm.getArray(self.obsMagADUHandle)[:] = obs['MAG'][:]
        snmm.getArray(self.obsMagADUErrHandle)[:] = obs['MAGERR'][:]
        snmm.getArray(self.obsMagStdHandle)[:] = obs['MAG'][:]

        self.fgcmLog.log('DEBUG','Applying sigma0Phot = %.4f to mag errs' %
                         (self.sigma0Phot))

        obsMagADUErr = snmm.getArray(self.obsMagADUErrHandle)
        obsMagADUErr = np.sqrt(obsMagADUErr**2. + self.sigma0Phot**2.)

        ## Question: do we need to confirm that errors are positive first?
        ##  eg. how sanitized is the input?

        a,b=esutil.numpy_util.match(self.expArray,
                                    snmm.getArray(self.obsExpHandle)[:])
        snmm.getArray(self.obsExpIndexHandle)[b] = a

        # and match bands to indices
        bandStrip = np.core.defchararray.strip(obs['BAND'][:])
        for i in xrange(self.nBands):
            use,=np.where(bandStrip == self.bands[i])
            if (use.size == 0):
                raise ValueError("No observations in band %s!" % (self.bands[i]))
            snmm.getArray(self.obsBandIndexHandle)[use] = i



        obs=None

        pos=fitsio.read(self.indexFile,ext='POS')

        #  nStars: total number of unique stars
        self.nStars = pos.size

        self.fgcmLog.log('INFO','Loaded %d unique stars.' % (self.nStars))

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

        snmm.getArray(self.objIDHandle)[:] = pos['FGCM_ID'][:]
        snmm.getArray(self.objRAHandle)[:] = pos['RA'][:]
        snmm.getArray(self.objDecHandle)[:] = pos['DEC'][:]
        snmm.getArray(self.objObsIndexHandle)[:] = pos['OBSINDEX'][:]
        snmm.getArray(self.objNobsHandle)[:] = pos['NOBS'][:]


        #  minObjID: minimum object ID
        self.minObjID = np.min(snmm.getArray(self.objIDHandle))
        #  maxObjID: maximum object ID
        self.maxObjID = np.max(snmm.getArray(self.objIDHandle))

        #  obsObjIDIndex: object ID Index of each observation
        #    (to get objID, then objID[obsObjIDIndex]

        self.obsObjIDIndexHandle = snmm.createArray(self.nStarObs,dtype='i4')
        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        objID = snmm.getArray(self.objIDHandle)
        obsIndex = snmm.getArray(self.obsIndexHandle)
        objObsIndex = snmm.getArray(self.objObsIndexHandle)
        objNobs = snmm.getArray(self.objNobsHandle)
        for i in xrange(self.nStars):
            obsObjIDIndex[obsIndex[objObsIndex[i]:objObsIndex[i]+objNobs[i]]] = i

        pos=None
        obsObjIDIndex = None
        objID = None
        obsIndex = None
        objObsIndex = None
        objNobs = None

        # and create a objFlag which flags bad stars as they fall out...

        self.objFlagHandle = snmm.createArray(self.nStars,dtype='i2')

        # And we need to record the mean mag, error, SED slopes...

        #  objMagStdMean: mean standard magnitude of each object, per band
        self.objMagStdMeanHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4')
        #  objMagStdMeanErr: error on the mean standard mag of each object, per band
        self.objMagStdMeanErrHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4')
        #  objSEDSlope: linearized approx. of SED slope of each object, per band
        self.objSEDSlopeHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4')


        # note: if this takes too long it can be moved to the star computation,
        #       but it seems pretty damn fast (which may raise the question of
        #       why it needs to be precomputed...)
        # compute secZenith for every observation

        objRARad = np.radians(snmm.getArray(self.objRAHandle))
        objDecRad = np.radians(snmm.getArray(self.objDecHandle))
        hi,=np.where(objRARad > np.pi)
        objRARad[hi] -= 2*np.pi
        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        obsIndex = snmm.getArray(self.obsIndexHandle)
        objHARad = (fgcmPars.expTelHA[obsExpIndex[obsIndex]] +
                    fgcmPars.expTelRA[obsExpIndex[obsIndex]] -
                    objRARad[obsObjIDIndex[obsIndex]])
        snmm.getArray(self.obsSecZenithHandle)[:] = 1./(np.sin(objDecRad[obsObjIDIndex[obsIndex]]) *
                                                        fgcmPars.sinLatitude +
                                                        np.cos(objDecRad[obsObjIDIndex[obsIndex]]) *
                                                        fgcmPars.cosLatitude *
                                                        np.cos(objHARad[obsObjIDIndex[obsIndex]]))



    def selectStarsMinObs(self,goodExps=None,goodExpsIndex=None):
        """
        """

        if (goodExps is None and goodExpsIndex is None):
            raise ValueError("Must supply *one* of goodExps or goodExpsIndex")
        if (goodExps is not None and goodExpsIndex is not None):
            raise ValueError("Must supply one of goodExps *or* goodExpsIndex")


        # Given a list of good exposures, which stars have at least minObs observations
        #  in each required band?

        obsExp = snmm.getArray(self.obsExpHandle)
        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsIndex = snmm.getArray(self.obsIndexHandle)
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
        obsObjIDIndex = snmm.getArray(self.obsObjIDIndexHandle)
        objNGoodObs = snmm.getArray(self.objNGoodObsHandle)
        objID = snmm.getArray(self.objIDHandle)

        if (goodExps is not None):
            a,b=esutil.numpy_util.match(goodExps,obsExp[obsIndex])
        else:
            a,b=esutil.numpy_util.match(goodExpsIndex,obsExpIndex[obsIndex])

        #req,=np.where(self.bandRequired)

        # Even better version
        objNGoodObs[:,:] = 0
        np.add.at(objNGoodObs,
                  (obsObjIDIndex[obsIndex[b]],
                   obsBandIndex[obsIndex[b]]),
                  1)

        minObs = objNGoodObs[:,self.bandRequiredIndex].min(axis=1)

        #snmm.getArray(self.objFlagHandle)[:] = 0
        #bad,=np.where(minObs < self.minPerBand)
        #snmm.getArray(self.objFlagHandle)[bad] = 1
        objFlag = snmm.getArray(self.objFlagHandle)
        bad,=np.where(minObs < self.minPerBand)
        objFlag[bad] |= objFlagDict['TOO_FEW_OBS']

        self.fgcmLog.log('INFO','Flagging %d of %d stars with TOO_FEW_OBS' % (bad.size,self.nStars))

    def computeObjectSEDSlope(self,objIndex):
        """
        """

        thisObjMagStdMean = snmm.getArray(self.objMagStdMeanHandle)[objIndex,:]
        objSEDSlope = snmm.getArray(self.objSEDSlopeHandle)
        #objSEDSlopeOld = snmm.getArray(self.objSEDSlopeOldHandle)


        # check that we have valid mags for all the required bands
        if (np.max(thisObjMagStdMean[self.bandRequired]) > 90.0):
            # cannot compute
            objSEDSlope[objIndex,:] = 0.0
        else:
            # we can compute S for everything, even if we don't use it.
            #  makes the indexing simpler

            # this is the flux "color"
            S=np.zeros(self.nBands-1,dtype='f4')
            for i in xrange(self.nBands-1):
                S[i] = (-1/self.magConstant) * (thisObjMagStdMean[i+1] - thisObjMagStdMean[i])/(self.lambdaStd[i+1] - self.lambdaStd[i])

            # first, handle the required bands.
            #  edge bands use a second derivative expansion
            #  central bands use a straight mean
            #  all have the possibility for a fudge factor

            ## FIXME: will have to handle u band "extra band"

            # handle the first required one...
            tempIndex=self.bandRequiredIndex[0]
            objSEDSlope[objIndex,tempIndex] = (
                S[tempIndex] + self.sedFitBandFudgeFactors[0] * (
                    (self.lambdaStd[tempIndex+1] - self.lambdaStd[tempIndex]) /
                    (self.lambdaStd[tempIndex+2] - self.lambdaStd[tempIndex])) *
                (S[tempIndex+1]-S[tempIndex]))

            # and the middle ones...
            #  these are straight averages
            for tempIndex in self.bandRequiredIndex[1:-1]:
                objSEDSlope[objIndex,tempIndex] = (S[tempIndex-1] + S[tempIndex]) / 2.0

            # and the last one...
            tempIndex=self.bandRequiredIndex[-1]
            objSEDSlope[objIndex,tempIndex] = (
                S[tempIndex-1] + self.sedFitBandFudgeFactors[-1] * (
                    (self.lambdaStd[tempIndex] - self.lambdaStd[tempIndex-1]) /
                    (self.lambdaStd[tempIndex] - self.lambdaStd[tempIndex-2])) *
                (S[tempIndex-1] - S[tempIndex-2]))

            # and the extra bands ... only redward now
            # we stick with the reddest band
            tempIndex = self.bandRequiredIndex[-1]
            extra,=np.where(thisObjMagStdMean[self.bandExtraIndex] < 90.0)
            for i in xrange(extra.size):
                objSEDSlope[objIndex,self.bandExtraIndex[extra[i]]] = (
                    S[tempIndex-1] + self.sedExtraBandFudgeFactors[extra[i]] * (
                        (self.lambdaStd[tempIndex] - self.lambdaStd[tempIndex-1]) /
                        (self.lambdaStd[tempIndex] - self.lambdaStd[tempIndex-2])) *
                    (S[tempIndex-1] - S[tempIndex-2]))



    def performColorCuts(self):
        """
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

            self.fgcmLog.log('INFO','Flag %d stars of %d with BAD_COLOR' % (bad.size,self.nStars))

    def applySuperStarFlat(self,fgcmPars):
        """
        """

        self.fgcmLog.log('INFO','Applying SuperStarFlat to raw magnitudes')

        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)
        obsCCDIndex = snmm.getArray(self.obsCCDHandle) - self.ccdStartIndex

        obsMagADU = snmm.getArray(self.obsMagADUHandle)

        obsMagADU += fgcmPars.expCCDSuperStar[obsExpIndex,
                                              obsCCDIndex]

    def applyApertureCorrections(self,fgcmPars):
        """
        """

        self.fgcmLog.log('INFO','Applying ApertureCorrections to raw magnitudes')

        obsExpIndex = snmm.getArray(self.obsExpIndexHandle)

        obsMagADU = snmm.getArray(self.obsMagADUHandle)

        obsMagADU += fgcmPars.expApertureCorrection[obsExpIndex]



    ## The following does not work in a multiprocessing environment.
    ##  further study is required...
    #def __del__(self):
    #    snmm.freeArray(self.obsIndexHandle)
    #    snmm.freeArray(self.obsExpHandle)
    #    snmm.freeArray(self.obsExpIndexHandle)
    #    snmm.freeArray(self.obsCCDHandle)
    #    snmm.freeArray(self.obsBandIndexHandle)
    #    snmm.freeArray(self.obsRAHandle)
    #    snmm.freeArray(self.obsDecHandle)
    #    snmm.freeArray(self.obsSecZenithHandle)
    #    snmm.freeArray(self.obsMagADUHandle)
    #    snmm.freeArray(self.obsMagADUErrHandle)
    #    snmm.freeArray(self.obsMagStdHandle)
    #    snmm.freeArray(self.objIDHandle)
    #    snmm.freeArray(self.objRAHandle)
    #    snmm.freeArray(self.objDecHandle)
    #    snmm.freeArray(self.objObsIndexHandle)
    #    snmm.freeArray(self.objNobsHandle)
    #    snmm.freeArray(self.objNGoodObsHandle)
    #    snmm.freeArray(self.obsObjIDIndexHandle)
    #    snmm.freeArray(self.objFlagHandle)
    #    snmm.freeArray(self.objMagStdMeanHandle)
    #    snmm.freeArray(self.objMagStdMeanErrHandle)
    #    snmm.freeArray(self.objSEDSlopeHandle)

