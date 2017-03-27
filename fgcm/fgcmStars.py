from __future__ import print_function

import numpy as np
import fitsio
import esutil

from fgcmUtilities import _pickle_method

import types
import copy_reg
#import sharedmem as shm

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmStars(object):
    def __init__(self,fgcmConfig,computeNobs=True):
        self.obsFile = fgcmConfig.obsFile
        self.indexFile = fgcmConfig.indexFile

        self.bands = fgcmConfig.bands
        self.nBands = fgcmConfig.bands.size
        self.minPerBand = fgcmConfig.minObsPerBand
        self.fitBands = fgcmConfig.fitBands

        self.lambdaStd = fgcmConfig.lambdaStd

        self.bandRequired = np.zeros(self.nBands,dtype=np.bool)
        for i in xrange(self.nBands):
            if (self.bands[i] in self.fitBands):
                self.bandRequired[i] = True

        self._loadStars()

        if (computeNobs):
            allExp = np.arange(fgcmConfig.expRange[0],fgcmConfig.expRange[1],dtype='i4')
            self.selectStarsMinObs(allExp)

    def _loadStars(self):

        # read in the observational indices
        index=fitsio.read(self.indexFile,ext='INDEX')

        # sort them for reference
        indexSort = np.argsort(index['OBSINDEX'])

        # and only read these entries from the obs table
        obs=fitsio.read(self.obsFile,ext=1,rows=index['OBSINDEX'][indexSort])

        # and fill in new, cut indices
        #self.obsIndex = shm.zeros(index.size,dtype='i4')
        #self.obsIndex[:] = np.searchsorted(index['OBSINDEX'][indexSort],index['OBSINDEX'])
        self.obsIndexHandle = snmm.createArray(index.size,dtype='i4')
        snmm.getArray(self.obsIndexHandle)[:] = np.searchsorted(index['OBSINDEX'][indexSort],index['OBSINDEX'])


        # need to stuff into shared memory objects.
        self.nStarObs = obs.size

        self.obsExpHandle = snmm.createArray(self.nStarObs,dtype='i4')
        self.obsCCDHandle = snmm.createArray(self.nStarObs,dtype='i2')
        self.obsBandIndexHandle = snmm.createArray(self.nStarObs,dtype='i2')
        self.obsRAHandle = snmm.createArray(self.nStarObs,dtype='f8')
        self.obsDecHandle = snmm.createArray(self.nStarObs,dtype='f8')
        self.obsMagADUHandle = snmm.createArray(self.nStarObs,dtype='f4')
        self.obsMagADUErrHandle = snmm.createArray(self.nStarObs,dtype='f4')
        self.obsMagStdHandle = snmm.createArray(self.nStarObs,dtype='f4')

        snmm.getArray(self.obsExpHandle)[:] = obs['EXPNUM'][:]
        snmm.getArray(self.obsCCDHandle)[:] = obs['CCDNUM'][:]
        snmm.getArray(self.obsRAHandle)[:] = obs['RA'][:]
        snmm.getArray(self.obsDecHandle)[:] = obs['DEC'][:]
        snmm.getArray(self.obsMagADUHandle)[:] = obs['MAG'][:]
        snmm.getArray(self.obsMagADUErrHandle)[:] = obs['MAGERR'][:]
        snmm.getArray(self.obsMagStdHandle)[:] = obs['MAG'][:]

        # and match bands to indices
        bandStrip = np.core.defchararray.strip(obs['BAND'][:])
        for i in xrange(self.nBands):
            use,=np.where(bandStrip == self.bands[i])
            if (use.size == 0):
                raise ValueError("No observations in band %s!" % (self.bands[i]))
            snmm.getArray(self.obsBandIndexHandle)[use] = i

        obs=None

        pos=fitsio.read(self.indexFile,ext='POS')

        self.nStars = pos.size

        self.objIDHandle = snmm.createArray(self.nStars,dtype='i4')
        self.objRAHandle = snmm.createArray(self.nStars,dtype='f8')
        self.objDecHandle = snmm.createArray(self.nStars,dtype='f8')
        self.objObsIndexHandle = snmm.createArray(self.nStars,dtype='i4')
        self.objNobsHandle = snmm.createArray(self.nStars,dtype='i4')
        self.objNGoodObsHandle = snmm.createArray((self.nStars,self.nBands),dtype='i4')

        snmm.getArray(self.objIDHandle)[:] = pos['FGCM_ID'][:]
        snmm.getArray(self.objRAHandle)[:] = pos['RA'][:]
        snmm.getArray(self.objDecHandle)[:] = pos['DEC'][:]
        snmm.getArray(self.objObsIndexHandle)[:] = pos['OBSINDEX'][:]
        snmm.getArray(self.objNobsHandle)[:] = pos['NOBS'][:]


        self.minObjID = np.min(snmm.getArray(self.objIDHandle))
        self.maxObjID = np.max(snmm.getArray(self.objIDHandle))

        self.obsObjIDHandle = snmm.createArray(self.nStarObs,dtype='i4')
        obsObjID = snmm.getArray(self.obsObjIDHandle)
        objID = snmm.getArray(self.objIDHandle)
        obsIndex = snmm.getArray(self.obsIndexHandle)
        objObsIndex = snmm.getArray(self.objObsIndexHandle)
        objNobs = snmm.getArray(self.objNobsHandle)
        for i in xrange(self.nStars):
            obsObjID[obsIndex[objObsIndex[i]:objObsIndex[i]+objNobs[i]]] = objID[i]

        pos=None
        obsObjID = None
        objID = None
        obsIndex = None
        objObsIndex = None
        objNobs = None

        # and create a starFlag which flags bad stars as they fall out...
        # 0: good
        # 1: bad...
        # 2: ???

        self.starFlagHandle = snmm.createArray(self.nStars,dtype='i2')

        # And we need to record the mean mag, error, SED slopes...

        self.objMagStdMeanHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4')
        self.objMagStdMeanErrHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4')
        self.objSEDSlopeHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4')

    def selectStarsMinObs(self,goodExps):
        """
        """

        # Given a list of good exposures, which stars have at least minObs observations
        #  in each required band?

        obsExp = snmm.getArray(self.obsExpHandle)
        obsIndex = snmm.getArray(self.obsIndexHandle)
        obsBandIndex = snmm.getArray(self.obsBandIndexHandle)
        obsObjID = snmm.getArray(self.obsObjIDHandle)
        objNGoodObs = snmm.getArray(self.objNGoodObsHandle)
        objID = snmm.getArray(self.objIDHandle)

        a,b=esutil.numpy_util.match(goodExps,obsExp[obsIndex])

        req,=np.where(self.bandRequired)

        #nobs = np.zeros((req.size, self.nStars),dtype='i4')
        #for i in xrange(req.size):
        #    use,=np.where(obsBandIndex[obsIndex[b]] == req[i])
        #    hist=esutil.stat.histogram(obsObjID[obsIndex[b[use]]],
        #                               min=self.minObjID,max=self.maxObjID)
            #nobs[i,:] = hist[self.objID - self.minObjID]
        #    objNGoodObs[:,i] = hist[objID - self.minObjID]

        # Better indexed version
        objNGoodObs[:,:] = 0.0
        tempObjID = np.arange(self.minObjID,self.maxObjID+1,dtype='i8')
        tempObjIndex = np.searchsorted(objID,tempObjID)
        np.add.at(objNGoodObs,
                  (tempObjIndex[obsObjID[obsIndex[b]] - self.minObjID],
                   obsBandIndex[obsIndex[b]]),
                  1)

        minObs = objNGoodObs[:,req].min(axis=1)

        snmm.getArray(self.starFlagHandle)[:] = 0
        bad,=np.where(minObs < self.minPerBand)
        snmm.getArray(self.starFlagHandle)[bad] = 1

    def computeObjectSEDSlope(self,objIndex):
        """
        """

        thisObjMagStdMean = snmm.getArray(self.objMagStdMeanHandle)[objIndex,:]
        objSEDSlope = snmm.getArray(self.objSEDSlopeHandle)

        ## FIXME
        #   work with fit bands and fudge factors

        if (np.max(thisObjMagStdMean) > 90.0):
            # cannot compute
            objSEDSlope[objIndex,:] = 0.0
        else:
            S=np.zeros(self.nBands-1,dtype='f4')
            for i in xrange(self.nBands-1):
                S[i] = -0.921 * (thisObjMagStdMean[i+1] - thisObjMagStdMean[i])/(self.lambdaStd[i+1] - self.lambdaStd[i])

            # this is hacked for now
            objSEDSlope[objIndex,0] = S[0] - 1.0 * ((self.lambdaStd[1] - self.lambdaStd[0])/(self.lambdaStd[2]-self.lambdaStd[0])) * (S[1]-S[0])
            objSEDSlope[objIndex,1] = (S[0] + S[1])/2.0
            objSEDSlope[objIndex,2] = (S[1] + S[2])/2.0
            objSEDSlope[objIndex,3] = S[2] + 0.5 * ((self.lambdaStd[3]-self.lambdaStd[2])/(self.lambdStd[3]-self.lambdaStd[1])) * (S[2] - S[1])
            if ((thisObjMagStdMean[4]) < 90.0):
                objSEDSlope[objIndex,4] = S[2] + 1.0 * ((self.lambdaStd[3]-self.lambdaStd[2])/(self.lambdaStd[3]-self.lambdaStd[1])) * (S[2]-S[1])

        


