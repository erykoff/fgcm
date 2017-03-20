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
    def __init__(self,fgcmConfig):
        self.obsFile = fgcmConfig.obsFile
        self.indexFile = fgcmConfig.indexFile

        self.bands = fgcmConfig.bands
        self.nBands = fgcmConfig.bands.size
        self.minPerBand = fgcmConfig.minObsPerBand
        self.fitBands = fgcmConfig.fitBands

        self.bandRequired = np.zeros(self.nBands,dtype=np.bool)
        for i in xrange(self.nBands):
            if (self.bands[i] in self.fitBands):
                self.bandRequired[i] = True

        self._loadStars()

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

        #self.obsExp = shm.zeros(self.nStarObs,dtype='i4')
        #self.obsCCD = shm.zeros(self.nStarObs,dtype='i2')
        #self.obsBandIndex = shm.zeros(self.nStarObs,dtype='i2')
        #self.obsRA = shm.zeros(self.nStarObs,dtype='f8')
        #self.obsDec = shm.zeros(self.nStarObs,dtype='f8')
        #self.obsMagObs = shm.zeros(self.nStarObs,dtype='f4')
        #self.obsMagObsErr = shm.zeros(self.nStarObs,dtype='f4')
        #self.obsMagStd = shm.zeros(self.nStarObs,dtype='f4')

        self.obsExpHandle = snmm.createArray(self.nStarObs,dtype='i4')
        self.obsCCDHandle = snmm.createArray(self.nStarObs,dtype='i2')
        self.obsBandIndexHandle = snmm.createArray(self.nStarObs,dtype='i2')
        self.obsRAHandle = snmm.createArray(self.nStarObs,dtype='f8')
        self.obsDecHandle = snmm.createArray(self.nStarObs,dtype='f8')
        self.obsMagObsHandle = snmm.createArray(self.nStarObs,dtype='f4')
        self.obsMagObsErrHandle = snmm.createArray(self.nStarObs,dtype='f4')
        self.obsMagStdHandle = snmm.createArray(self.nStarObs,dtype='f4')

        #self.obsExp[:] = obs['EXPNUM'][:]
        #self.obsCCD[:] = obs['CCDNUM'][:]
        #self.obsRA[:] = obs['RA'][:]
        #self.obsDec[:] = obs['DEC'][:]
        #self.obsMagObs[:] = obs['MAG'][:]
        #self.obsMagObsErr[:] = obs['MAGERR'][:]
        #self.obsMagStd[:] = obs['MAG'][:]  # default

        snmm.getArray(self.obsExpHandle)[:] = obs['EXPNUM'][:]
        snmm.getArray(self.obsCCDHandle)[:] = obs['CCDNUM'][:]
        snmm.getArray(self.obsRAHandle)[:] = obs['RA'][:]
        snmm.getArray(self.obsDecHandle)[:] = obs['DEC'][:]
        snmm.getArray(self.obsMagObsHandle)[:] = obs['MAG'][:]
        snmm.getArray(self.obsMagObsErrHandle)[:] = obs['MAGERR'][:]
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

        #self.objID = shm.zeros(self.nStars,dtype='i4')
        #self.objRA = shm.zeros(self.nStars,dtype='f8')
        #self.objDec = shm.zeros(self.nStars,dtype='f8')
        #self.objObsIndex = shm.zeros(self.nStars,dtype='i4')
        #self.objNobs = shm.zeros(self.nStars,dtype='i4')

        self.objIDHandle = snmm.createArray(self.nStars,dtype='i4')
        self.objRAHandle = snmm.createArray(self.nStars,dtype='f8')
        self.objDecHandle = snmm.createArray(self.nStars,dtype='f8')
        self.objObsIndexHandle = snmm.createArray(self.nStars,dtype='i4')
        self.objNobsHandle = snmm.createArray(self.nStars,dtype='i4')

        #self.objID[:] = pos['FGCM_ID'][:]
        #self.objRA[:] = pos['RA'][:]
        #self.objDec[:] = pos['DEC'][:]
        #self.objObsIndex[:] = pos['OBSINDEX'][:]
        #self.objNobs[:] = pos['NOBS'][:]

        snmm.getArray(self.objIDHandle)[:] = pos['FGCM_ID'][:]
        snmm.getArray(self.objRAHandle)[:] = pos['RA'][:]
        snmm.getArray(self.objDecHandle)[:] = pos['DEC'][:]
        snmm.getArray(self.objObsIndexHandle)[:] = pos['OBSINDEX'][:]
        snmm.getArray(self.objNobsHandle)[:] = pos['NOBS'][:]


        self.minObjID = np.min(self.objID)
        self.maxObjID = np.max(self.objID)

        # and create reverse look-up ids
        #self.obsObjID = shm.zeros(self.nStarObs,dtype='i4')
        #for i in xrange(self.nStars):
        #    self.obsObjID[self.obsIndex[self.objObsIndex[i]:self.objObsIndex[i]+self.objNobs[i]]] = self.objID[i]

        self.obsObjIDHandle = snmm.createArray(self.nStarObs,dtype='i4')
        obsObjID = snmm.getArray(self.obsObjIDHandle)
        objID = snmm.getArray(self.objIDHandle)
        obsIndex = snmm.getArray(self.obsIndexHandle)
        objObsIndex = snmm.getArray(self.objObsIndexHandle)
        objNobs = snmm.getArray(self.objNobs)
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

        #self.starFlag = shm.zeros(self.nStars,dtype=np.int16)
        self.starFlagHandle = snmm.createArray(self.nStars,dtype='i2')

        # And we need to record the mean mag, error, SED slopes...

        #self.objMagStdMean = shm.zeros((self.nStars,self.nBands),dtype='f4')
        #self.objMagStdMeanErr = shm.zeros((self.nStars,self.nBands),dtype='f4')
        #self.objSEDSlope = shm.zeros((self.nStars,self.nBands),dtype='f4')

        self.objMagStdMeanHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4')
        self.objMagStdMeanErrHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4')
        self.objSEDSlopeHandle = snmm.createArray((self.nStars,self.nBands),dtype='f4')

    def selectStars(self,goodExps):
        """
        """

        # want to select out stars that are seen on the good exposures
        # and we need to know the number of observations in each...

        # some testing...
        test=fitsio.read('/nfs/slac/g/ki/ki19/des/erykoff/des/y3a1/calibration_testing/fgcm_refactor_tests/pars1/y3a1_exposure_info.fits',ext=1)
        goodExps = test['EXPNUM']

        obsExp = snmm.getArray(self.obsExpHandle)
        obsIndex = snmm.getArray(self.obsIndexHandle)
        obsObjID = snmm.getArray(self.obsObjIDHandle)

        a,b=esutil.numpy_util.match(goodExps,obsExp[obsIndex])

        req,=np.where(self.bandRequired)

        nobs = np.zeros((req.size, self.nStars),dtype='i4')
        for i in xrange(req.size):
            use,=np.where(self.obsBandIndex[obsIndex[b]] == req[i])
            hist=esutil.stat.histogram(obsObjID[obsIndex[b[use]]],
                                       min=self.minObjID,max=self.maxObjID)
            nobs[i,:] = hist[self.objID - self.minObjID]

        minObs = nobs.min(axis=0)

        snmm.getArray(self.starFlagHandle)[:] = 0
        #self.starFlag[:] = 0
        bad,=np.where(minObs < self.minPerBand)
        #self.starFlag[bad] = 1
        snmm.getArray(self.starFlagHandle)[bad] = 1

        # and further selection?

