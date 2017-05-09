from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import glob
import smatch
import healpy as hp

class FgcmMakeStars(object):
    """
    """
    def __init__(self,starConfig):
        self.starConfig = starConfig

        requiredKeys=['exposureFile','fileGlobs',
                      'blacklistFile','brightStarFile',
                      'bands','requiredFlag',
                      'minPerBand','matchRadius',
                      'isolationRadius','nside',
                      'maxPerPixel','referenceBand',
                      'zpDefault','starSelectionMode',
                      'nCCD',
                      'starfileBase']

        for key in requiredKeys:
            if (key not in starConfig):
                raise ValueError("required %s not in starConfig" % (key))

        self.starConfig['observationFile'] = self.starConfig['starfileBase']+'_observations.fits'
        self.starConfig['starPrePositionFile'] = self.starConfig['starfileBase']+'_prepositions.fits'
        self.starConfig['obsIndexFile'] = self.starConfig['starfileBase']+'_obs_index.fits'

        self.nside=4096

    def run(self,clobber=False):
        """
        """

        self.makeObservationFile(clobber=clobber)
        self.makeReferenceStars(clobber=clobber)
        self.makeMatchedStars(clobber=clobber)

    def makeObservationFile(self,clobber=False):
        """
        """

        # a bunch of assumptions about the input...
        # we have a query that cuts low s/n observations
        # and edges of ccds, etc.
        # this may need to be revisited in the future

        if (not clobber):
            if (os.path.isfile(self.starConfig['observationFile'])):
                print("Found %s" % (self.starConfig['observationFile']))
                return


        # read in the exposure file
        print("Reading in exposure file...")

        ## FIXME: change to new exposure file format!

        expInfo = fitsio.read(self.starConfig['exposureFile'],ext=1)

        # read in the blacklist file (if available)
        useBlacklist = False
        if (self.starConfig['blacklistFile'] is not None):
            print("Reading blacklist file...")
            useBlacklist = True
            blacklist = fitsio.read(self.starConfig['blacklistFile'],ext=1)
            blackHash = (self.starConfig['nCCD'] + 1) * blacklist['EXPNUM'] + blacklist['CCDNUM']

        # make our file
        dtype = [('EXPNUM','i4'),
                 ('CCDNUM','i2'),
                 ('BAND','a2'),
                 ('RA','f8'),
                 ('DEC','f8'),
                 ('MAG','f4'),
                 ('MAGERR','f4')]

        inputFiles=[]
        for g in self.starConfig['fileGlobs']:
            files = glob.glob(g)
            inputFiles.extend(files)

        inputFiles.sort()

        print("Found %d files." % (len(inputFiles)))

        fits = fitsio.FITS(self.starConfig['observationFile'],mode='rw',clobber=True)
        fits.create_table_hdu(dtype=dtype)

        for f in inputFiles:
            # read in the file
            print("Reading %s" % (f))
            inObs = fitsio.read(f,ext=1,upper=True)

            # check that these have the right bands...
            mark = np.zeros(inObs.size,dtype='b1')
            for b in self.starConfig['bands']:
                mark[np.where(inObs['BAND'] == b)[0]] = True

            # check against blacklist
            if (useBlacklist):
                expHash = (self.starConfig['nCCD'] + 1) * inObs['EXPNUM'] + inObs['CCDNUM']

                # match exp/ccd pairs 
                _,badObs = esutil.numpy_util.match(blackHash,expHash)

                print("Removing %d observations due to blacklist." % (badObs.size))
                if (badObs.size > 0) :
                    mark[badObs] = False

            # and check stars in reference band...
            if (self.starConfig['starSelectionMode'] == 0):
                # DES Selection
                bad,=np.where((inObs['BAND'] == self.starConfig['referenceBand']) &
                              ((np.abs(inObs['SPREAD_MODEL']) > 0.003) |
                               (inObs['CLASS_STAR'] < 0.75)))
                ## FIXME: need imaflags_iso (oops)
                print("Removing %d observations due to DES star selection." % (bad.size))
                if (bad.size > 0):
                    mark[bad] = False

            # make sure these are in the exposure list
            expIndex,obsIndex = esutil.numpy_util.match(expInfo['EXPNUM'],inObs['EXPNUM'])

            stars,=np.where(mark[obsIndex])
            obsIndex = obsIndex[stars]
            expIndex = expIndex[stars]

            # these can all be config variables on the RHS
            tempObs = np.zeros(obsIndex.size,dtype=dtype)
            tempObs['EXPNUM'] = inObs['EXPNUM'][obsIndex].astype(np.int32)
            tempObs['CCDNUM'] = inObs['CCDNUM'][obsIndex].astype(np.int16)
            tempObs['BAND'] = inObs['BAND'][obsIndex]
            tempObs['RA'] = inObs['RA'][obsIndex]
            tempObs['DEC'] = inObs['DEC'][obsIndex]
            tempObs['MAG'] = -2.5*np.log10(inObs['FLUX_PSF'][obsIndex]) + 2.5*np.log10(expInfo['EXPTIME'][expIndex]) + self.starConfig['zpDefault']
            tempObs['MAGERR'] = (2.5/np.log(10))*inObs['FLUXERR_PSF'][obsIndex]/inObs['FLUX_PSF'][obsIndex]

            # and check ra/dec ranges
            bad,=np.where(tempObs['RA'] < 0.0)
            if (bad.size > 0):
                tempObs['RA'][bad] = tempObs['RA'][bad] + 360.0
            bad,=np.where(tempObs['RA'] > 360.0)
            if (bad.size > 0):
                tempObs['RA'][bad] = tempObs['RA'][bad] - 360.0

            fits[1].append(tempObs)

        print("Done building meta-observation table")
        fits.close()

    def makeReferenceStars(self,clobber=False):
        """
        """
        if (not clobber):
            if (os.path.isfile(self.starConfig['starPrePositionFile'])):
                print("Found %s" % (self.starConfig['starPrePositionFile']))
                return

        # currently takes a lot of memory!

        # only read in the reference band objects, we only need the positions
        # assuming we had a s/n cut on the original query

        fits = fitsio.FITS(self.starConfig['observationFile'])
        w=fits[1].where('band == "%s"' % (self.starConfig['referenceBand']))

        obsCat = fits[1].read(columns=['RA','DEC'],upper=True,rows=w)


        dtype=[('FGCM_ID','i4'),
               ('RA','f8'),
               ('DEC','f8')]

        print("Matching referenceBand catalog to itself...")
        matches=smatch.match(obsCat['RA'],obsCat['DEC'],self.starConfig['matchRadius']/3600.0,obsCat['RA'],obsCat['DEC'],nside=self.nside,maxmatch=0)

        fakeId = np.arange(obsCat.size)
        hist,rev = esutil.stat.histogram(fakeId[matches['i1']],rev=True)

        if (hist.max() == 1):
            raise ValueError("Only a single unique list of objects.")

        maxObs = hist.max()

        # how many unique objects?
        histTemp = hist.copy()
        count=0
        for j in xrange(histTemp.size):
            jj = fakeId[j]
            if (histTemp[jj] >= self.starConfig['minPerBand']):
                i1a=rev[rev[jj]:rev[jj+1]]
                histTemp[matches['i2'][i1a]] = 0
                count=count+1

        # make the object catalog
        objCat = np.zeros(count,dtype=dtype)
        objCat['FGCM_ID'] = np.arange(count)+1

        # rotate.  This works for DES, but we have to think about optimizing this...
        raTemp = obsCat['RA']

        hi,=np.where(raTemp > 180.0)
        if (hi.size > 0) :
            raTemp[hi] = raTemp[hi] - 360.0

        # get mean ra/dec
        index = 0
        for j in xrange(hist.size):
            jj = fakeId[j]
            if (hist[jj] >= self.starConfig['minPerBand']):
                i1a=rev[rev[jj]:rev[jj+1]]
                starInd=matches['i2'][i1a]
                # make sure this doesn't get used again
                hist[starInd] = 0
                objCat['RA'][index] = np.sum(raTemp[starInd])/starInd.size
                objCat['DEC'][index] = np.sum(obsCat['DEC'][starInd])/starInd.size
                index = index+1

        # restore negative RAs
        lo,=np.where(objCat['RA'] < 0.0)
        if (lo.size > 0):
            objCat['RA'][lo] = objCat['RA'][lo] + 360.0

        # here we crop out stars near very bright stars...
        if (self.starConfig['brightStarFile'] is not None):
            brightStars = fitsio.read(self.starConfig['brightStarFile'],ext=1,upper=True)

            matches=smatch.match(brightStars['RA'],brightStars['DEC'],brightStars['RADIUS'],objCat['RA'],objCat['DEC'],nside=self.nside,maxmatch=0)

            objCat = np.delete(objCat,matches['i2'])

        # and we get rid of stars with near neighbors
        matches=smatch.match(objCat['RA'],objCat['DEC'],self.starConfig['isolationRadius']/3600.0,objCat['RA'],objCat['DEC'],nside=self.nside,maxmatch=0)

        use,=np.where(matches['i1'] != matches['i2'])
        if (use.size > 0):
            neighbored = np.unique(matches['i2'][use])

            objCat = np.delete(objCat,neighbored)

        # and save the catalog...
        fitsio.write(self.starConfig['starPrePositionFile'],objCat)

    def makeMatchedStars(self,clobber=False):
        """
        """

        if (not clobber):
            if (os.path.isfile(self.starConfig['obsIndexFile'])):
                print("Found %s" % (self.starConfig['obsIndexFile']))
                return

        # we need the star list...

        print("Reading in reference positions...")
        objCat = fitsio.read(self.starConfig['starPrePositionFile'],ext=1)
        print("There are %d reference stars." % (objCat.size))

        # and the full observation list...

        print("Reading in observation file...")
        obsCat = fitsio.read(self.starConfig['observationFile'],ext=1,columns=['RA','DEC','BAND'])
        obsBand = np.core.defchararray.strip(obsCat['BAND'])

        print("Matching positions to observations...")
        m=smatch.match(objCat['RA'],objCat['DEC'],self.starConfig['matchRadius']/3600.0,obsCat['RA'],obsCat['DEC'],nside=self.nside,maxmatch=0)

        print("Collating observations...")
        nObsPerObj, obsInd = esutil.stat.histogram(m['i1'], rev=True)

        if (nObsPerObj.size != objCat.size):
            raise ValueError("Number of reference stars does not match observation file.")

        # which stars have at least minPerBand observations in each of the required bands?
        req, = np.where(np.array(self.starConfig['requiredFlag']) == 1)
        reqBands = np.array(self.starConfig['bands'])[req]

        print("Computing number of observations per band...")
        nobs = np.zeros((reqBands.size, objCat.size),dtype='i4')
        for i in xrange(reqBands.size):
            use,=np.where(obsBand[m['i2']] == reqBands[i])
            # need to make sure it's aligned
            hist = esutil.stat.histogram(m['i1'][use],min=0,max=objCat.size-1)
            nobs[i,:] = hist

        # cut the star list to those with enough per band
        minObs = nobs.min(axis=0)

        # this is our classifier...
        #   1 is a good star, 0 is bad.
        objClass = np.zeros(objCat.size,dtype='i2')

        gd,=np.where(minObs >= self.starConfig['minPerBand'])
        objClass[gd] = 1
        print("There are %d stars with at least %d observations in each required band." % (gd.size, self.starConfig['minPerBand']))

        # cut the density of stars down with sampling...only the good ones

        theta = (90.0 - objCat['DEC'][gd])*np.pi/180.
        phi = objCat['RA'][gd]*np.pi/180.

        ipring = hp.ang2pix(self.starConfig['nside'],theta,phi)
        hist,rev=esutil.stat.histogram(ipring,rev=True)

        high,=np.where(hist > self.starConfig['maxPerPixel'])
        ok,=np.where(hist > 0)
        print("There are %d/%d pixels with high stellar density" % (high.size, ok.size))
        for i in xrange(high.size):
            i1a=rev[rev[high[i]]:rev[high[i]+1]]
            cut=np.random.choice(i1a,size=i1a.size-self.starConfig['maxPerPixel'],replace=False)
            objClass[gd[cut]] = 0

        # save the star positions
        # redo the gd selection
        gd,=np.where(objClass == 1)

        print("Writing out %d potential calibration stars." % (gd.size))

        fits = fitsio.FITS(self.starConfig['obsIndexFile'],mode='rw',clobber=True)

        # note that these are 4-byte integers.
        #  to handle more than 2 billion observations will require a restructure of the code
        objCatIndex = np.zeros(gd.size,dtype=[('FGCM_ID','i4'),
                                              ('RA','f8'),
                                              ('DEC','f8'),
                                              ('OBSARRINDEX','i4'),
                                              ('NOBS','i4')])
        objCatIndex['FGCM_ID'][:] = objCat['FGCM_ID'][gd]
        objCatIndex['RA'][:] = objCat['RA'][gd]
        objCatIndex['DEC'][:] = objCat['DEC'][gd]
        # this is definitely the number of observations
        objCatIndex['NOBS'][:] = nObsPerObj[gd]
        # and the index is the cumulative sum...
        objCatIndex['OBSARRINDEX'][1:] = np.cumsum(nObsPerObj[gd])[:-1]

        # first extension is the positions
        fits.create_table_hdu(data=objCatIndex,extname='POS')
        fits[1].write(objCatIndex)

        # next extension is the obsCat indices...
        dtype=[('OBSINDEX','i4')]
        fits.create_table_hdu(dtype=dtype,extname='INDEX')

        print("Spooling out %d observation indices." % (np.sum(objCatIndex['NOBS'])))
        for i in gd:
            tempIndices = np.zeros(nObsPerObj[i],dtype=dtype)
            tempIndices['OBSINDEX'][:] = m['i2'][obsInd[obsInd[i]:obsInd[i+1]]]
            fits[2].append(tempIndices)

        fits.close()

        # note that we're not updating the total observation table, though this
        # could be done to save some speed on reading it in later on
        #  (or just grab the rows from the index...need to check on fitsio there.)

        # and we're done



