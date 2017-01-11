from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import glob
import smatch

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
        self.starConfig['starPositionFile'] = self.starConfig['starfileBase']+'_positions.fits'

        self.nside=4096

    def run(self):
        pass

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

        fits = fitsio.FITS(self.starConfig['observationFile'],mode='rw',clobber=True)

        # read in the exposure file
        expInfo = fitsio.read(self.starConfig['exposureFile'],ext=1)

        # read in the blacklist file (if available)
        useBlacklist = False
        if (self.starConfig['blacklistFile'] is not None):
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

        for f in inputFiles:
            # read in the file
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
                mark[ind2] = False

            # and check stars in reference band...
            if (self.starConfig['starSelectionMode'] == 0):
                # DES Selection
                bad,=np.where((inObs['BAND'] == self.starConfig['referenceBand']) &
                              ((np.abs(inObs['SPREAD_MODEL']) > 0.003) |
                               (inObs['CLASS_STAR'] < 0.75)))
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
        fitsio.write(objCat,self.starConfig['starPrePositionFile'])

    def makeMatchedStars(self,clobber=False):
        """
        """

        if (not clobber):
            if (os.path.isfile(self.starConfig['obsIndexFile'])):
                print("Found %s" % (self.starConfig['obsIndexFile']))
                return

        # we need the star list...

        objCat = fitsio.read(self.starConfig['starPrePositionFile'],ext=1)

        # and the full observation list...

        obsCat = fitsio.read(self.starConfig['observationFile'],ext=1,columns=['RA','DEC','BAND'])

        m=smatch.match(objCat['RA'],objCat['DEC'],self.starConfig['matchRadius']/3600.0,obsCat['RA'],obsCat['DEC'],nside=self.nside,maxmatch=0)

        objInd, obsInd = esutil.stat.histogram(m['i1'], rev=True)

        if (objInd.size != objCat.size):
            raise ValueError("Number of reference stars does not match observation file.")

        # which stars have at least minPerBand observations in each of the required bands?
        req, = np.where(self.starConfig['requiredFlag'] == 1)
        reqBands = self.starConfig['bands'][req]

        nobs = np.array((reqBands.size, objCat.size),dtype='i4')
        for i in xrange(reqBands.size):
            use,=np.where(obsCat['BAND'][m['i2']] == reqBands[i])
            # need to make sure it's aligned
            hist = esutil.stat.histogram(m['i1'][use],min=0,max=blah)
            nobs[i,:] = hist

        # cut the star list to those with enough per band

        # cut the density of stars down with sampling

        # save the star positions

        # spool out the indices -- can we do this without rematching?  Cleanly?

        # and we're done


    #def cutReferenceStars(self):
    #    pass

    

