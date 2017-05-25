from __future__ import print_function

import numpy as np
import os
import sys
import esutil
import glob
import healpy as hp

class FgcmMakeStars(object):
    """
    """
    def __init__(self,starConfig):
        self.starConfig = starConfig

        requiredKeys=['bands','requiredFlag',
                      'minPerBand','matchRadius',
                      'isolationRadius','densNSide',
                      'densMaxPerPixel','referenceBand',
                      'zpDefault','matchNSide']

        for key in requiredKeys:
            if (key not in starConfig):
                raise ValueError("required %s not in starConfig" % (key))

        self.objCat = None

    def runFromFits(self, clobber=False):
        """
        """

        if 'starfileBase' not in self.starConfig:
            raise ValueError("Required starfileBase not in starConfig")

        observationFile = self.starConfig['starfileBase']+'_observations.fits'

        if (not os.path.isfile(observationFile)):
            raise IOError("Could not find observationFile %s" % (observationFile))

        obsIndexFile = self.starConfig['starfileBase']+'_obs_index.fits'

        self.makeReferenceStarsFromFits(observationFile)
        self.makeMatchedStarsFromFits(observationFile, obsIndexFile, clobber=clobber)

    def makeReferenceStarsFromFits(self, observationFile):
        """
        """

        import fitsio

        fits = fitsio.FITS(observationFile)
        w=fits[1].where('band == "%s"' % (self.starConfig['referenceBand']))

        obsCat = fits[1].read(columns=['RA','DEC'],upper=True,rows=w)

        if ('brightStarFile' in self.starConfig):
            brightStarCat = fitsio.read(self.starConfig['brightStarFile'],ext=1,upper=True)

            brightStarRA = brightStarCat['RA']
            brightStarDec = brightStarCat['DDEC']
            brightStarRaidus = brightStarCat['RADIUS']

        else :
            brightStarRA = None
            brightStarDec = None
            brightStarRadius = None

        self.makeReferenceStars(obsCat['RA'], obsCat['DEC'], bandSelected=True,
                                brightStarRA = brightStarRA,
                                brightStarDec = brightStarDec,
                                brightStarRadius = brightStarRadius)


    def makeMatchedStarsFromFits(self, observationFile, obsIndexFile, clobber=False):
        """
        """

        import fitsio

        if (not clobber):
            if (os.path.isfile(obsIndexFile)):
                print("Found %s " % (obsIndexFile))
                return


        obsCat = fitsio.read(observationFile, ext=1,
                             columns=['RA','DEC','BAND'])

        bandArray = np.core.defchararray.strip(obsCat['BAND'])

        self.makeMatchedStars(obsCat['RA'], obsCat['DEC'], bandArray)

        # and save the outputs...
        fits=fitsio.FITS(obsIndexFile, mode='w', clobber=True)
        fits.create_table_hdu(data=self.objIndexCat, extname='POS')
        fits[1].write(self.objIndexCat)

        fits.create_table_hdu(data=self.obsIndexCat, extname='INDEX')
        fits[2].write(self.obsIndexCat)



    def makeReferenceStars(self, raArray, decArray, bandArray=None, bandSelected=False,
                           brightStarRA=None, brightStarDec=None, brightStarRadius=None):
        """
        """

        # can we use the better smatch code?
        try:
            import smatch
            hasSmatch = True
            print("Good news!  smatch is available.")
        except:
            hasSmatch = False
            print("Bad news.  smatch not found.")

        if (raArray.size != decArray.size):
            raise ValueError("raArray, decArray must be same length.")

        if (not bandSelected):
            if (bandArray is None):
                raise ValueError("Must provide bandArray if bandSelected == True")
            if (bandArray.size != raArray.size):
                raise ValueError("bandArray must be same lenght as raArray")

            # down-select
            use,=np.where(bandArray == self.starConfig['referenceBand'])
            raArray = raArray[use]
            decArray = decArray[use]

        if (brightStarRA is not None and brightStarDec is not None and
            brightStarRadius is not None):
            if (brightStarRA.size != brightStarDec.size or
                brightStarRA.size != brightStarRadius.size):
                raise ValueError("brightStarRA/Dec/Radius must have same length")
            cutBrightStars = True
        else:
            cutBrightStars = False

        print("Matching %s observations in the referenceBand catalog to itself" %
              (raArray.size))

        if (hasSmatch):
            # faster smatch...
            matches = smatch.match(raArray, decArray, self.starConfig['matchRadius']/3600.0,
                                   raArray, decArray, nside=self.starConfig['matchNSide'], maxmatch=0)

            i1 = matches['i1']
            i2 = matches['i2']
        else:
            # slower htm matching...
            htm = esutil.htm.HTM(11)

            matcher = esutil.htm.Matcher(11, raArray, decArray)
            matches = matcher.match(raArray, decArray,
                                    self.starConfig['matchRadius']/3600.0,
                                    maxmatch=0)

            i1 = matches[0]
            i2 = matches[1]


        fakeId = np.arange(raArray.size)
        hist,rev = esutil.stat.histogram(fakeId[i1],rev=True)

        if (hist.max() == 1):
            raise ValueError("No matches found!")

        maxObs = hist.max()

        # how many unique objects do we have?
        histTemp = hist.copy()
        count=0
        for j in xrange(histTemp.size):
            jj = fakeId[j]
            if (histTemp[jj] >= self.starConfig['minPerBand']):
                i1a=rev[rev[jj]:rev[jj+1]]
                histTemp[matches['i2'][i1a]] = 0
                count=count+1

        print("Found %d unique objects with >= %d observations in %s band." %
              (count, self.starConfig['minPerBand'], self.starConfig['referenceBand']))

        # make the object catalog
        dtype=[('FGCM_ID','i4'),
               ('RA','f8'),
               ('DEC','f8')]

        self.objCat = np.zeros(count,dtype=dtype)
        self.objCat['FGCM_ID'] = np.arange(count)+1

        # rotate.  This works for DES, but we have to think about optimizing this...
        raTemp = raArray.copy()

        hi,=np.where(raTemp > 180.0)
        if (hi.size > 0) :
            raTemp[hi] = raTemp[hi] - 360.0

        # compute mean ra/dec
        index = 0
        for j in xrange(hist.size):
            jj = fakeId[j]
            if (hist[jj] >= self.starConfig['minPerBand']):
                i1a=rev[rev[jj]:rev[jj+1]]
                starInd=i2[i1a]
                # make sure this doesn't get used again
                hist[starInd] = 0
                self.objCat['RA'][index] = np.sum(raTemp[starInd])/starInd.size
                self.objCat['DEC'][index] = np.sum(decArray[starInd])/starInd.size
                index = index+1

        # restore negative RAs
        lo,=np.where(self.objCat['RA'] < 0.0)
        if (lo.size > 0):
            self.objCat['RA'][lo] = self.objCat['RA'][lo] + 360.0

        if (cutBrightStars):
            if (hasSmatch):
                # faster smatch...

                matches = smatch.match(brightStarsRA, brightStarsDec, brightStarsRadius,
                                       self.objCat['RA'], self.objCat['DEC'], nside=self.starConfig['matchNSide'],
                                       maxmatch=0)
                i1=matches['i1']
                i2=matches['i2']
            else:
                # slower htm matching...
                htm = esutil.htm.HTM(11)

                matcher = esutil.htm.Matcher(10, brightStarsRA, brightStarsDec)
                matches = matcher.match(raArray, decArray, brightStarsRadius,
                                        maxmatch=0)
                i1=matches[0]
                i2=matches[1]

            self.objCat = np.delete(self.objCat,i2)

        # and remove stars with near neighbors
        if (hasSmatch):
            # faster smatch...

            matches=smatch.match(self.objCat['RA'], self.objCat['DEC'], self.starConfig['isolationRadius']/3600.0,
                                 self.objCat['RA'], self.objCat['DEC'], nside=self.starConfig['matchNSide'], maxmatch=0)
            i1=matches['i1']
            i2=matches['i2']
        else:
            # slower htm matching...
            htm = esutil.htm.HTM(11)

            matcher = esutil.htm.Matcher(self.objCat['RA'], self.objCat['DEC'])
            matches = matcher.match(self.objCat['RA'], self.objCat['DEC'],
                                    self.starConfig['isolationRadius']/3600.0,
                                    maxmatch = 0)
            i1=matches[0]
            i2=matches[1]

        use,=np.where(i1 != i2)
        if (use.size > 0):
            neighbored = np.unique(i2[use])
            self.objCat = np.delete(self.objCat, neighbored)

        # and we're done

    def makeMatchedStars(self, raArray, decArray, bandArray):
        """
        """

        if (self.objCat is None):
            raise ValueError("Must run makeReferenceStars first")

        # can we use the better smatch code?
        try:
            import smatch
            hasSmatch = True
        except:
            hasSmatch = False

        if (raArray.size != decArray.size or
            raArray.size != bandArray.size):
            raise ValueError("raArray, decArray, bandArray must be same length")

        print("Matching positions to observations...")

        if (hasSmatch):
            # faster smatch...

            matches=smatch.match(self.objCat['RA'], self.objCat['DEC'],
                                 self.starConfig['matchRadius']/3600.0,
                                 raArray, decArray, nside=self.starConfig['matchNSide'], maxmatch=0)
            i1=matches['i1']
            i2=matches['i2']
        else:
            # slower htm matching...
            htm = esutil.htm.HTM(11)

            matcher = esutil.htm.Matcher(11, self.objCat['RA'], self.objCat['DEC'])
            matches = matcher.match(raArray, decArray,
                                    self.starConfig['matchRadius']/3600.,
                                    maxmatch=0)
            i1 = matches[0]
            i2 = matches[1]

        print("Collating observations")
        nObsPerObj, obsInd = esutil.stat.histogram(i1, rev=True)

        if (nObsPerObj.size != self.objCat.size):
            raise ValueError("Number of reference stars does not match observations.")

        # which stars have at least minPerBand observations in each required band?
        req, = np.where(np.array(self.starConfig['requiredFlag']) == 1)
        reqBands = np.array(self.starConfig['bands'])[req]

        # this could be made more efficient
        print("Computing number of observations per band")
        nObs = np.zeros((reqBands.size, self.objCat.size), dtype='i4')
        for i in xrange(reqBands.size):
            use,=np.where(bandArray[i2] == reqBands[i])
            hist = esutil.stat.histogram(i1[use], min=0, max=self.objCat.size-1)
            nObs[i,:] = hist

        # cut the star list to those with enough per band
        minObs = nObs.min(axis=0)

        # and our simple classifier
        #    1 is a good star, 0 is bad.
        objClass = np.zeros(self.objCat.size, dtype='i2')

        # make sure we have enough per band
        gd,=np.where(minObs >= self.starConfig['minPerBand'])
        objClass[gd] = 1
        print("There are %d stars with at least %d observations in each required band." %
              (gd.size, self.starConfig['minPerBand']))


        # cut the density of stars down with sampling.

        theta = (90.0 - self.objCat['DEC'][gd])*np.pi/180.
        phi = self.objCat['RA'][gd]*np.pi/180.

        ipring = hp.ang2pix(self.starConfig['densNSide'], theta, phi)
        hist, rev = esutil.stat.histogram(ipring, rev=True)

        high,=np.where(hist > self.starConfig['densMaxPerPixel'])
        ok,=np.where(hist > 0)
        print("There are %d/%d pixels with high stellar density" % (high.size, ok.size))
        for i in xrange(high.size):
            i1a=rev[rev[high[i]]:rev[high[i]+1]]
            cut=np.random.choice(i1a,size=i1a.size-self.starConfig['densMaxPerPixel'],replace=False)
            objClass[gd[cut]] = 0

        # redo the good object selection after sampling
        gd,=np.where(objClass == 1)

        # create the object catalog index
        self.objIndexCat = np.zeros(gd.size, dtype=[('FGCM_ID','i4'),
                                                    ('RA','f8'),
                                                    ('DEC','f8'),
                                                    ('OBSARRINDEX','i4'),
                                                    ('NOBS','i4')])
        self.objIndexCat['FGCM_ID'][:] = self.objCat['FGCM_ID'][gd]
        self.objIndexCat['RA'][:] = self.objCat['RA'][gd]
        self.objIndexCat['DEC'][:] = self.objCat['DEC'][gd]
        # this is the number of observations per object
        self.objIndexCat['NOBS'][:] = self.nObsPerObj[gd]
        # and the index is given by the cumulative sum
        self.objIndexCat['OBSARRINDEX'][1:] = np.cumsum(nObsPerObj[gd])[:-1]

        # and we need to create the observation indices from the OBSARRINDEX

        nTotObs = self.objIndexCat['OBSARRINDEX'][-1]

        self.obsIndexCat = np.zeros(nTotObs,
                                    dtype=[('OBSINDEX','i4')])
        ctr = 0
        print("Spooling out %d observation indices." % (nTotObs))
        for i in gd:
            self.obsIndexCat[ctr:ctr+nObsPerObj[i]] = i2[obsInd[obsInd[i]:obsInd[i+1]]]

        # and we're done

    def makeReferenceStars0(self,clobber=False):
        """
        """
        import fitsio

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

    def makeMatchedStars0(self,clobber=False):
        """
        """

        import fitsio

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



