from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import os
import sys
import esutil
import glob
import healpy as hp

from .fgcmLogger import FgcmLogger


class FgcmMakeStars(object):
    """
    Class to match and build star inputs for FGCM.

    parameters
    ----------
    starConfig: dict
       Dictionary with config values

    Config variables (in starConfig)
    ----------------
    filterToBand: dict
       Dictionary with one entry per filter (string) pointing to a band (string)
       e.g. {'g':'g', 'r':'r', 'i':'i', 'i2':'i'}
    requiredBands: string list
       List of required bands
    minPerBand: int
       Minimum number of observations per required band for a star to be considered
    matchRadius: float
       Match radius in arcseconds
    isolationRadius: float
       Distance to nearest star to reject both, in arcseconds
    densNSide: int
       Healpix nside for computing density
    densMaxPerPixel: int
       Maximum number of stars in each healpix.  Will randomly sample down to this density.
    referenceBand: string
       Name of reference band
    zpDefault: float
       Zeropoint to apply to fluxes get numbers to be normal-ish.
    matchNSide: int
       Healpix nside to do smatch matching.  Should just be 4096.
    coarseNSide: int
       Healpix nside to break down into coarse pixels (save memory)
    brightStarFile: string, optional
       File with (very) bright stars (RA/DEC/RADIUS) for masking
    """

    def __init__(self,starConfig):
        self.starConfig = starConfig

        requiredKeys=['filterToBand','requiredBands',
                      'minPerBand','matchRadius',
                      'isolationRadius','densNSide',
                      'densMaxPerPixel','referenceBand',
                      'zpDefault','matchNSide','coarseNSide']

        for key in requiredKeys:
            if (key not in starConfig):
                raise ValueError("required %s not in starConfig" % (key))

        self.objCat = None

        # Note that the order doesn't matter for the making of the stars
        self.filterNames = starConfig['filterToBand'].keys()

        # check that the requiredBands are there...
        for reqBand in starConfig['requiredBands']:
            found=False
            for filterName in self.filterNames:
                if (starConfig['filterToBand'][filterName] == reqBand):
                    found = True
                    break
            if not found:
                raise ValueError("requiredBand %s not in filterToBand!" % (reqBand))

        if 'logger' in starConfig:
            self.fgcmLog = starConfig['logger']
        else:
            self.fgcmLog = FgcmLogger('dummy.log', 'INFO', printLogger=True)


    def runFromFits(self, clobber=False):
        """
        Do the star matching, loading observations from fits files.

        parameters
        ----------
        clobber: bool, default=False
           Should existing files be clobbered?
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
        Make reference stars, loading observations from fits.

        parameters
        ----------
        observationFile: string

        Observation file is a fits table with the following fields:
           'FILTERNAME': Name of the filter used
           'RA': RA
           'DEC': Dec
        In addition, the FGCM run will require:
           'MAG': raw magnitude computed from ADU
           'MAGERR': raw magnitude error computed from ADU
           expField: a field name which specifies the exposure
           ccdField: a field name which specifies the ccd
           'X': x-position on CCD (optional)
           'Y': y-position on CCD (optional)
        """

        import fitsio

        fits = fitsio.FITS(observationFile)
        #w=fits[1].where('band == "%s"' % (self.starConfig['referenceBand']))
        fitsWhere = None
        for filterName in self.filterNames:
            if (self.starConfig['filterToBand'][filterName] == self.starConfig['referenceBand']):
                clause = '(filtername == "%s")' % (filterName)
                if fitsWhere is None:
                    fitsWhere = clause
                else:
                    fitsWhere = fitsWhere + ' || ' + clause
        w=fits[1].where(fitsWhere)

        obsCat = fits[1].read(columns=['RA','DEC'],upper=True,rows=w)

        if ('brightStarFile' in self.starConfig):
            brightStarCat = fitsio.read(self.starConfig['brightStarFile'],ext=1,upper=True)

            brightStarRA = brightStarCat['RA']
            brightStarDec = brightStarCat['DEC']
            brightStarRadius = brightStarCat['RADIUS']

        else :
            brightStarRA = None
            brightStarDec = None
            brightStarRadius = None

        self.makeReferenceStars(obsCat['RA'], obsCat['DEC'], bandSelected=True,
                                brightStarRA = brightStarRA,
                                brightStarDec = brightStarDec,
                                brightStarRadius = brightStarRadius)

        fitsio.write(self.starConfig['starfileBase']+'_prepositions.fits',self.objCat,clobber=True)


    def makeMatchedStarsFromFits(self, observationFile, obsIndexFile, clobber=False):
        """
        Make matched stars, loading observations from fits.

        parameters
        ----------
        observationFile: string
        obsIndexFile: string
           File output from makeReferenceStarsFromFits
        """

        import fitsio

        if (not clobber):
            if (os.path.isfile(obsIndexFile)):
                self.fgcmLog.info("Found %s " % (obsIndexFile))
                return


        obsCat = fitsio.read(observationFile, ext=1,
                             columns=['RA','DEC','FILTERNAME'])

        filterNameArray = np.core.defchararray.strip(obsCat['FILTERNAME'])

        self.makeMatchedStars(obsCat['RA'], obsCat['DEC'], filterNameArray)

        # and save the outputs...
        fits=fitsio.FITS(obsIndexFile, mode='rw', clobber=True)
        fits.create_table_hdu(data=self.objIndexCat, extname='POS')
        fits[1].write(self.objIndexCat)

        fits.create_table_hdu(data=self.obsIndexCat, extname='INDEX')
        fits[2].write(self.obsIndexCat)



    def makeReferenceStars(self, raArray, decArray, bandSelected=False,
                           filterNameArray=None,
                           brightStarRA=None, brightStarDec=None, brightStarRadius=None):
        """
        Make reference stars, from pre-loaded arrays

        parameters
        ----------
        raArray: double array
           RA for each observation
        decArray: double array
           Dec for each observation
        bandSelected: bool, default=False
           Has the input raArray/decArray been pre-selected by band?
        filterNameArray: string array
           Array of filterNames.  Required if bandSelected==False
        brightStarRA: double array, optional
           RA for bright stars for mask
        brightStarDec: double array, optional
           Dec for bright stars for mask
        brightStarRadius: float array, optional
           Radius for bright stars for mask

        Output attributes
        -----------------
        objCat: numpy recarray
           Catalog of unique objects selected from reference band
        """

        # can we use the better smatch code?
        try:
            import smatch
            hasSmatch = True
            self.fgcmLog.info("Good news!  smatch is available.")
        except:
            hasSmatch = False
            self.fgcmLog.info("Bad news.  smatch not found.")

        if (raArray.size != decArray.size):
            raise ValueError("raArray, decArray must be same length.")

        if (not bandSelected):
            if (filterNameArray is None):
                raise ValueError("Must provide filterNameArray if bandSelected == False")
            if (filterNameArray.size != raArray.size):
                raise ValueError("filterNameArray must be same length as raArray")

            # down-select
            #use,=np.where(bandArray == self.starConfig['referenceBand'])
            #raArray = raArray[use]
            #decArray = decArray[use]

            # We select based on the aliased *band* not on the filter name
            useFlag = None
            for filterName in self.filterNames:
                if (self.starConfig['filterToBand'][filterName] == self.starConfig['referenceBand']):
                    if useFlag is None:
                        useFlag = (filterNameArray == filterName.encode('utf-8'))
                    else:
                        useFlag |= (filterNameArray == filterName.encode('utf-8'))

            raArray = raArray[useFlag]
            decArray = decArray[useFlag]

        if (brightStarRA is not None and brightStarDec is not None and
            brightStarRadius is not None):
            if (brightStarRA.size != brightStarDec.size or
                brightStarRA.size != brightStarRadius.size):
                raise ValueError("brightStarRA/Dec/Radius must have same length")
            cutBrightStars = True
        else:
            cutBrightStars = False

        self.fgcmLog.info("Matching %s observations in the referenceBand catalog to itself" %
                          (raArray.size))

        dtype=[('FGCM_ID','i4'),
               ('RA','f8'),
               ('DEC','f8')]

        objCats = []

        # need to split into parts here

        ipring=hp.ang2pix(self.starConfig['coarseNSide'],
                          (90.0-decArray)*np.pi/180.,
                          raArray*np.pi/180.)
        hpix,revpix=esutil.stat.histogram(ipring,rev=True)

        gdpix,=np.where(hpix >= 2)
        for ii,gpix in enumerate(gdpix):
            p1a=revpix[revpix[gpix]:revpix[gpix+1]]

            if (hasSmatch):
                # faster smatch...
                matches = smatch.match(raArray[p1a], decArray[p1a],
                                       self.starConfig['matchRadius']/3600.0,
                                       raArray[p1a], decArray[p1a],
                                       nside=self.starConfig['matchNSide'], maxmatch=0)

                i1 = matches['i1']
                i2 = matches['i2']
            else:
                # slower htm matching...
                htm = esutil.htm.HTM(11)

                matcher = esutil.htm.Matcher(11, raArray[p1a], decArray[p1a])
                matches = matcher.match(raArray[p1a], decArray[p1a],
                                        self.starConfig['matchRadius']/3600.0,
                                        maxmatch=0)

                i1 = matches[1]
                i2 = matches[0]


            fakeId = np.arange(p1a.size)
            hist,rev = esutil.stat.histogram(fakeId[i1],rev=True)

            if (hist.max() == 1):
                self.fgcmLog.info("Warning: No matches found!")
                continue

            maxObs = hist.max()

            # how many unique objects do we have?
            histTemp = hist.copy()
            count=0
            for j in xrange(histTemp.size):
                jj = fakeId[j]
                if (histTemp[jj] >= self.starConfig['minPerBand']):
                    i1a=rev[rev[jj]:rev[jj+1]]
                    histTemp[i2[i1a]] = 0
                    count=count+1

            self.fgcmLog.info("Found %d unique objects in pixel %d (%d of %d)." %
                              (count, ipring[p1a[0]], ii, gdpix.size))

            # make the object catalog

            #self.objCat = np.zeros(count,dtype=dtype)
            #self.objCat['FGCM_ID'] = np.arange(count)+1
            objCatTemp = np.zeros(count,dtype=dtype)

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
                    #self.objCat['RA'][index] = np.sum(raTemp[starInd])/starInd.size
                    #self.objCat['DEC'][index] = np.sum(decArray[starInd])/starInd.size
                    objCatTemp['RA'][index] = np.sum(raTemp[p1a[starInd]])/starInd.size
                    objCatTemp['DEC'][index] = np.sum(decArray[p1a[starInd]])/starInd.size
                    index = index+1

            # restore negative RAs
            #lo,=np.where(self.objCat['RA'] < 0.0)
            #if (lo.size > 0):
            #    self.objCat['RA'][lo] = self.objCat['RA'][lo] + 360.0
            lo,=np.where(objCatTemp['RA'] < 0.0)
            if lo.size > 0:
                objCatTemp['RA'][lo] = objCatTemp['RA'][lo] + 360.0

            # and append...
            objCats.append(objCatTemp.copy())

            # clear memory?
            objCatTemp = None
            matches = None
            fakeId = None
            hist = None
            rev = None
            histTemp = None
            raTemp = None
            #resourceUsage('pixel %d' % (ipring[p1a[0]]))

        # now assemble into a total objCat
        count = 0
        for objCatTemp in objCats:
            count += objCatTemp.size

        self.objCat = np.zeros(count, dtype=dtype)
        ctr = 0
        for objCatTemp in objCats:
            self.objCat[ctr:ctr+objCatTemp.size] = objCatTemp
            ctr += objCatTemp.size
            # and clear memory
            objCatTemp = None

        self.objCat['FGCM_ID'] = np.arange(count)+1

        self.fgcmLog.info("Found %d unique objects with >= %d observations in %s band." %
                          (count, self.starConfig['minPerBand'], self.starConfig['referenceBand']))


        if (cutBrightStars):
            self.fgcmLog.info("Matching to bright stars for masking...")
            if (hasSmatch):
                # faster smatch...

                matches = smatch.match(brightStarRA, brightStarDec, brightStarRadius,
                                       self.objCat['RA'], self.objCat['DEC'], nside=self.starConfig['matchNSide'],
                                       maxmatch=0)
                i1=matches['i1']
                i2=matches['i2']
            else:
                # slower htm matching...
                htm = esutil.htm.HTM(11)

                matcher = esutil.htm.Matcher(10, brightStarRA, brightStarDec)
                matches = matcher.match(self.objCat['RA'], self.objCat['DEC'], brightStarRadius,
                                        maxmatch=0)
                # matches[0] -> m1 -> array from matcher.match() call (self.objCat)
                # matches[1] -> m2 -> array from htm.Matcher() (brightStar)
                i1=matches[1]
                i2=matches[0]

            self.fgcmLog.info("Cutting %d objects too near bright stars." % (i2.size))
            self.objCat = np.delete(self.objCat,i2)

        # and remove stars with near neighbors
        self.fgcmLog.info("Matching stars to neighbors...")
        if (hasSmatch):
            # faster smatch...

            matches=smatch.match(self.objCat['RA'], self.objCat['DEC'],
                                 self.starConfig['isolationRadius']/3600.0,
                                 self.objCat['RA'], self.objCat['DEC'],
                                 nside=self.starConfig['matchNSide'], maxmatch=0)
            i1=matches['i1']
            i2=matches['i2']
        else:
            # slower htm matching...
            htm = esutil.htm.HTM(11)

            matcher = esutil.htm.Matcher(11, self.objCat['RA'], self.objCat['DEC'])
            matches = matcher.match(self.objCat['RA'], self.objCat['DEC'],
                                    self.starConfig['isolationRadius']/3600.0,
                                    maxmatch = 0)
            i1=matches[1]
            i2=matches[0]

        use,=np.where(i1 != i2)

        if (use.size > 0):
            neighbored = np.unique(i2[use])
            self.fgcmLog.info("Cutting %d objects within %.2f arcsec of a neighbor" %
                  (neighbored.size, self.starConfig['isolationRadius']))
            self.objCat = np.delete(self.objCat, neighbored)

        # and we're done

    def makeMatchedStars(self, raArray, decArray, filterNameArray):
        """
        Make matched stars, from pre-loaded arrays.  Requires self.objCat was
         generated from makeReferenceStars().

        parameters
        ----------
        raArray: double array
           RA for each observation
        decArray: double array
           Dec for each observation
        filterNameArray: numpy string array
           filterName for each array
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
            raArray.size != filterNameArray.size):
            raise ValueError("raArray, decArray, filterNameArray must be same length")

        # translate filterNameArray to bandArray ... can this be made faster, or
        #  does it need to be?
        bandArray = np.zeros_like(filterNameArray)
        for filterName in self.filterNames:
            use,=np.where(filterNameArray == filterName.encode('utf-8'))
            bandArray[use] = self.starConfig['filterToBand'][filterName]


        self.fgcmLog.info("Matching positions to observations...")

        if (hasSmatch):
            # faster smatch...

            matches=smatch.match(self.objCat['RA'], self.objCat['DEC'],
                                 self.starConfig['matchRadius']/3600.0,
                                 raArray, decArray,
                                 nside=self.starConfig['matchNSide'],
                                 maxmatch=0)
            i1=matches['i1']
            i2=matches['i2']
        else:
            # slower htm matching...
            htm = esutil.htm.HTM(11)

            matcher = esutil.htm.Matcher(11, self.objCat['RA'], self.objCat['DEC'])
            matches = matcher.match(raArray, decArray,
                                    self.starConfig['matchRadius']/3600.,
                                    maxmatch=0)
            # matches[0] -> m1 -> array from matcher.match() call (ra/decArray)
            # matches[1] -> m2 -> array from htm.Matcher() (self.objCat)
            i2 = matches[0]
            i1 = matches[1]

        self.fgcmLog.info("Collating observations")
        nObsPerObj, obsInd = esutil.stat.histogram(i1, rev=True)

        if (nObsPerObj.size != self.objCat.size):
            raise ValueError("Number of reference stars (%d) does not match observations (%d)." %
                             (self.objCat.size, nObsPerObj.size))

        # which stars have at least minPerBand observations in each required band?
        reqBands = np.array(self.starConfig['requiredBands'], dtype=bandArray.dtype)

        # this could be made more efficient
        self.fgcmLog.info("Computing number of observations per band")
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
        self.fgcmLog.info("There are %d stars with at least %d observations in each required band." %
              (gd.size, self.starConfig['minPerBand']))


        # cut the density of stars down with sampling.

        theta = (90.0 - self.objCat['DEC'][gd])*np.pi/180.
        phi = self.objCat['RA'][gd]*np.pi/180.

        ipring = hp.ang2pix(self.starConfig['densNSide'], theta, phi)
        hist, rev = esutil.stat.histogram(ipring, rev=True)

        high,=np.where(hist > self.starConfig['densMaxPerPixel'])
        ok,=np.where(hist > 0)
        self.fgcmLog.info("There are %d/%d pixels with high stellar density" % (high.size, ok.size))
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
        self.objIndexCat['NOBS'][:] = nObsPerObj[gd]
        # and the index is given by the cumulative sum
        self.objIndexCat['OBSARRINDEX'][1:] = np.cumsum(nObsPerObj[gd])[:-1]

        # and we need to create the observation indices from the OBSARRINDEX

        nTotObs = self.objIndexCat['OBSARRINDEX'][-1] + self.objIndexCat['NOBS'][-1]

        self.obsIndexCat = np.zeros(nTotObs,
                                    dtype=[('OBSINDEX','i4')])
        ctr = 0
        self.fgcmLog.info("Spooling out %d observation indices." % (nTotObs))
        for i in gd:
            self.obsIndexCat[ctr:ctr+nObsPerObj[i]] = i2[obsInd[obsInd[i]:obsInd[i+1]]]
            ctr+=nObsPerObj[i]

        # and we're done




