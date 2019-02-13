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
    referenceBands: string list
       List of reference bands, in wavelength order
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
    primaryBands: string
       List of primary bands
    matchNSide: int
       Healpix nside to do smatch matching.  Should just be 4096.
    coarseNSide: int
       Healpix nside to break down into coarse pixels (save memory)
    brightStarFile: string, optional
       File with (very) bright stars (ra/dec/radius) for masking
    """

    def __init__(self,starConfig):
        self.starConfig = starConfig

        requiredKeys=['filterToBand','requiredBands','referenceBands',
                      'minPerBand','matchRadius',
                      'isolationRadius','densNSide',
                      'densMaxPerPixel','primaryBands',
                      'matchNSide','coarseNSide']

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

        for referenceBand in starConfig['referenceBands']:
            found = False
            for filterName in self.filterNames:
                if (starConfig['filterToBand'][filterName] == referenceBand):
                    found = True
                    break
            if not found:
                raise ValueError("band %s not in filterToBand!" % (referenceBand))

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

        self.makePrimaryStarsFromFits(observationFile)
        self.makeMatchedStarsFromFits(observationFile, obsIndexFile, clobber=clobber)

    def makePrimaryStarsFromFits(self, observationFile):
        """
        Make primary stars, loading observations from fits.

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
        fitsWhere = None
        for filterName in self.filterNames:
            for primaryBand in self.starConfig['primaryBands']:
                if (self.starConfig['filterToBand'][filterName] == primaryBand):
                    clause = '(filtername == "%s")' % (filterName)
                    if fitsWhere is None:
                        fitsWhere = clause
                    else:
                        fitsWhere = fitsWhere + ' || ' + clause
        w=fits[1].where(fitsWhere)

        obsCat = fits[1].read(columns=['ra','dec','filtername'],lower=True,rows=w)

        if ('brightStarFile' in self.starConfig):
            brightStarCat = fitsio.read(self.starConfig['brightStarFile'],ext=1,lower=True)

            brightStarRA = brightStarCat['ra']
            brightStarDec = brightStarCat['dec']
            brightStarRadius = brightStarCat['radius']

        else :
            brightStarRA = None
            brightStarDec = None
            brightStarRadius = None

        filterNameArray = np.core.defchararray.strip(obsCat['filtername'])

        self.makePrimaryStars(obsCat['ra'], obsCat['dec'], filterNameArray,
                                brightStarRA=brightStarRA,
                                brightStarDec=brightStarDec,
                                brightStarRadius=brightStarRadius)

        fitsio.write(self.starConfig['starfileBase']+'_prepositions.fits',self.objCat,clobber=True)


    def makeMatchedStarsFromFits(self, observationFile, obsIndexFile, clobber=False):
        """
        Make matched stars, loading observations from fits.

        parameters
        ----------
        observationFile: string
        obsIndexFile: string
           File output from makePrimaryStarsFromFits
        """

        import fitsio

        if (not clobber):
            if (os.path.isfile(obsIndexFile)):
                self.fgcmLog.info("Found %s " % (obsIndexFile))
                return


        obsCat = fitsio.read(observationFile, ext=1, lower=True,
                             columns=['ra','dec','filtername'])

        filterNameArray = np.core.defchararray.strip(obsCat['filtername'])

        self.makeMatchedStars(obsCat['ra'], obsCat['dec'], filterNameArray)

        # and save the outputs...
        fits=fitsio.FITS(obsIndexFile, mode='rw', clobber=True)
        fits.create_table_hdu(data=self.objIndexCat, extname='POS')
        fits[1].write(self.objIndexCat)

        fits.create_table_hdu(data=self.obsIndexCat, extname='INDEX')
        fits[2].write(self.obsIndexCat)

    def makeReferenceMatchesFromFits(self, refLoader, clobber=False):
        """
        Make an absolute reference match catalog, saving to fits.

        Parameters
        ----------
        refLoader: `object`
           Object which has refLoader.getFgcmReferenceStarsHealpix
        clobber: `bool`, optional
           Clobber existing absref catalog?  Default is False.
        """

        import fitsio

        refFile = self.starConfig['starfileBase'] + '_refcat.fits'

        if not clobber:
            if os.path.isfile(refFile):
                self.fgcmLog.info("Found %s" % (refFile))
                return refFile

        self.makeReferenceMatches(refLoader)

        fitsio.write(refFile, self.referenceCat, clobber=True)

        return refFile

    def makePrimaryStars(self, raArray, decArray, filterNameArray,
                           bandSelected=False,
                           brightStarRA=None, brightStarDec=None, brightStarRadius=None):
        """
        Make primary stars, from pre-loaded arrays

        parameters
        ----------
        raArray: double array
           RA for each observation
        decArray: double array
           Dec for each observation
        filterNameArray: string array
           Array of filterNames.
        bandSelected: bool, default=False
           Has the input raArray/decArray been pre-selected by band?
        brightStarRA: double array, optional
           RA for bright stars for mask
        brightStarDec: double array, optional
           Dec for bright stars for mask
        brightStarRadius: float array, optional
           Radius for bright stars for mask

        Output attributes
        -----------------
        objCat: numpy recarray
           Catalog of unique objects selected from primary band
        """

        # can we use the better smatch code?
        try:
            import smatch
            hasSmatch = True
            self.fgcmLog.info("Using smatch for matching.")
        except ImportError:
            hasSmatch = False
            self.fgcmLog.info("Using htm for matching.")

        if (raArray.size != decArray.size):
            raise ValueError("raArray, decArray must be same length.")
        if (raArray.size != filterNameArray.size):
            raise ValueError("raArray, filterNameArray must be same length.")

        # Prepare bright stars if necessary...
        if (brightStarRA is not None and brightStarDec is not None and
            brightStarRadius is not None):
            if (brightStarRA.size != brightStarDec.size or
                brightStarRA.size != brightStarRadius.size):
                raise ValueError("brightStarRA/Dec/Radius must have same length")
            cutBrightStars = True
        else:
            cutBrightStars = False

        # Define the dtype
        dtype=[('fgcm_id', 'i4'),
               ('ra', 'f8'),
               ('dec', 'f8')]

        pixelCats = []

        # Split into pixels
        ipring = hp.ang2pix(self.starConfig['coarseNSide'],
                            (90.0 - decArray) * np.pi / 180.,
                            raArray * np.pi / 180.)
        hpix, revpix = esutil.stat.histogram(ipring, rev=True)

        gdpix, = np.where(hpix > 0)
        for ii, gpix in enumerate(gdpix):
            # This is the array of all the observations in the coarse pixel
            p1a=revpix[revpix[gpix]: revpix[gpix + 1]]

            if p1a.size == 0:
                continue

            bandPixelCat = None

            # loop over bands...
            for primaryBand in self.starConfig['primaryBands']:
                # We first need to select based on the band, not on the filter name
                useFlag = None
                for filterName in self.filterNames:
                    if (self.starConfig['filterToBand'][filterName] == primaryBand):
                        if useFlag is None:
                            useFlag = (filterNameArray[p1a] == filterName.encode('utf-8'))
                        else:
                            useFlag |= (filterNameArray[p1a] == filterName.encode('utf-8'))

                raArrayUse = raArray[p1a[useFlag]]
                decArrayUse = decArray[p1a[useFlag]]

                if raArrayUse.size == 0:
                    self.fgcmLog.info("Nothing found for pixel %d" % (ipring[p1a[0]]))
                    continue

                if hasSmatch:
                    # faster match...
                    matches = smatch.match(raArrayUse, decArrayUse,
                                           self.starConfig['matchRadius'] / 3600.0,
                                           raArrayUse, decArrayUse,
                                           nside=self.starConfig['matchNSide'], maxmatch=0)
                    i1 = matches['i1']
                    i2 = matches['i2']
                else:
                    # slower htm matching...
                    htm = esutil.htm.HTM(11)

                    matcher = esutil.htm.Matcher(11, raArrayUse, decArrayUse)
                    matches = matcher.match(raArrayUse, decArrayUse,
                                            self.starConfig['matchRadius'] / 3600.0,
                                            maxmatch=0)
                    i1 = matches[1]
                    i2 = matches[0]

                fakeId = np.arange(p1a.size)
                hist, rev = esutil.stat.histogram(fakeId[i1], rev=True)

                if (hist.max() == 1):
                    self.fgcmLog.info("Warning: No matches found for pixel %d, band %s!" %
                                      (ipring[p1a[0]], primaryBand))
                    continue

                maxObs = hist.max()

                # how many unique objects do we have?
                histTemp = hist.copy()
                count=0
                for j in xrange(histTemp.size):
                    jj = fakeId[j]
                    if (histTemp[jj] >= self.starConfig['minPerBand']):
                        i1a = rev[rev[jj]: rev[jj + 1]]
                        histTemp[i2[i1a]] = 0
                        count = count + 1

                # make a temporary catalog...
                bandPixelCatTemp = np.zeros(count, dtype=dtype)

                # Rotate.  This works for DES, but maybe not in general?
                raTemp = raArrayUse.copy()

                hi, = np.where(raTemp > 180.0)
                raTemp[hi] -= 360.0

                # Compute mean ra/dec
                index = 0
                for j in xrange(hist.size):
                    jj = fakeId[j]
                    if (hist[jj] >= self.starConfig['minPerBand']):
                        i1a = rev[rev[jj]: rev[jj + 1]]
                        starInd = i2[i1a]
                        # make sure this doesn't get used again
                        hist[starInd] = 0
                        bandPixelCatTemp['ra'][index] = np.sum(raTemp[starInd]) / starInd.size
                        bandPixelCatTemp['dec'][index] = np.sum(decArrayUse[starInd]) / starInd.size
                        index = index + 1

                # Restore negative RAs
                lo, = np.where(bandPixelCatTemp['ra'] < 0.0)
                bandPixelCatTemp['ra'][lo] += 360.0

                # Match to previously pixel catalog if available, and remove dupes
                if bandPixelCat is None:
                    # First time through, these are all new objects
                    bandPixelCat = bandPixelCatTemp
                    self.fgcmLog.info(" Found %d primary stars in %s band" % (bandPixelCatTemp.size, primaryBand))
                else:
                    # We already have objects, need to match/append
                    if hasSmatch:
                        bandMatches = smatch.match(bandPixelCat['ra'], bandPixelCat['dec'],
                                                   self.starConfig['matchRadius'] / 3600.0,
                                                   bandPixelCatTemp['ra'], bandPixelCatTemp['dec'],
                                                   maxmatch=0)
                        i1b = matches['i1']
                        i2b = matches['i2']
                    else:
                        matcher = esutil.htm.Matcher(11, bandPixelCat['ra'], bandPixelCat['dec'])
                        matches = matcher.match(bandPixelCatTemp['ra'], bandPixelCatTemp['dec'],
                                                self.starConfig['matchRadius'] / 3600.0,
                                                maxmatch=0)
                        i1b = matches[1]
                        i2b = matches[0]

                    # Remove all matches from the temp catalog
                    bandPixelCatTemp = np.delete(bandPixelCatTemp, i2b)
                    self.fgcmLog.info(" Found %d new primary stars in %s band" % (bandPixelCatTemp.size, primaryBand))

                    bandPixelCat = np.append(bandPixelCat, bandPixelCatTemp)

            if bandPixelCat is not None:
                # Append to list of catalogs...
                pixelCats.append(bandPixelCat)

                self.fgcmLog.info("Found %d unique objects in pixel %d (%d of %d)." %
                                  (bandPixelCat.size, ipring[p1a[0]], ii, gdpix.size - 1))

        # now assemble into a total objCat
        count = 0
        for pixelCat in pixelCats:
            count += pixelCat.size

        self.objCat = np.zeros(count, dtype=dtype)
        ctr = 0
        for pixelCat in pixelCats:
            self.objCat[ctr:ctr + pixelCat.size] = pixelCat
            ctr += pixelCat.size
            # and clear memory
            pixelCat = None

        self.objCat['fgcm_id'] = np.arange(count) + 1

        self.fgcmLog.info("Found %d unique objects with >= %d observations." %
                          (count, self.starConfig['minPerBand']))

        if (cutBrightStars):
            self.fgcmLog.info("Matching to bright stars for masking...")
            if (hasSmatch):
                # faster smatch...
                matches = smatch.match(brightStarRA, brightStarDec, brightStarRadius,
                                       self.objCat['ra'], self.objCat['dec'], nside=self.starConfig['matchNSide'],
                                       maxmatch=0)
                i1=matches['i1']
                i2=matches['i2']
            else:
                # slower htm matching...
                htm = esutil.htm.HTM(11)

                matcher = esutil.htm.Matcher(10, brightStarRA, brightStarDec)
                matches = matcher.match(self.objCat['ra'], self.objCat['dec'], brightStarRadius,
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

            matches=smatch.match(self.objCat['ra'], self.objCat['dec'],
                                 self.starConfig['isolationRadius']/3600.0,
                                 self.objCat['ra'], self.objCat['dec'],
                                 nside=self.starConfig['matchNSide'], maxmatch=0)
            i1=matches['i1']
            i2=matches['i2']
        else:
            # slower htm matching...
            htm = esutil.htm.HTM(11)

            matcher = esutil.htm.Matcher(11, self.objCat['ra'], self.objCat['dec'])
            matches = matcher.match(self.objCat['ra'], self.objCat['dec'],
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
         generated from makePrimaryStars().

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
            raise ValueError("Must run makePrimaryStars first")

        # can we use the better smatch code?
        try:
            import smatch
            hasSmatch = True
        except ImportError:
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

            matches=smatch.match(self.objCat['ra'], self.objCat['dec'],
                                 self.starConfig['matchRadius']/3600.0,
                                 raArray, decArray,
                                 nside=self.starConfig['matchNSide'],
                                 maxmatch=0)
            i1=matches['i1']
            i2=matches['i2']
        else:
            # slower htm matching...
            htm = esutil.htm.HTM(11)

            matcher = esutil.htm.Matcher(11, self.objCat['ra'], self.objCat['dec'])
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
            raise ValueError("Number of primary stars (%d) does not match observations (%d)." %
                             (self.objCat.size, nObsPerObj.size))

        # and our simple classifier
        #    1 is a good star, 0 is bad.
        objClass = np.zeros(self.objCat.size, dtype='i2')

        # We may have no "required" bands beyond being in one of the primary bands
        if len(self.starConfig['requiredBands']) > 0:

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

            # make sure we have enough per band
            gd,=np.where(minObs >= self.starConfig['minPerBand'])
            objClass[gd] = 1
        else:
            objClass[:] = 1
            gd, = np.where(objClass == 1)

        self.fgcmLog.info("There are %d stars with at least %d observations in each required band." %
              (gd.size, self.starConfig['minPerBand']))


        # cut the density of stars down with sampling.

        theta = (90.0 - self.objCat['dec'][gd])*np.pi/180.
        phi = self.objCat['ra'][gd]*np.pi/180.

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
        self.objIndexCat = np.zeros(gd.size, dtype=[('fgcm_id','i4'),
                                                    ('ra','f8'),
                                                    ('dec','f8'),
                                                    ('obsarrindex','i4'),
                                                    ('nobs','i4')])
        self.objIndexCat['fgcm_id'][:] = self.objCat['fgcm_id'][gd]
        self.objIndexCat['ra'][:] = self.objCat['ra'][gd]
        self.objIndexCat['dec'][:] = self.objCat['dec'][gd]
        # this is the number of observations per object
        self.objIndexCat['nobs'][:] = nObsPerObj[gd]
        # and the index is given by the cumulative sum
        self.objIndexCat['obsarrindex'][1:] = np.cumsum(nObsPerObj[gd])[:-1]

        # and we need to create the observation indices from the obsarrindex

        nTotObs = self.objIndexCat['obsarrindex'][-1] + self.objIndexCat['nobs'][-1]

        self.obsIndexCat = np.zeros(nTotObs,
                                    dtype=[('obsindex','i4')])
        ctr = 0
        self.fgcmLog.info("Spooling out %d observation indices." % (nTotObs))
        for i in gd:
            self.obsIndexCat[ctr:ctr+nObsPerObj[i]] = i2[obsInd[obsInd[i]:obsInd[i+1]]]
            ctr+=nObsPerObj[i]

        # and we're done

    def makeReferenceMatches(self, refLoader):
        """
        Make an absolute reference match catalog.

        Parameters
        ----------
        refLoader: `object`
           Object which has refLoader.getFgcmReferenceStarsHealpix
        """

        # can we use the better smatch code?
        try:
            import smatch
            hasSmatch = True
        except ImportError:
            hasSmatch = False

        ipring = hp.ang2pix(self.starConfig['coarseNSide'],
                            np.radians(90.0 - self.objIndexCat['dec']),
                            np.radians(self.objIndexCat['ra']))
        hpix, revpix = esutil.stat.histogram(ipring, rev=True)

        pixelCats = []
        nBands = len(self.starConfig['referenceBands'])

        dtype = [('fgcm_id', 'i4'),
                 ('refMag', 'f4', nBands),
                 ('refMagErr', 'f4', nBands)]

        gdpix, = np.where(hpix > 0)
        for ii, gpix in enumerate(gdpix):
            p1a = revpix[revpix[gpix]: revpix[gpix + 1]]

            # Load in reference catalog
            refCat = refLoader.getFgcmReferenceStarsHealpix(self.starConfig['coarseNSide'],
                                                            ipring[p1a[0]],
                                                            self.starConfig['referenceBands'])

            if refCat.size == 0:
                # No stars in this pixel.  That's okay.
                continue

            if hasSmatch:
                matches = smatch.match(self.objIndexCat['ra'][p1a],
                                       self.objIndexCat['dec'][p1a],
                                       self.starConfig['matchRadius']/3600.0,
                                       refCat['ra'], refCat['dec'],
                                       nside=self.starConfig['matchNSide'],
                                       maxmatch=1)
                i1 = matches['i1']
                i2 = matches['i2']
            else:
                htm = esutil.htm.HTM(11)

                matcher = esutil.htm.Matcher(11,
                                             self.objIndexCat['ra'][p1a],
                                             self.objIndexCat['dec'][p1a])
                matches = matcher.match(refCat['ra'], refCat['dec'],
                                        self.starConfig['matchRadius']/3600.0,
                                        maxmatch=1)

                # matches[0] -> m1 -> array from matcher.match() call (refCat)
                # matches[1] -> m2 -> array from htm.Matcher() (self.objIndexCat)
                i2 = matches[0]
                i1 = matches[1]

            # i1 -> objIndexCat[p1a]
            # i2 -> refCat

            if i1.size == 0:
                # No matched stars in this pixel.  That's okay.
                continue

            pixelCat = np.zeros(i1.size, dtype=dtype)
            pixelCat['fgcm_id'] = self.objIndexCat['fgcm_id'][p1a[i1]]
            pixelCat['refMag'][:, :] = refCat['refMag'][i2, :]
            pixelCat['refMagErr'][:, :] = refCat['refMagErr'][i2, :]

            pixelCats.append(pixelCat)

            self.fgcmLog.info("Found %d reference matches in pixel %d (%d of %d)." %
                              (pixelCat.size, ipring[p1a[0]], ii, gdpix.size - 1))

        # Now assemble
        count = 0
        for pixelCat in pixelCats:
            count += pixelCat.size

        self.referenceCat = np.zeros(count, dtype=dtype)
        ctr = 0
        for pixelCat in pixelCats:
            self.referenceCat[ctr: ctr + pixelCat.size] = pixelCat
            ctr += pixelCat.size
            # and clear memory
            pixelCat = None
