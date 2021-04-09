import numpy as np
import os
import sys
import esutil
import glob
import healpy as hp

from .fgcmLogger import FgcmLogger
from .fgcmUtilities import Matcher


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
    referenceFilterNames: string list
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
    coarseNSide: int
       Healpix nside to break down into coarse pixels (save memory)
    brightStarFile: string, optional
       File with (very) bright stars (ra/dec/radius) for masking
    """

    def __init__(self,starConfig):
        self.starConfig = starConfig

        requiredKeys=['filterToBand','requiredBands','referenceFilterNames',
                      'minPerBand','matchRadius',
                      'isolationRadius','densNSide',
                      'densMaxPerPixel','primaryBands',
                      'matchNSide','coarseNSide']

        for key in requiredKeys:
            if (key not in starConfig):
                raise ValueError("required %s not in starConfig" % (key))

        if 'quantitiesToAverage' not in starConfig:
            starConfig['quantitiesToAverage'] = []

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

        for referenceFilterName in starConfig['referenceFilterNames']:
            if referenceFilterName not in starConfig['filterToBand']:
                raise ValueError("referenceFilterName %s not in filterToBand filters!" % (referenceFilterName))

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

        prepositionFile = self.starConfig['starfileBase'] + '_prepositions.fits'

        if not clobber and os.path.isfile(prepositionFile):
            import fitsio

            self.objCat = fitsio.read(prepositionFile, ext=1)
        else:
            self.makePrimaryStarsFromFits(observationFile)

        obsIndexFile = self.starConfig['starfileBase']+'_obs_index.fits'

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

        columns = ['ra', 'dec', 'filtername']
        if len(self.starConfig['quantitiesToAverage']) > 0:
            extraColumns = []
            for quant in self.starConfig['quantitiesToAverage']:
                extraColumns.extend([quant.lower(), quant.lower() + '_err'])
            columns.extend(extraColumns)

        obsCat = fits[1].read(columns=columns,lower=True,rows=w)

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

        if len(self.starConfig['quantitiesToAverage']) > 0:
            extraQuantityArrays = obsCat[extraColumns]
        else:
            extraQuantityArrays = None

        self.makePrimaryStars(obsCat['ra'], obsCat['dec'], filterNameArray,
                              extraQuantityArrays=extraQuantityArrays,
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
                         extraQuantityArrays=None,
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
        extraQuantityArrays: numpy recarray, optional
           Record array of extra quantities to average.  Default None.
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

        hasExtraQuantities = False
        if len(self.starConfig['quantitiesToAverage']) > 0:
            if extraQuantityArrays is None:
                raise RuntimeError("Cannot set quantitiesToAverage without passing extraQuantityArrays")
            hasExtraQuantities = True
            for quant in self.starConfig['quantitiesToAverage']:
                dtype.extend([(quant, 'f4')])
                if quant not in extraQuantityArrays.dtype.names:
                    raise RuntimeError("quantity to average %s not in extraQuantityArrays" % (quant))

        pixelCats = []

        # Split into pixels
        ipring = hp.ang2pix(self.starConfig['coarseNSide'],
                            (90.0 - decArray) * np.pi / 180.,
                            raArray * np.pi / 180.)
        hpix, revpix = esutil.stat.histogram(ipring, rev=True)

        gdpix, = np.where(hpix > 0)
        self.fgcmLog.info("Matching primary stars in %d pixels" % (gdpix.size))

        for ii, gpix in enumerate(gdpix):
            # This is the array of all the observations in the coarse pixel
            p1a=revpix[revpix[gpix]: revpix[gpix + 1]]

            if p1a.size == 0:
                continue

            bandPixelCat = None

            filterNameArrayIsEncoded = False
            try:
                test = filterNameArray[0].decode('utf-8')
                filterNameArrayIsEncoded = True
            except AttributeError:
                pass

            # loop over bands...
            for primaryBand in self.starConfig['primaryBands']:
                # We first need to select based on the band, not on the filter name
                useFlag = None
                for filterName in self.filterNames:
                    if (self.starConfig['filterToBand'][filterName] == primaryBand):
                        if useFlag is None:
                            if filterNameArrayIsEncoded:
                                useFlag = (filterNameArray[p1a] == filterName.encode('utf-8'))
                            else:
                                useFlag = (filterNameArray[p1a] == filterName)
                        else:
                            if filterNameArrayIsEncoded:
                                useFlag |= (filterNameArray[p1a] == filterName.encode('utf-8'))
                            else:
                                useFlag = (filterNameArray[p1a] == filterName)

                raArrayUse = raArray[p1a[useFlag]]
                decArrayUse = decArray[p1a[useFlag]]

                if hasExtraQuantities:
                    extraQuantityArraysUse = extraQuantityArrays[p1a[useFlag]]

                if raArrayUse.size == 0:
                    self.fgcmLog.info("Nothing found for pixel %d" % (ipring[p1a[0]]))
                    continue

                with Matcher(raArrayUse, decArrayUse) as matcher:
                    idx = matcher.query_self(self.starConfig['matchRadius']/3600.,
                                             min_match=self.starConfig['minPerBand'])
#                idx = match_to_self(raArrayUse, decArrayUse,
#                                    self.starConfig['matchRadius']/3600.,
#                                    min_match=self.starConfig['minPerBand'])

                count = len(idx)

                bandPixelCatTemp = np.zeros(count, dtype=dtype)

                # Rotate if necessary...
                rotated = False
                if raArrayUse.min() > 60.0 and raArrayUse.max() > 300.0:
                    # Even for very large pixels it looks like we span the 360.0 pole
                    raTemp = raArrayUse.copy()
                    raTemp[raTemp > 180.0] -= 360.0
                else:
                    raTemp = raArrayUse

                for i, row in enumerate(idx):
                    row = np.array(row)
                    bandPixelCatTemp['ra'][i] = np.sum(raArrayUse[row])/len(row)
                    bandPixelCatTemp['dec'][i] = np.sum(decArrayUse[row])/len(row)

                    if hasExtraQuantities:
                        for quant in self.starConfig['quantitiesToAverage']:
                            ok, = np.where(extraQuantityArraysUse[quant + '_err'][row] > 0.0)
                            wt = 1./extraQuantityArraysUse[quant + '_err'][row[ok]]
                            bandPixelCatTemp[quant][i] = np.sum(wt*extraQuantityArraysUse[quant][row[ok]])/np.sum(wt)

                # Restore negative RAs
                bandPixelCatTemp['ra'][bandPixelCatTemp['ra'] < 0.0] += 360.0

                # Match to previous band pixel catalog if available, and remove duplicates
                if bandPixelCat is None:
                    bandPixelCat = bandPixelCatTemp
                    self.fgcmLog.info(" Found %d primary stars in %s band" % (bandPixelCatTemp.size, primaryBand))
                else:
                    with Matcher(bandPixelCatTemp['ra'], bandPixelCatTemp['dec']) as matcher:
                        idx = matcher.query_radius(bandPixelCat['ra'], bandPixelCat['dec'],
                                                   self.starConfig['matchRadius']/3600.0)

#                    idx = match_to_other(bandPixelCatTemp['ra'], bandPixelCatTemp['dec'],
#                                         self.starConfig['matchRadius']/3600.0,
#                                         bandPixelCat['ra'], bandPixelCat['dec'])

                    # Any object that has a match should be removed
                    matchedIndices = np.array([i for i in range(len(idx)) if len(idx[i]) > 0])
                    bandPixelCatTemp = np.delete(bandPixelCatTemp, matchedIndices)
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

            with Matcher(self.objCat['ra'], self.objCat['dec']) as matcher:
                idx = matcher.query_radius(brightStarRA, brightStarDec,
                                           np.max(brightStarRadius))

#            idx = match_to_other(self.objCat['ra'], self.objCat['dec'],
#                                 np.max(brightStarRadius),
#                                 brightStarRA, brightStarDec)

            matched = np.zeros(self.objCat.size, dtype=bool)
            for j in range(len(idx)):
                if len(idx[j]) > 0:
                    ds = esutil.coords.sphdist(self.objCat['ra'][j], self.objCat['dec'][j],
                                               brightStarRA[idx[j]], brightStarDec[idx[j]])
                    ok, = np.where(ds < brightStarRadius[idx[j]])
                    if ok.size > 0:
                        matched[j] = True

            self.fgcmLog.info("Cutting %d objects too near bright stars." % (i2.size))
            self.objCat = np.delete(self.objCat, np.where(matched)[0])

        # and remove stars with near neighbors
        self.fgcmLog.info("Matching stars to neighbors...")

        # Only those objects that have more than 1 match (self + other)
        # have neighbors and should be removed.
        with Matcher(self.objCat['ra'], self.objCat['dec']) as matcher:
            idx = matcher.query_self(self.starConfig['isolationRadius']/3600.0, min_match=2)

        # idx = match_to_self(self.objCat['ra'], self.objCat['dec'],
        #                    self.starConfig['isolationRadius']/3600.0,
        #                    min_match=2)

        neighbor_indices = []
        for row in idx:
            neighbor_indices.extend(row)

        if len(neighbor_indices) > 0:
            neighbored = np.unique(neighbor_indices)
            self.fgcmLog.info("Cutting %d objects within %.2f arcsec of a neighbor" %
                              (neighbored.size, self.starConfig['isolationRadius']))

            self.objCat = np.delete(self.objCat, neighbored)

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

        if (raArray.size != decArray.size or
            raArray.size != filterNameArray.size):
            raise ValueError("raArray, decArray, filterNameArray must be same length")

        # translate filterNameArray to bandArray ... can this be made faster, or
        #  does it need to be?

        filterNameArrayIsEncoded = False
        try:
            test = filterNameArray[0].decode('utf-8')
            filterNameArrayIsEncoded = True
        except AttributeError:
            pass

        bandArray = np.zeros_like(filterNameArray)
        for filterName in self.filterNames:
            if filterNameArrayIsEncoded:
                use, = np.where(filterNameArray == filterName.encode('utf-8'))
            else:
                use, = np.where(filterNameArray == filterName)
            bandArray[use] = self.starConfig['filterToBand'][filterName]

        self.fgcmLog.info("Matching positions to observations...")

        with Matcher(self.objCat['ra'], self.objCat['dec']) as matcher:
            idx, i1, i2, dist = matcher.query_radius(raArray, decArray,
                                                     self.starConfig['matchRadius']/3600.0,
                                                     return_indices=True)
        # idx = match_to_other(self.objCat['ra'], self.objCat['dec'],
        #                      self.starConfig['matchRadius']/3600.0,
        #                     raArray, decArray)

        nObsPerObj = np.array([len(row) for row in idx])

        # Our simple classifier
        #    1 is a good star, 0 is bad.
        objClass = np.zeros(self.objCat.size, dtype='i2')

        if len(self.starConfig['requiredBands']) > 0:
            # Which stars have at least minPerBand observations in each required band?
            reqBands = np.array(self.starConfig['requiredBands'], dtype=bandArray.dtype)

            # We need indices for this code
            # i1 = np.zeros(nObsPerObj.sum(), dtype=np.int32)
            # i2 = np.zeros_like(i1)
            # counter = 0
            # for i, row in enumerate(idx):
            #     i1[counter: counter + nObsPerObj[i]] = i
            #     i2[counter: counter + nObsPerObj[i]] = row
            #     counter += nObsPerObj[i]

            # This could be made more efficient
            self.fgcmLog.info("Computing number of observations per band")
            nObs = np.zeros((reqBands.size, self.objCat.size), dtype='i4')
            for i in range(reqBands.size):
                use, = np.where(bandArray[i2] == reqBands[i])

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
        ipring = hp.ang2pix(self.starConfig['densNSide'], self.objCat['ra'], self.objCat['dec'], lonlat=True)
        hist, rev = esutil.stat.histogram(ipring, rev=True)

        high,=np.where(hist > self.starConfig['densMaxPerPixel'])
        ok,=np.where(hist > 0)
        self.fgcmLog.info("There are %d/%d pixels with high stellar density" % (high.size, ok.size))
        for i in range(high.size):
            i1a=rev[rev[high[i]]:rev[high[i]+1]]
            cut=np.random.choice(i1a,size=i1a.size-self.starConfig['densMaxPerPixel'],replace=False)
            objClass[gd[cut]] = 0

        # redo the good object selection after sampling
        gd, = np.where(objClass == 1)

        dtype = [('fgcm_id','i4'),
                 ('ra','f8'),
                 ('dec','f8'),
                 ('obsarrindex','i4'),
                 ('nobs','i4')]

        hasExtraQuantities = False
        if len(self.starConfig['quantitiesToAverage']) > 0:
            hasExtraQuantities = True
            for quant in self.starConfig['quantitiesToAverage']:
                dtype.extend([(quant, 'f4')])

        # create the object catalog index
        self.objIndexCat = np.zeros(gd.size, dtype=dtype)

        self.objIndexCat['fgcm_id'][:] = self.objCat['fgcm_id'][gd]
        self.objIndexCat['ra'][:] = self.objCat['ra'][gd]
        self.objIndexCat['dec'][:] = self.objCat['dec'][gd]
        # this is the number of observations per object
        self.objIndexCat['nobs'][:] = nObsPerObj[gd]
        # and the index is given by the cumulative sum
        self.objIndexCat['obsarrindex'][1:] = np.cumsum(nObsPerObj[gd])[:-1]

        # Copy in the extra quantities
        if hasExtraQuantities:
            for quant in self.starConfig['quantitiesToAverage']:
                self.objIndexCat[quant][:] = self.objCat[quant][gd]

        # and we need to create the observation indices from the obsarrindex

        nTotObs = self.objIndexCat['obsarrindex'][-1] + self.objIndexCat['nobs'][-1]

        self.obsIndexCat = np.zeros(nTotObs,
                                    dtype=[('obsindex','i4')])
        ctr = 0
        self.fgcmLog.info("Spooling out %d observation indices." % (nTotObs))
        for i in gd:
            self.obsIndexCat[ctr: ctr + nObsPerObj[i]] = idx[i]
            ctr += nObsPerObj[i]

    def makeReferenceMatches(self, refLoader):
        """
        Make an absolute reference match catalog.

        Parameters
        ----------
        refLoader: `object`
           Object which has refLoader.getFgcmReferenceStarsHealpix
        """
        ipring = hp.ang2pix(self.starConfig['coarseNSide'],
                            np.radians(90.0 - self.objIndexCat['dec']),
                            np.radians(self.objIndexCat['ra']))
        hpix, revpix = esutil.stat.histogram(ipring, rev=True)

        pixelCats = []
        nBands = len(self.starConfig['referenceFilterNames'])

        dtype = [('fgcm_id', 'i4'),
                 ('refMag', 'f4', nBands),
                 ('refMagErr', 'f4', nBands)]

        gdpix, = np.where(hpix > 0)
        for ii, gpix in enumerate(gdpix):
            p1a = revpix[revpix[gpix]: revpix[gpix + 1]]

            # Choose the center of the stars...
            raWrap = self.objIndexCat['ra'][p1a]
            if (raWrap.min() < 10.0) and (raWrap.max() > 350.0):
                hi, = np.where(raWrap > 180.0)
                raWrap[hi] -= 360.0
                meanRA = np.mean(raWrap)
                if meanRA < 0.0:
                    meanRA += 360.0
            else:
                meanRA = np.mean(raWrap)
            meanDec = np.mean(self.objIndexCat['dec'][p1a])

            dist = esutil.coords.sphdist(meanRA, meanDec,
                                         self.objIndexCat['ra'][p1a], self.objIndexCat['dec'][p1a])
            rad = dist.max()

            # Note nside2resol returns radians of the pixel along a side...
            if rad < np.degrees(hp.nside2resol(self.starConfig['coarseNSide'])/2.):
                # If it's a smaller radius, read the circle
                refCat = refLoader.getFgcmReferenceStarsSkyCircle(meanRA, meanDec, rad,
                                                                  self.starConfig['referenceFilterNames'])
            else:
                # Otherwise, this will always work
                refCat = refLoader.getFgcmReferenceStarsHealpix(self.starConfig['coarseNSide'],
                                                                ipring[p1a[0]],
                                                                self.starConfig['referenceFilterNames'])

            if refCat.size == 0:
                # No stars in this pixel.  That's okay.
                continue

            with Matcher(self.objIndexCat['ra'][p1a], self.objIndexCat['dec'][p1a]) as matcher:
                idx, i1, i2, d = matcher.query_knn(refCat['ra'], refCat['dec'], k=1,
                                                   distance_upper_bound=self.starConfig['matchRadius']/3600.0,
                                                   return_indices=True)

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
