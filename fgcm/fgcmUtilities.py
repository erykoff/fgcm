import numpy as np


# Dictionary of object flags
objFlagDict = {'TOO_FEW_OBS':2**0,
               'BAD_COLOR':2**1,
               'VARIABLE':2**2,
               'TEMPORARY_BAD_STAR':2**3,
               'RESERVED':2**4,
               'REFSTAR_OUTLIER': 2**5,
               'BAD_QUANTITY': 2**6}

# Dictionary of observation flags
obsFlagDict = {'NO_EXPOSURE': 2**0,
               'BAD_ERROR': 2**1,
               'SUPERSTAR_OUTLIER': 2**2,
               'NO_ZEROPOINT': 2**4,
               'BAD_MAG': 2**5,
               'BAD_AIRMASS': 2**6,
               'FOCALPLANE_OUTLIER': 2**7,
               'FOCALPLANE_OUTLIER_REF': 2**8}

# Dictionary of exposure flags
expFlagDict = {'TOO_FEW_STARS':2**0,
               'EXP_GRAY_TOO_NEGATIVE':2**1,
               'VAR_GRAY_TOO_LARGE':2**2,
               'TOO_FEW_EXP_ON_NIGHT':2**3,
               'NO_STARS':2**4,
               'BAND_NOT_IN_LUT':2**5,
               'TEMPORARY_BAD_EXPOSURE':2**6,
               'EXP_GRAY_TOO_POSITIVE':2**7,
               'BAD_ZPFLAG':2**8}

# Dictionary of zeropoint flags
zpFlagDict = {'PHOTOMETRIC_FIT_EXPOSURE':2**0,
              'PHOTOMETRIC_NOTFIT_EXPOSURE':2**1,
              'NONPHOTOMETRIC_FIT_NIGHT':2**2,
              'NOFIT_NIGHT':2**3,
              'CANNOT_COMPUTE_ZEROPOINT':2**4,
              'TOO_FEW_STARS_ON_CCD':2**5}

# Dictionary of retrieval flags
retrievalFlagDict = {'EXPOSURE_RETRIEVED':2**0,
                     'EXPOSURE_INTERPOLATED':2**1,
                     'EXPOSURE_STANDARD':2**2}

logDict = {'NONE': 0,
           'WARN': 1,
           'INFO': 2,
           'DEBUG': 3}

class MaxFitIterations(Exception):
    """
    Raise when we've hit the maximum number of iterations
    """

    pass


def getMemoryString(location):
    """
    Get a string for memory usage (current and peak) for logging.

    parameters
    ----------
    location: string
       A short string which denotes where in the code the memory was recorded.
    """

    status = None
    result = {'peak':0, 'rss':0}
    memoryString = ''
    try:
        with open('/proc/self/status') as status:
            for line in status:
                parts = line.split()
                key = parts[0][2:-1].lower()
                if key in result:
                    result[key] = int(parts[1])/1000

            memoryString = 'Memory usage at %s: %d MB current; %d MB peak.' % (
                location, result['rss'], result['peak'])
    except:
        memoryString = 'Could not get process status for memory usage at %s!' % (location)

    return memoryString

def dataBinner(x,y,binSize,xRange,nTrial=100,xNorm=-1.0,minPerBin=5):
    """
    Bin data and compute errors via bootstrap resampling.  All median statistics.

    parameters
    ----------
    x: float array
       x values
    y: float array
       y values
    binSize: float
       Bin size
    xRange: float array [2]
       x limits for binning
    nTrial: float, optional, default=100
       Number of bootstraps
    xNorm: float, optional, default=-1
       Set the y value == 0 when x is equan to xNorm.  if -1.0 then no norm.
    minPerBin: int, optional, default=5
       Minimum number of points per bin

    returns
    -------
    binStruct: recarray, length nbins
       'X_BIN': Left edge of each bin
       'X': median x in bin
       'X_ERR': width of x distribution in bin
       'X_ERR_MEAN': error on the mean of the x's in bin
       'Y': mean y in bin
       'Y_WIDTH': width of y distribution in bin
       'Y_ERR': error on the mean of the y's in the bin

    """


    import esutil

    hist,rev=esutil.stat.histogram(x,binsize=binSize,min=xRange[0],max=xRange[1]-0.0001,rev=True)
    binStruct=np.zeros(hist.size,dtype=[('X_BIN','f4'),
                                        ('X','f4'),
                                        ('X_ERR_MEAN','f4'),
                                        ('X_ERR','f4'),
                                        ('Y','f4'),
                                        ('Y_WIDTH','f4'),
                                        ('Y_ERR','f4'),
                                        ('N','i4')])
    binStruct['X_BIN'] = np.linspace(xRange[0],xRange[1],hist.size)

    for i in range(hist.size):
        if (hist[i] >= minPerBin):
            i1a=rev[rev[i]:rev[i+1]]

            binStruct['N'][i] = i1a.size

            medYs=np.zeros(nTrial,dtype='f8')
            medYWidths=np.zeros(nTrial,dtype='f8')
            medXs=np.zeros(nTrial,dtype='f8')
            medXWidths=np.zeros(nTrial,dtype='f8')

            for t in range(nTrial):
                r=(np.random.random(i1a.size)*i1a.size).astype('i4')

                medYs[t] = np.median(y[i1a[r]])
                medYWidths[t] = 1.4826*np.median(np.abs(y[i1a[r]] - medYs[t]))

                medXs[t] = np.median(x[i1a[r]])
                medXWidths[t] = 1.4826*np.median(np.abs(x[i1a[r]] - medXs[t]))

            binStruct['X'][i] = np.median(medXs)
            binStruct['X_ERR'][i] = np.median(medXWidths)
            binStruct['X_ERR_MEAN'][i] = 1.4826*np.median(np.abs(medXs - binStruct['X'][i]))
            binStruct['Y'][i] = np.median(medYs)
            binStruct['Y_WIDTH'][i] = np.median(medYWidths)
            binStruct['Y_ERR'][i] = 1.4826*np.median(np.abs(medYs - binStruct['Y'][i]))

    if (xNorm >= 0.0) :
        ind=np.clip(np.searchsorted(binStruct['X_BIN'],xnorm),0,binStruct.size-1)
        binStruct['Y'] = binStruct['Y'] - binStruct['Y'][ind]

    return binStruct

def gaussFunction(x, *p):
    """
    A gaussian function for fitting

    parameters
    ----------
    x: float array
       x values
    *p: Tuple (A, mu, sigma)
    """

    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2./(2.*sigma**2))

def histoGauss(ax,array):
    """
    Plot a histogram and fit a Gaussian to it.  Modeled after IDL histogauss.pro.

    parameters
    ----------
    ax: Plot axis object
       If None, return coefficients but do not plot
    array: float array to plot

    returns
    -------
    coeff: tuple (A, mu, sigma, fail)
       Fail is 0 if the fit succeeded
    """

    import scipy.optimize
    import matplotlib.pyplot as plt
    import esutil

    if array.size < 5:
        return np.array([1.0, 0.0, 1.0])

    q13 = np.percentile(array,[25,75])
    binsize=2*(q13[1] - q13[0])*array.size**(-1./3.)

    hist=esutil.stat.histogram(array,binsize=binsize,more=True)

    p0=[array.size,
        np.median(array),
        np.std(array)]

    try:
        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore")

            # This fit might throw a warning, which we don't need now.
            # Note that in the future if we use the output from this fit in an
            # automated way we'll need to tell the parent that we didn't converge
            coeff, varMatrix = scipy.optimize.curve_fit(gaussFunction, hist['center'],
                                                        hist['hist'], p0=p0)

        coeff = np.append(coeff, 0)
    except:
        # set to starting values...
        coeff = np.append(p0, 1)

    hcenter=hist['center']
    hhist=hist['hist']

    rangeLo = coeff[1] - 5*coeff[2]
    rangeHi = coeff[1] + 5*coeff[2]

    lo,=np.where(hcenter < rangeLo)
    ok,=np.where(hcenter > rangeLo)
    hhist[ok[0]] += np.sum(hhist[lo])

    hi,=np.where(hcenter > rangeHi)
    ok,=np.where(hcenter < rangeHi)
    hhist[ok[-1]] += np.sum(hhist[hi])

    if ax is not None:
        ax.plot(hcenter[ok],hhist[ok],'b-',linewidth=3)
        ax.set_xlim(rangeLo,rangeHi)

        if coeff[3] == 0:
            xvals=np.linspace(rangeLo,rangeHi,1000)
            yvals=gaussFunction(xvals,*coeff[: -1])

            ax.plot(xvals,yvals,'k--',linewidth=3)
        ax.locator_params(axis='x',nbins=6)  # hmmm

    return coeff


class Cheb2dField(object):
    """
    Chebyshev 2d Field class.
    """
    def __init__(self, xSize, ySize, pars):
        """
        Instantiate a Cheb2dField

        Parameters
        ----------
        xSize: `int`
           Size of bounding box in x direction
        ySize: `int`
           Size of bounding box in y direction
        pars: `np.array`
           Parameters may be 2d (order + 1, order + 1) or
           1d (order + 1) * (order + 1)
        """
        self.xSize = xSize
        self.ySize = ySize

        if len(pars.shape) == 1:
            # this is a 1-d flat pars
            self.order = int(np.sqrt(len(pars))) - 1
            self.pars = pars.reshape((self.order + 1, self.order + 1))
        else:
            # This is a 2-d pars
            self.order = pars.shape[0] - 1
            self.pars = pars

    @classmethod
    def fit(cls, xSize, ySize, order, x, y, value, valueErr=None, triangular=True):
        """
        Construct a Cheb2dField by fitting a field of x/y/value

        Parameters
        ----------
        xSize: `int`
           Size of bounding box in x direction
        ySize: `int`
           Size of bounding box in y direction
        order: `int`
           Chebyshev order of fit
        x: `np.array`
           Float array of x values
        y: `np.array`
           Float array of y values
        value: `np.array`
           Float array of dependent values to fit
        valueErr: `np.array`, optional
           Float array of dependent value errors to fit.
           Default is None (unweighted fit)
        triangular: `bool`, optional
           Fit should suppress high-order cross terms.  Default is True

        Returns
        -------
        cheb2dField: `fgcm.Cheb2dField`
           The Cheb2dField object
        """

        fit = np.zeros((order + 1) * (order + 1))

        if triangular:
            iind = np.repeat(np.arange(order + 1), order + 1)
            jind = np.tile(np.arange(order + 1), order + 1)
            lowInds, = np.where((iind + jind) <= order)
        else:
            lowInds = np.arange(fit.size)

        # We add a 0.5 here because of lsst stack compatibility
        xScaled = (x + 0.5 - xSize/2.) / (xSize / 2.)
        yScaled = (y + 0.5 - ySize/2.) / (ySize / 2.)

        V = np.polynomial.chebyshev.chebvander2d(yScaled, xScaled, [order, order])

        if triangular:
            V = V[:, lowInds]

        if valueErr is not None:
            w = 1./valueErr**2.
            Vprime = np.matmul(np.diag(w), V)
        else:
            w = np.ones(value.size)
            Vprime = V

        fit[lowInds] = np.matmul(np.matmul(np.linalg.inv(np.matmul(Vprime.T,
                                                                   Vprime)),
                                           Vprime.T), value * w)

        return cls(xSize, ySize, fit.reshape((order + 1, order + 1)))

    def evaluate(self, x, y, flatPars=None):
        """
        Evaluate the chebyshev field at a given position.  Optionally can
        substitute a set of fit parameters (used in curve fitter).

        Parameters
        ----------
        x: `np.array`
           Float array of x values
        y: `np.array`
           Float array of y values
        flatPars: 'list' or `np.array`, optional
           Replacement parameters, must have size (order + 1) * (order + 1)
           If None, use self.pars

        Returns
        -------
        values: `np.array`
           Float array of Chebyshev field evaluated at x, y
        """

        if flatPars is not None:
            c = np.array(flatPars).reshape(self.order + 1, self.order + 1)
        else:
            c = self.pars

        xScaled = (x + 0.5 - self.xSize/2.) / (self.xSize / 2.)
        yScaled = (y + 0.5 - self.ySize/2.) / (self.ySize / 2.)

        return np.polynomial.chebyshev.chebval2d(yScaled, xScaled, c)

    def evaluateCenter(self):
        """
        Evaluate the chebyshev field at the center.

        Returns
        -------
        value: `float`
           Float value of Chebyshev field evaluated at center.
        """

        return float(self.evaluate(self.xSize/2. - 0.5, self.ySize/2. - 0.5))

    def __call__(self, xy, *flatpars):
        """
        Evaluate for use in a fitter.

        Parameters
        ----------
        yx: `np.array`
           Numpy vstack (2, nvalues)
        *flatpars: `float`
           Chebyshev parameters.

        Returns
        -------
        values: `np.array`
           Value of function evaluated at x = xy[0, :], y = xy[1, :]
        """

        return self.evaluate(xy[0, :], xy[1, :], flatpars)


def plotCCDMap(ax, deltaMapper, values, cbLabel, loHi=None):
    """
    Plot CCD map with single values for each CCD.

    Parameters
    ----------
    ax :
        plot axis object
    deltaMapper : `np.recarray`
        Delta x/y, ra/dec mapper.
    values : `np.ndarray`
        Values for each ccd.
    cbLabel :  `str`
        Color bar label
    loHi : tuple [2], optional
        (lo, hi) if set.  Otherwise, scaling is computed from data.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    cm = plt.get_cmap('rainbow')

    plotRaRange = [np.min(deltaMapper['delta_ra']) - 0.02,
                   np.max(deltaMapper['delta_ra']) + 0.02]
    plotDecRange = [np.min(deltaMapper['delta_dec']) - 0.02,
                    np.max(deltaMapper['delta_dec']) + 0.02]

    if loHi is None:
        st = np.argsort(values)

        lo = values[st[int(0.02*st.size)]]
        hi = values[st[int(0.98*st.size)]]
    else:
        lo = loHi[0]
        hi = loHi[1]

    cNorm = colors.Normalize(vmin=lo, vmax=hi)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    Z = [[0, 0], [0, 0]]
    levels = np.linspace(lo, hi, num=150)
    CS3 = plt.contourf(Z, levels, cmap=cm)

    ax.clear()

    ax.set_xlim(plotRaRange[0], plotRaRange[1])
    ax.set_ylim(plotDecRange[0], plotDecRange[1])
    ax.set_xlabel(r'$\delta\,\mathrm{R.A.}$', fontsize=16)
    ax.set_ylabel(r'$\delta\,\mathrm{Dec.}$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    zGrid = np.zeros_like(deltaMapper['x'])
    for k in range(deltaMapper.size):
        zGrid[k, :] = values[k]

    plt.scatter(deltaMapper['delta_ra'].ravel(), deltaMapper['delta_dec'].ravel(),
                s=0.1,
                c=zGrid.ravel(),
                vmin=lo, vmax=hi)

    cb = None
    cb = plt.colorbar(CS3, ticks=np.linspace(lo, hi, 5))
    cb.set_label('%s' % (cbLabel), fontsize=14)

    return None


def plotCCDMap2d(ax, deltaMapper, parArray, cbLabel, loHi=None):
    """
    Plot CCD map with Chebyshev fits for each CCD.

    Parameters
    ----------
    ax :
        plot axis object
    deltaMapper : `np.recarray`
        Delta x/y, ra/dec mapper.
    parArray : `np.ndarray`
        Chebyshev parameter array (nCCD, nPars)
    cbLabel :  `str`
        Color bar label
    loHi : tuple [2], optional
        (lo, hi) if set.  Otherwise, scaling is computed from data.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    cm = plt.get_cmap('rainbow')
    plt.set_cmap('rainbow')

    plotRaRange = [np.min(deltaMapper['delta_ra']) - 0.02,
                   np.max(deltaMapper['delta_ra']) + 0.02]
    plotDecRange = [np.min(deltaMapper['delta_dec']) - 0.02,
                    np.max(deltaMapper['delta_dec']) + 0.02]

    centralValues = np.zeros(len(deltaMapper))
    for i in range(deltaMapper.size):
        field = Cheb2dField(deltaMapper['x_size'][i],
                            deltaMapper['y_size'][i],
                            parArray[i, :])
        centralValues[i] = -2.5*np.log10(field.evaluateCenter())*1000.0

    if loHi is None:
        st = np.argsort(centralValues)
        lo = centralValues[st[int(0.02*st.size)]]
        hi = centralValues[st[int(0.98*st.size)]]
    else:
        lo = loHi[0]
        hi = loHi[1]

    cNorm = colors.Normalize(vmin=lo, vmax=hi)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    Z = [[0, 0], [0, 0]]
    levels = np.linspace(lo, hi, num=150)
    CS3 = plt.contourf(Z, levels, cmap=cm)

    ax.clear()

    ax.set_xlim(plotRaRange[0], plotRaRange[1])
    ax.set_ylim(plotDecRange[0], plotDecRange[1])
    ax.set_xlabel(r'$\delta\,\mathrm{R.A.}$', fontsize=16)
    ax.set_ylabel(r'$\delta\,\mathrm{Dec.}$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    zGrid = np.zeros_like(deltaMapper['x'])
    for k in range(deltaMapper.size):
        field = Cheb2dField(deltaMapper['x_size'][k],
                            deltaMapper['y_size'][k],
                            parArray[k, :])
        zGrid[k, :] = -2.5*np.log10(np.clip(field.evaluate(deltaMapper['x'][k, :],
                                                           deltaMapper['y'][k, :]),
                                                           0.1,
                                                           None))*1000.0

    plt.scatter(deltaMapper['delta_ra'].ravel(), deltaMapper['delta_dec'].ravel(),
                s=0.1,
                c=zGrid.ravel(),
                vmin=lo, vmax=hi)

    cb = None
    cb = plt.colorbar(CS3, ticks=np.linspace(lo, hi, 5))
    cb.set_label('%s' % (cbLabel), fontsize=14)

    return None


def plotCCDMapBinned2d(ax, deltaMapper, binnedArray, cbLabel, loHi=None, illegalValue=-9999):
    """
    Plot CCD map with binned values for each CCD.

    Parameters
    ----------
    ax :
        plot axis object
    deltaMapper : `np.recarray`
        Delta x/y, ra/dec mapper.
    binnedArray : `np.ndarray`
        Binned values (nCCD, nx, ny)
    cbLabel :  `str`
        Color bar label
    loHi : tuple [2], optional
        (lo, hi) if set.  Otherwise, scaling is computed from data.
    illegalValue: float
       Sentinel for blank values
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    cm = plt.get_cmap('rainbow')
    plt.set_cmap('rainbow')

    plotRaRange = [np.min(deltaMapper['delta_ra']) - 0.02,
                   np.max(deltaMapper['delta_ra']) + 0.02]
    plotDecRange = [np.min(deltaMapper['delta_dec']) - 0.02,
                    np.max(deltaMapper['delta_dec']) + 0.02]

    if loHi is None:
        flatArray = binnedArray.ravel()
        gd, = np.where(flatArray > illegalValue)

        if gd.size < 2:
            # Nothing to plot here
            return

        st = np.argsort(flatArray[gd])
        lo = flatArray[gd[st[int(0.02*st.size)]]]
        hi = flatArray[gd[st[int(0.98*st.size)]]]
    else:
        lo = loHi[0]
        hi = loHi[1]

    cNorm = colors.Normalize(vmin=lo, vmax=hi)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    Z = [[0, 0], [0, 0]]
    levels = np.linspace(lo, hi, num=150)
    CS3 = plt.contourf(Z, levels, cmap=cm)

    ax.clear()

    ax.set_xlim(plotRaRange[0], plotRaRange[1])
    ax.set_ylim(plotDecRange[0], plotDecRange[1])
    ax.set_xlabel(r'$\delta\,\mathrm{R.A.}$', fontsize=16)
    ax.set_ylabel(r'$\delta\,\mathrm{Dec.}$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    zGrid = np.zeros_like(deltaMapper['x'])
    for k in range(deltaMapper.size):
        xBin = np.floor(deltaMapper[k]['x']/deltaMapper[k]['x_size']).astype(np.int32)
        yBin = np.floor(deltaMapper[k]['y']/deltaMapper[k]['y_size']).astype(np.int32)

        zGrid[k, :] = binnedArray[k, xBin, yBin]

    plt.scatter(deltaMapper['delta_ra'].ravel(), deltaMapper['delta_dec'].ravel(),
                s=0.1,
                c=zGrid.ravel(),
                vmin=lo, vmax=hi)

    cb = None
    cb = plt.colorbar(CS3, ticks=np.linspace(lo, hi, 5))
    cb.set_label('%s' % (cbLabel), fontsize=14)

    return None


def logFlaggedExposuresPerBand(log, fgcmPars, flagName, raiseAllBad=True):
    """
    Log how many exposures per band were flagged with a given flag.

    Parameters
    ----------
    log: fgcmLog
    fgcmPars: `fgcm.fgcmParameters`
    flagName: `str`
       Name of flag
    raiseAllBad: `bool`, optional
       Raise if a band has all bad.  Default is True.
    """

    for i, band in enumerate(fgcmPars.bands):
        if not fgcmPars.hasExposuresInBand[i]:
            continue
        inBand, = np.where(fgcmPars.expBandIndex == i)
        inBandBad, = np.where((fgcmPars.expFlag[inBand] &
                               expFlagDict[flagName]) > 0)
        log.info(' %s: %s band: %d of %d exposures' %
                 (flagName, band, inBandBad.size, inBand.size))
        if raiseAllBad and inBandBad.size == inBand.size:
            raise RuntimeError("FATAL: All observations in %s band have been cut with %s" % (band, flagName))


def checkFlaggedExposuresPerBand(log, fgcmPars):
    """
    Raise if any band has had all exposures removed.

    Parameters
    ----------
    log: fgcmLog
    fgcmPars: `fgcm.fgcmParameters`

    """

    for i, band in enumerate(fgcmPars.bands):
        if not fgcmPars.hasExposuresInBand[i]:
            continue
        inBand, = np.where(fgcmPars.expBandIndex == i)
        inBandBad, = np.where(fgcmPars.expFlag[inBand] > 0)
        if inBandBad.size >= (inBand.size - 1):
            raise RuntimeError("FATAL: All observations in %s band have been cut!")


def computeDeltaRA(a, b, dec=None, degrees=False):
    """
    Compute a - b for RAs.  Optionally scale by cos(dec).

    Parameters
    ----------
    a : `float` or `np.ndarray`
    b : `float` or `np.ndarray`
    dec : `float` or `np.ndarray`
    degrees : `bool`, optional
    """
    if degrees:
        rotVal = 360.0
    else:
        rotVal = 2.0*np.pi

    delta = (np.atleast_1d(a) - np.atleast_1d(b)) % rotVal
    test, = np.where(delta > rotVal/2.)
    delta[test] -= rotVal

    if dec is not None and degrees:
        delta *= np.cos(np.deg2rad(dec))
    elif dec is not None:
        delta *= np.cos(dec)

    return delta


class FocalPlaneProjectorFromOffsets(object):
    """Create a focalPlaneProjector that returns the formatted
    ccdOffsetsTable.

    This is for backwards compatibility and straight DECam processing.

    Parameters
    ----------
    ccdOffsets : `np.ndarray`
        CCD offset table
    """
    def __init__(self, ccdOffsets):
        # Convert from input to internal format.
        dtype = ccdOffsets.dtype.descr
        dtype.extend([('XRA', bool),
                      ('RASIGN', 'i2'),
                      ('DECSIGN', 'i2')])
        self.ccdOffsets = np.zeros(ccdOffsets.size, dtype=dtype)
        for name in ccdOffsets.dtype.names:
            self.ccdOffsets[name][:] = ccdOffsets[name][:]

        self._deltaMapper = None
        self._computedSigns = False

    def __call__(self, orientation, nstep=50):
        """
        Make a focal plane projection mapping.

        Parameters
        ----------
        orientation : `int`
            Camera "orientation".  Unused for FocalPlaneProjectorFromOffsets
            which does not support arbitrary orientations.
        nstep : `int`
            Number of steps in x/y per detector for the mapping.

        Returns
        -------
        projectionMapping : `np.darray`
            A projection mapping object with x, y, x_size, y_size,
            delta_ra_cent, delta_dec_cent, delta_ra, delta_dec for
            each detector id.
        """
        if self._deltaMapper is not None:
            # Return cached value
            return self._deltaMapper

        if not self._computedSigns:
            raise RuntimeError("Cannot make a projection mapping without first "
                               "running computeCCDOffsetSigns")

        deltaMapper = np.zeros(len(self.ccdOffsets), dtype=[('id', 'i4'),
                                                            ('x', 'f8', nstep**2),
                                                            ('y', 'f8', nstep**2),
                                                            ('x_size', 'i4'),
                                                            ('y_size', 'i4'),
                                                            ('delta_ra_cent', 'f8'),
                                                            ('delta_dec_cent', 'f8'),
                                                            ('delta_ra', 'f8', nstep**2),
                                                            ('delta_dec', 'f8', nstep**2)])
        deltaMapper['id'] = np.arange(len(self.ccdOffsets))

        for i in range(len(self.ccdOffsets)):
            xSize = int(self.ccdOffsets[i]['X_SIZE'])
            ySize = int(self.ccdOffsets[i]['Y_SIZE'])

            xValues = np.linspace(0.0, xSize, nstep)
            yValues = np.linspace(0.0, ySize, nstep)

            deltaMapper['x'][i, :] = np.repeat(xValues, yValues.size)
            deltaMapper['y'][i, :] = np.tile(yValues, xValues.size)
            deltaMapper['x_size'][i] = xSize
            deltaMapper['y_size'][i] = ySize

            deltaMapper['delta_ra_cent'][i] = self.ccdOffsets[i]['DELTA_RA']
            deltaMapper['delta_dec_cent'][i] = self.ccdOffsets[i]['DELTA_DEC']

            # And translate these into delta_ra, delta_dec
            raValues = np.linspace(self.ccdOffsets['DELTA_RA'][i] -
                                   self.ccdOffsets['RASIGN'][i]*self.ccdOffsets['RA_SIZE'][i]/2.,
                                   self.ccdOffsets['DELTA_RA'][i] +
                                   self.ccdOffsets['RASIGN'][i]*self.ccdOffsets['RA_SIZE'][i]/2.,
                                   nstep)
            decValues = np.linspace(self.ccdOffsets['DELTA_DEC'][i] -
                                    self.ccdOffsets['DECSIGN'][i]*self.ccdOffsets['DEC_SIZE'][i]/2.,
                                    self.ccdOffsets['DELTA_DEC'][i] +
                                    self.ccdOffsets['DECSIGN'][i]*self.ccdOffsets['DEC_SIZE'][i]/2.,
                                    nstep)

            if not self.ccdOffsets['XRA'][i]:
                # Swap axes
                deltaMapper['delta_ra'][i, :] = np.tile(raValues, decValues.size)
                deltaMapper['delta_dec'][i, :] = np.repeat(decValues, raValues.size)
            else:
                deltaMapper['delta_ra'][i, :] = np.repeat(raValues, decValues.size)
                deltaMapper['delta_dec'][i, :] = np.tile(decValues, raValues.size)
        self._deltaMapper = deltaMapper

        return deltaMapper

    def computeCCDOffsetSigns(self, fgcmStars):
        """Compute plotting signs for x/y to ra/dec conversions.

        This does not support rotations; you need a proper
        focalPlaneProjector for that (only supported via lsst dm).

        Parameters
        ----------
        fgcmStars : `fgcmStars`
        """
        import scipy.stats
        import esutil
        from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

        obsObjIDIndex = snmm.getArray(fgcmStars.obsObjIDIndexHandle)
        obsCCDIndex = snmm.getArray(fgcmStars.obsCCDHandle) - fgcmStars.ccdStartIndex
        obsExpIndex = snmm.getArray(fgcmStars.obsExpIndexHandle)

        obsX = snmm.getArray(fgcmStars.obsXHandle)
        obsY = snmm.getArray(fgcmStars.obsYHandle)
        objRA = snmm.getArray(fgcmStars.objRAHandle)
        objDec = snmm.getArray(fgcmStars.objDecHandle)

        h, rev = esutil.stat.histogram(obsCCDIndex, rev=True)

        for i in range(h.size):
            if h[i] == 0: continue

            i1a = rev[rev[i]:rev[i+1]]

            cInd = obsCCDIndex[i1a[0]]

            if self.ccdOffsets['RASIGN'][cInd] == 0:
                # choose a good exposure to work with
                hTest, revTest = esutil.stat.histogram(obsExpIndex[i1a], rev=True)
                maxInd = np.argmax(hTest)
                testStars = revTest[revTest[maxInd]:revTest[maxInd + 1]]

                testRA = objRA[obsObjIDIndex[i1a[testStars]]]
                testDec = objDec[obsObjIDIndex[i1a[testStars]]]
                testX = obsX[i1a[testStars]]
                testY = obsY[i1a[testStars]]

                corrXRA,_ = scipy.stats.pearsonr(testX, testRA)
                corrYRA,_ = scipy.stats.pearsonr(testY, testRA)

                if (np.abs(corrXRA) > np.abs(corrYRA)):
                    self.ccdOffsets['XRA'][cInd] = True
                else:
                    self.ccdOffsets['XRA'][cInd] = False

                if self.ccdOffsets['XRA'][cInd]:
                    # x is correlated with RA
                    if corrXRA < 0:
                        self.ccdOffsets['RASIGN'][cInd] = -1
                    else:
                        self.ccdOffsets['RASIGN'][cInd] = 1

                    corrYDec,_ = scipy.stats.pearsonr(testY, testDec)
                    if corrYDec < 0:
                        self.ccdOffsets['DECSIGN'][cInd] = -1
                    else:
                        self.ccdOffsets['DECSIGN'][cInd] = 1
                else:
                    # y is correlated with RA
                    if corrYRA < 0:
                        self.ccdOffsets['RASIGN'][cInd] = -1
                    else:
                        self.ccdOffsets['RASIGN'][cInd] = 1

                    corrXDec,_ = scipy.stats.pearsonr(testX, testDec)
                    if corrXDec < 0:
                        self.ccdOffsets['DECSIGN'][cInd] = -1
                    else:
                        self.ccdOffsets['DECSIGN'][cInd] = 1

        self._computedSigns = True
