from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

# Dictionary of object flags
objFlagDict = {'TOO_FEW_OBS':2**0,
               'BAD_COLOR':2**1,
               'VARIABLE':2**2,
               'TEMPORARY_BAD_STAR':2**3,
               'RESERVED':2**4,
               'REFSTAR_OUTLIER': 2**5,
               'BAD_QUANTITY': 2**6}

# Dictionary of observation flags
obsFlagDict = {'NO_EXPOSURE':2**0,
               'BAD_ERROR':2**1,
               'SUPERSTAR_OUTLIER':2**2,
               'NO_ZEROPOINT': 2**4}

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

logDict = {'NONE':0,
           'INFO':1,
           'DEBUG':2}

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

    for i in xrange(hist.size):
        if (hist[i] >= minPerBin):
            i1a=rev[rev[i]:rev[i+1]]

            binStruct['N'][i] = i1a.size

            medYs=np.zeros(nTrial,dtype='f8')
            medYWidths=np.zeros(nTrial,dtype='f8')
            medXs=np.zeros(nTrial,dtype='f8')
            medXWidths=np.zeros(nTrial,dtype='f8')

            for t in xrange(nTrial):
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

def plotCCDMap(ax, ccdOffsets, values, cbLabel, loHi=None):
    """
    Plot the map of CCDs.

    parameters
    ----------
    ax: plot axis object
    ccdOffsets: ccdOffset recarray
    values: float array
       Values for each ccd
    cbLabel: string
       Color bar label
    loHi: tuple [2], optional
       (lo, hi) or else scaling is computed from data.
    """

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    cm = plt.get_cmap('rainbow')

    plotRARange = [ccdOffsets['DELTA_RA'].min() - ccdOffsets['RA_SIZE'].max()/2.,
                   ccdOffsets['DELTA_RA'].max() + ccdOffsets['RA_SIZE'].max()/2.]
    plotDecRange = [ccdOffsets['DELTA_DEC'].min() - ccdOffsets['DEC_SIZE'].max()/2.,
                    ccdOffsets['DELTA_DEC'].max() + ccdOffsets['DEC_SIZE'].max()/2.]

    if (loHi is None):
        st=np.argsort(values)

        lo=values[st[int(0.02*st.size)]]
        hi=values[st[int(0.98*st.size)]]
    else:
        lo = loHi[0]
        hi = loHi[1]

    cNorm = colors.Normalize(vmin=lo, vmax=hi)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    Z=[[0,0],[0,0]]
    levels=np.linspace(lo,hi,num=150)
    CS3=plt.contourf(Z,levels,cmap=cm)

    ax.clear()

    ax.set_xlim(plotRARange[0]-0.05,plotRARange[1]+0.05)
    ax.set_ylim(plotDecRange[0]-0.05,plotDecRange[1]+0.05)
    ax.set_xlabel(r'$\delta\,\mathrm{R.A.}$',fontsize=16)
    ax.set_ylabel(r'$\delta\,\mathrm{Dec.}$',fontsize=16)
    ax.tick_params(axis='both',which='major',labelsize=14)

    for k in xrange(values.size):
        off=[ccdOffsets['DELTA_RA'][k],
             ccdOffsets['DELTA_DEC'][k]]

        ax.add_patch(
            patches.Rectangle(
                (off[0]-ccdOffsets['RA_SIZE'][k]/2.,
                 off[1]-ccdOffsets['DEC_SIZE'][k]/2.),
                ccdOffsets['RA_SIZE'][k],
                ccdOffsets['DEC_SIZE'][k],
                edgecolor="none",
                facecolor=scalarMap.to_rgba(values[k]))
            )

    cb=None
    cb = plt.colorbar(CS3,ticks=np.linspace(lo,hi,5))

    cb.set_label('%s' % (cbLabel), fontsize=14)

    return None

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

def plotCCDMap2d(ax, ccdOffsets, parArray, cbLabel, loHi=None):
    """
    Plot CCD map with Chebyshev fits for each CCD

    parameters
    ----------
    ax: plot axis object
    ccdOffsets: ccdOffset recarray
    parArray: float array (nCCD, nPars)
    cbLabel: string
       Color bar label
    loHi: tuple [2], optional
       (lo, hi) or else scaling is computed from data.
    """

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    cm = plt.get_cmap('rainbow')
    plt.set_cmap('rainbow')

    plotRARange = [ccdOffsets['DELTA_RA'].min() - ccdOffsets['RA_SIZE'].max()/2.,
                   ccdOffsets['DELTA_RA'].max() + ccdOffsets['RA_SIZE'].max()/2.]
    plotDecRange = [ccdOffsets['DELTA_DEC'].min() - ccdOffsets['DEC_SIZE'].max()/2.,
                    ccdOffsets['DELTA_DEC'].max() + ccdOffsets['DEC_SIZE'].max()/2.]

    # compute central values...
    centralValues = np.zeros(ccdOffsets.size)

    for i in xrange(ccdOffsets.size):
        field = Cheb2dField(ccdOffsets['X_SIZE'][i], ccdOffsets['Y_SIZE'][i], parArray[i, :])
        centralValues[i] = -2.5 * np.log10(field.evaluateCenter()) * 1000.0

    if (loHi is None):
        st=np.argsort(centralValues)

        lo = centralValues[st[int(0.02*st.size)]]
        hi = centralValues[st[int(0.98*st.size)]]
    else:
        lo = loHi[0]
        hi = loHi[1]

    cNorm = colors.Normalize(vmin=lo, vmax=hi)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    Z=[[0,0],[0,0]]
    levels=np.linspace(lo,hi,num=150)
    CS3=plt.contourf(Z,levels,cmap=cm)

    ax.clear()

    ax.set_xlim(plotRARange[0]-0.05,plotRARange[1]+0.05)
    ax.set_ylim(plotDecRange[0]-0.05,plotDecRange[1]+0.05)
    ax.set_xlabel(r'$\delta\,\mathrm{R.A.}$',fontsize=16)
    ax.set_ylabel(r'$\delta\,\mathrm{Dec.}$',fontsize=16)
    ax.tick_params(axis='both',which='major',labelsize=14)

    for k in xrange(ccdOffsets.size):
        xValues = np.linspace(0.0, ccdOffsets['X_SIZE'][i], 50)
        yValues = np.linspace(0.0, ccdOffsets['Y_SIZE'][i], 50)

        xGrid = np.repeat(xValues, yValues.size)
        yGrid = np.tile(yValues, xValues.size)

        field = Cheb2dField(ccdOffsets['X_SIZE'][i], ccdOffsets['Y_SIZE'][i], parArray[k, :])
        zGrid = -2.5 * np.log10(np.clip(field.evaluate(xGrid, yGrid), 0.1, None)) * 1000.0

        # This seems to be correct
        extent = [ccdOffsets['DELTA_RA'][k] -
                  ccdOffsets['RASIGN'][k]*ccdOffsets['RA_SIZE'][k]/2.,
                  ccdOffsets['DELTA_RA'][k] +
                  ccdOffsets['RASIGN'][k]*ccdOffsets['RA_SIZE'][k]/2.,
                  ccdOffsets['DELTA_DEC'][k] -
                  ccdOffsets['DECSIGN'][k]*ccdOffsets['DEC_SIZE'][k]/2.,
                  ccdOffsets['DELTA_DEC'][k] +
                  ccdOffsets['DECSIGN'][k]*ccdOffsets['DEC_SIZE'][k]/2.]

        zGridPlot = zGrid.reshape(xValues.size, yValues.size)
        if ccdOffsets['XRA'][k]:
            zGridPlot = zGridPlot.T

        plt.imshow(zGridPlot,
                   interpolation='bilinear',
                   origin='lower',
                   extent=extent,
                   norm=cNorm)

    cb=None
    cb = plt.colorbar(CS3,ticks=np.linspace(lo,hi,5))

    cb.set_label('%s' % (cbLabel), fontsize=14)

    return None

