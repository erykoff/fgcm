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
               'RESERVED':2**4}

# Dictionary of observation flags
obsFlagDict = {'NO_EXPOSURE':2**0,
               'BAD_ERROR':2**1}

# Dictionary of exposure flags
expFlagDict = {'TOO_FEW_STARS':2**0,
               'EXP_GRAY_TOO_NEGATIVE':2**1,
               'VAR_GRAY_TOO_LARGE':2**2,
               'TOO_FEW_EXP_ON_NIGHT':2**3,
               'NO_STARS':2**4,
               'BAND_NOT_IN_LUT':2**5,
               'TEMPORARY_BAD_EXPOSURE':2**6,
               'EXP_GRAY_TOO_POSITIVE':2**7}

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
    array: float array to plot

    returns
    -------
    coeff: tuple (A, mu, sigma)
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
    except:
        # set to starting values...
        coeff = p0

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

    ax.plot(hcenter[ok],hhist[ok],'b-',linewidth=3)
    ax.set_xlim(rangeLo,rangeHi)

    xvals=np.linspace(rangeLo,rangeHi,1000)
    yvals=gaussFunction(xvals,*coeff)

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

def poly2dFunc(xy, p0, p1, p2, p3, p4, p5):
    """
    2d polynomial fitting function.  Up to 2nd order, higher orders dropped.

    parameters
    ----------
    xy: numpy vstack (2,nvalues)
    p0: Const
    p1: x term
    p2: y term
    p3: x**2 term
    p4: y**2 term
    p5: x*y term
    """

    return p0 + p1*xy[0,:] + p2*xy[1,:] + p3*xy[0,:]**2. + p4*xy[1,:]**2. + p5*xy[0,:]*xy[1,:]

def cheb2dFunc(xy, *cpars):
    """
    2d Chebyshev polynomial fitting function.

    parameters
    ----------
    xy: numpy vstack (2, nvalues)
    *cpars: Chebyshev parameters

    Note that the degree of the polynomials in inferred from the number of *cpars.
    The array is reshaped to hand to np.polynomial.chebyshev.chebval2d

    returns
    -------
    Value of function evaluated at xy[0, :], xy[1, :]
    """

    degplus1 = int(np.sqrt(len(cpars)))
    c = np.array(cpars).reshape(degplus1, degplus1)

    return np.polynomial.chebyshev.chebval2d(xy[0, :], xy[1, :], c)


def plotCCDMap2d(ax, ccdOffsets, parArray, cbLabel, loHi=None, usePoly2d=False):
    """
    Plot CCD map with Chebyshev or polynomial fits for each CCD

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
        xy = np.vstack((ccdOffsets['X_SIZE'][i]/2.,
                        ccdOffsets['Y_SIZE'][i]/2.))
        centralValues[i] = poly2dFunc(xy, *parArray[i,:])

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
        xRange = np.array([0, ccdOffsets['X_SIZE'][k]])
        yRange = np.array([0, ccdOffsets['Y_SIZE'][k]])

        xValues = np.arange(xRange[0], xRange[1], 50)
        yValues = np.arange(yRange[0], yRange[1], 50)

        xGrid = np.repeat(xValues, yValues.size)
        yGrid = np.tile(yValues, xValues.size)

        if usePoly2d:
            zGrid = poly2dFunc(np.vstack((xGrid, yGrid)),
                               *parArray[k, :])
        else:
            zGrid = cheb2dFunc(np.vstack((xGrid, yGrid)),
                               *parArray[k, :])

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
