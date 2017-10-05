from __future__ import print_function

import numpy as np


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

objFlagDict = {'TOO_FEW_OBS':2**0,
               'BAD_COLOR':2**1,
               'VARIABLE':2**2,
               'TEMPORARY_BAD_STAR':2**3,
               'RESERVED':2**4}

obsFlagDict = {'NO_EXPOSURE':2**0,
               'BAD_ERROR':2**1}

expFlagDict = {'TOO_FEW_STARS':2**0,
               'EXP_GRAY_TOO_NEGATIVE':2**1,
               'VAR_GRAY_TOO_LARGE':2**2,
               'TOO_FEW_EXP_ON_NIGHT':2**3,
               'NO_STARS':2**4,
               'BAND_NOT_IN_LUT':2**5,
               'TEMPORARY_BAD_EXPOSURE':2**6,
               'EXP_GRAY_TOO_POSITIVE':2**7}


zpFlagDict = {'PHOTOMETRIC_FIT_EXPOSURE':2**0,
              'PHOTOMETRIC_EXTRA_EXPOSURE':2**1,
              'NONPHOTOMETRIC_FIT_NIGHT':2**2,
              'NOFIT_NIGHT':2**3,
              'CANNOT_COMPUTE_ZEROPOINT':2**4,
              'TOO_FEW_STARS_ON_CCD':2**5}

logDict = {'NONE':0,
           'INFO':1,
           'DEBUG':2}

#def resourceUsage(where):
#    status = None
#    result = {'peak':0, 'rss':0}
#    try:
#        status = open('/proc/self/status')
#        for line in status:
#            parts = line.split()
#            key = parts[0][2:-1].lower()
#            if key in result:
#                result[key] = int(parts[1])/1000
#        print('Memory usage at %s:  %d MB current and %d MB peak.'  %(where, result['rss'], result['peak']))
#    except:
#        print('Could not find self status.')
#    return

def dataBinner(x,y,binSize,xRange,nTrial=100,xNorm=-1.0,minPerBin=5):
    """
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
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2./(2.*sigma**2))

def histoGauss(ax,array):
    """
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
        coeff,varMatrix = scipy.optimize.curve_fit(gaussFunction, hist['center'],
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
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    cm = plt.get_cmap('rainbow')
    plt.set_cmap('rainbow')

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


