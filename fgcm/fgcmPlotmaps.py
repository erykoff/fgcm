import os
import matplotlib
try:             os.environ['DISPLAY']
except KeyError: matplotlib.use('Agg')
from collections import OrderedDict as odict

import fitsio
import numpy as np
import scipy.ndimage as nd
import pylab as plt
import matplotlib.colors as colors
import healpy
import esutil

from mpl_toolkits.axisartist import Subplot
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from fgcmBasemap import FgcmBasemap

def set_cmap(name='viridis'):
    try:
        import matplotlib.colormaps as cmaps
        plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    except ImportError:
        # this is the local version
        import colormaps as cmaps
        plt.register_cmap(name='viridis',cmap=cmaps.viridis)
    finally:
        plt.set_cmap(name)
        # For some reason set_cmap creates a figure, so close it.
        plt.close(plt.gcf())

def draw_peak(peak,**kwargs):
    kwargs.setdefault('ls','--')
    kwargs.setdefault('label','%.1f '%(peak))
    ax = plt.gca()
    ax.axvline(peak,**kwargs)

def draw_hist(skymap,fit_gaussian=False,**kwargs):
    ax = plt.gca()

    if isinstance(skymap,np.ma.MaskedArray):
        pix = np.where(~skymap.mask)
    else:
        pix = np.where((np.isfinite(skymap)) & (skymap != healpy.UNSEEN))

    data = skymap[pix]

    vmin = kwargs.pop('vmin',np.percentile(data,q=1.0))
    vmax = kwargs.pop('vmax',np.percentile(data,q=99.0))
    nbins = kwargs.pop('nbins',100)
    defaults = dict(bins=np.linspace(vmin,vmax,nbins),
                    histtype='step',normed=True,lw=1.5,
                    peak=False,quantiles=False)
    set_defaults(kwargs,defaults)

    do_peak = kwargs.pop('peak')
    do_quantiles = kwargs.pop('quantiles')

    n,b,p = ax.hist(data,**kwargs)
    ret = dict()

    #peak = ((b[1:]+b[:-1])/2.)[np.argmax(n)]
    peak = np.median(data)
    ret['peak'] = peak
    if do_peak:
        draw_peak(peak,color='k',label='%.1f'%(peak))

    ret['mean'] = np.mean(data)
    ret['std']  = np.std(data)

    quantiles = [5,16,50,84,95]
    percentiles = np.percentile(data,quantiles)
    ret['quantiles']   = quantiles
    ret['percentiles'] = percentiles
    for p,q in zip(percentiles,quantiles):
        ret['q%02d'%q] = p

    if do_quantiles:
        for q,p in zip(quantiles,percentiles):
            draw_peak(p,color='r',label='%.1f (%g%%)'%(p,100-q))

    if (fit_gaussian):
        #import esutil
        import scipy.optimize

        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2./(2.*sigma**2))

        p0=[data.size, ret['mean'], ret['std']]

        hist_fit_x = (np.array(b[0:-1])+np.array(b[1:]))/2.
        hist_fit_y = np.array(n)

        coeff,var_matrix = scipy.optimize.curve_fit(gauss, hist_fit_x, hist_fit_y, p0=p0)

        xvals=np.linspace(-5*coeff[2],5*coeff[2],1000)
        yvals=gauss(xvals,*coeff)

        ax.plot(xvals,yvals,'k--',linewidth=3)
        ret['gauss_norm'] = coeff[0]
        ret['gauss_mean'] = coeff[1]
        ret['gauss_sigma'] = coeff[2]


    ax.set_xlim(kwargs['bins'].min(),kwargs['bins'].max())
    return ret

def plot_hpxmap_hist(hpxmap,raRange=[-180,180],decRange=[-90,90],lonRef=0.0,
                     cbar_kwargs=dict(),hpxmap_kwargs=dict(),
                     hist_kwargs=dict(),fit_gaussian=False):

    hist_defaults = dict(peak=True)
    set_defaults(hist_kwargs,hist_defaults)

    if isinstance(hpxmap,basestring):
        hpxmap = healpy.read_map(f)

    fig = plt.figure(10,figsize=(12,4))
    fig.clf()
    gridspec=plt.GridSpec(1, 3)

    bmap = FgcmBasemap(lonRef=lonRef,raMin=raRange[0],raMax=raRange[1],
                       decMin=decRange[0],decMax=decRange[1])
    bmap.create_axes(rect=gridspec[0:2])
    im = bmap.draw_hpxmap(hpxmap,**hpxmap_kwargs)
    bmap.draw_inset_colorbar(**cbar_kwargs)
    ax1 = plt.gca()
    ax1.axis['right'].major_ticklabels.set_visible(False)
    ax1.axis['top'].major_ticklabels.set_visible(False)

    ax2 = Subplot(fig,gridspec[2])
    fig.add_subplot(ax2)
    plt.sca(ax2)
    ret = draw_hist(hpxmap,fit_gaussian=fit_gaussian,**hist_kwargs)
    ax2.yaxis.set_major_locator(MaxNLocator(6,prune='both'))
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    ax2.axis['left'].major_ticklabels.set_visible(False)
    ax2.axis['right'].major_ticklabels.set_visible(True)
    ax2.axis['right'].label.set_visible(True)
    ax2.axis['right'].label.set_text(r'Normalized Area (a.u.)')
    ax2.axis['bottom'].label.set_visible(True)

    plt.subplots_adjust(bottom=0.15,top=0.95)

    return fig,[ax1,ax2],ret

def plot_hpxmap(hpxmap,raRange=[-180,180],decRange=[-90,90],lonRef=0.0,
                cbar_kwargs=dict(),hpxmap_kwargs=dict()):
    if isinstance(hpxmap,basestring):
        hpxmap = healpy.read_map(f)

    fig = plt.figure(10,figsize=(8,4))
    fig.clf()
    gridspec=plt.GridSpec(1,2)

    bmap = FgcmBasemap(lonRef=lonRef,raMin=raRange[0],raMax=raRange[1],
                       decMin=decRange[0],decMax=decRange[1])
    bmap.create_axes(rect=gridspec[0:2])
    im = bmap.draw_hpxmap(hpxmap,**hpxmap_kwargs)
    bmap.draw_inset_colorbar(**cbar_kwargs)
    ax = plt.gca()
    ax.axis['right'].major_ticklabels.set_visible(False)
    ax.axis['top'].major_ticklabels.set_visible(False)

    fig.subplots_adjust(bottom=0.15,top=0.95)

    return fig,ax

def set_defaults(kwargs,defaults):
    for k,v in defaults.items():
        kwargs.setdefault(k,v)
    return kwargs
