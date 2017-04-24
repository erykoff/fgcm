from __future__ import print_function

"""
Adapted from Alex Drlica-Wagner
"""

import copy
import os
from collections import OrderedDict as odict

import numpy as np
import pylab as plt
import healpy
import healpy.projector

from mpl_toolkits.basemap import Basemap
from mpl_toolkits import basemap
from  mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
from mpl_toolkits.axisartist import Subplot
import mpl_toolkits.axisartist as axisartist
import  mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class FormatterFgcm(angle_helper.FormatterDMS):
    def __call__(self, direction, factor, values):
        values = np.asarray(values)
        ss = np.where(values>=0, 1, -1)
        values = np.mod(np.abs(values),360)
        values -= 360*(values > 180)
        return [self.fmt_d % (s*int(v),) for (s, v) in zip(ss, values)]

class FgcmBasemap(Basemap):
    """
    """

    def __init__(self, *args, **kwargs):
        self.lonRef = kwargs.pop('lonRef',0.0)
        self.raMin = kwargs.pop('raMin',-180.0)
        self.raMax = kwargs.pop('raMax',180.0)
        self.decMin = kwargs.pop('decMin',-90.0)
        self.decMax = kwargs.pop('decMax',90.0)

        defaults = dict(projection='mbtfpq', lon_0=self.lonRef,
                        rsphere=1.0, celestial=True)
        self.set_defaults(kwargs,defaults)
        super(FgcmBasemap,self).__init__(*args, **kwargs)


    @basemap._transform1d
    def annotate(self, x, y, text, *args, **kwargs):
        defaults = dict(zorder=10,fontsize=12,
                        bbox=dict(boxstyle='round,pad=0',fc='w',ec='none',
                                  alpha=0.25))
        self.set_defaults(kwargs,defaults)

        ax = kwargs.pop('ax', None) or self._check_ax()
        # allow callers to override the hold state by passing hold=True|False
        b = ax.ishold()
        h = kwargs.pop('hold',None)
        if h is not None: ax.hold(h)

        try:
            ret = ax.annotate(s=text, xy=(x,y), *args, **kwargs)
        except:
            raise
        finally:
            ax.hold(b)
        return ret

    def label(self,labels,**kwargs):
        defaults = dict(latlon=True)
        self.set_defaults(kwargs,defaults)
        for k,v in labels.items():
            kw = copy.deepcopy(kwargs)
            v['text'] = k
            self.set_defaults(kw,v)
            self.annotate(**kw)

    def draw_hpxmap(self, hpxmap, xsize=800, **kwargs):
        """
        Use pcolormesh to draw healpix map
        """
        if not isinstance(hpxmap,np.ma.MaskedArray):
            mask = ~np.isfinite(hpxmap) | (hpxmap==healpy.UNSEEN)
            hpxmap = np.ma.MaskedArray(hpxmap,mask=mask)

        vmin,vmax = np.percentile(hpxmap.compressed(),[0.1,99.9])

        print(vmin,vmax)

        defaults = dict(latlon=True, rasterized=True, vmin=vmin, vmax=vmax)
        self.set_defaults(kwargs,defaults)

        ax = plt.gca()

        lon = np.linspace(0, 360., xsize) 
        lat = np.linspace(-90., 90., xsize) 
        lon, lat = np.meshgrid(lon, lat)

        nside = healpy.get_nside(hpxmap.data)
        theta = np.radians(90.0-lat)
        phi = np.radians(lon)
        pix = healpy.ang2pix(nside, theta,phi)

        values = hpxmap[pix]
        im = self.pcolormesh(lon,lat,values,**kwargs)

        return im

    def draw_inset_colorbar(self,format=None):
        ax = plt.gca()
        im = plt.gci()
        cax = inset_axes(ax, width="25%", height="5%", loc=7)
        cmin,cmax = im.get_clim()
        cmed = (cmax+cmin)/2.
        delta = (cmax-cmin)/10.

        ticks = np.array([cmin+delta,cmed,cmax-delta])
        tmin = np.min(np.abs(ticks[0]))
        tmax = np.max(np.abs(ticks[1]))

        if format is None:
            if (tmin < 1e-2) or (tmax > 1e3):
                format = '%.1e'
            elif (tmin > 0.1) and (tmax < 100):
                format = '%.1f'
            elif (tmax > 100):
                format = '%i'
            else:
                format = '%.2g'

        cbar = plt.colorbar(cax=cax,orientation='horizontal',
                            ticks=ticks,format=format)
        cax.xaxis.set_ticks_position('top')
        cax.tick_params(axis='x', labelsize=10)
        plt.sca(ax)
        return cbar

    def set_axes_limits(self, ax=None):
        if ax is None: ax = plt.gca()

        FRAME = [
            [self.raMin,self.raMin,self.raMax,self.raMax],
            [self.decMin,self.decMax,self.decMin,self.decMax],
            ]
        #ax.set_xlim(self.raMin,self.raMax)
        #ax.set_ylim(self.decMin,self.decMax)
        #ax.grid(True)
        #return ax.get_xlim(),ax.get_ylim()

        x,y = self(*FRAME)
        #print(min(x),max(x))
        #print(min(y),max(y))
        #print("or")
        #print(self.raMin,self.raMax)
        #print(self.decMin,self.decMax)

        ax.set_xlim(min(x),max(x))
        ax.set_ylim(min(y),max(y))
        ax.grid(True)
        return ax.get_xlim(),ax.get_ylim()


    def create_axes(self,rect=111):
        """
        Create a special AxisArtist to overlay grid coordinates.

        Much of this taken from the examples here:
        http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html
        """

        # from curved coordinate to rectlinear coordinate.
        def tr(x, y):
            x, y = np.asarray(x), np.asarray(y)
            return self(x,y)

        # from rectlinear coordinate to curved coordinate.
        def inv_tr(x,y):
            x, y = np.asarray(x), np.asarray(y)
            return self(x,y,inverse=True)

        # Cycle the coordinates
        extreme_finder = angle_helper.ExtremeFinderCycle(20, 20)

        # Find a grid values appropriate for the coordinate.
        # The argument is a approximate number of grid lines.
        grid_locator1 = angle_helper.LocatorD(8,include_last=False)
        grid_locator2 = angle_helper.LocatorD(6,include_last=False)

        # Format the values of the grid
        tick_formatter1 = FormatterFgcm()
        tick_formatter2 = angle_helper.FormatterDMS()

        grid_helper = GridHelperCurveLinear((tr, inv_tr),
                                            extreme_finder=extreme_finder,
                                            grid_locator1=grid_locator1,
                                            grid_locator2=grid_locator2,
                                            tick_formatter1=tick_formatter1,
                                            tick_formatter2=tick_formatter2,
        )

        fig = plt.gcf()
        ax = axisartist.Subplot(fig,rect,grid_helper=grid_helper)
        fig.add_subplot(ax)

        ax.axis['left'].major_ticklabels.set_visible(True)
        ax.axis['right'].major_ticklabels.set_visible(True)
        ax.axis['bottom'].major_ticklabels.set_visible(True)
        ax.axis['top'].major_ticklabels.set_visible(True)

        ax.set_xlabel("Right Ascension")
        ax.set_ylabel("Declination")

        return fig,ax

    def set_defaults(self,kwargs,defaults):
        for k,v in defaults.items():
            kwargs.setdefault(k,v)
        return kwargs


