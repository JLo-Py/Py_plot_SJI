# Juraj Lorincik, LMSAL, Jan 27 - 30 2026

import irispy.io 

import numpy as np
import dateutil.parser as parser
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface
import sunpy.map as sm
from astropy.time import Time

import astropy.units as u 
from matplotlib.ticker import (MultipleLocator)

def diffrot(in_map, out_time):
    
    out_frame = Helioprojective(observer='earth', obstime=out_time, rsun=in_map.coordinate_frame.rsun)
    out_center = SkyCoord(in_map.reference_coordinate.Tx, in_map.reference_coordinate.Ty, frame=out_frame)
    out_header = sm.make_fitswcs_header(in_map.data.shape, out_center, scale=u.Quantity(in_map.scale))
    out_wcs = WCS(out_header)
    
    with propagate_with_solar_surface():
        out_map = in_map.reproject_to(out_wcs)
        
    return out_map

def decode(isotime):
    if isinstance(isotime, (bytes, np.bytes_)):
        isotime = isotime.decode("utf-8")
    outtime = parser.parse(isotime).isoformat()
    
    return outtime

def _load_sjimap_fIDL(sji_in):
    
    data    = sji_in['DATA']
    XC      = float(sji_in['XC'])
    YC      = float(sji_in['YC'])
    time = decode(sji_in['TIME'])

    nx = data.shape[1]
    ny = data.shape[0]

    ds = float(sji_in['DX'])
    
    xl = XC-ds*nx/2-ds/2
    xr = XC+ds*nx/2+ds/2
    yt = YC+ds*ny/2+ds/2
    yb = YC-ds*ny/2-ds/2

    extent = [xl, xr, yt, yb]
        
    return data, time, extent 

def _get_sjimap_extent(sjimap):
        
    sx, sy = sjimap.scale       
    ds = sx.to(u.arcsec/u.pix)      # sx = sy for IRIS/SJI
    ds = ds.value
    
    xl = float(sjimap.bottom_left_coord.Tx.value)   -ds/2
    xr = float(sjimap.top_right_coord.Tx.value)     +ds/2
    yt = float(sjimap.top_right_coord.Ty.value)     +ds/2
    yb = float(sjimap.bottom_left_coord.Ty.value)   -ds/2
    
    extent = [xl, xr, yb, yt]
    
    return extent 

def plot_sji(sjimap, show_time = True, log = True, mode = 'extent', **kwargs):
                
    cmap = kwargs.get('cmap', 'irissjiFUV')
    
    sjidata = sjimap.data
    sjitime = sjimap.date.value
    # sjitime = Time(sjimap.meta['date-obs'], scale = 'utc', format = 'isot')
    sjiextent = _get_sjimap_extent(sjimap)
    
    sat = kwargs.get('sat', [np.nanmin(sjidata [sjidata > 0])*5, np.nanmax(sjidata)*0.5])

    fig = plt.figure(figsize = kwargs.get('figsize', (8, 8)), dpi = kwargs.get('dpi', 60))
    
    if mode == 'extent':
        ax = plt.axes(kwargs.get('position', [0.15, 0.15, 0.8, 0.8]))
        
        if log:
            im = ax.imshow(sjidata, cmap = cmap, norm = colors.LogNorm(sat[0], sat[1]), extent = sjiextent, origin = 'lower')
        else: 
            im = ax.imshow(sjidata, cmap = cmap, vmin = sat[0], vmax = sat[1], extent = sjiextent, origin = 'lower')
        
        if 'xrange' in kwargs:
            ax.set_xlim(kwargs['xrange'])
        if 'yrange' in kwargs:
            ax.set_ylim(kwargs['yrange'])
            
        if 'tickinterval' in kwargs:
            ax.xaxis.set_major_locator(MultipleLocator(kwargs['tickinterval']))
            ax.yaxis.set_major_locator(MultipleLocator(kwargs['tickinterval']))
        
        plt.tick_params(axis = 'both', length = 5, top = True, right = True)
        
    else:
        ax = plt.axes(kwargs.get('position', [0.15, 0.15, 0.8, 0.8]), projection = sjimap)
        
        if log:
            im = ax.imshow(sjidata, cmap = cmap, norm = colors.LogNorm(sat[0], sat[1]))
        else: 
            im = ax.imshow(sjidata, cmap = cmap, vmin = sat[0], vmax = sat[1])      
              
        if 'xrange' in kwargs:
                        
            xlims_world = kwargs['xrange']*u.arcsec
            ylims_world = kwargs['yrange']*u.arcsec

            world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame = sjimap.coordinate_frame)
            pixel_coords = sjimap.world_to_pixel(world_coords)

            xlims_pixel = pixel_coords.x.value
            ylims_pixel = pixel_coords.y.value
                        
            ax.set_xlim(xlims_pixel)
            ax.set_ylim(ylims_pixel)
            
        ax.coords[0].set_major_formatter("s", show_decimal_unit=False)
        ax.coords[1].set_major_formatter("s", show_decimal_unit=False)

        if 'tickinterval' in kwargs:
            ax.coords[0].set_ticks(spacing=kwargs['tickinterval'] * u.arcsec)
            ax.coords[1].set_ticks(spacing=kwargs['tickinterval'] * u.arcsec)

        plt.tick_params(axis = 'both', length = 5)
        
    ax.set_xlabel('Solar X [arc sec]')
    ax.set_ylabel('Solar Y [arc sec]')

    if show_time:
        ax.text(0.03, 0.03, sjitime[11:19]+' UT', transform = ax.transAxes, color = 'white')

    return fig, ax, im

def plot_sji_fIDL(sjimap, show_time = True, log = True, **kwargs):
            
    sjidata, sjitime, sjiextent = _load_sjimap_fIDL(sjimap)
    
    cmap = kwargs.get('cmap', 'irissjiFUV')
    
    fig = plt.figure(figsize = kwargs.get('figsize', (8, 8)), dpi = kwargs.get('dpi', 60))
    ax = plt.axes(kwargs.get('position', [0.15, 0.15, 0.8, 0.8]))

    sat = kwargs.get('sat', [np.nanmin(sjidata [sjidata > 0])*5, np.nanmax(sjidata)*0.5])

    if log:
        im = ax.imshow(sjidata, cmap = cmap, norm = colors.LogNorm(sat[0], sat[1]), extent = sjiextent)
    else: 
        im = ax.imshow(sjidata, cmap = cmap, vmin = sat[0], vmax = sat[1], extent = sjiextent)

    if 'xrange' in kwargs:
        ax.set_xlim(kwargs['xrange'])
    else: 
        ax.set_xlim(sjiextent[0], sjiextent[1])
        
    if 'yrange' in kwargs:
        ax.set_ylim(kwargs['yrange'])
    else: 
        ax.set_ylim(sjiextent[3], sjiextent[2])

    ax.set_xlabel('Solar X [arc sec]')
    ax.set_ylabel('Solar Y [arc sec]')

    if show_time:
        ax.text(0.03, 0.03, sjitime[11:19]+' UT', transform = ax.transAxes, color = 'white')

    return fig, ax, im
