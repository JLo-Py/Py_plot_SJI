# Juraj Lorincik, LMSAL, Jan 27 - 30 2026

import irispy.io 

import numpy as np
import dateutil.parser as parser
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.coordinates import SkyCoord
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

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    def norm(self):
        return self.dot(self)**0.5
    def normalized(self):
        norm = self.norm()
        return Vector(self.x / norm, self.y / norm)
    def perp(self):
        return Vector(1, -self.x / self.y)
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    def __str__(self):
        return f'({self.x}, {self.y})'


def _tick_vector(vector, point, tick_len):
    
    A = Vector(vector[0, 0], vector[0, 1])
    B = Vector(vector[1, 0], vector[1, 1])
    C = Vector(point[0], point[1])

    AB = B - A  

    AB_perp_normed = AB.perp().normalized()

    P1 = C + AB_perp_normed * tick_len
    P2 = C - AB_perp_normed * tick_len
    
    return [P1.x, P2.x], [P1.y, P2.y]

def _get_ticks(xt_scoord, cut_xi, cut_yi, interval = 10, tick_len = 10):

    scmax = int(np.nanmax(xt_scoord))

    ticks = np.arange(0, scmax, interval)
    nticks = len(ticks)

    tickind = np.zeros((nticks))

    for iti in range(nticks):
        tickind[iti] = np.argmin(abs(xt_scoord - ticks[iti]))

    tickind = tickind.astype('int')

    tick_x = cut_xi[tickind]
    tick_y = cut_yi[tickind]

    node_vec_x = np.zeros((2, nticks))
    node_vec_y = np.zeros((2, nticks))

    for iti in range(nticks):
        if tickind[iti] == 0:
            node_vec_x[0, iti] = cut_xi[0]
            node_vec_x[1, iti] = cut_xi[1]
            node_vec_y[0, iti] = cut_yi[0]
            node_vec_y[1, iti] = cut_yi[1]
        else: 
            node_vec_x[0, iti] = cut_xi[tickind[iti]-1]
            node_vec_x[1, iti] = cut_xi[tickind[iti]+1]
            node_vec_y[0, iti] = cut_yi[tickind[iti]-1]
            node_vec_y[1, iti] = cut_yi[tickind[iti]+1]     
            
    node_vec = np.zeros((nticks, 2, 2))
    tick_coord = np.zeros((nticks, 2))  

    for iti in range(nticks): 
        node_vec[iti, 0, :] = [node_vec_x[0, iti], node_vec_y[0, iti]]
        node_vec[iti, 1, :] = [node_vec_x[1, iti], node_vec_y[1, iti]]
        
    for iti in range(nticks): 
        tick_coord[iti, :] = [tick_x[iti], tick_y[iti]]
        
    tick_vec = np.zeros((nticks, 2, 2)) 

    for iti in range(nticks): 
        uu = node_vec[iti, :, :]
        vv = tick_coord[iti, :]
        tick_vec[iti, :, :] = _tick_vector(uu, vv, tick_len = tick_len)
        
    return ticks, tick_x, tick_y, tick_vec

# always get the cut including the full extent

def _get_cut(xt_xap, xt_yap):
    
    n_perpsteps = xt_xap.shape[0]
    
    if n_perpsteps % 2 == 1:
        center = int(n_perpsteps/2)
        center_cut_xi = xt_xap[center, :]
        center_cut_yi = xt_yap[center, :]        
    else:
        raise ValueError('Even number of perpendicular steps. Case not yet coded, interpolation needed to get central cut.')
      
    edge1_cut_xi = xt_xap[0, :]
    edge1_cut_yi = xt_yap[0, :]
        
    edge2_cut_xi = xt_xap[-1, :]
    edge2_cut_yi = xt_yap[-1, :]
        
    cut_xi = np.vstack((center_cut_xi, edge1_cut_xi, edge2_cut_xi))
    cut_yi = np.vstack((center_cut_yi, edge1_cut_yi, edge2_cut_yi))
        
    return cut_xi, cut_yi

def plot_cut(axis, xt_scoord, xt_xap, xt_yap, mode = 'extent', **kwargs):
    
    abi = kwargs.get('abi', 10)
    arr_param = kwargs.get('kwargs', [0.1, 3, 3, 'cyan'])
    interval = kwargs.get('interval', 10)
    tick_len = kwargs.get('tick_len', 1)   
    
    cut_xi, cut_yi = _get_cut(xt_xap, xt_yap)
    ncoord = cut_xi.shape[1]
    
    # index 0 here for cut_xi so that ticks are plotted across the central (not edge) cut 
    
    ticks, tick_x, tick_y, tick_vec = _get_ticks(xt_scoord, cut_xi[0], cut_yi[0], interval = interval, tick_len = tick_len)

    print(type(tick_x))

    if 'xtick_shift' in kwargs: 
        xtsh = kwargs['xtick_shift']
        tick_x += xtsh
        
    if 'ytick_shift' in kwargs:
        ytsh = kwargs['ytick_shift']
        tick_y += ytsh
    
    axis.plot(cut_xi[0][:-abi+1], cut_yi[0][:-abi+1], '--', color=arr_param[3], linewidth = arr_param[1])

    dx = cut_xi[0][ncoord - abi -1] - cut_xi[0][ncoord -1]
    dy = cut_yi[0][ncoord - abi -1] - cut_yi[0][ncoord -1]

    axis.arrow(cut_xi[0][ncoord-abi], cut_yi[0][ncoord-abi], -dx, -dy, length_includes_head = True, width=arr_param[0], head_width = arr_param[1], head_length = arr_param[2], color=arr_param[3])

    for iti in range(len(ticks)): 
        axis.plot(tick_vec[iti, 0, :], tick_vec[iti, 1, :], color = arr_param[3], linewidth = arr_param[1])
        axis.text(tick_x[iti], tick_y[iti]-3, str(ticks[iti]), color = arr_param[3], weight='bold')
        
    if mode == 'extent':
        axis.plot(cut_xi[1][:-abi+1], cut_yi[1][:-abi+1], ':', color=arr_param[3], linewidth = arr_param[1]-1)
        axis.plot(cut_xi[2][:-abi+1], cut_yi[2][:-abi+1], ':', color=arr_param[3], linewidth = arr_param[1]-1)