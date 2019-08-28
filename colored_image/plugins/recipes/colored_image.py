# -*- coding: utf-8 -*-
"""
Created on Tue Apr 02 14:22:15 2019

@author: kkc29


based off PYME recipes processing.py
"""



from PYME.recipes.base import ModuleBase, register_module, Filter
from PYME.recipes.traits import Input, Output, Float, Enum, CStr, Bool, Int, List, File

import numpy as np
from scipy import ndimage, optimize, signal, interpolate, stats
from PYME.IO.image import ImageStack
from PYME.IO.dataWrap import ListWrap
from PYME.IO.MetaDataHandler import NestedClassMDHandler

from PYME.recipes import processing
import time, multiprocessing
import matplotlib
from matplotlib import pyplot

import logging
logger=logging.getLogger(__name__)

from copy import copy
from collections import namedtuple


BinnedStatisticddResult = namedtuple('BinnedStatisticddResult',
                                 ('statistic', 'bin_edges',
                                  'binnumber'))

@register_module('GenerateImage')
class GenerateImage(ModuleBase):
    """

    """
    
    input_locs = Input('Localizations')
    
    number_of_processes = Int(1)
    render_method = Enum(["Gaussian", "Histogram"])
    
    axis_one = CStr("x")
    axis_one_pixel_size = Float(50.0)
    axis_one_range = List(Float, [0, 20000], 2, 2)
    axis_one_range_auto_minmax = Bool(True)
    axis_two = CStr("y")
    axis_two_pixel_size = Float(50.0)
    axis_two_range = List(Float, [0, 20000], 2, 2)
    axis_two_range_auto_minmax = Bool(True)
    
    axis_color = CStr("z")
    axis_color_range = List(Float, [-500, 500], 2, 2)
    cmap = Enum((["gist_rainbow",] if "gist_rainbow" in matplotlib.pyplot.colormaps() else []) + matplotlib.pyplot.colormaps())
    color_alpha = Float(0.02)
    
    gaussian_sigma = Float(5.0)
    
    cache_path = File("")
    
    output_bw_image = Output('bw_image')
    output_color_image = Output('colored_image')
    
    def execute(self, namespace):
        from PYME.util import mProfile
        mProfile.profileOn(['colored_image.py'])
#        try:
#            z_colors._mmap.close()
#            del namespace[self.output_color_image]
#        except:
#            pass
        
        if self.number_of_processes < 1:
            proccess_count = multiprocessing.cpu_count() + self.number_of_processes
        else:
            proccess_count = self.number_of_processes
        proccess_count = np.clip(proccess_count, 1, multiprocessing.cpu_count())
        
#        print "CPU {}".format(proccess_count)
        if proccess_count != 1:            
            self._pool = multiprocessing.Pool(processes=proccess_count)
        else:
            self._pool = None
        
        pipeline = namespace[self.input_locs]
        x = pipeline[self.axis_one]
        y = pipeline[self.axis_two]
        
        if self.axis_one_range_auto_minmax:
            x_min = x.min()
            x_max = x.max()
        else:
            x_min, x_max = self.axis_one_range
        if self.axis_two_range_auto_minmax:
            y_min = y.min()
            y_max = y.max()
        else:
            y_min, y_max = self.axis_two_range
        x_bins = np.arange(x_min, x_max + 0.5*self.axis_one_pixel_size, self.axis_one_pixel_size)
        y_bins = np.arange(y_min, y_max + 0.5*self.axis_two_pixel_size, self.axis_two_pixel_size)
#        x_bins = np.arange(x.min(), x.max() + 0.5*self.axis_one_pixel_size, self.axis_one_pixel_size)
#        y_bins = np.arange(y.min(), y.max() + 0.5*self.axis_two_pixel_size, self.axis_two_pixel_size)
        
        z = pipeline[self.axis_color]
        
#        z_norm = z - self.axis_color_range[0]
#        z_norm = z / self.axis_color_range[1]
#        np.clip(z_norm, 0, 1, out=z_norm)
#        z_colors = getattr(matplotlib.cm, self.cmap)(z_norm)       
        
        cmap = copy(matplotlib.cm.get_cmap(self.cmap))
        color_x = np.linspace(0., 1., 101)
#        color_interp = interpolate.interp1d(color_x, cmap(color_x), kind='linear', axis=0, bounds_error=False, fill_value=(cmap(0), cmap(1)))
#        color_interp = interpolate.RegularGridInterpolator(color_x, cmap(color_x), method="linear", bounds_error=False, fill_value=(cmap(0), cmap(1)))
        
        color_interp = interp1d_picklable(color_x, cmap(color_x), kind='linear', axis=0, bounds_error=False, fill_value=(cmap(0), cmap(1)))
        
#        blend_before_color = False
#        if blend_before_color:
#            bin_counts = stats.binned_statistic_2d(x, y, z, statistic='count', bins=(x_bins, y_bins))[0]
#            bin_z = stats.binned_statistic_2d(x, y, z, statistic='mean', bins=(x_bins, y_bins))[0]
#    #        bin_z = GenerateImage.binned_colors([x, y], z, bins=(x_bins, y_bins))[0]
#            bin_z -= self.axis_color_range[0]
#            bin_z /= (np.diff(self.axis_color_range)[0])
#            np.clip(bin_z, 0, 1, out=bin_z)
#            
##            cmap = copy(matplotlib.cm.get_cmap(self.cmap))
#            z_colors = cmap(bin_z)
#            z_colors[np.isnan(bin_z)] = 0
#            z_colors *= (bin_counts / bin_counts.max())[:,:,None]
#        else:
        if self.render_method == "Histogram":
            z_colors = GenerateImage.binned_colors([x, y], z, bins=(x_bins, y_bins), cmap=color_interp, crange=self.axis_color_range, color_alpha=self.color_alpha, cache_path=self.cache_path, pool=self._pool)[0]
        elif self.render_method == "Gaussian":
            z_colors = GenerateImage.gauss_colors([x, y], z, bins=(x_bins, y_bins), sigma=self.gaussian_sigma, cmap=color_interp, crange=self.axis_color_range, color_alpha=self.color_alpha, cache_path=self.cache_path, pool=self._pool)
            z_colors = z_colors
        

        
        if proccess_count != 1:
            self._pool.close()
            self._pool.join()
        
#        try:
#            del z_colors
#            del self._pool
#        except:
#            pass        
            
        try:
            mdh = NestedClassMDHandler()
            mdh['Rendering.Method'] = self.render_method
            if 'imageID' in pipeline.mdh.getEntryNames():
                mdh['Rendering.SourceImageID'] = pipeline.mdh['imageID']
            mdh['Rendering.SourceFilename'] = getattr(pipeline, 'filename', '')
            if 'DataFileID' in pipeline.mdh.getEntryNames():
                mdh['Rendering.SourceDataFileID'] = pipeline.mdh['DataFileID']
            
            mdh['Rendering.X'] = self.axis_one
            mdh['Rendering.Y'] = self.axis_two
            mdh['Rendering.C'] = self.axis_color
            mdh['Rendering.XBounds'] = [x_min, x_max]
            mdh['Rendering.YBounds'] = [y_min, y_max]
            mdh['Rendering.CBounds'] = self.axis_color_range
            mdh['Rendering.CMap'] = self.cmap
            mdh['Rendering.CAlpha'] = self.color_alpha
            if self.render_method == "Gaussian":
                mdh['Rendering.Sigma'] = self.gaussian_sigma
            
            mdh['voxelsize.x'] = self.axis_one_pixel_size * 1E-3
            mdh['voxelsize.y'] = self.axis_two_pixel_size * 1E-3
            mdh['voxelsize.z'] = 0
        except Exception as e:
            raise e
            
#        namespace[self.output_bw_image] = ImageStack(bin_counts)
        namespace[self.output_color_image] = ImageStack(z_colors[:,:,None,:], mdh=mdh)
        
        mProfile.profileOff()
        mProfile.report()
        
        import gc
        gc.collect()
    
    @staticmethod
    def gauss_colors(sample, values, bins, sigma, cmap, crange, color_alpha, cache_path, pool):
#        im = np.zeros((len(bins[0]), len(bins[1]), 4), dtype=np.float)
        if cache_path == "":
            im = np.zeros((len(bins[0]), len(bins[1]), 4), dtype=np.float)
        else:
            try:
                im = np.memmap(cache_path, dtype=np.float, mode='w+', shape=(len(bins[0]), len(bins[1]), 4))
            except:
                 im = np.memmap(cache_path, dtype=np.float, mode='r+', shape=(len(bins[0]), len(bins[1]), 4)) 
                 
        roiSize = np.ceil(4 * sigma / min(bins[0][1]-bins[0][0], bins[1][1]-bins[1][0])).astype(np.int)
        
        z_sorting = np.flipud(np.argsort(values))
        x = sample[0][z_sorting]
        y = sample[1][z_sorting]
        
        z_color = values[z_sorting]
        z_color -= crange[0]
        z_color *= 1./np.diff(crange)    
        z_color = cmap(z_color)
        
        # can be out of range
        xi = np.digitize(x, bins[0]) - 1
        yi = np.digitize(y, bins[1]) - 1
        
        xi_start = xi - roiSize
        np.clip(xi_start, 0, len(bins[0]-1), out=xi_start)
        xi_stop = xi + roiSize + 1
        np.clip(xi_stop, 0, len(bins[0]-1), out=xi_stop)
        
        yi_start = yi - roiSize
        np.clip(yi_start, 0, len(bins[1]-1), out=yi_start)
        yi_stop = yi + roiSize + 1
        np.clip(yi_stop, 0, len(bins[1]-1), out=yi_stop)
        
        X, Y = np.meshgrid(*bins, indexing="ij")
#        XY = np.stack([X, Y], axis=0)[:,:,None]
#        print XY.shape
        
        # maybe 3x speed if vectorized?
#        psfs = gaussian_2d_simple([x, sigma, y, sigma], np.stack([X[:15,:15], Y[:15,:15]], axis=0)[:,:,:,None])
#        print(psfs.shape)
        
#        for i in range(len(x)):
##            ix = np.absolute(bins[0] - x[i]).argmin()
##            iy = np.absolute(bins[1] - y[i]).argmin()
#            
##            psf = gaussian_nd_simple([x[i], sigma, y[i], sigma], [bins[0][xi_start[i]: xi_stop[i]], bins[1][yi_start[i]: yi_stop[i]]])
#            psf = gaussian_2d_simple([x[i], sigma, y[i], sigma], [dim[xi_start[i]: xi_stop[i], yi_start[i]: yi_stop[i]] for dim in [X, Y]])
#            if color_alpha <= 0:
#                im[xi_start[i]: xi_stop[i], yi_start[i]: yi_stop[i]] += z_color[i] * psf[:,:,None]
#            else:
#                im[xi_start[i]: xi_stop[i], yi_start[i]: yi_stop[i]] += color_alpha * z_color[i] * ((1-im[xi_start[i]: xi_stop[i], yi_start[i]: yi_stop[i], 3])*psf)[:,:,None]

        im_tmp = np.empty_like(im)                
        for chunk in xrange(0, len(x), 5000):
            chunk_end = min(len(x), chunk+5000)
            im_tmp.fill(0)
            
            for i in xrange(chunk, chunk_end):                
                psf = gaussian_2d_simple([x[i], sigma, y[i], sigma], [dim[xi_start[i]: xi_stop[i], yi_start[i]: yi_stop[i]] for dim in [X, Y]])
                if color_alpha <= 0:
                    im_tmp[xi_start[i]: xi_stop[i], yi_start[i]: yi_stop[i]] += z_color[i] * psf[:,:,None]
                else:
                    im_tmp[xi_start[i]: xi_stop[i], yi_start[i]: yi_stop[i]] += color_alpha * z_color[i] * ((1-im[xi_start[i]: xi_stop[i], yi_start[i]: yi_stop[i], 3])*psf)[:,:,None]
                    
            if color_alpha <= 0:
                im += im_tmp
            else:
                im += im_tmp * (1-im[:,:,3])[:,:,None]
                
        
        return im

    @staticmethod
    def binned_colors(sample, values, bins, cmap, crange, color_alpha, cache_path, pool):
        # copyied from scipy stats binned_statistic_dd
        
        try:
        # `sample` is an ND-array.
            Dlen, Ndim = sample.shape
        except (AttributeError, ValueError):
            # `sample` is a sequence of 1D arrays.
            sample = np.atleast_2d(sample).T
            Dlen, Ndim = sample.shape
            
        # Store initial shape of `values` to preserve it in the output
        values = np.asarray(values)
        input_shape = list(values.shape)
        # Make sure that `values` is 2D to iterate over rows
        values = np.atleast_2d(values)
        Vdim, Vlen = values.shape

        # Make sure `values` match `sample`
        if(Vlen != Dlen):
            raise AttributeError('The number of `values` elements must match the '
                                 'length of each `sample` dimension.')
    
        nbin = np.empty(Ndim, int)    # Number of bins in each dimension
        edges = Ndim * [None]         # Bin edges for each dim (will be 2D array)
        dedges = Ndim * [None]        # Spacing between edges (will be 2D array)
        
        try:
            M = len(bins)
            if M != Ndim:
                raise AttributeError('The dimension of bins must be equal '
                                     'to the dimension of the sample x.')
        except TypeError:
            bins = Ndim * [bins]
        
        smin = np.atleast_1d(np.array(sample.min(axis=0), float))
        smax = np.atleast_1d(np.array(sample.max(axis=0), float))
        
        # Make sure the bins have a finite width.
        for i in xrange(len(smin)):
            if smin[i] == smax[i]:
                smin[i] = smin[i] - .5
                smax[i] = smax[i] + .5
                
        # Create edge arrays
        for i in xrange(Ndim):
            if np.isscalar(bins[i]):
                nbin[i] = bins[i] + 2  # +2 for outlier bins
                edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1)
            else:
                edges[i] = np.asarray(bins[i], float)
                nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
            dedges[i] = np.diff(edges[i])
    
        nbin = np.asarray(nbin)
    
        # Compute the bin number each sample falls into, in each dimension
        sampBin = [
            np.digitize(sample[:, i], edges[i])
            for i in xrange(Ndim)
        ]
        
        # Using `digitize`, values that fall on an edge are put in the right bin.
        # For the rightmost bin, we want values equal to the right
        # edge to be counted in the last bin, and not as an outlier.
        for i in xrange(Ndim):
            # Find the rounding precision
            decimal = int(-np.log10(dedges[i].min())) + 6
            # Find which points are on the rightmost edge.
            on_edge = np.where(np.around(sample[:, i], decimal) ==
                               np.around(edges[i][-1], decimal))[0]
            # Shift these points one bin to the left.
            sampBin[i][on_edge] -= 1
    
        # Compute the sample indices in the flattened statistic matrix.
        binnumbers = np.ravel_multi_index(sampBin, nbin)
        
        ### modified to return in RGBA
        if cache_path == "":
            result = np.empty([Vdim, nbin.prod(), 4], float)
        else:
            try:
                result = np.memmap(cache_path, dtype=float, mode='w+', shape=(Vdim, nbin.prod(), 4))
            except:
                 result = np.memmap(cache_path, dtype=float, mode='r+', shape=(Vdim, nbin.prod(), 4))   
        
        result.fill(0)
#        flatcount = np.bincount(binnumbers, None)
#        a = flatcount.nonzero()
#        for vv in xrange(Vdim):
#            flatsum = np.bincount(binnumbers, values[vv])
#            result[vv, a] = np.tile(flatsum[a] / flatcount[a], (4,1)).T
        
        if color_alpha <= 0:
            weighting = None
        else:
#            alpha = 0.02
            cap = np.ceil(np.log(0.01)/np.log(1-color_alpha)).astype(np.int)    
            weighting = np.geomspace(color_alpha, color_alpha*(1-color_alpha)**(cap-1), num=cap)
        
        if pool == None:
            for i in np.unique(binnumbers):
                for vv in xrange(Vdim):
                    result[vv, i] = calculate_observed_color((None, values[vv, binnumbers == i], cmap, crange, weighting))[1]
        else:
            unique_binnumbers = np.unique(binnumbers)
            for vv in xrange(Vdim):
#                args = zip(unique_binnumbers, [values[vv, binnumbers == i] for i in unique_binnumbers], (cmap,)*len(unique_binnumbers), (crange,)*len(unique_binnumbers))
                args = [(unique_binnumbers[i], values[vv, binnumbers == j], cmap, crange, weighting) for i, j in enumerate(unique_binnumbers)]
                for i, (j, res) in enumerate(pool.imap_unordered(calculate_observed_color, args, chunksize=100)):                    
                    result[vv, j] = res
               
                    if ((i+1) % (len(unique_binnumbers)//100) == 0):
                        print("Completed calculating {} of {}.".format(i+1, len(unique_binnumbers)))
                
            
        # Shape into a proper matrix
        result = result.reshape(np.concatenate(([Vdim], nbin, [4])))
    
        # Remove outliers (indices 0 and -1 for each bin-dimension).
        core = tuple([slice(None)] + Ndim * [slice(1, -1)])
        result = result[core]
       
        if np.any(result.shape[1:-1] != nbin - 2):
            raise RuntimeError('Internal Shape Error')
    
        # Reshape to have output (`reulst`) match input (`values`) shape
        result = result.reshape(input_shape[:-1] + list(nbin-2) + [4,])
    
        return BinnedStatisticddResult(result, edges, binnumbers)
    
#@staticmethod
def calculate_observed_color(packed):
#    from PYME.util import mProfile
#    mProfile.profileOn(['colored_image.py'])
    
    index, values, cmap, crange, weighting = packed
    
    processed_values = np.sort(values[::-1])
#    print processed_values.dtype
    processed_values -= crange[0]
#    processed_values /= np.diff(crange)
    processed_values *= 1./np.diff(crange)
    
    processed_values = cmap(processed_values)
#        processed_values = np.mean(processed_values, axis=0)        
#        processed_values *= values.shape[0]
    
#    alpha = 0.02
#    cap = np.ceil(np.log(0.1)/np.log(1-alpha)).astype(np.int)
    counts = values.shape[0]
#    cap = min(cap, counts)
#    weighting = np.geomspace(1, (1-alpha)**(cap-1), num=cap)
    
#    processed_values = np.average(processed_values[:len(weighting)], axis=0, weights=weighting[:counts]) * counts
    if weighting is None:
        processed_values = processed_values.sum()
    else:
        processed_values = np.multiply(processed_values[:len(weighting)], weighting[:counts][:,None])
        processed_values = np.sum(processed_values, axis=0) * counts
    
    
    res = processed_values
#        print res
#    mProfile.profileOff()
#    mProfile.report()
        
    return (index, res)
#        return np.repeat(values.mean(), 4)
        
# from https://stackoverflow.com/questions/32883491/pickling-scipy-interp1d-spline    
class interp1d_picklable:
    def __init__(self, xi, yi, **kwargs):
        self.xi = xi
        self.yi = yi
        self.args = kwargs
        self.f = interpolate.interp1d(xi, yi, **kwargs)

    def __call__(self, xnew):
        return self.f(xnew)

    def __getstate__(self):
        return self.xi, self.yi, self.args

    def __setstate__(self, state):
        self.f = interpolate.interp1d(state[0], state[1], **state[2])
        
def gaussian_nd(p, dims):
    # p = A, bg, x, sig_x, y, sig_y, z, sig_z, ...
    A, bg = p[:2]
    dims_nd = np.meshgrid(*dims, indexing='ij')
    exponent = 0
    for i, dim in enumerate(dims_nd):
        exponent += (dim-p[2+2*i])**2/(2*p[2+2*i+1]**2)
    return A * np.exp(-exponent) + bg

def gaussian_nd_simple(p, dims):
    # p = A, bg, x, sig_x, y, sig_y, z, sig_z, ...
#    A, bg = p[:2]
#    dims_nd = np.meshgrid(*dims, indexing='ij')
#    dims_nd = dims
    exponent = 0
    for i, dim in enumerate(dims):
        exponent += (dim-p[2*i])**2/(2*p[2*i+1]**2)
    return np.exp(-exponent)

# simple --> faster
def gaussian_2d_simple(p, dims):
    # p = A, bg, x, sig_x, y, sig_y, z, sig_z, ...
#    A, bg = p[:2]
#    dims_nd = np.meshgrid(*dims, indexing='ij')
#    dims_nd = dims
        
    return np.exp(-((dims[0]-p[0])**2/(2*p[1]**2) + (dims[1]-p[2])**2/(2*p[3]**2)))