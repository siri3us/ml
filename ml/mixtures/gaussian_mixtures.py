import numpy as np
import copy
import scipy
import matplotlib.pyplot as plt

from collections import defaultdict
from helpers import Checker

class GaussianMixtureD1(Checker):
    def __init__(self, locs, scales, weights):
        """
        Одномерное нормальное распределение
        
        Аргументы
        :param locs: средние значения для каждой из K компонент
        :type locs:  list, np.array
        
        :param scales: стандартные отклонения для каждой из K компонент
        :type scales:  list, np.array
        
        :param weights: веса компонент
        :type weights:  list, np.array
        """
        self.weights = np.array(weights) / np.sum(weights)
        self.locs    = np.array(locs)
        self.scales  = np.array(scales)
        self.n_components = len(weights)
        
    def sample(self, size=None):
        if size is None:
            return self._sample()
        return np.array([self._sample() for _ in range(size)])
    def _sample(self):
        c = np.random.choice(self.n_components, p=self.weights)
        return np.random.normal(loc=self.locs[c], scale=self.scales[c])   
    def __call__(self, points):
        return self.pdf(points) 
    def pdf(self, points):
        points = np.array(points)[:, np.newaxis]
        A = np.exp(-0.5 * ((points - self.locs[np.newaxis, :]) / self.scales[np.newaxis, :])**2) / (np.sqrt(np.pi * 2) * self.scales[np.newaxis, :])
        return np.sum(self.weights[np.newaxis, :] * A, axis=1)

        
class GaussianMixtureD2(Checker):
    def __init__(self, xlocs, ylocs, xscales, yscales, weights):
        """
        Двумерное нормальное распределение
        
        Аргументы
        :param xlocs: средние значения по оси X для каждой из K компонент
        :type xlocs:  list, np.array
        
        :param ylocs: средние значения по оси Y для каждой из K компонент
        :type ylocs:  list, np.array
        
        :param xscales: стандартные отклонения по оси X для каждой из K компонент
        :type xscales:  list, np.array
        
        :param yscales: стандартные отклонения по оси Y для каждой из K компонент
        :type yscales:  list, np.array
        
        :param weights: веса компонент
        :type weights:  list, np.array
        """
        self.xlocs = copy.deepcopy(xlocs)
        self.xscales = copy.deepcopy(xscales)
        self.ylocs = copy.deepcopy(ylocs)
        self.yscales = copy.deepcopy(yscales)
        self.weights = copy.deepcopy(weights)
        self.weights = self.weights / np.sum(self.weights)
        assert len(self.xlocs) == len(self.xscales)
        assert len(self.ylocs) == len(self.yscales)
        assert len(self.xlocs) == len(self.ylocs)
        assert len(self.weights) == len(self.ylocs)
        assert np.allclose(np.sum(self.weights), 1)
        self.n_gens = len(self.xlocs)

    def sample(self, size=None):
        if size is None:
            return self._sample()
        else:
            numbers = np.zeros((size, 2))
            for i in range(size):
                numbers[i, :] = self._sample()
            return numbers
                
    def pdf(self, points):
        if len(points.shape) == 1:
            points = points[np.newaxis, :]
        xs, ys = points[:, 0], points[:, 1]
        pdfs = np.zeros(len(xs))
        for n_gen in range(self.n_gens):
            xloc = self.xlocs[n_gen]
            xscale = self.xscales[n_gen]
            xpdf = stats.norm.pdf(xs, xloc, xscale)
            
            yloc = self.ylocs[n_gen]
            yscale = self.yscales[n_gen]
            ypdf = stats.norm.pdf(ys, yloc, yscale)
            pdfs += self.weights[n_gen] * xpdf * ypdf
        return pdfs       
    
    def __call__(self, points):
        return self.pdf(points)
    
    def _sample(self):
        n_gen = np.random.choice(self.n_gens, p=self.weights)
        xloc = self.xlocs[n_gen]
        yloc = self.ylocs[n_gen]
        xscale = self.xscales[n_gen]
        yscale = self.yscales[n_gen]
        x = np.random.normal(xloc, xscale)
        y = np.random.normal(yloc, yscale)
        return np.array([x, y])
        
class GaussianMixture:
    def __init__(self, locs, scales, weights):
        """
        Смесь из K многомерных нормальных распределений в пространстве размерности D.
        
        Аргументы:
        :param locs:    D средних значений для каждой из K компонент смеси.
        :type locs:     список из K списков; каждый из K списков - список размера D со средними значениями вдоль 
            каждого из измерений.
            
        :param scales:  значения D стандартных отклонений для каждой из K компонент смеси.
        :type scales:   список из K список; каждый из K списков - список размера D со стандартными отклонениями вдоль
            каждого из измерений.
        
        :param weights: значения K весов компонент смеси
        :type weights:  list, np.array
        """
        
        assert len(locs) == len(scales)
        assert len(locs) == len(weights)
        assert len(locs) > 0
        self.n_components = len(locs)
        
        if hasattr(locs[0], '__iter__'):
            n_dim = len(locs[0])
            assert n_dim > 0
            for i in range(len(locs)):
                assert len(locs[i]) == n_dim
                assert len(scales[i]) == n_dim
            self.n_dim = n_dim
        else:
            locs   = [[l] for l in locs]
            scales = [[s] for s in scales]
            self.n_dim = 1

        self.locs    = copy.deepcopy(locs)
        self.scales  = copy.deepcopy(scales)
        self.weights = np.array(weights)
        self.weights = self.weights / np.sum(self.weights)
        assert np.allclose(np.sum(self.weights), 1)
        
    def sample(self, size=None, return_components=False):
        if size is None:
            return self._sample(return_components)
        else:
            points = np.zeros((size, self.n_dim))
            components = np.zeros(size, dtype=np.int32)
            for i in range(size):
                points[i], components[i] = self._sample(return_components=True)
            if return_components:
                return points, components
            return points
       
    def _sample(self, return_components=False):
        c = np.random.choice(self.n_components, p=self.weights)
        loc = self.locs[c]
        scale = self.scales[c]
        point = []
        for i in range(self.n_dim):
            point.append(np.random.normal(loc=loc[i], scale=scale[i]))
        if return_components:
            return np.array(point), c
        return np.array(point)
    
    def __call__(self, points):
        return self.pdf(points)
    
    def pdf(self, points):
        points = np.array(points)
        if len(points.shape) == 1:
            points = points[np.newaxis, :]
        n_points = points.shape[0]
        pdfs = np.zeros((self.n_dim, self.n_components, n_points))
        for d in range(self.n_dim):
            values = points[:, d]
            for k in range(self.n_components):
                scale = self.scales[k][d]
                loc = self.locs[k][d]
                pdfs[d, k] = scipy.stats.norm.pdf(values, loc, scale)
        pdfs = np.sum(np.prod(pdfs, axis=0) * self.weights[:, np.newaxis], axis=0)
        return pdfs
        
        
class ContourPlotter2D:
    def __init__(self, dist, x_range, y_range, **plotter_params):
        """
        Аргументы:
        :param  dist
        :type   dist
        
        :param x_range
        :type  x_range
        
        :param y_range
        :type  y_range
        
        :param plotter_params:
            :param plot_contour: [True, False]
            :param levels_style: ['continious', 'stepwise, 'none']
            :param cmap: default is 'afmhot'
            :param figsize: default is (8, 8)
            :param dashed: [True, False]
            :param vmin: default is None
            :param vmax: default is None 
            :param gamma: default is 1.0
            :param levels: default is TODO
        """
        self.x_range = np.array(x_range)
        self.y_range = np.array(y_range)
        self.dist = dist
        self.set_plotter_params(plotter_params)
        
    
    def set_plotter_params(self, plotter_params):
        self.plot_contour = plotter_params.setdefault('plot_contour', False)
        self.levels_style = plotter_params.setdefault('levels_style', 'stepwise')
        assert self.levels_style in ['continious', 'stepwise', 'none']
        self.cmap = plotter_params.setdefault('cmap', 'afmhot')
        self.linewidths = plotter_params.setdefault('linewidths', 1.0)
        self.figsize = plotter_params.setdefault('figsize', (8, 8))
        self.dashed = plotter_params.setdefault('dashed', False)
        self.vmin = plotter_params.setdefault('vmin', None)
        self.vmax = plotter_params.setdefault('vmax', None)
        self.gamma = plotter_params.setdefault('gamma', 1.0)
        self.levels = plotter_params.setdefault('levels', np.logspace(-4, 0, 15))
    
    def __call__(self, ax=None, **kwargs):
        if ax is None:
            plt.figure(figsize=self.figsize)
            ax = plt.gca()
        
        xx, yy = np.meshgrid(self.x_range, self.y_range)
        points = np.array(list(zip(xx.flatten(), yy.flatten())))
        zz = self.dist.pdf(points).reshape((len(self.x_range), len(self.y_range)))
        zz = zz ** self.gamma
        levels = self.levels ** self.gamma
        
        #scaled_zz = rescale(zz ** self.deg_scale, self.vmin * self.abs_scale, 1.0)
        #scaled_levels = rescale(self.levels ** self.deg_scale, self.vmin * self.abs_scale, 1.0)
        
        if self.levels_style == 'continious':
            ax.pcolormesh(xx, yy, zz, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
            if self.plot_contour:
                CS = ax.contour(xx, yy, zz, colors='k', linewidths=self.linewidths, levels=levels)
        elif self.levels_style == 'stepwise':
            CF = ax.contourf(xx, yy, zz, cmap=self.cmap, linewidths=self.linewidths, vmin=self.vmin, vmax=self.vmax, 
                             levels=levels)
            if self.plot_contour:
                CS = ax.contour(CF, levels=CF.levels, colors='k', origin='lower')
        else:
            if self.plot_contour:
                CS = ax.contour(xx, yy, zz, colors='k', linewidths=self.linewidths, 
                                levels=levels)
                
        if self.plot_contour & self.dashed:
            for c in CS.collections:
                c.set_dashes([(0, (2.5 * self.linewidths, 2.5 * self.linewidths))])
