"""
Utility classes for model operations

"""
import re
import sys
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt

from .core  import Model, Data, Parameters
from .minimizers import POIMinimizer, OptiMinimizer
from .model_tools import NPPruner

# -------------------------------------------------------------------------
class PlotNPs :
  
  def __init__(self, pars) :
    self.pars = pars

  def pull_plot(self, names : list = [], max_pull = None, left_adjust : float = 0.50, figsize : tuple = (5,10), fig : plt.Figure = None) :
    indices = np.arange(len(names))
    try :
      values = [ self.pars[name] for name in names ]
    except Exception as inst :
      print(inst)
      raise('Invalid parameter names, cannot plot pulls')
    if not fig :
      plt.figure(figsize=figsize)
    else :
      plt.figure(fig.number)
    plt.scatter(values, indices)
    if not fig :
      plt.xlabel('normalized pull')
      if max_pull is not None : plt.xlim(-max_pull, max_pull)
      plt.yticks(indices, names)
      plt.margins(0.01) # to ensure the top and bottom points don't get clipped by the axes
      if left_adjust is not None : plt.subplots_adjust(left=left_adjust) # give more space for axis labels
      plt.tight_layout()

class PlotResults :

  def __init__(self, pars, data) :
    self.pars = pars
    self.data = data

  def decorate_with_pois(self, axes: str = 'xy', reverse: bool = False) :
    indices = np.arange(len(self.data.model.pois)) if not reverse else np.arange(len(self.data.model.pois), 0, -1)
    if 'x' in axes : plt.xticks(indices, self.data.model.pois.keys(), rotation='vertical')
    if 'y' in axes : plt.yticks(indices, self.data.model.pois.keys())

  def make_canvas(self, figsize : tuple = (10,10), fig : plt.Figure = None) :
    if not fig :
      plt.figure(figsize=figsize)
    else :
      plt.figure(canvas.number)

  def plot_correlation(self, figsize : tuple = (10,10), fig : plt.Figure = None) :
    self.make_canvas(figsize, fig)
    img = plt.imshow(self.data.model.correlation_matrix(self.pars, self.data), vmin=-1, vmax=1, cmap='seismic')
    self.decorate_with_pois(axes='xy')
    plt.colorbar(img)
    plt.tight_layout()

  def plot_covariance(self, figsize : tuple = (10,10), fig : plt.Figure = None) :
    self.make_canvas(figsize, fig)
    img = plt.imshow(self.data.model.covariance_matrix(self.pars, self.data), cmap='seismic')
    self.decorate_with_pois(axes='xy')
    plt.colorbar(img)
    plt.tight_layout()
    
  def plot_best_fit(self, values: dict = None, sym_errors : dict = None, results : dict = None, figsize : tuple = (10,10), fig : plt.Figure = None) :
    self.make_canvas(figsize, fig)
    vals = np.zeros(len(self.data.model.pois))
    if values is not None :
      vals = np.array([ values[poi] for poi in self.data.model.pois ])
    elif results is not None :
      vals = np.array([ results[poi][0] for poi in self.data.model.pois ])
    elif self.pars is not None :
      vals = np.array([ self.pars[poi] for poi in self.data.model.pois ])
    else :
      raise ValueError("Parameter values should be provided either through the 'pars', 'values' or 'results' arguments.")
    if sym_errors is not None :
      sym_errs = [ sym_errors[poi] for poi in self.data.model.pois ]
      errs = np.array([ sym_errs, sym_errs ])
    elif results is not None :
      errs = np.array([[ results[poi][1] for poi in self.data.model.pois ], [ results[poi][2] for poi in self.data.model.pois ]])
    else :
      errs = np.zeros((2,len(self.data.model.pois)))
    plt.errorbar(vals, np.arange(len(self.data.model.pois), 0, -1), xerr=errs, fmt='o')
    self.decorate_with_pois(axes='y', reverse=True)
    plt.tight_layout()

class PlotImpacts :

  def __init__(self, poi, data, minimizer : POIMinimizer = None) :
    self.data = data
    self.poi = poi
    self.minimizer = minimizer if minimizer is not None else OptiMinimizer()
  
  def make_canvas(self, figsize : tuple = (10,10), fig : plt.Figure = None) :
    if not fig :
      plt.figure(figsize=figsize)
    else :
      plt.figure(canvas.number)

  def plot(self, prefit : bool = True, postfit : bool =True, impacts : bool = True, figsize : tuple = (10,10), fig : plt.Figure = None) :
    self.make_canvas(figsize, fig)
    self.plot_impacts(prefit=prefit, postfit=postfit)
    self.plot_pulls()
    plt.tight_layout()

  def plot_impacts(self, prefit : bool = True, postfit : bool = True) :
    pass # for now

  def impacts(self, is_postfit : bool = True) :
    impacts = []
    self.minimizer.minimize(self.data)
    min_pars = self.minimizer.min_pars.clone()
    if is_postfit : errors = self.data.model.parabolic_errors(min_pars, self.data)
    for p, par in enumerate(self.data.model.nps.values()) :
      sys.stderr.write("\rProcessing NP %4d of %4d [%30s]" % (p+1, len(self.data.model.nps), par.name[:30]))
      var = errors[par.name] if is_postfit else 1
      pruned_aux_obs = np.delete(self.data.aux_obs, p)
      #print(par.name, par.aux_obs, len(self.data.aux_obs), len(pruned_aux_obs))
      posvar_model = NPPruner(self.data.model).remove_nps({ par.name : min_pars[par.name] + var }, clone_model=True)
      pos_minimizer = self.minimizer.clone(posvar_model.ref_pars.clone())
      pos_minimizer.minimize(self.data.clone(posvar_model, aux_obs=pruned_aux_obs))
      negvar_model = NPPruner(self.data.model).remove_nps({ par.name : min_pars[par.name] - var }, clone_model=True)
      neg_minimizer = self.minimizer.clone(negvar_model.ref_pars.clone())
      neg_minimizer.minimize(self.data.clone(negvar_model, aux_obs=pruned_aux_obs))
      impacts.append((par.name, pos_minimizer.min_pars[self.poi] - min_pars[self.poi], neg_minimizer.min_pars[self.poi] - min_pars[self.poi]))
    sys.stderr.write("\n")
    return { trip[0] : (trip[1], trip[2]) for trip in sorted(impacts, key=lambda trip: (trip[1] - trip[2])/2) }  
 
