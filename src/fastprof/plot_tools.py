"""
Utility classes for plot-making

* :class:`PlotNPs`: draw NP pull plots 

* :class:`PlotResults`: plot POI best-fit values, covariance and correlation matrices.

* :class:`PlotImpacts`: draw NP impacts and pulls together
"""

import re
import sys
import math
import json
import numpy as np
import itertools
import matplotlib.pyplot as plt

from .core  import Model, Data, Parameters
from .minimizers import POIMinimizer, OptiMinimizer
from .model_tools import NPPruner

# -------------------------------------------------------------------------
class PlotNPs :
  """Utility class to plot NP pulls

  Atttributes:
    pars (Parameters) : a Parameters object with parameter (POI and NP) values
  """
  
  def __init__(self, pars : Parameters) :
    """Initialize the object

    Args:
      pars : a Parameters object with the parameter values
    """
    self.pars = pars

  def pull_plot(self, names : list = [], max_pull : float = None,
                left_adjust : float = 0.50, figsize : tuple = (5,10),
                fig : plt.Figure = None) :
    """Plot NP pulls

    Args:
      names : list of names of the NPs to plot
      max_pull : maximum pull value to define the x-axis range
      left_adjust : size of the left margin, to give space for axis labels (default: 0.50)
      figsize : figure size, as a (size_x, size_y) pair (default: (5,10))
      fig : figure object to use for plotting (if None, create a new one)
    """
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
  """Utility class to plot POI results

  So far 3 plots are implemented:
  - Correlation matrix plot
  - Covariance matrix plot
  - Bar plot with POI best-fit value and parabolic error

  Atttributes:
    pars (Parameters) : a Parameters object with parameter (POI and NP) values
    data (Data) : the dataset corresponding to the results, to compute covariances.
  """

  def __init__(self, pars, data) :
    """Initialize the object

    Args:
      pars : a Parameters object with the parameter values.
      data : the dataset corresponding to the results.
    """
    self.pars = pars
    self.data = data

  def decorate_with_pois(self, axes: str = 'xy', reverse: bool = False) :
    """Decorate the plot axes with POI names

    Sets the number of axis ticks to the number of POIs and label
    each tick with the POI name.

    Args:
      axes : which axes to decorate (default: 'xy')
      reverse: if True, decorate in reverse order (right to left or top to bottom)
    """
    indices = np.arange(len(self.data.model.pois)) if not reverse else np.arange(len(self.data.model.pois), 0, -1)
    if 'x' in axes : plt.xticks(indices, self.data.model.pois.keys(), rotation='vertical')
    if 'y' in axes : plt.yticks(indices, self.data.model.pois.keys())

  def make_canvas(self, figsize : tuple = (10,10), fig : plt.Figure = None) :
    """Provide a figure for plotting

    If the fig argument is passed, then the same figure is returned.
    Otherwise create a new one with the specified size.

    Args:
      figsize : figure size, as a (size_x, size_y) pair (default: (5,10))
      fig : figure object to use (if None, create a new one)

    Returns:
      the figure
    """
    if not fig :
      self.figure = plt.figure(figsize=figsize)
    else :
      self.figure = plt.figure(fig.number)

  def plot_correlation(self, figsize : tuple = (10,10), fig : plt.Figure = None) :
    """Plot the correlation matrix of the POIs

    Args:
      figsize : figure size, as a (size_x, size_y) pair (default: (5,10))
      fig : figure object to use (if None, create a new one)
    """
    self.make_canvas(figsize, fig)
    img = plt.imshow(self.data.model.correlation_matrix(self.pars, self.data), vmin=-1, vmax=1, cmap='seismic')
    self.decorate_with_pois(axes='xy')
    plt.colorbar(img)
    plt.tight_layout()

  def plot_covariance(self, figsize : tuple = (10,10), fig : plt.Figure = None) :
    """Plot the covariance matrix of the POIs

    Args:
      figsize : figure size, as a (size_x, size_y) pair (default: (5,10))
      fig : figure object to use (if None, create a new one)
    """
    self.make_canvas(figsize, fig)
    img = plt.imshow(self.data.model.covariance_matrix(self.pars, self.data), cmap='seismic')
    self.decorate_with_pois(axes='xy')
    plt.colorbar(img)
    plt.tight_layout()
    
  def plot_best_fit(self, values: dict = None, sym_errors : dict = None, results : dict = None,
                    figsize : tuple = (10,10), fig : plt.Figure = None) :
    """Plot the best-fit results of the POIs

    Plots the best-fit values and symmetric (parabolic) errors
    for the specified POIs. The values and errors can be given as
    separate dicts (values and sym_errors); or as a dict of results
    in the form { par_name : (val, err) }, as in the FitResult class.
    The inputs are checked in this order. If none are provided, the
    values from self.pars are used instead.

    Args:
      values : POI values as a {name: value} dict
      sym_errors : parabolic errors on the POIs, as a {name: value} dict
      results : dict of fit results in the form { par_name : (val, err) }
      figsize : figure size, as a (size_x, size_y) pair (default: (5,10))
      fig : figure object to use (if None, create a new one)
    """
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
  """Utility class to make pull/impact plots

  Atttributes:
    data (Data) : the observed dataset.
    poi (str) : the POI on whih to compute the impacts
    minimizer (OptiMinimizer) : the minimizer algorithm to use for fitting.
    verbosity (int) : the verbosity of the output
  """

  def __init__(self, poi : str, data : Data, minimizer : POIMinimizer = None, verbosity : int = 0) :
    """Initialize the object

    Args:
      data : the observed dataset.
      poi : the POI on whih to compute the impacts
      minimizer : the minimizer algorithm to use for fitting.
      verbosity : the verbosity of the output
    """
    self.data = data
    self.poi = poi
    self.minimizer = minimizer if minimizer is not None else OptiMinimizer(verbosity=verbosity)
    self.verbosity = verbosity
  
  def make_canvas(self, figsize : tuple = (10,10), fig : plt.Figure = None) :
    """Provide a figure for plotting

    If the fig argument is passed, then the same figure is returned.
    Otherwise create a new one with the specified size.

    Args:
      figsize : figure size, as a (size_x, size_y) pair (default: (5,10))
      fig : figure object to use (if None, create a new one)

    Returns:
      the figure
    """
    if not fig :
      self.figure = plt.figure(figsize=figsize)
    else :
      self.figure = plt.figure(canvas.number)

  def plot(self, n_max : int = 20, figsize : tuple = (10,10), fig : plt.Figure = None, output : str = None) :
    """Plot NP pulls and impacts

    If the fig argument is passed, then the same figure is returned.
    Otherwise create a new one with the specified size.

    Args:
      n_max : number of impacts to show (after sorting from largest to smallest)
      figsize : figure size, as a (size_x, size_y) pair (default: (5,10))
      fig : figure object to use (if None, create a new one)
      output : the output file prefix ('_impacts.xxx' extensions will be added)
    """
    self.make_canvas(figsize, fig)
    ax_impc = self.figure.subplots()
    ax_pull = plt.twiny()
    impacts = self.impacts()
    for impact in impacts.values() : print("%20s : pull = %+.4f, impact = %+g %+g" % (impact['name'], impact['best_fit'], impact['pos_impact'], impact['neg_impact']))
    impacts = self.plot_impacts(ax_impc, impacts, n_max)
    self.plot_pulls(ax_pull, impacts, n_max)
    plt.tight_layout()
    if output is not None :
      plt.savefig(output + '_impacts.png')
      plt.savefig(output + '_impacts.pdf')
      with open(output + '_impacts.json', 'w') as fd :
        json.dump(impacts, fd, ensure_ascii=True, indent=3)

  def plot_impacts(self, ax : plt.Axes, impacts : dict, n_max : int = 20) :
    """Plot NP impacts

    Args:
      ax : the figure axes on which to plot
      impact : the dict of NP impacts
      n_max : number of impacts to show (after sorting from largest to smallest)
    """
    names = [ name for name in impacts ][:n_max]
    pos_impacts = [ impact['pos_impact'] for impact in impacts.values() ][:n_max]
    neg_impacts = [ impact['neg_impact'] for impact in impacts.values() ][:n_max]
    indices = np.arange(len(names) + 1, 0, -1)
    ax.fill_betweenx(indices, np.zeros(len(names) + 1), [0] + pos_impacts, step='pre', clim=(0, len(names)), color='r', alpha=0.15)
    ax.fill_betweenx(indices, np.zeros(len(names) + 1), [0] + neg_impacts, step='pre', clim=(0, len(names)), color='b', alpha=0.15)
    xmax = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
    ax.set_xlim(-xmax, xmax)
    plt.yticks(ticks=indices + 0.5, labels=[''] + names)
    return impacts

  def plot_pulls(self, ax : plt.Axes, impacts : dict, n_max : int = 20) :
    """Plot NP pulls

    Args:
      ax : the figure axes on which to plot
      impact : the dict of NP impacts
      n_max : number of impacts to show (after sorting from largest to smallest)
    """
    names = [ name for name in impacts ][:n_max]
    values = [ impact['best_fit'] for impact in impacts.values() ][:n_max]
    errors = np.ones(len(names))
    indices = np.arange(len(names), 0, -1)
    ax.errorbar(x=values, y=indices + 0.5, xerr=errors, fmt='ko')
    ax.set_xlim(-3, 3)

  def impacts(self, pars : list = None) :
    """Compute NP impacts

    Computes the impacts of the specified NPs on the
    POI. The results are returned as a dict mapping the 
    POI name to a dict containing the best-fit value of
    the NP and its positive and negative impacts.

    Args:
      pars : list of names of NPs to process

    Returns:
      dict of impact values
    """
    impacts = []
    self.minimizer.minimize(self.data)
    min_pars = self.minimizer.min_pars.clone()
    if self.verbosity > 3 :
      with open('_impact_test_best_fit.json', 'w') as fd :
        json.dump(min_pars.dict(), fd, ensure_ascii=True, indent=3)
    par_names = pars if pars is not None else self.data.model.nps.keys()
    for p, par_name in enumerate(par_names) :
      par = self.data.model.nps[par_name]
      if self.verbosity == 0 :
        sys.stderr.write("\rProcessing NP %4d of %4d [%30s]" % (p+1, len(self.data.model.nps), par.name[:30]))
      var = 1
      pruned_aux_obs = np.delete(self.data.aux_obs, self.data.model.np_indices[par_name])
      posvar_model = NPPruner(self.data.model, self.verbosity).remove_nps({ par.name : par.unscaled_value(min_pars[par.name] + var) }, clone_model=True)
      pos_minimizer = self.minimizer.clone(posvar_model.ref_pars.clone())
      pos_minimizer.minimize(self.data.clone(posvar_model, aux_obs=pruned_aux_obs))
      negvar_model = NPPruner(self.data.model, self.verbosity).remove_nps({ par.name : par.unscaled_value(min_pars[par.name] - var) }, clone_model=True)
      neg_minimizer = self.minimizer.clone(negvar_model.ref_pars.clone())
      neg_minimizer.minimize(self.data.clone(negvar_model, aux_obs=pruned_aux_obs))
      if self.verbosity > 0 :
        print("Parameter %s : %s = %g (nominal), %s = %g (%+g), %s = %g (%+g)" % (par.name,
                                                                                  self.poi, min_pars[self.poi],
                                                                                  self.poi, pos_minimizer.min_pars[self.poi], +var,
                                                                                  self.poi, neg_minimizer.min_pars[self.poi], -var))
      if self.verbosity > 3 :
        with open('_impact_test_pos_%s.json' % par_name, 'w') as fd :
          json.dump(pos_minimizer.min_pars.dict(), fd, ensure_ascii=True, indent=3)
        with open('_impact_test_neg_%s.json' % par_name, 'w') as fd :
          json.dump(neg_minimizer.min_pars.dict(), fd, ensure_ascii=True, indent=3)
      impacts.append({ 'name' : par.name,
                      'pos_impact' : pos_minimizer.min_pars[self.poi] - min_pars[self.poi],
                      'neg_impact' : neg_minimizer.min_pars[self.poi] - min_pars[self.poi],
                      'best_fit' : min_pars[par.name] })
    if self.verbosity == 0 : sys.stderr.write("\n")
    return { data['name'] : data for data in sorted(impacts, key=lambda data: abs(data['pos_impact'] - data['neg_impact'])/2, reverse=True) }  
 
