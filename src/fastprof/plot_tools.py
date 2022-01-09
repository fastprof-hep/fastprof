"""
Utility classes for model operations

"""
import re
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt

from .core  import Model, Data, Parameters
  
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

  def decorate_matrix(self) :
    plt.xticks(np.arange(len(self.data.model.pois)), self.data.model.pois.keys(), rotation='vertical')
    plt.yticks(np.arange(len(self.data.model.pois)), self.data.model.pois.keys())

  def make_canvas(self, figsize : tuple = (10,10), fig : plt.Figure = None) :
    if not fig :
      plt.figure(figsize=figsize)
    else :
      plt.figure(canvas.number)

  def plot_correlation(self, figsize : tuple = (10,10), fig : plt.Figure = None) :
    self.make_canvas()
    img = plt.imshow(self.data.model.correlation_matrix(self.pars, self.data), vmin=-1, vmax=1, cmap='seismic')
    self.decorate_matrix()
    plt.colorbar(img)
    plt.tight_layout()

  def plot_covariance(self, figsize : tuple = (10,10), fig : plt.Figure = None) :
    self.make_canvas()
    img = plt.imshow(self.data.model.covariance_matrix(self.pars, self.data), cmap='seismic')
    self.decorate_matrix()
    plt.colorbar(img)
    plt.tight_layout()
