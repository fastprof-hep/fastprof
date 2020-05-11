#! /usr/bin/env python

__doc__ = "Perform a fit to a fast model"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fastprof import Model, Data, OptiMinimizer, JSONSerializable
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import math

####################################################################################################################################
###

parser = ArgumentParser("plot.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-m", "--model-file"     , type=str, required=True, help="Name of JSON file defining model")
parser.add_argument("-v", "--validation-data", type=str, required=True, help="Name of JSON file containing validation data")
parser.add_argument("-b", "--bins"           , type=str, required=True, help="List of bins for which to plot validation data")

options = parser.parse_args()
if not options : 
  parser.print_help()
  sys.exit(0)

try :
  bins = [ int(b) for b in options.bins.split(',') ]
except Exception as inst :
  print(inst)
  raise ValueError('Invalid bin specification %s : the format should be bin1,bin2,...' % options.bins)

class ValidationData (JSONSerializable) :
  def __init__(self, filename = '') :
    super().__init__()
    if filename != '' : self.load(filename)

  def load_jdict(self, jdict) :
    self.points     = jdict['points']
    self.variations = {}
    for par in model.alphas + model.betas + model.gammas :
      if not par in jdict : 
        print('No validation data found for NP %s' % par)
      else :
        self.variations[par] = np.array(jdict[par])
    return self

  def dump_jdict(self) :
    return {}  

model = Model.create(options.model_file)
if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)

data = ValidationData(options.validation_data)

plt.ion()
nplots = model.n_nps
nc = math.ceil(math.sqrt(nplots))
nr = math.ceil(nplots/nc)

cont_x = np.linspace(data.points[0], data.points[-1], 100)

pars = model.expected_pars(1)
sig0 = model.s_exp(pars)
bkg0 = model.b_exp(pars)
nexp0 = model.n_exp(pars)

for b in bins :
  fig = plt.figure(figsize=(8, 8), dpi=96)
  fig.suptitle('Bin [%g, %g] linearity checks'  % (model.bins[b]['lo_edge'], model.bins[b]['hi_edge']))
  gs = gridspec.GridSpec(nrows=nr, ncols=nc, wspace=0.3, hspace=0.3, top=0.9, bottom=0.05, left=0.1, right=0.95)
  for i, par in enumerate(model.alphas + model.betas + model.gammas) :
    
    sgs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i//nc, i % nc], wspace=0.1, hspace=0.1)

    pars = model.expected_pars(1)
    model.linear_nps = True
    vars_lin = [ model.n_exp(pars.set(par, x))[b]/nexp0[b] - 1 for x in cont_x ]
    rvar_lin = [ -((model.n_exp(pars.set(par, x))[b] - nexp0[b])/nexp0[b])**2 for x in cont_x ]
    model.linear_nps = False
    vars_nli = [ model.n_exp(pars.set(par, x))[b]/nexp0[b] - 1 for x in cont_x ]
    rvar_nli = [ -((model.n_exp(pars.set(par, x))[b] - nexp0[b])/nexp0[b])**2 for x in cont_x ]

    ax_var = fig.add_subplot(sgs[0])
    ax_inv = fig.add_subplot(sgs[1], sharex=ax_var)
    ax_var.set_title(par)
    ax_var.plot(data.points, data.variations[par][b, :]-1, 'ko')
    ax_var.plot(cont_x, vars_lin, 'r--')
    ax_var.plot(cont_x, vars_nli, 'b')
    ax_inv.plot(cont_x, rvar_lin, 'r--')
    ax_inv.plot(cont_x, rvar_nli, 'b')

  fig.canvas.set_window_title('Bin %g linearity checks' % b)

  
