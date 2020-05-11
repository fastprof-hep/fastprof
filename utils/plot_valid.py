#! /usr/bin/env python

__doc__ = "Perform a fit to a fast model"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fastprof import Model, Data, OptiMinimizer, JSONSerializable
import matplotlib.pyplot as plt
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
  fig_var = plt.figure(figsize=(8, 6), dpi=96)
  fig_inv = plt.figure(figsize=(8, 6), dpi=96)
  gs = plt.GridSpec(nrows=nr, ncols=nc)
  fig_var.suptitle('Bin [%g, %g] parameter variations' % (model.bins[b]['lo_edge'], model.bins[b]['hi_edge']))
  fig_inv.suptitle('Bin [%g, %g] 1/N linearity check'  % (model.bins[b]['lo_edge'], model.bins[b]['hi_edge']))
  for i, par in enumerate(model.alphas + model.betas + model.gammas) :
    pars = model.expected_pars(1)
    #if par in model.alphas :
      #model.linear_nps = True
      #vars_lin = [ model.s_exp(pars.set(par, x))[b]/sig0 for x in cont_x ]
      #model.linear_nps = False
      #vars_nonlin = [ model.s_exp(pars.set(par, x))[b]/sig0 for x in cont_x ]
    #else:
      #model.linear_nps = True
      #vars_lin = [ model.b_exp(pars.set(par, x))[b]/bkg0 for x in cont_x ]
      #model.linear_nps = False
      #vars_nonlin = [ model.b_exp(pars.set(par, x))[b]/bkg0 for x in cont_x ]
    model.linear_nps = True
    vars_lin = [ model.n_exp(pars.set(par, x))[b]/nexp0[b] for x in cont_x ]
    rvar_lin = [ 1 - ((model.n_exp(pars.set(par, x))[b] - nexp0[b])/nexp0[b])**2 for x in cont_x ]
    model.linear_nps = False
    vars_nli = [ model.n_exp(pars.set(par, x))[b]/nexp0[b] for x in cont_x ]
    rvar_nli = [ 1 - ((model.n_exp(pars.set(par, x))[b] - nexp0[b])/nexp0[b])**2 for x in cont_x ]

    ax_var = fig_var.add_subplot(gs[i//nc, i % nc])
    ax_var.set_title(par)
    ax_var.plot(data.points, data.variations[par][b, :], 'ko')
    ax_var.plot(cont_x, vars_lin, 'r--')
    ax_var.plot(cont_x, vars_nli, 'b')
    
    ax_inv = fig_inv.add_subplot(gs[i//nc, i % nc])
    ax_inv.set_title(par)
    ax_inv.plot(cont_x, rvar_lin, 'r--')
    ax_inv.plot(cont_x, rvar_nli, 'b')

  fig_var.tight_layout()
  fig_var.subplots_adjust(top=0.9)
  fig_var.canvas.set_window_title('Bin %g parameter variations' % b)
  
  fig_inv.tight_layout()
  fig_inv.subplots_adjust(top=0.9)
  fig_inv.canvas.set_window_title('Bin %g 1/N linearity check' % b)

  
