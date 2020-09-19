#! /usr/bin/env python

__doc__ = "Plot validation data generated with convert_ws"
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
parser.add_argument("-m", "--model-file"     , type=str  , required=True, help="Name of JSON file defining model")
parser.add_argument("-v", "--validation-data", type=str  , required=True, help="Name of JSON file containing validation data")
parser.add_argument("-c", "--channel"        , type=str  , default=None , help="Name of selected channel (default: first one in the model)")
parser.add_argument("-s", "--sample"         , type=str  , default=None , help="Name of selected sample (default: first one in the channel)")
parser.add_argument("-b", "--bins"           , type=str  , required=True, help="List of bins for which to plot validation data")
parser.add_argument("-y", "--yrange"         , type=str  , default=''   , help="Vertical range for variations, in the form min,max")
parser.add_argument("-i", "--inv-range"      , type=float, default=None , help="Vertical range for inversion impact")
parser.add_argument(      "--cutoff"         , type=float, default=None , help="Cutoff to regularize the impact of NPs")
parser.add_argument(      "--vars-only"      , action='store_true'      , help="Only plot variations, not inversion impact")
parser.add_argument(      "--no-nli"         , action='store_true'      , help="Only plot linear reference")
parser.add_argument("-o", "--output-file"    , type=str  , default=''   , help="Output file name")

options = parser.parse_args()
if not options : 
  parser.print_help()
  sys.exit(0)

try :
  bins = [ int(b) for b in options.bins.split(',') ]
except Exception as inst :
  print(inst)
  raise ValueError('Invalid bin specification %s : the format should be bin1,bin2,...' % options.bins)

if options.yrange != '' :
  try:
    y_min, y_max = [ float(p) for p in options.yrange.split(',') ]
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid range specification %s, expected y_min,y_max' % options.yrange)

class ValidationData (JSONSerializable) :
  def __init__(self, model, filename = '') :
    super().__init__()
    self.model = model
    if filename != '' : self.load(filename)

  def load_jdict(self, jdict) :
    self.poi        = jdict[list(self.model.pois)[0]]
    self.points     = jdict['points']
    self.variations = {}
    for par in self.model.nps.keys() :
      if not par in jdict : 
        print('No validation data found for NP %s' % par)
      else :
        self.variations[par] = np.array(jdict[par])
    return self

  def dump_jdict(self) :
    return {}  

model = Model.create(options.model_file)
if model is None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
if not options.cutoff is None : model.cutoff = options.cutoff

if options.channel != None :
  channel = model.channel(options.channel)
  if not channel : raise KeyError('Channel %s not found in model.' % options.channel)
else :
  channel = list(model.channels.values())[0]

if options.sample != None :
  sample = channel.sample(options.sample)
  if not sample : raise KeyError('Sample %s not found in channel %s.' % (channel.name, options.sample))
else :
  sample = list(channel.samples.values())[0]

data = ValidationData(model, options.validation_data)
print('Validating for POI value %s = %g' % (list(model.pois)[0], data.poi))
plt.ion()
nplots = model.nnps
nc = math.ceil(math.sqrt(nplots))
nr = math.ceil(nplots/nc)

cont_x = np.linspace(data.points[0], data.points[-1], 100)

pars = model.expected_pars(data.poi)
channel_offset = model.channel_offsets[channel.name]
sample_index = model.sample_indices[sample.name]
nexp0 = model.n_exp(pars)[sample_index, channel_offset:channel_offset + channel.dim()]
ax_vars = []
ax_invs = []

def nexp_var(pars, par, x) :
  return model.n_exp(pars.set(par, x))[sample_index, channel_offset:channel_offset + channel.dim()]

for b in bins :
  fig = plt.figure(figsize=(8, 8), dpi=96)
  fig.suptitle('Linearity checks for sample %s, bin [%g, %g]'  % (sample.name, channel.bins[b]['lo_edge'], channel.bins[b]['hi_edge']))
  gs = gridspec.GridSpec(nrows=nr, ncols=nc, wspace=0.3, hspace=0.3, top=0.9, bottom=0.05, left=0.1, right=0.95)
  for i, par in enumerate(model.nps.keys()) :
    if not options.vars_only :
      sgs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i//nc, i % nc], wspace=0.1, hspace=0.1)
    else :
      sgs = [ gs[i//nc, i % nc] ]
    pars = model.expected_pars(data.poi)
    model.linear_nps = True
    vars_lin = [ nexp_var(pars, par, x)[b]/nexp0[b] - 1 for x in cont_x ]
    rvar_lin = [ -((nexp_var(pars, par, x)[b] - nexp0[b])/nexp0[b])**2 for x in cont_x ]
    model.linear_nps = False
    vars_nli = [ nexp_var(pars, par, x)[b]/nexp0[b] - 1 for x in cont_x ]
    rvar_nli = [ -((nexp_var(pars, par, x)[b] - nexp0[b])/nexp0[b])**2 for x in cont_x ]

    ax_var = fig.add_subplot(sgs[0])
    ax_var.set_title(par)
    ax_var.plot(data.points, data.variations[par][sample_index, b, :] - 1, 'ko')
    ax_var.plot(cont_x, vars_lin, 'r--')
    ax_vars.append(ax_var)
    if not options.no_nli : ax_var.plot(cont_x, vars_nli, 'b')
    if options.yrange : ax_var.set_ylim(y_min, y_max)
    if not options.vars_only :
      ax_inv = fig.add_subplot(sgs[1], sharex=ax_var)
      ax_inv.plot(cont_x, rvar_lin, 'r--')
      ax_inv.plot(cont_x, rvar_nli, 'b')
      if options.inv_range : ax_inv.set_ylim(-options.inv_range, 0)
    ax_invs.append(ax_inv)
  fig.canvas.set_window_title('Linearity checks for sample %s, bin  %g' % (sample.name, b))

if options.output_file != '' : plt.savefig(options.output_file)
