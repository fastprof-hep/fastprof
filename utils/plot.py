#! /usr/bin/env python

__doc__ = "Plot a model and dataset"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fastprof import Model, Data, OptiMinimizer
import matplotlib.pyplot as plt
import math

####################################################################################################################################
###

parser = ArgumentParser("plot.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-m", "--model-file" , type=str  , required=True , help="Name of JSON file defining model")
parser.add_argument("-d", "--data-file"  , type=str  , default=''    , help="Name of JSON file defining the dataset (optional, otherwise taken from model file)")
parser.add_argument("-a", "--asimov"     , type=float, default=None  , help="Fit an Asimov dataset for the specified POI value")
parser.add_argument("-p", "--poi-value"  , type=float, required=True , help="POI initial value")
parser.add_argument("-r", "--poi-range"  , type=str  , default='0,20', help="POI allowed range, in the form min,max")
parser.add_argument("-x", "--x-range"    , type=str  , default=''    , help="X-axis range, in the form min,max")
parser.add_argument("-y", "--y-range"    , type=str  , default=''    , help="Y-axis range, in the form min,max")
parser.add_argument(      "--profile"    , action='store_true'       , help="Perform a conditional fit for the provided POI value before plotting")
parser.add_argument("-l", "--log-scale"  , action='store_true'       , help="Use log scale for plotting")
parser.add_argument("-s", "--variations" , type=str  , default=''    , help="Plot variations for parameters par1=val1[:color],par2=val2[:color]... or a single value for all parameters")
parser.add_argument("-o", "--output-file", type=str  , default=''    , help="Output file name")
parser.add_argument("-v", "--verbosity"  , type=int  , default=0     , help="Verbosity level")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

model = Model.create(options.model_file)
if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
if options.data_file :
  data = Data(model).load(options.data_file)
  if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
  print('Using dataset stored in file %s.' % options.data_file)
elif options.asimov != None :
  data = model.generate_expected(sets)
  print('Using Asimov dataset with POI = %g' % options.asimov)
else :
  data = Data(model).load(options.model_file)

if options.profile :
  try:
    poi_min, poi_max = [ float(p) for p in options.poi_range.split(',') ]
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid POI range specification %s, expected poi_min,poi_max' % options.poi_range)
  mini = OptiMinimizer(data, options.poi_value, (poi_min, poi_max))
  nll_min, min_mu = mini.minimize()
  print('Minimum: nll = %g @ POI = %g, NP values :' % (nll_min, min_mu))
  pars = mini.mini_pars
else :
  pars = model.expected_pars(options.poi_value)

plt.ion()
plt.figure(1)
model.plot(pars, data=data)
if options.log_scale : plt.yscale('log')

varias_a = []
varias_b = []
varias_c = []
varias = []
if options.variations != '' :
  # First try the comma-separated format
  varias = []
  try:
    for spec in options.variations.split(',') :
      try :
        varval,color = spec.split(':')
      except:
        varval = spec
        color = None
      var,val = varval.split('=')
      try :
        val = float(val)
      except:
        raise ValueError('Invalid numerical value %s.' % val)
      if var in model.alphas :
        if color == None : color = 'r'
      elif var in model.betas :
        if color == None : color = 'r'
      elif var in model.gammas :
        if color == None : color = 'g'
      else:
        raise KeyError('Parameter %s is not defined in the model.' % var)
      varias.append( (var, val, color,) )
  except:
    varias = None
    try:
      var_val = float(options.variations)
    except:
      raise ValueError('Invalid variations specification %s : should be a comma-separated list of var=val[:color] items, or a single number')

if varias != None :
  plt.figure(1)
  model.plot(pars, data=data, variations=varias)
  if options.log_scale : plt.yscale('log')
else :
  if model.na > 0 :
    nc = math.ceil(math.sqrt(model.na))
    fig_a, ax_a = plt.subplots(nrows=nc, ncols=nc)
    for alpha, ax in zip(model.alphas, ax_a.flatten()) :
      model.plot(pars, data=data, variations = [ (alpha, var_val, 'r'), (alpha, -var_val, 'r') ], canvas=ax)
      if options.log_scale : ax.set_yscale('log')
  if model.nb > 0 :
    nc = math.ceil(math.sqrt(model.nb))
    fig_b, ax_b = plt.subplots(nrows=nc, ncols=nc)
    for beta, ax in zip(model.betas, ax_b.flatten()) :
      model.plot(pars, data=data, variations = [ (beta, var_val, 'r'), (beta, -var_val, 'r.') ], canvas=ax)
      if options.log_scale : ax.set_yscale('log')
  if model.nc > 0 :
    nc = math.ceil(math.sqrt(model.nc))
    fig_c, ax_c = plt.subplots(nrows=nc, ncols=nc)
    for gamma, ax in zip(model.gammas, ax_c.flatten()) :
      model.plot(pars, data=data, variations = [ (gamma, var_val, 'g'), (gamma, -var_val, 'g') ], canvas=ax)
      if options.log_scale : ax.set_yscale('log')

if options.x_range :
  try:
    x_min, x_max = [ float(p) for p in options.x_range.split(',') ]
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid X-axis range specification %s, expected x_min,x_max' % options.x_range)
  plt.xlim(x_min, x_max)

if options.y_range :
  try:
    y_min, y_max = [ float(p) for p in options.y_range.split(',') ]
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid Y-axis range specification %s, expected y_min,y_max' % options.y_range)
  plt.ylim(y_min, y_max)

if options.output_file != '' : plt.savefig(options.output_file)
