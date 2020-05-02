#! /usr/bin/env python

__doc__ = "Plot a fast model"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fastprof import Model, Data
import matplotlib.pyplot as plt

####################################################################################################################################
###

def fit_fast() :
  """fit_fast"""
  
  parser = ArgumentParser("fit_fast.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file",        required=True,     help="Name of JSON file defining model", type=str)
  parser.add_argument("-d", "--data-file",         default='',        help="Name of JSON file defining the dataset (optional, otherwise taken from model file)", type=str)
  parser.add_argument("-a", "--asimov",            default=None,      help="Fit an Asimov dataset for the specified POI value", type=float)
  parser.add_argument("-i", "--poi-initial-value", default=0,         help="POI initial value", type=float)
  parser.add_argument("-r", "--poi-range",         default='0,20',    help="POI allowed range, in the form min,max", type=str)
  parser.add_argument("-l", "--log-scale",    action='store_true',    help="Use log scale for plotting")
  #parser.add_argument("-o", "--output-file",       required=True,     help="Name of output file", type=str)
  parser.add_argument("-v", "--verbosity",         default=0,         help="Verbosity level", type=int)
  
  options = parser.parse_args()
  if not options : 
    parser.print_help()
    return

  model = Model.create(options.model_file)
  if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
  if options.data_file :
    data = Data(model).load(options.data_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    print('Using dataset stored in file %s.' % options.data_file)
  elif options.asimov != None :
    data = Data(model).set_expected(model.expected_pars(options.asimov))
    print('Using Asimov dataset with %s = %g' % (full_results.poi_name, options.asimov))
  else :
    data = Data(model).load(options.model_file)
    
  if options.poi_range != '' :
    try:
      poi_min, poi_max = [ float(p) for p in options.poi_range.split(',') ]
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid POI range specification %s, expected poi_min,poi_max' % options.poi_range)


  mini = OptiMinimizer(data, options.poi_initial_value, (poi_min, poi_max))
  nll_min, min_mu = mini.minimize()
  print('Minimum: nll = %g @ POI = %g, NP values :' % (nll_min, min_mu))
  print(mini.min_pars)


  plt.ion()
  plt.figure(1)
  model.plot(mini.min_pars, data=data)
  if options.log_scale : plt.yscale('log')


fit_fast()
