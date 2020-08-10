#! /usr/bin/env python

__doc__ = "Check asymptotic results of the fast model against those of the full model"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import json

from fastprof import Model, Data, FitResults, QMuCalculator, QMuTildaCalculator, OptiMinimizer

####################################################################################################################################
###

parser = ArgumentParser("check_model.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-m", "--model-file"    , type=str  , default=''     , help="Name of JSON file defining model")
parser.add_argument("-d", "--data-file"     , type=str  , default=''     , help="Name of JSON file defining the dataset (optional, otherwise taken from model file)")
parser.add_argument("-a", "--asimov"        , type=float, default=None   , help="Fit an Asimov dataset for the specified POI value")
parser.add_argument("-f", "--fits-file"     , type=str  , default=''     , help="Name of JSON file containing full-model fit results")
parser.add_argument(      "--regularize"    , type=float, default=None   , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
parser.add_argument("-t", "--test-statistic", type=str  , default='q~mu' , help="Test statistic to use in the check")
parser.add_argument(      "--marker"        , type=str  , default=''     , help="Marker type for plots")
parser.add_argument("-b", "--batch-mode"    , action='store_true'        , help="Batch mode: no plots shown")
parser.add_argument("-v", "--verbosity"     , type=int  , default=0      , help="Verbosity level")
parser.add_argument("-o", "--output-file"   , type=str  , default='check', help="Output file name")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

model = Model.create(options.model_file)
if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
if options.regularize != None : model.set_gamma_regularization(options.regularize)

results = FitResults(model, options.fits_file)

if options.data_file :
  data = Data(model).load(options.data_file)
  if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
  print('Using dataset stored in file %s.' % options.data_file)
elif options.asimov != None :
  data = Data(model).set_expected(model.expected_pars(options.asimov))
  print('Using Asimov dataset with %s = %g.' % (results.poi_name, options.asimov))
else :
  print('Using dataset stored in file %s.' % options.model_file)
  data = Data(model).load(options.model_file)

if options.test_statistic == 'q~mu' :
  calc = QMuTildaCalculator(OptiMinimizer(data, results.poi_initial_value, (results.poi_min, results.poi_max)), results)
elif options.test_statistic == 'q_mu' :
  calc = QMuCalculator(OptiMinimizer(data, results.poi_initial_value, (results.poi_min, results.poi_max)), results)
else :
  raise ValueError('Unknown test statistic %s' % options.test_statistic)
calc.fill_qpv()
calc.fill_fast_results()
results.print(verbosity = options.verbosity)

# Plot results
if not options.batch_mode :
  plt.ion()
  fig1 = plt.figure(1)
  plt.suptitle('$CL_{s+b}$')
  plt.xlabel(model.poi_name)
  plt.ylabel('$CL_{s+b}$')
  plt.plot(results.hypos, [ fit_result['pv']      for fit_result in results.fit_results ], options.marker + 'r:' , label = 'Full model')
  plt.plot(results.hypos, [ fit_result['fast_pv'] for fit_result in results.fit_results ], options.marker + 'g-' , label = 'Fast model')
  plt.legend()

  fig2 = plt.figure(2)
  plt.suptitle('$CL_s$')
  plt.xlabel(model.poi_name)
  plt.ylabel('$CL_s$')
  plt.plot(results.hypos, [ fit_result['cls']      for fit_result in results.fit_results ], options.marker + 'r:' , label = 'Full model')
  plt.plot(results.hypos, [ fit_result['fast_cls'] for fit_result in results.fit_results ], options.marker + 'g-' , label = 'Fast model')
  plt.legend()
  fig1.savefig(options.output_file + '_clsb.pdf')
  fig2.savefig(options.output_file + '_cls.pdf')
  plt.show()
