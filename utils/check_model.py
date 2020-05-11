#! /usr/bin/env python

__doc__ = "Check asymptotic results of the fast model against those of the full model"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json

from fastprof import Model, Data, FitResults, QMuTildaCalculator, OptiMinimizer

####################################################################################################################################
###

def check_model() :
  """convert """
  
  parser = ArgumentParser("check_model.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file", default='',      help="Name of JSON file defining model", type=str)
  parser.add_argument("-d", "--data-file",  default='',      help="Name of JSON file defining the dataset (optional, otherwise taken from model file)", type=str)
  parser.add_argument("-a", "--asimov",     default=None,    help="Fit an Asimov dataset for the specified POI value", type=float)
  parser.add_argument("-f", "--fits-file",  default='',      help="Name of JSON file containing full-model fit results", type=str)
  parser.add_argument("-v", "--verbosity",  default=0,       help="Verbosity level", type=int)
  
  options = parser.parse_args()
  if not options :
    parser.print_help()
    return
 
  results = FitResults(options.fits_file)

  model = Model.create(options.model_file)
  if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)

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
  calc = QMuTildaCalculator(OptiMinimizer(data, results.poi_initial_value, (results.poi_min, results.poi_max)), results)
  calc.fill_qcl()
  calc.fill_fast_results()
  results.print(verbosity = options.verbosity)

check_model()
