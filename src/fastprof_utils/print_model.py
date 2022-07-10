#! /usr/bin/env python

__doc__ = """
*Print out model [and dataset] information*

Prints out information on a model (defined using the `--model-file` argument)
and optionally  a dataset (`--data-file` argument).
"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fastprof import Model, Data, OptiMinimizer, NPMinimizer, QMuTildaCalculator, PlotResults
from fastprof_utils import process_setvals, process_setranges
import matplotlib.pyplot as plt
import numpy as np
import json
import time

####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("fit_model.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"       , type=str  , required=True   , help="Name of markup file defining model")
  parser.add_argument("-d", "--data-file"        , type=str  , default=''      , help="Name of markup file defining the dataset (optional, otherwise taken from model file)")
  parser.add_argument("-v", "--verbosity"        , type = int, default=1       , help="Verbosity level")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args()
  if not options :
    parser.print_help()
    return

  model = Model.create(options.model_file, verbosity=options.verbosity)
  if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)

  if options.data_file :
    data = Data(model).load(options.data_file)
    if data is None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
  else :
    data = Data(model).load(options.model_file)

  print('Model ' + model.string_repr(verbosity=options.verbosity))
  if data is not None : 
    print('\n============================\n')
    print('Data ' + data.string_repr(verbosity=options.verbosity))


if __name__ == '__main__' : run()
