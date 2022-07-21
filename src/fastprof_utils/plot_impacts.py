#! /usr/bin/env python

__doc__ = """
*Plot nuisance parameter impacts*

"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import json

from fastprof import Model, Data, PlotImpacts


####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("prune_model.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"        , type=str  , required=True    , help="Name of markup file defining model")
  parser.add_argument("-d", "--data-file"         , type=str  , default=None     , help="Use the dataset stored in the specified markup file")
  parser.add_argument("-o", "--output-file"       , type=str  , required=True    , help="Name of output file")
  parser.add_argument("-p", "--poi"               , type=str  , required=True    , help="Parameter of interest on which to compute impacts")
  parser.add_argument("-n", "--max_npar"          , type=int  , default=20       , help="Maximum number of NPs to plot")
  parser.add_argument("-v", "--verbosity"         , type=int  , default=0        , help="Verbosity level")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args(argv)
  if not options :
    parser.print_help()
    sys.exit(0)
  if options.verbosity >= 1 : print('Loading model from file %s' % options.model_file)
  model = Model.create(options.model_file, verbosity=options.verbosity)
  if model is None : raise ValueError('No valid model definition found in file %s.' % options.model_file)

  if options.output_file.endswith('.json') : options.output_file = options.output_file[:-5]
  if options.data_file :
    data = Data(model).load(options.data_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    if options.verbosity >= 1 : print('Using dataset stored in file %s.' % options.data_file)
  else :
    data = Data(model).load(options.model_file)

  impacts = PlotImpacts(options.poi, data, verbosity=options.verbosity)
  impacts.plot(options.max_npar, output=options.output_file)

if __name__ == '__main__' : run()
