#! /usr/bin/env python

__doc__ = """
*Prune model NPs*

"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import json

from fastprof import Model, Data, NPPruner


####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("prune_nps.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"    , type=str  , required=True    , help="Name of markup file defining model")
  parser.add_argument("-d", "--data-file"     , type=str  , default=None     , help="Use the dataset stored in the specified markup file")
  parser.add_argument("-o", "--output-file"   , type=str  , required=True    , help="Name of output file")
  parser.add_argument("-p", "--nps"           , type=str  , required=True    , help="List of NPs to prune")
  parser.add_argument("-v", "--verbosity"     , type=int  , default=0        , help="Verbosity level")
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


  nps = options.nps.replace(' ', '').replace('\n', '').split(',')
  NPPruner(model).remove_nps([ 'lumi', 'shape*', 'Fakes*', 'theory*', 'FF*', 'EL*', 'MUON*', 'JET*', 'MET*', 'param*', 'mu*' ])
  model.save('%s.json' % options.output_file)

  if options.data_file :
    data = Data(model).load(options.data_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    if options.verbosity >= 1 : print('Using dataset stored in file %s.' % options.data_file)
    data.save('%s_data.json' % options.output_file)

if __name__ == '__main__' : run()
