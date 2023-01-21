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

from fastprof import Model, Data, NPPruner, SamplePruner
from fastprof_utils import process_setvals


####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("prune_model.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"        , type=str  , required=True    , help="Name of markup file defining model")
  parser.add_argument("-d", "--data-file"         , type=str  , default=None     , help="Use the dataset stored in the specified markup file")
  parser.add_argument("-o", "--output-file"       , type=str  , required=True    , help="Name of output file")
  parser.add_argument("-p", "--nps"               , type=str  , default=None     , help="List of NPs to prune (par1=val1,par2=val2...)")
  parser.add_argument("-i", "--min-np-impact"     , type=float, default=None     , help="Prune away NPs with an impact below the specified threshold")  
  parser.add_argument("-z", "--min-sample-signif" , type=float, default=None     , help="Prune away samples with significance below the specified threshold")  
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

  if options.min_sample_signif is not None :
    SamplePruner(model, options.verbosity).prune(options.min_sample_signif)

  if options.nps is not None :
    par_dict = process_setvals(options.nps, model, scale_nps = True)
    NPPruner(model, options.verbosity).remove_nps(par_dict)
  
  if options.min_np_impact is not None :
    NPPruner(model, options.verbosity).prune(options.min_np_impact)

  if options.data_file :
    data = Data(model).load(options.data_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    if options.verbosity >= 1 : print('Using dataset stored in file %s.' % options.data_file)
  else :
    data = Data(model).load(options.model_file)

  if data is not None :
    data.save_with_model('%s.json' % options.output_file)
  else :
    model.save('%s.json' % options.output_file)

if __name__ == '__main__' : run()
