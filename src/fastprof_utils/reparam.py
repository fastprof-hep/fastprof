#! /usr/bin/env python

__doc__ = """
*Reparameterize a model with different POIs and normalization terms*

"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import json

from fastprof import Model, Data, ModelPOI, ModelReparam, NumberNorm, ParameterNorm, FormulaNorm
from fastprof_utils import process_setval_list, process_setvals, process_setranges, process_pois


####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("reparam.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"  , type=str  , required=True, help="Name of markup file defining model")
  parser.add_argument("-o", "--output-file" , type=str  , required=True, help="Name of output file")
  parser.add_argument("-n", "--norms"       , type=str  , default=None , help="New normalization terms to apply, in the form channel1:sample1:norm1,...")
  parser.add_argument("-f", "--file"        , type=str  , default=None , help="Best-fit computation: at all points (all), at best point (single) or just the best fixed fit (best_fixed)")
  parser.add_argument("-a", "--add"         , type=str  , default=None , help="POI(s) to be added or modified (comma-separated list of )")
  parser.add_argument("-r", "--remove"      , type=str  , default=None , help="POI(s) to be removed (comma-separated)")
  parser.add_argument("-v", "--verbosity"   , type=int  , default=0    , help="Verbosity level")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args(argv)
  if not options :
    parser.print_help()
    sys.exit(0)
  if options.verbosity >= 1 : print('Initializing model from file %s' % options.model_file)
  model = Model.create(options.model_file, verbosity=options.verbosity)
  if model is None : raise ValueError('No valid model definition found in file %s.' % options.model_file)

  reparam = ModelReparam(model)
  norms = {}

  if options.norms is not None :
    try:
      norm_specs = options.norms.replace(' ', '').replace('\n', '').split(',')
      for spec in norm_specs :
        if spec == '' : continue
        spec_names = spec.split(':')
        if len(spec_names) == 1 : spec_names = [ '.*' ] + spec_names
        if len(spec_names) == 2 : spec_names = [ '.*' ] + spec_names
        norms[(spec_names[0], spec_names[1])] = spec_names[2]
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid normalization factor specification %s : should be of the form chan1:samp1:norm1,chan2:samp2:norm2,...' % options.norms)

  if options.file is not None :
    sdict = {}
    try :
      with open(options.file, 'r') as fd :
        sdict = json.load(fd)
    except FileNotFoundError as inst :
      print("File '%s' not found." % options.file)
      return
    except Exception as inst :
      print(inst)
      print("Could not load data from '%s'." % options.file)
      return
  
  if options.file is not None and 'norms' in sdict :
    try:
      for norm in sdict['norms'] :
        norm_spec = norm['norm']
        channel = norm['channel'] if 'channel' in norm else ''
        sample = norm['sample'] if 'sample' in norm else ''
        norms[(channel, sample)] = norm_spec
    except Exception as inst :
      print(inst)
      print("Could not load norm data from '%s'." % options.file)
      return

  if len(norms) > 0 :
    parsed_norms = {}
    for (channel, sample), norm in norms.items() :
      if norm in model.pois or norm in model.nps :
        parsed_norms[(channel, sample)] = ParameterNorm(norm)
        continue
      if isinstance(norm, (int, float)) or norm == '' :
        parsed_norms[(channel, sample)] = NumberNorm(norm)
        continue
      try:
        float(norm)
        parsed_norms[(channel, sample)] = NumberNorm(float(norm))
      except Exception as inst:
        parsed_norms[(channel, sample)] = FormulaNorm(norm)
    reparam.update_norms(parsed_norms, verbosity=options.verbosity)

  add_pois = []
  remove_pois = []
  change_pois = []

  if options.add is not None :
    pois = process_pois(options.add, model, check_pars=False)
    for poi in pois :
      if poi.name in model.pois :
        change_pois.append(poi)
      else :
        add_pois.append(poi)
    
  if options.remove is not None :
    remove_pois = [ v.replace(' ', '') for v in options.remove.split(',') ]

  if options.file is not None and 'POIs' in sdict :
    new_pois = {}
    try :
      for poi in sdict['POIs'] :
        name = poi['name']
        new_poi = ModelPOI().load_dict(poi)
        new_pois[new_poi.name] = new_poi
    except Exception as inst :
      print(inst)
      print("Could not load POI data from '%s'." % options.file)
      return
    for new_poi in new_pois.values() :
      if new_poi.name not in model.pois :
        add_pois.append(new_poi)
      else :
        change_pois.append(new_poi)
    for poi in model.pois.values() :
      if poi.name not in new_pois :
        remove_pois.append(poi.name)

  if len(add_pois) > 0 :
    reparam.add_pois(add_pois, verbosity=options.verbosity)
  if len(change_pois) > 0 :
    for poi in change_pois : 
      if options.verbosity > 0 : print("Modifying POI '%s', now '%s'." % (poi.name, str(poi)))
      model.pois[poi.name] = poi
  if len(remove_pois) > 0 :
    reparam.remove_pois(remove_pois, verbosity=options.verbosity)

  if options.verbosity >= 1 : print('Saving model to file %s' % options.output_file)
  model.save(options.output_file)

if __name__ == '__main__' : run()
