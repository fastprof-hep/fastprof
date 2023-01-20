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

from fastprof import Model, Data, ModelPOI, Expression, ModelReparam, Norm
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
  parser.add_argument("-e", "--expressions" , type=str  , default=None , help="New expressions to add (comma-separated list of the form name:type:data)")
  parser.add_argument("-a", "--add"         , type=str  , default=None , help="POI(s) to be added or modified (comma-separated list)")
  parser.add_argument("-r", "--remove"      , type=str  , default=None , help="POI(s) to be removed (comma-separated list)")
  parser.add_argument(      "--replacements", type=str  , default=None , help="Replacement values for removed POIs, in the form name:value [default: use initial values]")
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

  reparam = ModelReparam(model, verbosity=options.verbosity)
  norms = {}

  # load options from JSON file instead of the command line, if specified
  fdict = {}
  if options.file is not None :
    try :
      with open(options.file, 'r') as fd :
        fdict = json.load(fd)
    except FileNotFoundError as inst :
      print("File '%s' not found." % options.file)
      return
    except Exception as inst :
      print(inst)
      print("Could not load data from '%s'." % options.file)
      return

  if 'norms' in fdict : # from file
    try :
      for norm_dict in fdict['norms'] :
        norms[(norm_dict['channel'], norm_dict['sample'])] = Norm.instantiate({ 'norm' : norm_dict['norm'] })
    except Exception as inst :
      print(inst)
      print("Could not load norm data from '%s'." % options.file)
      return

  if options.norms is not None :
    try:
      norm_specs = options.norms.replace(' ', '').replace('\n', '').split(',')
      for spec in norm_specs :
        if spec == '' : continue
        spec_names = spec.split(':')
        if len(spec_names) == 1 : spec_names = [ '.*' ] + spec_names
        if len(spec_names) == 2 : spec_names = [ '.*' ] + spec_names
        norms[(spec_names[0], spec_names[1])] = Norm.instantiate({ 'norm' : spec_names[2] })
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid normalization factor specification %s : should be of the form chan1:samp1:norm1,chan2:samp2:norm2,...' % options.norms)

  add_expressions = []
  
  if 'expressions' in fdict : # from file
    try :
      for expr_dict in fdict['expressions'] :
        add_expressions.append(Expression.instantiate(expr_dict))
    except Exception as inst :
      print(inst)
      print("Could not load expression data from '%s'." % options.file)
      return
  
  if options.expressions is not None :
    expr_dict = {}
    try:
      expr_specs = options.expressions.replace(' ', '').replace('\n', '').split(',')
      for spec in expr_specs :
        if spec == '' : continue
        spec_names = spec.split(':')
        if len(spec_names) < 2 : raise ValueError('Only %d element(s) found in the specification' % len(spec_names))
        if len(spec_names) == 2 : spec_names = [ spec_names[0], 'formula', spec_names[1] ]
        expr_dict = { 'name' : spec_names[0], 'type' : spec_names[1] }
        if spec_names[1] == 'formula' : 
          expr_dict['formula'] = spec_names[2]
        elif spec_names[1] == 'linear_combination' : 
          expr_dict['coeffs'] = {}
          for pc_spec in spec_names[2].split('#') :
            par_name, coeff_str = pc_spec.split('*')
            expr_dict['coeffs'][par_name] = float(coeff_str)
        else :
          raise ValueError("Invalid expression type '%s'." % spec_names[1])
        expr = Expression.instantiate(expr_dict, load_data=True)
        add_expressions.append(expr)
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid expression specification %s : should be of the form name1:type1:data1,...' % options.expressions)    

  add_pois = []
  remove_pois = []
  change_pois = []
  new_pois = {}
  replacements = {}

  if 'replacements' in fdict : # from file
    try :
      for par, value in fdict['replacements'].items() :
        replacements[par] = value
    except Exception as inst :
      print(inst)
      print("Could not load replacement data from '%s'." % options.file)
      return

  if options.replacements is not None :
    try :
      for repl_spec in options.replacements.split(',') :
        par, value = repl_spec.replace(' ', '').split(':')
        replacements[par] = value
    except Exception as inst :
      print(inst)
      print("Could not parse replacement data")
      return

  if 'POIs' in fdict : # from file
    try :
      for poi in fdict['POIs'] :
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

  if options.add is not None :
    pois = process_pois(options.add, model, check_pars=False)
    for poi in pois :
      if poi.name in model.pois :
        change_pois.append(poi)
      else :
        add_pois.append(poi)
    
  if options.remove is not None :
    remove_pois = [ v.replace(' ', '') for v in options.remove.split(',') ]

  if len(add_pois) > 0 :
    reparam.add_pois(add_pois)
  if len(add_expressions) > 0 :
    reparam.add_expressions(add_expressions)
  if len(norms) > 0 :
    reparam.update_norms(norms)
  if len(change_pois) > 0 :
    for poi in change_pois :
      if options.verbosity > 0 : print("Modifying POI '%s', now '%s'." % (poi.name, str(poi)))
      model.pois[poi.name] = poi
  if len(remove_pois) > 0 :
    for poi in remove_pois :
      if poi not in replacements :
        replacements[poi] = model.pois[poi].initial_value
        if options.verbosity > 0 : print("Using default replacement '%s=%g' when removing POI '%s'." % (poi, replacements[poi], poi))
    reparam.remove_pois(remove_pois, values=replacements)

  # If we specified a full list of POIs, reorder the POIs to match the list order
  if len(new_pois) > 0 : model.pois = {name : model.pois[name] for name in new_pois}

  if options.verbosity >= 1 : print('Saving model to file %s' % options.output_file)
  model.save(options.output_file)

if __name__ == '__main__' : run()
