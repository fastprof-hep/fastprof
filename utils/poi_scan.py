#! /usr/bin/env python

__doc__ = """
*Perform a PLR scan over one or more parameters*

"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import json

from fastprof import Model, Data, Parameters, OptiMinimizer, Raster, TMuCalculator, ParBound, PLRScan
from utils import process_setval_list


####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("compute_limits_fast.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"    , type=str  , required=True , help="Name of markup file defining model")
  parser.add_argument("-d", "--data-file"     , type=str  , default=None  , help="Use the dataset stored in the specified markup file")
  parser.add_argument("-a", "--asimov"        , type=str  , default=None  , help="Use an Asimov dataset for the specified POI values (format: 'poi1=xx,poi2=yy'")
  parser.add_argument("-y", "--hypos"         , type=str  , required=True , help="List of POI hypothesis values (poi1=val1,poi2=val2#...)")
  parser.add_argument("-n", "--nsigmas"       , type=float, default=1     , help="Confidence level at which to compute the limit")
  parser.add_argument("-c", "--cl"            , type=float, default=None  , help="Confidence level at which to compute the limit")
  parser.add_argument("-o", "--output-file"   , type=str  , required=True , help="Name of output file")
  parser.add_argument("-i", "--iterations"    , type=int  , default=1     , help="Number of iterations to perform for NP computation")
  parser.add_argument(      "--regularize"    , type=float, default=None  , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
  parser.add_argument(      "--cutoff"        , type=float, default=None  , help="Cutoff to regularize the impact of NPs")
  parser.add_argument(      "--bounds"        , type=str  , default=None  , help="Parameter bounds in the form name1=[min]#[max],name2=[min]#[max],...")
  parser.add_argument(      "--marker"        , type=str  , default=''    , help="Marker type for plots")
  parser.add_argument(      "--batch-mode"    , action='store_true'       , help="Batch mode: no plots shown")
  parser.add_argument("-v", "--verbosity"     , type=int  , default=1     , help="Verbosity level")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args(argv)
  if not options :
    parser.print_help()
    sys.exit(0)

  model = Model.create(options.model_file)
  if model is None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
  if not options.regularize is None : model.set_gamma_regularization(options.regularize)
  if not options.cutoff is None : model.cutoff = options.cutoff

  if options.data_file :
    data = Data(model).load(options.data_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    print('Using dataset stored in file %s.' % options.data_file)
  elif options.asimov != None :
    try:
      sets = [ v.replace(' ', '').split('=') for v in options.asimov.split(',') ]
      data = model.generate_expected(sets)
    except Exception as inst :
      print(inst)
      raise ValueError("Cannot define an Asimov dataset from options '%s'." % options.asimov)
    print('Using Asimov dataset with POIs %s.' % str(sets))
  else :
    data = Data(model).load(options.model_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    print('Using dataset stored in file %s.' % options.model_file)

  try :
    hypos = [ Parameters(setval_dict, model=model) for setval_dict in process_setval_list(options.hypos, model) ]
  except Exception as inst :
    print(inst)
    raise ValueError("Could not parse list of hypothesis values '%s' : expected colon-separated list of variable assignments" % options.hypos)

  par_bounds = []
  if options.bounds :
    bound_specs = options.bounds.split(',')
    try :
      for spec in bound_specs :
        var_range = spec.split('=')
        range_spec = var_range[1].split('#')
        if len(range_spec) == 2 :
          par_bounds.append(ParBound(var_range[0], float(range_spec[0]) if range_spec[0] != '' else None, float(range_spec[1]) if range_spec[1] != '' else None))
        elif len(range_spec) == 1 :
          par_bounds.append(ParBound(var_range[0], float(range_spec[0]), float(range_spec[0]))) # case of fixed parameter
    except Exception as inst:
      print(inst)
      raise ValueError('Could not parse parameter bound specification "%s", expected in the form name1=[min]#[max],name2=[min]#[max],...' % options.bounds)


  # Compute the tmu values
  if len(model.pois) > 1 : raise ValueError('Currently not supporting more than 1 POI for this operation')
  calc = TMuCalculator(OptiMinimizer(niter=options.iterations).set_pois_from_model(model, par_bounds))
  print('Producing PLR scan with POI(s) %s, bounds %s and niter=%d.' % (str(calc.minimizer.free_pois()), str(calc.minimizer.bounds), calc.minimizer.niter))
  poi_name = calc.minimizer.free_pois()[0]
  if len(calc.minimizer.free_pois()) > 1 : raise ValueError('Currently not supporting more than 1 POI for this operation')
  raster = calc.compute_fast_results(hypos, data)
  #print(raster)

  raster.print(keys=[ 'tmu', 'pv' ], verbosity=1)

  poi_scan = PLRScan(raster, 'tmu', name='PLR Scan for %s' % poi_name, nsigmas=options.nsigmas, cl=options.cl)
  interval = poi_scan.interval(print_result=True)
  if interval is None : return
  # Plot results
  if not options.batch_mode :
    plt.ion()
    fig1 = plt.figure(1)
    poi_scan.plot(plt, marker=options.marker + 'b-', label='PRL', smooth=100)
    plt.ylim(0, None)
    plt.axhline(y=1 - poi_scan.cl, color='k', linestyle='dotted')
    plt.show()

  jdict = {}
  jdict['cl'] = poi_scan.cl
  jdict['poi_name'] = poi_name
  jdict['poi_unit'] = model.pois[poi_name].unit
  jdict['central_value']  = interval[0]
  jdict['uncertainty_up'] = interval[1]
  jdict['uncertainty_dn'] = interval[2]

  with open(options.output_file + '_results.json', 'w') as fd:
    json.dump(jdict, fd, ensure_ascii=True, indent=3)

if __name__ == '__main__' : run()
