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

from fastprof import Model, Data, Parameters, OptiMinimizer, Raster, TMuCalculator, ParBound, PLRScan1D, PLRScan2D
from utils import process_setval_list, process_setvals


####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("poi_scan.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"    , type=str  , required=True , help="Name of markup file defining model")
  parser.add_argument("-d", "--data-file"     , type=str  , default=None  , help="Use the dataset stored in the specified markup file")
  parser.add_argument("-a", "--asimov"        , type=str  , default=None  , help="Use an Asimov dataset for the specified POI values (format: 'poi1=xx,poi2=yy'")
  parser.add_argument("-y", "--hypos"         , type=str  , required=True , help="List of POI hypothesis values (poi1=val1,poi2=val2#...)")
  parser.add_argument("-n", "--nsigmas"       , type=float, default=1     , help="Confidence level at which to compute the limit")
  parser.add_argument("-c", "--cl"            , type=float, default=None  , help="Confidence level at which to compute the limit")
  parser.add_argument("-o", "--output-file"   , type=str  , required=True , help="Name of output file")
  parser.add_argument("-b", "--best-fit-mode" , type=str  , default='all' , help="Best-fit computation: at all points (all), at best point (single) or just the best fixed fit (best_fixed)")
  parser.add_argument("-i", "--iterations"    , type=int  , default=1     , help="Number of iterations to perform for NP computation")
  parser.add_argument(      "--regularize"    , type=float, default=None  , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
  parser.add_argument(      "--cutoff"        , type=float, default=None  , help="Cutoff to regularize the impact of NPs")
  parser.add_argument(      "--bounds"        , type=str  , default=None  , help="Parameter bounds in the form name1=[min]#[max],name2=[min]#[max],...")
  parser.add_argument(      "--marker"        , type=str  , default='+'   , help="Marker type for plots")
  parser.add_argument(      "--smoothing"     , type=int  , default=0     , help="Smoothing for contours (0=no smoothing)")
  parser.add_argument(      "--batch-mode"    , action='store_true'       , help="Batch mode: no plots shown")
  parser.add_argument("-v", "--verbosity"     , type=int  , default=1     , help="Verbosity level")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args(argv)
  if not options :
    parser.print_help()
    sys.exit(0)
  if options.verbosity > 1 : print('Initializing model from file %s' % options.model_file)
  model = Model.create(options.model_file)
  if model is None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
  if not options.regularize is None : model.set_gamma_regularization(options.regularize)
  if not options.cutoff is None : model.cutoff = options.cutoff

  results_file = options.output_file + '_results.json'
  raster_file = options.output_file + '_raster.json'

  if options.data_file :
    if options.verbosity > 1 : print('Initializing data from file %s.' % options.data_file)
    data = Data(model).load(options.data_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    print('Using dataset stored in file %s.' % options.data_file)
  elif options.asimov != None :
    try :
      sets = process_setvals(options.asimov, model)
    except Exception as inst :
      print(inst)
      raise ValueError("ERROR : invalid POI specification string '%s'." % options.asimov)
    data = model.generate_expected(sets)
    if options.verbosity > 1 : print('Using Asimov dataset with parameters %s' % str(sets))
  else :
    if options.verbosity > 1 : print('Initializing data from file %s.' % options.model_file)
    data = Data(model).load(options.model_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    print('Using dataset stored in file %s.' % options.model_file)

  try :
    hypos = [ Parameters(setval_dict, model=model) for setval_dict in process_setval_list(options.hypos, model) ]
  except Exception as inst :
    print(inst)
    raise ValueError("Could not parse list of hypothesis values '%s' : expected #-separated list of variable assignments" % options.hypos)
  if options.verbosity > 1 : print('Will scan the following hypotheses : %s' % '\n- '.join([str(hypo) for hypo in process_setval_list(options.hypos, model)]))

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
  if len(model.pois) > 2 : raise ValueError('Currently not supporting more than 2 POIs for this operation')
  calc = TMuCalculator(OptiMinimizer(niter=options.iterations).set_pois_from_model(model, par_bounds))
  print('Producing PLR scan with POI(s) %s, bounds %s and niter=%d.' % (str(calc.minimizer.free_pois()), str(calc.minimizer.bounds), calc.minimizer.niter))
  if len(calc.minimizer.free_pois()) > 2 : raise ValueError('Currently not supporting more than 2 POIs for this operation')
  do_computation = True
  try :
    raster = Raster('fast', model=model)
    raster.load(raster_file)
    do_computation = False
  except FileNotFoundError :
    pass
  
  if do_computation :
    raster = calc.compute_fast_results(hypos, data, verbosity=options.verbosity, free_fit=options.best_fit_mode)
    raster.save(raster_file)

  raster.print(keys=[ 'tmu' ], verbosity=options.verbosity)
  jdict = {}

  if len(model.pois) == 1 :
    poi_name = calc.minimizer.free_pois()[0]
    poi_scan = PLRScan1D(raster, 'tmu', name='PLR Scan for %s' % poi_name, ts_name='t_{\mu}', nsigmas=options.nsigmas, cl=options.cl)
    interval = poi_scan.interval(print_result=True)
    # Plot results
    if not options.batch_mode :
      plt.ion()
      fig1 = plt.figure(1)
      poi_scan.plot(plt, marker='b-', label='PRL_smooth', smooth=100)
      poi_scan.plot(plt, marker=options.marker, label='PRL')
      plt.ylim(0, None)
      plt.axhline(y=poi_scan.ts_level, color='k', linestyle='dotted')
      plt.show()
    jdict = {}
    jdict['cl'] = poi_scan.cl()
    jdict['poi_name'] = poi_name
    jdict['poi_unit'] = model.pois[poi_name].unit
    jdict['central_value']  = interval[0] if interval is not None else None
    jdict['uncertainty_up'] = interval[1] if interval is not None else None
    jdict['uncertainty_dn'] = interval[2] if interval is not None else None
  else :
    poi1_name = calc.minimizer.free_pois()[0]
    poi2_name = calc.minimizer.free_pois()[1]
    poi_scan = PLRScan2D(raster, 'tmu', name='PLR Scan for (%s,%s)' % (poi1_name, poi2_name), ts_name='t_{\mu}', nsigmas=options.nsigmas, cl=options.cl)
    best_fit = poi_scan.best_fit(print_result=True)
    if not options.batch_mode :
      plt.ion()
      fig1 = plt.figure(1)
      poi_scan.plot(plt, label='PRL', best_fit=True, marker=options.marker, smoothing=options.smoothing)
      plt.show()

  with open(results_file, 'w') as fd:
    json.dump(jdict, fd, ensure_ascii=True, indent=3)

if __name__ == '__main__' : run()
