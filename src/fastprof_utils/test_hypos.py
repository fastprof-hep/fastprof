#! /usr/bin/env python

__doc__ = """
*Test hypotheses using likelihood ratio test statistic*

"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import copy
import json
import time

from fastprof import POIHypo, Parameters, Model, Data, Samples, CLsSamples, PValueSampler, OptiMinimizer, NPMinimizer, Raster, TMuCalculator, ParBound, UpperLimitScan

from fastprof_utils import make_model, make_data, make_hypos, init_calc, try_loading_results


####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("compute_limits_fast.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"    , type=str  , required=True , help="Name of markup file defining model")
  parser.add_argument("-y", "--hypos"         , type=str  , required=True , help="List of POI hypothesis values (poi1=val1,poi2=val2|...)")
  parser.add_argument("-n", "--ntoys"         , type=int  , default=0     , help="Number of pseudo-datasets to produce")
  parser.add_argument("-s", "--seed"          , type=int  , default='0'   , help="Seed to use for random number generation")
  parser.add_argument("-o", "--output-file"   , type=str  , required=True , help="Name of output file")
  parser.add_argument("-%", "--print-freq"    , type=int  , default=1000  , help="Verbosity level")
  parser.add_argument("-d", "--data-file"     , type=str  , default=None  , help="Use the dataset stored in the specified markup file")
  parser.add_argument("-a", "--asimov"        , type=str  , default=None  , help="Use an Asimov dataset for the specified POI values (format: 'poi1=xx,poi2=yy'")
  parser.add_argument("-i", "--iterations"    , type=int  , default=1     , help="Number of iterations to perform for NP computation")
  parser.add_argument(      "--regularize"    , type=float, default=None  , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
  parser.add_argument(      "--cutoff"        , type=float, default=None  , help="Cutoff to regularize the impact of NPs")
  parser.add_argument(      "--bounds"        , type=str  , default=None  , help="Parameter bounds in the form name1=[min]#[max],name2=[min]#[max],...")
  parser.add_argument(      "--break-locks"   , action='store_true'       , help="Allow breaking locks from other sample production jobs")
  parser.add_argument(      "--debug"         , action='store_true'       , help="Produce debugging output")
  parser.add_argument(      "--show-timing"   , action='store_true'       , help="Enables printout of timing information")
  parser.add_argument("-x", "--overwrite"     , action='store_true'       , help="Allow overwriting output file")
  parser.add_argument("-v", "--verbosity"     , type=int  , default=1     , help="Verbosity level")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args(argv)
  if not options :
    parser.print_help()
    return 0

  if options.show_timing : start_time = time.time()

  results_file = options.output_file + '_results.json'
  raster_file = options.output_file + '_raster.json'

  model = make_model(options)
  if not options.regularize is None : model.set_gamma_regularization(options.regularize)
  if not options.cutoff is None : model.cutoff = options.cutoff
  data = make_data(model, options)
  hypos = make_hypos(model, options)

  if len(model.pois) > 1 : raise ValueError('Currently not supporting more than 1 POI for this operation')
  calc = TMuCalculator(OptiMinimizer(verbosity=options.verbosity), verbosity=options.verbosity)
  par_bounds = init_calc(calc, model, options)
  faster = try_loading_results(model, raster_file, options)

  if options.show_timing : comp_start_time = time.time()
  if faster is None :
    full_hypos = { hypo : model.expected_pars(hypo.pars) for hypo in hypos }
    faster = calc.compute_fast_results(hypos, data, full_hypos)
    faster.save(raster_file)
  if options.show_timing : comp_stop_time = time.time()
  faster.print(verbosity = options.verbosity)

  if options.ntoys == 0 : return
  print('Checking CL computed from fast model against those of the full model (a large difference would require to correct the sampling distributions) :')

  if options.seed != None : np.random.seed(options.seed)
  niter = options.iterations
  samplers_clsb = []
  samplers_cl_b = []
  
  # Disabling optimization actually speeds up processing in some cases (small matrices ?)
  # NPMinimizer.optimize_einsum = False
  
  calc.minimizer.set_pois(model)
  print('Running with POI %s, bounds %s, and %d iteration(s).' % (str(calc.minimizer.init_pois.dict(pois_only=True)), str(calc.minimizer.bounds), niter))

  for fast_plr_data in faster.plr_data.values() :
    test_hypo = fast_plr_data.hypo
    gen_hypo = fast_plr_data.full_hypo
    tmu_A0 = fast_plr_data.test_statistics['tmu_A0']
    gen0_hypo = gen_hypo.clone().set(model.poi(0).name, 0)
    clsb = PValueSampler(model, test_hypo, print_freq=options.print_freq, bounds=par_bounds, debug=options.debug, niter=niter, tmu_Amu=tmu_A0, tmu_A0=tmu_A0, gen_hypo=gen_hypo)
    samplers_clsb.append(clsb)

  if options.truncate_dist : opti_samples.cut(None, options.truncate_dist)

  for plr_data in faster.plr_data.values() :
    plr_data.pvs['sampling_pv' ] = opti_samples.clsb.pv(plr_data.hypo, plr_data.pvs['pv'], with_error=True)

  if options.bands :
    sampling_bands = opti_samples.bands(options.bands)

  faster.print(keys=[ 'sampling_pv' ], verbosity=1)

  scan_asy_fast_clsb = UpperLimitScan(faster, 'pv'          , name='CLsb, asymptotics, fast model', cl=options.cl, cl_name='CL_{s+b}')
  scan_sampling_clsb = UpperLimitScan(faster, 'sampling_pv' , name='CLsb, sampling   , fast model', cl=options.cl, cl_name='CL_{s+b}')

  jdict = {}

  with open(results_file, 'w') as fd:
    json.dump(jdict, fd, ensure_ascii=True, indent=3)
  
  return 0
  
if __name__ == '__main__' : run()
