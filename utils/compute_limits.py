#! /usr/bin/env python

__doc__ = "Ccompute limits from sampling distributions"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

from fastprof import Model, Data, QMu, Samples, CLsSamples, OptiSampler, OptiMinimizer, FitResults

####################################################################################################################################
###

def compute_limits() :
  """convert """
  
  parser = ArgumentParser("compute_limits.py")
  parser.description = __doc__
  parser.add_argument("-m", "--model-file",     required=True,   help="Name of JSON file defining model", type=str)
  parser.add_argument("-d", "--data-file",      default='',      help="Name of JSON file defining the dataset (optional, otherwise taken from model file)", type=str)
  parser.add_argument("-f", "--fits-file",      required=True,   help="Name of JSON file containing full-model fit results", type=str)
  parser.add_argument("-n", "--ntoys",          default=10000,   help="Name of output file", type=int)
  parser.add_argument("-s", "--seed",           default='0',     help="Name of output file", type=int)
  parser.add_argument("-o", "--output-file",    required=True,   help="Name of output file", type=str)
  parser.add_argument("-v", "--verbosity",      default=0,       help="Verbosity level", type=int)

  options = parser.parse_args()
  if not options : 
    parser.print_help()
    return

  model = Model.create(options.model_file)

  full_results = FitResults(options.fits_file)
  fit_results = full_results.fit_results
  fr = full_results
 
  if options.seed != None : np.random.seed(options.seed)

  opti_samples = CLsSamples(
    Samples(OptiSampler(model, mu0=fr.poi_initial_value, bounds=(fr.poi_min, fr.poi_max))               , options.output_file),
    Samples(OptiSampler(model, mu0=fr.poi_initial_value, bounds=(fr.poi_min, fr.poi_max), do_CLb = True), options.output_file + '_clb')).generate_and_save(fr.hypos, options.ntoys)

  full_results.fill() # fill asymptotic results
  for fit_result in fit_results :
    fit_result['sampling_cl' ] = opti_samples.clsb.cl(fit_result['cl'], fit_result[fr.poi_name])
    fit_result['sampling_cls'] = opti_samples.cl     (fit_result['cl'], fit_result[fr.poi_name])

  # Check the fastprof CLs against the ones in the reference: in principle this should match well,
  # otherwise it means what we generate isn't exactly comparable to the observation, which would be a problem...
  print('Check CL computed from fast model against those of the full model (a large difference would require to correct the sampling distributions) :')
  if options.data_file == '' :
    print('Loading data from %s.' % options.model_file)
    data = Data(model).load(options.model_file)
  else :
    print('Loading data from %s.' % options.data_file)
    data = Data(model).load(options.data_file)
  full_results.check(data)

  # Print results
  print('Asymptotics, CLsb : UL(95) =', fr.solve('cl'          , 0.05, log_scale = True))
  print('Sampling,    CLsb : UL(95) =', fr.solve('sampling_cl' , 0.05, log_scale = True))
  print('Asymptotics, CLs  : UL(95) =', fr.solve('cls'         , 0.05, log_scale = True))
  print('Sampling,    CLs  : UL(95) =', fr.solve('sampling_cls', 0.05, log_scale = True))

  # Plot results
  fig1 = plt.figure(1)
  plt.suptitle('$CL_{s+b}$')
  plt.xlabel('$\mu$')
  plt.ylabel('$CL_{s+b}$')
  plt.plot(fr.hypos, [ fit_result['cl']          for fit_result in fit_results ], 'r:' , label = 'Asymptotics')
  plt.plot(fr.hypos, [ fit_result['sampling_cl'] for fit_result in fit_results ], 'b'  , label = 'Sampling')
  plt.legend()
  
  fig2 = plt.figure(2)
  plt.suptitle('$CL_s$')
  plt.xlabel('$\mu$')
  plt.ylabel('$CL_s$')
  plt.plot(fr.hypos, [ fit_result['cls']          for fit_result in fit_results ], 'r:' , label = 'Asymptotics')
  plt.plot(fr.hypos, [ fit_result['sampling_cls'] for fit_result in fit_results ], 'b'  , label = 'Sampling')
  plt.legend()

  fig1.savefig(options.output_file + '_clsb.pdf')
  fig2.savefig(options.output_file + '_cls.pdf')


compute_limits()
