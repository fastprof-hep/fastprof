#! /usr/bin/env python

__doc__ = "Ccompute limits from sampling distributions"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import copy
import json

from fastprof import Model, Data, Samples, CLsSamples, OptiSampler, OptiMinimizer, FitResults, QMuCalculator, QMuTildaCalculator

####################################################################################################################################
###

parser = ArgumentParser("compute_limits.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-m", "--model-file"    , type=str  , required=True , help="Name of JSON file defining model")
parser.add_argument("-f", "--fits-file"     , type=str  , required=True , help="Name of JSON file containing full-model fit results")
parser.add_argument("-n", "--ntoys"         , type=int  , default=10000 , help="Number of pseudo-datasets to produce")
parser.add_argument("-s", "--seed"          , type=int  , default='0'   , help="Seed to use for random number generation")
parser.add_argument("-c", "--cl"            , type=float, default=0.95  , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
parser.add_argument("-o", "--output-file"   , type=str  , required=True , help="Name of output file")
parser.add_argument("-%", "--print-freq"    , type=int  , default=1000  , help="Verbosity level")
parser.add_argument("-d", "--data-file"     , type=str  , default=''    , help="Perform checks using the dataset stored in the specified JSON file")
parser.add_argument("-a", "--asimov"        , type=float, default=None  , help="Perform checks using an Asimov dataset for the specified POI value")
parser.add_argument("-i", "--iterations"    , type=int  , default=1     , help="Number of iterations to perform for NP computation")
parser.add_argument(      "--regularize"    , type=float, default=None  , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
parser.add_argument("-t", "--test-statistic", type=str  , default='q~mu', help="Test statistic to use")
parser.add_argument(      "--break-locks"   , action='store_true'       , help="Allow breaking locks from other sample production jobs")
parser.add_argument(      "--debug"         , action='store_true'       , help="Produce debugging output")
parser.add_argument(      "--bands"         , type=int  , default=None  , help="Number of bands to show")
parser.add_argument(      "--marker"        , type=str  , default=''    , help="Marker type for plots")
parser.add_argument("-b", "--batch-mode"    , action='store_true'       , help="Batch mode: no plots shown")
parser.add_argument(      "--truncate_dist" , type=float, default=None  , help="Truncate high p-values (just below 1) to get reasonable bands")
parser.add_argument("-v", "--verbosity"     , type=int  , default=0     , help="Verbosity level")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

model = Model.create(options.model_file)
if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
if options.regularize != None : model.set_gamma_regularization(options.regularize)

res = FitResults(model, options.fits_file)
fit_results = res.fit_results

if options.data_file :
  data = Data(model).load(options.data_file)
  if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
  print('Using dataset stored in file %s.' % options.data_file)
elif options.asimov != None :
  data = Data(model).set_expected(model.expected_pars(options.asimov))
  print('Using Asimov dataset with %s = %g.' % (res.poi_name, options.asimov))
else :
  data = Data(model).load(options.model_file)
  if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
  print('Using dataset stored in file %s.' % options.model_file)

mu0 = res.poi_initial_value
bounds = (res.poi_min, res.poi_max)

# Check the fastprof CLs against the ones in the reference: in principle this should match well,
# otherwise it means what we generate isn't exactly comparable to the observation, which would be a problem...
print('Check CL computed from fast model against those of the full model (a large difference would require to correct the sampling distributions) :')
if options.test_statistic == 'q~mu' :
  calc = QMuTildaCalculator(OptiMinimizer(data, mu0, bounds), res)
elif options.test_statistic == 'q_mu' :
  calc = QMuCalculator(OptiMinimizer(data, mu0, bounds), res)
else:
  raise ValueError('Unknown test statistic %s.' % options.test_statistic)
calc.fill_qpv()
calc.fill_fast_results()
res.print(verbosity=1)

if options.seed != None : np.random.seed(options.seed)
niter = options.iterations
samplers_clsb = []
samplers_cl_b = []
for fit_result in res.fit_results :
  test_hypo = fit_result['hypo_pars']
  tmu_0 = fit_result['fast_tmu_0']
  gen0_hypo = copy.deepcopy(test_hypo).set_poi(0)
  samplers_clsb.append(OptiSampler(model, test_hypo, mu0=mu0, bounds=bounds, print_freq=options.print_freq, debug=options.debug, niter=niter, tmu_A=tmu_0, tmu_0=tmu_0))
  samplers_cl_b.append(OptiSampler(model, test_hypo, mu0=mu0, bounds=bounds, print_freq=options.print_freq, debug=options.debug, niter=niter, tmu_A=tmu_0, tmu_0=tmu_0, gen_hypo=gen0_hypo))

opti_samples = CLsSamples( \
  Samples(samplers_clsb, options.output_file), \
  Samples(samplers_cl_b, options.output_file + '_clb')) \
  .generate_and_save(options.ntoys, break_locks=options.break_locks)

if options.truncate_dist : opti_samples.cut(None, options.truncate_dist)

for fit_result in fit_results :
  fit_result['sampling_pv' ] = opti_samples.clsb.pv(fit_result[res.poi_name], fit_result['pv'])
  fit_result['sampling_clb'] = opti_samples.cl_b.pv(fit_result[res.poi_name], fit_result['pv'])
  fit_result['sampling_cls'] = opti_samples.pv     (fit_result[res.poi_name], fit_result['pv'])

if options.bands :
  sampling_bands = opti_samples.bands(options.bands)
  for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
    for fit_result, band_point in zip(fit_results, sampling_bands[band]) : fit_result['sampling_cls_%+d' % band] = band_point

res.print(verbosity=1)

limit_asy_full_clsb = calc.limit('pv'          , options.cl)
limit_asy_fast_clsb = calc.limit('fast_pv'     , options.cl)
limit_sampling_clsb = calc.limit('sampling_pv' , options.cl)
limit_asy_full_cls  = calc.limit('cls'         , options.cl)
limit_asy_fast_cls  = calc.limit('fast_cls'    , options.cl)
limit_sampling_cls  = calc.limit('sampling_cls', options.cl)

if options.bands :
  limit_sampling_cls_bands = {}
  for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
    limit_sampling_cls_bands[band] = calc.limit('sampling_cls_%+d' % band, options.cl)

# Print results
if limit_asy_full_clsb : print('Asymptotics, full model, CLsb : UL(%g%%) = %g (N_signal = %g)' % (100*options.cl, limit_asy_full_clsb, np.sum(model.s_exp(model.expected_pars(limit_asy_full_clsb)))))
if limit_asy_fast_clsb : print('Asymptotics, fast model, CLsb : UL(%g%%) = %g (N_signal = %g)' % (100*options.cl, limit_asy_fast_clsb, np.sum(model.s_exp(model.expected_pars(limit_asy_fast_clsb)))))
if limit_sampling_clsb : print('Sampling   , fast model, CLsb : UL(%g%%) = %g (N_signal = %g)' % (100*options.cl, limit_sampling_clsb, np.sum(model.s_exp(model.expected_pars(limit_sampling_clsb)))))
if limit_asy_full_cls  : print('Asymptotics, full model, CLs  : UL(%g%%) = %g (N_signal = %g)' % (100*options.cl, limit_asy_full_cls , np.sum(model.s_exp(model.expected_pars(limit_asy_full_cls )))))
if limit_asy_fast_cls  : print('Asymptotics, fast model, CLs  : UL(%g%%) = %g (N_signal = %g)' % (100*options.cl, limit_asy_fast_cls , np.sum(model.s_exp(model.expected_pars(limit_asy_fast_cls )))))
if limit_sampling_cls  : print('Sampling   , fast model, CLs  : UL(%g%%) = %g (N_signal = %g)' % (100*options.cl, limit_sampling_cls , np.sum(model.s_exp(model.expected_pars(limit_sampling_cls )))))

# Plot results
if not options.batch_mode :
  plt.ion()
  fig1 = plt.figure(1)
  plt.suptitle('$CL_{s+b}$')
  plt.xlabel(model.poi_name)
  plt.ylabel('$CL_{s+b}$')
  plt.plot(res.hypos, [ fit_result['pv']          for fit_result in fit_results ], options.marker + 'r:' , label = 'Asymptotics')
  plt.plot(res.hypos, [ fit_result['sampling_pv'] for fit_result in fit_results ], options.marker + 'b-'  , label = 'Sampling')
  plt.legend()

  fig2 = plt.figure(2)
  plt.suptitle('$CL_s$')
  plt.xlabel(model.poi_name)
  plt.ylabel('$CL_s$')
  if options.bands :
    opti_samples.plot_bands(options.bands)
  plt.plot(res.hypos, [ fit_result['cls']          for fit_result in fit_results ], options.marker + 'r:' , label = 'Asymptotics')
  plt.plot(res.hypos, [ fit_result['sampling_cls'] for fit_result in fit_results ], options.marker + 'b-'  , label = 'Sampling')
  plt.legend()
  fig1.savefig(options.output_file + '_clsb.pdf')
  fig2.savefig(options.output_file + '_cls.pdf')
  plt.show()

jdict = {}
jdict['cl'] = options.cl
jdict['poi'] = model.poi_name
jdict['limit_sampling_CLs'] = limit_sampling_cls
jdict['limit_asymptotics_CLs'] = limit_asy_full_cls
jdict['limit_asymptotics_CLs_fast'] = limit_asy_fast_cls
jdict['limit_sampling_CLsb'] = limit_sampling_clsb
jdict['limit_asymptotics_CLsb'] = limit_asy_full_clsb
jdict['limit_asymptotics_CLsb_fast'] = limit_asy_fast_clsb

if options.bands :
  for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
    jdict['limit_sampling_CLs_expected_band_%+d' % band] = limit_sampling_cls_bands[band]

with open(options.output_file + '_results.json', 'w') as fd:
  json.dump(jdict, fd, ensure_ascii=True, indent=3)
