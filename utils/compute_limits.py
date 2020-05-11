#! /usr/bin/env python

__doc__ = "Ccompute limits from sampling distributions"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import numpy as np

from fastprof import Model, Data, Samples, CLsSamples, OptiSampler, OptiMinimizer, FitResults, QMuCalculator, QMuTildaCalculator

####################################################################################################################################
###

parser = ArgumentParser("compute_limits.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-m", "--model-file"    , type=str  , required=True      , help="Name of JSON file defining model")
parser.add_argument("-f", "--fits-file"     , type=str  , required=True      , help="Name of JSON file containing full-model fit results")
parser.add_argument("-n", "--ntoys"         , type=int  , default=10000      , help="Name of output file")
parser.add_argument("-s", "--seed"          , type=int  , default='0'        , help="Name of output file")
parser.add_argument("-o", "--output-file"   , type=str  , required=True      , help="Name of output file")
parser.add_argument("-%", "--print-freq"    , type=int  , default=1000       , help="Verbosity level")
parser.add_argument("-d", "--data-file"     , type=str  , default=''         , help="Perform checks using the dataset stored in the specified JSON file")
parser.add_argument("-a", "--asimov"        , type=float, default=None       , help="Perform checks using an Asimov dataset for the specified POI value")
parser.add_argument("-i", "--iterations"    , type=int  , default=1          , help="Numer of iterations to perform for NP computation")
parser.add_argument("-r", "--regularize"    , type=float, default=None       , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
parser.add_argument("-t", "--test-statistic", type=str  , default='qmu_tilda', help="Test statistic to use")
parser.add_argument("-b", "--break-locks"   , action='store_true'            , help="Allow breaking locks from other sample production jobs")
parser.add_argument(      "--debug"         , action='store_true'            , help="Produce debugging output")
parser.add_argument("-v", "--verbosity"     , type=int  , default=0          , help="Verbosity level")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

model = Model.create(options.model_file)
if options.regularize != None : model = model.regularize(options.regularize)

res = FitResults(options.fits_file)
fit_results = res.fit_results

if options.data_file :
  data = Data(model).load(options.data_file)
  if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
  print('Using dataset stored in file %s.' % options.data_file)
elif options.asimov != None :
  data = Data(model).set_expected(model.expected_pars(options.asimov))
  print('Using Asimov dataset with %s = %g.' % (res.poi_name, options.asimov))
else :
  print('Using dataset stored in file %s.' % options.model_file)
  data = Data(model).load(options.model_file)

# Check the fastprof CLs against the ones in the reference: in principle this should match well,
# otherwise it means what we generate isn't exactly comparable to the observation, which would be a problem...
print('Check CL computed from fast model against those of the full model (a large difference would require to correct the sampling distributions) :')
if options.test_statistic == 'qmu_tilda' :
  calc = QMuCalculator(OptiMinimizer(data, res.poi_initial_value, (res.poi_min, res.poi_max)), res)
elif options.test_statistic == 'qmu' :
  calc = QMuCalculator(OptiMinimizer(data, res.poi_initial_value, (res.poi_min, res.poi_max)), res)
else:
  raise ValueError('Unknown test statistic %s.' % options.test_statistic)
calc.fill_qcl()
calc.fill_fast_results()
res.print(verbosity=1)

if options.seed != None : np.random.seed(options.seed)

opti_samples = CLsSamples(
  Samples(res.hypos, OptiSampler(model, mu0=res.poi_initial_value, bounds=(res.poi_min, res.poi_max), print_freq=options.print_freq, debug=options.debug, niter=options.iterations)          , options.output_file),
  Samples(res.hypos, OptiSampler(model, mu0=res.poi_initial_value, bounds=(res.poi_min, res.poi_max), print_freq=options.print_freq, debug=options.debug, niter=options.iterations, gen_mu=0), options.output_file + '_clb')) \
  .generate_and_save(options.ntoys, break_locks=options.break_locks)

for fit_result in fit_results :
  fit_result['sampling_cl' ] = opti_samples.clsb.cl(fit_result[res.poi_name], fit_result['cl'])
  fit_result['sampling_clb'] = opti_samples.cl_b.cl(fit_result[res.poi_name], fit_result['cl'])
  fit_result['sampling_cls'] = opti_samples.cl     (fit_result[res.poi_name], fit_result['cl'])
res.print(verbosity=1)

limit_asymptot_clsb = res.solve('cl'          , 0.05, log_scale = True)
limit_sampling_clsb = res.solve('sampling_cl' , 0.05, log_scale = True)
limit_asymptot_cls  = res.solve('cls'         , 0.05, log_scale = True)
limit_sampling_cls  = res.solve('sampling_cls', 0.05, log_scale = True)

# Print results
print('Asymptotics, CLsb : UL(95) = %g (N_signal = %g)' % (limit_asymptot_clsb, np.sum(model.s_exp(model.expected_pars(limit_asymptot_clsb)))))
print('Sampling,    CLsb : UL(95) = %g (N_signal = %g)' % (limit_sampling_clsb, np.sum(model.s_exp(model.expected_pars(limit_sampling_clsb)))))
print('Asymptotics, CLs  : UL(95) = %g (N_signal = %g)' % (limit_asymptot_cls , np.sum(model.s_exp(model.expected_pars(limit_asymptot_cls )))))
print('Sampling,    CLs  : UL(95) = %g (N_signal = %g)' % (limit_sampling_cls , np.sum(model.s_exp(model.expected_pars(limit_sampling_cls )))))

# Plot results
plt.ion()
fig1 = plt.figure(1)
plt.suptitle('$CL_{s+b}$')
plt.xlabel('$\mu$')
plt.ylabel('$CL_{s+b}$')
plt.plot(res.hypos, [ fit_result['cl']          for fit_result in fit_results ], 'r:' , label = 'Asymptotics')
plt.plot(res.hypos, [ fit_result['sampling_cl'] for fit_result in fit_results ], 'b'  , label = 'Sampling')
plt.legend()

fig2 = plt.figure(2)
plt.suptitle('$CL_s$')
plt.xlabel('$\mu$')
plt.ylabel('$CL_s$')
opti_samples.plot_bands(2)
plt.plot(res.hypos, [ fit_result['cls']          for fit_result in fit_results ], 'r:' , label = 'Asymptotics')
plt.plot(res.hypos, [ fit_result['sampling_cls'] for fit_result in fit_results ], 'b'  , label = 'Sampling')
plt.legend()
plt.show()

fig1.savefig(options.output_file + '_clsb.pdf')
fig2.savefig(options.output_file + '_cls.pdf')
