#! /usr/bin/env python

__doc__ = "Ccompute limits from sampling distributions"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import numpy as np

from fastprof import Model, Data, Samples, CLsSamples, OptiSampler, OptiMinimizer, FitResults, QMuCalculator

####################################################################################################################################
###

parser = ArgumentParser("compute_limits.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-m", "--model-file",     required=True,   help="Name of JSON file defining model", type=str)
parser.add_argument("-f", "--fits-file",      required=True,   help="Name of JSON file containing full-model fit results", type=str)
parser.add_argument("-n", "--ntoys",          default=10000,   help="Name of output file", type=int)
parser.add_argument("-s", "--seed",           default='0',     help="Name of output file", type=int)
parser.add_argument("-o", "--output-file",    required=True,   help="Name of output file", type=str)
parser.add_argument("-%", "--print-freq",     default=1000,    help="Verbosity level", type=int)
parser.add_argument("-d", "--data-file",      default='',      help="Perform checks using the dataset stored in the specified JSON file", type=str)
parser.add_argument("-a", "--asimov",         default=None,    help="Perform checks using an Asimov dataset for the specified POI value", type=float)
parser.add_argument("-v", "--verbosity",      default=0,       help="Verbosity level", type=int)

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

model = Model.create(options.model_file)
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
calc = QMuCalculator(OptiMinimizer(data, res.poi_initial_value, (res.poi_min, res.poi_max)), res)
calc.fill_qcl('qmu', 'cl', 'cls')
calc.fill_fast()
res.print(verbosity=1)

if options.seed != None : np.random.seed(options.seed)

opti_samples = CLsSamples(
  Samples(res.hypos, OptiSampler(model, mu0=res.poi_initial_value, bounds=(res.poi_min, res.poi_max), print_freq=options.print_freq)          , options.output_file),
  Samples(res.hypos, OptiSampler(model, mu0=res.poi_initial_value, bounds=(res.poi_min, res.poi_max), print_freq=options.print_freq, gen_mu=0), options.output_file + '_clb')) \
  .generate_and_save(options.ntoys)

for fit_result in fit_results :
  fit_result['sampling_cl' ] = opti_samples.clsb.cl(fit_result[res.poi_name], fit_result['cl'])
  fit_result['sampling_clb'] = opti_samples.cl_b.cl(fit_result[res.poi_name], fit_result['cl'])
  fit_result['sampling_cls'] = opti_samples.cl     (fit_result[res.poi_name], fit_result['cl'])
res.print(verbosity=1)

# Print results
print('Asymptotics, CLsb : UL(95) =', res.solve('cl'          , 0.05, log_scale = True))
print('Sampling,    CLsb : UL(95) =', res.solve('sampling_cl' , 0.05, log_scale = True))
print('Asymptotics, CLs  : UL(95) =', res.solve('cls'         , 0.05, log_scale = True))
print('Sampling,    CLs  : UL(95) =', res.solve('sampling_cls', 0.05, log_scale = True))

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
