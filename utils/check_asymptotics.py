#! /usr/bin/env python

__doc__ = "Check asymptotic results of the fast model against those of the full model"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math

from fastprof import Model, FitResults, QMu, CLsSamples, Samples, OptiSampler, QMuCalculator, SamplingDistribution

####################################################################################################################################
###

parser = ArgumentParser("check_asymptotics.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-m", "--model-file", type=str  , default=''   , help="Name of JSON file defining model")
parser.add_argument("-f", "--fits-file" , type=str  , default=''   , help="Name of JSON file containing full-model fit results")
parser.add_argument("-s", "--samples"   , type=str  , default=''   , help="Name of JSON file containing full-model fit results")
parser.add_argument("-b", "--nbins"     , type=int  , default=100  , help="Number of bins to use")
parser.add_argument("-n", "--ntoys"     , type=int  , default=10000, help="Number of pseudo-datasets to produce")
parser.add_argument("-v", "--verbosity" , type=int  , default=0    , help="Verbosity level")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

model = Model.create(options.model_file)
if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)

res = FitResults(model, options.fits_file)
fit_results = res.fit_results
calc = QMuCalculator(None, res)
calc.fill_qcl()

opti_samples = CLsSamples(
  Samples(pois=res.hypos, file_root=options.samples),
  Samples(pois=res.hypos, file_root=options.samples + '_clb')).load()

def sampling_cl(samples, acl) :
  sd = SamplingDistribution()
  sd.samples = np.array(samples)
  sd.sort()
  return sd.cl(acl)

plt.ion()
for fit_result in fit_results :
  hypo = fit_result[res.poi_name]
  print('Checking POI = %g' % hypo)
  tmu_A = fit_result['tmu_A']
  sigma_A = hypo/math.sqrt(tmu_A) if tmu_A > 0 else fit_result['best_fit_err']
  fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8, 8), dpi=96)
  fig.suptitle('POI = %g'  % hypo)
  #bias = -0.065*hypo # test!!
  bias = 0
  asym_clsb = [ QMu(hypo, tmu_A*(hypo-poi)**2/hypo**2, poi).asymptotic_cl() for poi in norm.rvs(hypo + bias, sigma_A, options.ntoys) ]
  asym_cl_b = [ QMu(hypo, tmu_A*(hypo-poi)**2/hypo**2, poi).asymptotic_cl() for poi in norm.rvs(       bias, sigma_A, options.ntoys) ]
  print('clsb CL: asymptotic value     = %g' % fit_result['cl'])
  print('clsb CL: sampling from asympt = %g' % sampling_cl(asym_clsb, fit_result['cl']))
  print('clsb CL: sampling from toys   = %g' % opti_samples.clsb.dists[hypo].cl(fit_result['cl']))
  print('---')
  print('cl_b CL: asymptotic value     = %g' % fit_result['clb'])
  print('cl_b CL: sampling from asympt = %g' % sampling_cl(asym_cl_b, fit_result['cl']))
  print('cl_b CL: sampling from toys   = %g' % opti_samples.cl_b.dists[hypo].cl(fit_result['cl']))
  print('---')
  ax1.hist(opti_samples.clsb.dists[hypo].samples, bins=options.nbins)
  ax1.hist(asym_clsb, bins=options.nbins, histtype='step')  
  ax2.hist(opti_samples.cl_b.dists[hypo].samples, bins=options.nbins)
  ax2.hist(asym_cl_b, bins=options.nbins, histtype='step')
  ax1.set_yscale('log')
  ax2.set_yscale('log')
