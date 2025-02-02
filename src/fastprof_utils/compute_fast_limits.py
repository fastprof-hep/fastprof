#! /usr/bin/env python

__doc__ = """
*Upper limit computation using sampling distributions*

Runs the full procedure to set upper limits on a single parameter of a
linear statistical model.

* The code computes the limit at the specified CL (`--cl` argument), by
  default 95%. Both frequentist and modified-frequentist :math:`CL_s` limits
  are computed. The test statistic can be set usign the `-t` option. By
  default, the q_mu~ of arXiv:1007.1727 is used.

* The main inputs are the model file (`--model-file` argument) and the
  *fits* file providing results from the full model (`--fits-file`).
  The fits file also defines the parameter hypotheses at which the
  sampling distributions are generated.

* The limit is computed using the test statistic values stored in the
  fits file. The corresponding p-values are computed using sampling
  distributions which are randomly generated at each of the parameter
  hypotheses listed in the fits file (`--ntoys` pseudo-datasets in
  each sample). A lock file system allows several processes to generate
  the distributions in parallel, or resume an interrupted job. The
  `--break-locks` option allows to break locks left by dead jobs.

* An interpolation between the hypotheses is then performed to obtain
  the upper limit value. Output is given as p-values, :math:`CL_s` and
  :math:`CL_b` values at each hypothesis, and interpolated frequentist
  and :math:`CL_s` upper limits.
  Results for the full model in the asymptotic approximation are also shown
  for comparison, as well as those of the linear model if a dataset is
  supplied either as observed data (`--data-file` argument) or an Asimov
  dataset (`--asimov`).
  The results are stored in a markup file (`-o` argument)
  Plots of CL vs hypothesis are also shown, as well as the expected
  limit and its uncertainty bands for the null hypothesis if the `--bands`
  option is specified.

* Several options can be specified to account for non linear effects (`--iterations`)
  or regularize the model (`--regularize`, `--cutoff`, `--bounds`, `--sethypo`),
  as described in the package documentation.
"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import copy
import json
import time

from fastprof import POIHypo, Parameters, Model, Data, Samples, CLsSamples, PValueSampler, OptiMinimizer, NPMinimizer, Raster, QMuCalculator, QMuTildaCalculator, ParBound, UpperLimitScan, PlotBands

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
  parser.add_argument("-c", "--cl"            , type=float, default=0.95  , help="Confidence level at which to compute the limit")
  parser.add_argument(      "--clsb"          , action='store_true'       , help="Also show CLsb results")
  parser.add_argument("-o", "--output-file"   , type=str  , required=True , help="Name of output file")
  parser.add_argument("-%", "--print-freq"    , type=int  , default=1000  , help="Verbosity level")
  parser.add_argument("-d", "--data-file"     , type=str  , default=None  , help="Use the dataset stored in the specified markup file")
  parser.add_argument("-a", "--asimov"        , type=str  , default=None  , help="Use an Asimov dataset for the specified POI values (format: 'poi1=xx,poi2=yy'")
  parser.add_argument("-i", "--iterations"    , type=int  , default=1     , help="Number of iterations to perform for NP computation")
  parser.add_argument(      "--regularize"    , type=float, default=None  , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
  parser.add_argument(      "--cutoff"        , type=float, default=None  , help="Cutoff to regularize the impact of NPs")
  parser.add_argument(      "--bounds"        , type=str  , default=None  , help="Parameter bounds in the form name1=[min]#[max],name2=[min]#[max],...")
  parser.add_argument("-t", "--test-statistic", type=str  , default='q~mu', help="Test statistic to use")
  parser.add_argument(      "--break-locks"   , action='store_true'       , help="Allow breaking locks from other sample production jobs")
  parser.add_argument("-x", "--overwrite"     , action='store_true'       , help="Allow overwriting output file")
  parser.add_argument(      "--debug"         , action='store_true'       , help="Produce debugging output")
  parser.add_argument("-b", "--bands"         , type=int  , default=0     , help="Number of bands to show")
  parser.add_argument(      "--marker"        , type=str  , default=''    , help="Marker type for plots")
  parser.add_argument(      "--batch-mode"    , action='store_true'       , help="Batch mode: no plots shown")
  parser.add_argument(      "--truncate-dist" , type=float, default=None  , help="Truncate high p-values (just below 1) to get reasonable bands")
  parser.add_argument(      "--show-timing"   , action='store_true'       , help="Enables printout of timing information")
  parser.add_argument("-v", "--verbosity"     , type=int  , default=0     , help="Verbosity level")
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
  data = make_data(model, options)
  hypos = make_hypos(model, options)

  if options.test_statistic == 'q~mu' :
    if len(model.pois) > 1 : raise ValueError('Currently not supporting more than 1 POI for this operation')
    calc = QMuTildaCalculator(OptiMinimizer(verbosity=options.verbosity), verbosity=options.verbosity)
  elif options.test_statistic == 'q_mu' :
    if len(model.pois) > 1 : raise ValueError('Currently not supporting more than 1 POI for this operation')
    calc = QMuCalculator(OptiMinimizer(verbosity=options.verbosity), verbosity=options.verbosity)
  else :
    raise ValueError('Unknown test statistic %s' % options.test_statistic)
  par_bounds = init_calc(calc, model, options)
  
  raster = try_loading_results(model, raster_file, options, hypos)
  if options.show_timing : comp_start_time = time.time()
  if raster is None :
    full_hypos = { hypo : model.expected_pars(hypo.pars) for hypo in hypos }
    raster = calc.compute_fast_results(hypos, data, full_hypos, bands=options.bands)
    raster.save(raster_file)
  if options.show_timing : comp_stop_time = time.time()
  raster.print(verbosity = options.verbosity)
  scan_asy_fast_cls  = UpperLimitScan(raster, 'cls', name='CL_s [asymptotic]', cl=options.cl, cl_name='CL_s' )
  limit_asy_fast_cls  = scan_asy_fast_cls .limit(print_result=True, bands=options.bands)
  if options.clsb :
    scan_asy_fast_clsb = UpperLimitScan(raster, 'pv' , name='CLsb [asymptotic]', cl=options.cl, cl_name='CL_{s+b}')
    limit_asy_fast_clsb = scan_asy_fast_clsb.limit(print_result=True, bands=options.bands)

  if options.ntoys > 0 :
    if options.seed != None : np.random.seed(options.seed)
    niter = options.iterations
    samplers_clsb = []
    samplers_cl_b = []
    
    # Disabling optimization actually speeds up processing in some cases (small matrices ?)
    # NPMinimizer.optimize_einsum = False    
    print('Running with POI %s, bounds %s, and %d iteration(s).' % (str(calc.minimizer.init_pois.dict(pois_only=True)), str(calc.minimizer.bounds), niter))
  
    for fast_plr_data in raster.plr_data.values() :
      test_hypo = fast_plr_data.hypo
      gen_hypo = fast_plr_data.full_hypo
      tmu_A0 = fast_plr_data.test_statistics['tmu_A0']
      gen0_hypo = gen_hypo.clone().set(model.poi(0).name, 0)
      clsb = PValueSampler(model, test_hypo, print_freq=options.print_freq, bounds=par_bounds, debug=options.debug, niter=niter, tmu_Amu=tmu_A0, tmu_A0=tmu_A0, gen_hypo=gen_hypo)
      cl_b = PValueSampler(model, test_hypo, print_freq=options.print_freq, bounds=par_bounds, debug=options.debug, niter=niter, tmu_Amu=tmu_A0, tmu_A0=tmu_A0, gen_hypo=gen0_hypo)
      samplers_clsb.append(clsb)
      samplers_cl_b.append(cl_b)
  
    opti_samples = CLsSamples( \
      Samples(samplers_clsb, options.output_file), \
      Samples(samplers_cl_b, options.output_file + '_clb')) \
      .generate_and_save(options.ntoys, break_locks=options.break_locks)
  
    if options.truncate_dist : opti_samples.cut(None, options.truncate_dist)

    for plr_data in raster.plr_data.values() :
      # We always use pv which represents the q_mu value. This is the pv that is sampled in all cases, both
      # clsb and clb, so the lookup is also done in terms of pv. Of course the sampling p-values do reflect
      # the different distributions  of clsb and clb
      plr_data.pvs['sampling_pv' ] = opti_samples.clsb.pv(plr_data.hypo, plr_data.pvs['pv'], with_error=True)
      plr_data.pvs['sampling_clb'] = opti_samples.cl_b.pv(plr_data.hypo, plr_data.pvs['pv'], with_error=True)
      plr_data.pvs['sampling_cls'] = opti_samples.pv     (plr_data.hypo, plr_data.pvs['pv'], with_error=True)

    if options.bands :
      sampling_bands_pv  = opti_samples.bkg_hypo_bands(options.bands, clsb = True)
      sampling_bands_cls = opti_samples.bkg_hypo_bands(options.bands)
      for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
        for plr_data, band_point in zip(raster.plr_data.values(), sampling_bands_cls[band]) : 
          plr_data.pvs['sampling_cls_%+d' % band] = band_point
        if options.clsb :
          for plr_data, band_point in zip(raster.plr_data.values(), sampling_bands_pv[band]) : 
            plr_data.pvs['sampling_pv_%+d' % band] = band_point

    if options.bands >= 2 :
      raster.print(keys=[ 'sampling_cls_-2', 'sampling_cls_-1', 'sampling_cls_+0', 'sampling_cls_+1', 'sampling_cls_+2' ], verbosity=1)
      if options.clsb : raster.print(keys=[ 'sampling_pv_-2', 'sampling_pv_-1', 'sampling_pv_+0', 'sampling_pv_+1', 'sampling_pv_+2' ], verbosity=1)
    raster.print(keys=[ 'sampling_pv', 'sampling_cls', 'sampling_clb' ], verbosity=1)
    scan_sampling_cls = UpperLimitScan(raster, 'sampling_cls', name='CLs [sampling]', cl=options.cl, cl_name='CL_s' )
    limit_sampling_cls = scan_sampling_cls.limit(print_result=True)
    if options.clsb :
      scan_sampling_clsb = UpperLimitScan(raster, 'sampling_pv' , name='CLsb [sampling]', cl=options.cl, cl_name='CL_{s+b}')
      limit_sampling_clsb = scan_sampling_clsb.limit(print_result=True)
    if options.bands :
      scan_sampling_cls_bands = {}
      limit_sampling_cls_bands = {}
      for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
        scan_sampling_cls_bands[band] = UpperLimitScan(raster, 'sampling_cls_%+d' % band, name='Expected limit band, %+d sigma band' % band)
        limit_sampling_cls_bands[band] = scan_sampling_cls_bands[band].limit(print_result=True)

  # Show results
  if not options.batch_mode :
    plt.ion()
    fig_cls, ax_cls = plt.subplots(constrained_layout=True, num='CLs limit scan')
    if options.ntoys > 0 : 
      if options.bands :
        PlotBands(opti_samples.par_hypos(), opti_samples.bkg_hypo_bands(options.bands)).plot(options.bands, label='Expected CL_s')
      scan_sampling_cls.plot(fig_cls, marker=options.marker + 'b-', label='Sampling $CL_s$', with_errors=True)
    scan_asy_fast_cls.plot(fig_cls, marker=options.marker + 'r:', label='Asymptotic $CL_s$', bands=options.bands)
    plt.legend(loc=1) # 1 -> upper right
    plt.axhline(y=1 - options.cl, color='k', linestyle='dotted')
    fig_cls.savefig(options.output_file + '_cls.pdf')
    fig_cls.savefig(options.output_file + '_cls.png')

    if options.clsb :
      fig_clsb, ax_clsb = plt.subplots(constrained_layout=True, num='CLsb limit scan')
      if options.ntoys > 0 : scan_sampling_clsb.plot(fig_clsb, marker=options.marker + 'b-', label='Sampling $CL_{s+b}$', with_errors=True)
      scan_asy_fast_clsb.plot(fig_clsb, marker=options.marker + 'r:', label='Asymptotic $CL_{s+b}$', bands=options.bands)
      plt.legend(loc=1) # 1 -> upper right
      plt.axhline(y=1 - options.cl, color='k', linestyle='dotted')
      fig_clsb.savefig(options.output_file + '_clsb.pdf')

    plt.show()

  jdict = {}
  jdict['cl'] = options.cl
  jdict['poi_name'] = model.poi(0).name
  jdict['poi_unit'] = model.poi(0).unit
  if options.ntoys > 0 : 
    jdict['limit_sampling_CLs']    = limit_sampling_cls
  jdict['limit_asymptotic_CLs'] = limit_asy_fast_cls
  if options.clsb : 
    if options.ntoys > 0 : 
      jdict['limit_sampling_CLsb']    = limit_sampling_clsb
    jdict['limit_asymptotic_CLsb'] = limit_asy_fast_clsb

  if options.ntoys > 0 : 
    if options.bands :
      for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
        jdict['limit_sampling_CLs_expected_band_%+d' % band] = limit_sampling_cls_bands[band]

  if options.show_timing :
    stop_time = time.time()
    print("##           Setup time : %g s" % (comp_start_time - start_time))
    print("##     Computation time : %g s" % (comp_stop_time - comp_start_time))
    print("## Post-processing time : %g s" % (stop_time - comp_stop_time))

  with open(results_file, 'w') as fd:
    json.dump(jdict, fd, ensure_ascii=True, indent=3)
  
  return 0
  
if __name__ == '__main__' : run()
