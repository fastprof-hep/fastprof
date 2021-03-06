#! /usr/bin/env python

__doc__ = """
*Upper limit computation using sampling distributions*

Runs the full procedure to set upper limits on a single parameter of a
linear statistical model.

* The code computes the limit at the specified CL (`--cl` argument), by
  default 95%. Both frequentist and modified-frequentist :math:`CL_s` limits
  are computed. The test statistic can be set usign the `-t` option. By
  default, the q_mu~ of [arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>]_
  is used.

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
  The results are stored in a JSON file (`-o` argument)
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

from fastprof import Parameters, Model, Data, Samples, CLsSamples, OptiSampler, OptiMinimizer, Raster, QMuCalculator, QMuTildaCalculator, ParBound, process_setvals

####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("compute_limits_fast.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"    , type=str  , required=True , help="Name of JSON file defining model")
  parser.add_argument("-y", "--hypos"         , type=str  , required=True , help="List of POI hypothesis values (poi1=val1,poi2=val2:...)")
  parser.add_argument("-n", "--ntoys"         , type=int  , default=0     , help="Number of pseudo-datasets to produce")
  parser.add_argument("-s", "--seed"          , type=int  , default='0'   , help="Seed to use for random number generation")
  parser.add_argument("-c", "--cl"            , type=float, default=0.95  , help="Confidence level at which to compute the limit")
  parser.add_argument("-o", "--output-file"   , type=str  , required=True , help="Name of output file")
  parser.add_argument("-%", "--print-freq"    , type=int  , default=1000  , help="Verbosity level")
  parser.add_argument("-d", "--data-file"     , type=str  , default=None  , help="Use the dataset stored in the specified JSON file")
  parser.add_argument("-a", "--asimov"        , type=str  , default=None  , help="Use an Asimov dataset for the specified POI values (format: 'poi1=xx,poi2=yy'")
  parser.add_argument("-i", "--iterations"    , type=int  , default=1     , help="Number of iterations to perform for NP computation")
  parser.add_argument(      "--regularize"    , type=float, default=None  , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
  parser.add_argument(      "--cutoff"        , type=float, default=None  , help="Cutoff to regularize the impact of NPs")
  parser.add_argument(      "--bounds"        , type=str  , default=None  , help="Parameter bounds in the form name1:[min]:[max],name2:[min]:[max],...")
  parser.add_argument("-t", "--test-statistic", type=str  , default='q~mu', help="Test statistic to use")
  parser.add_argument(      "--break-locks"   , action='store_true'       , help="Allow breaking locks from other sample production jobs")
  parser.add_argument(      "--debug"         , action='store_true'       , help="Produce debugging output")
  parser.add_argument("-b", "--bands"         , type=int  , default=None  , help="Number of bands to show")
  parser.add_argument(      "--marker"        , type=str  , default=''    , help="Marker type for plots")
  parser.add_argument(      "--batch-mode"    , action='store_true'       , help="Batch mode: no plots shown")
  parser.add_argument(      "--truncate-dist" , type=float, default=None  , help="Truncate high p-values (just below 1) to get reasonable bands")
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

  try :
    hypos = [ Parameters(process_setvals(spec, model), model=model) for spec in options.hypos.split(':') ]
  except Exception as inst :
    print(inst)
    raise ValueError("Could not parse list of hypothesis values '%s' : expected colon-separated list of variable assignments" % options.hypos)

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

  gen_bounds = []
  if options.bounds :
    bound_specs = options.bounds.split(',')
    try :
      for spec in bound_specs :
        fields = spec.split(':')
        gen_bounds.append(ParBound(fields[0], float(fields[1]) if fields[1] != '' else None, float(fields[2]) if fields[2] != '' else None))
    except Exception as inst:
      print('ERROR: could not parse parameter bound specification "%s", expected in the form name1:[min]:[max],name2:[min]:[max],...' % options.bounds)
      raise(inst)

  if options.test_statistic == 'q~mu' :
    if len(model.pois) > 1 : raise ValueError('Currently not supporting more than 1 POI for this operation')
    calc = QMuTildaCalculator(OptiMinimizer().set_pois_from_model(model))
  elif options.test_statistic == 'q_mu' :
    if len(model.pois) > 1 : raise ValueError('Currently not supporting more than 1 POI for this operation')
    calc = QMuCalculator(OptiMinimizer().set_pois_from_model(model))
  else :
    raise ValueError('Unknown test statistic %s' % options.test_statistic)

  # Check the fastprof CLs against the ones in the reference: in principle this should match well,
  # otherwise it means what we generate isn't exactly comparable to the observation, which would be a problem...
  if options.ntoys > 0 : 
    print('Check CL computed from fast model against those of the full model (a large difference would require to correct the sampling distributions) :')
  faster = calc.compute_fast_results(hypos, data)
  faster.print(verbosity = options.verbosity)
  if options.ntoys == 0 : return

  if options.seed != None : np.random.seed(options.seed)
  niter = options.iterations
  samplers_clsb = []
  samplers_cl_b = []

  print('Running with POI %s, bounds %s, and %d iteration(s).' % (str(calc.minimizer.init_pois.dict(pois_only=True)), str(calc.minimizer.bounds), niter))

  for fast_plr_data in faster.plr_data.values() :
    test_hypo = fast_plr_data.hypo
    gen_hypo = test_hypo
    tmu_A0 = fast_plr_data.test_statistics['tmu_A0']
    gen0_hypo = gen_hypo.clone().set(model.poi(0).name, 0)
    clsb = OptiSampler(model, test_hypo, print_freq=options.print_freq, bounds=gen_bounds, debug=options.debug, niter=niter, tmu_Amu=tmu_A0, tmu_A0=tmu_A0, gen_hypo=gen_hypo)
    cl_b = OptiSampler(model, test_hypo, print_freq=options.print_freq, bounds=gen_bounds, debug=options.debug, niter=niter, tmu_Amu=tmu_A0, tmu_A0=tmu_A0, gen_hypo=gen0_hypo)
    clsb.minimizer.set_pois_from_model(model)
    cl_b.minimizer.set_pois_from_model(model)
    samplers_clsb.append(clsb)
    samplers_cl_b.append(cl_b)

  opti_samples = CLsSamples( \
    Samples(samplers_clsb, options.output_file), \
    Samples(samplers_cl_b, options.output_file + '_clb')) \
    .generate_and_save(options.ntoys, break_locks=options.break_locks)

  if options.truncate_dist : opti_samples.cut(None, options.truncate_dist)

  poi = faster.pois()[list(faster.pois())[0]]

  for plr_data in faster.plr_data.values() :
    plr_data.pvs['sampling_pv' ] = opti_samples.clsb.pv(plr_data.hypo, plr_data.pvs['pv'], with_error=True)
    plr_data.pvs['sampling_clb'] = opti_samples.cl_b.pv(plr_data.hypo, plr_data.pvs['pv'], with_error=True)
    plr_data.pvs['sampling_cls'] = opti_samples.pv     (plr_data.hypo, plr_data.pvs['pv'], with_error=True)

  if options.bands :
    sampling_bands = opti_samples.bands(options.bands)
    for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
      for plr_data, band_point in zip(faster.plr_data.values(), sampling_bands[band]) : plr_data.pvs['sampling_cls_%+d' % band] = band_point

  def limit(rast, key, description, with_error=False) :
    limit_result = rast.contour(key, 1 - options.cl, with_error=with_error)
    limit_value = limit_result if not with_error else limit_result[0]
    error_str = ''
    if with_error :
      limit_error = (limit_result[1] - limit_result[2])/2 if limit_result[1] is not None and limit_result[2] is not None else None
      error_str = '+/- %g' % limit_error if not limit_error is None else ''
    if not limit_value is None : print(description + ' : UL(%g%%) = %g %s (N = %s)' % (100*options.cl, limit_value, error_str, str(model.n_exp(model.expected_pars(limit_value)).sum(axis=1))) )
    return limit_result

  faster.print(keys=[ 'sampling_pv', 'sampling_cls', 'sampling_clb' ], verbosity=1)

  limit_asy_full_clsb = limit(faster, 'pv'          , 'Asymptotics, fast model, CLsb')
  limit_sampling_clsb = limit(faster, 'sampling_pv' , 'Sampling   , fast model, CLsb', with_error=True)
  limit_asy_full_cls  = limit(faster, 'cls'         , 'Asymptotics, fast model, CLs ')
  limit_sampling_cls  = limit(faster, 'sampling_cls', 'Sampling   , fast model, CLs ', with_error=True)

  if options.bands :
    limit_sampling_cls_bands = {}
    for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
      limit_sampling_cls_bands[band] = limit(faster, 'sampling_cls_%+d' % band, 'Expected limit band, fast model, %+d sigma band' % band)

  # Plot results
  if not options.batch_mode :
    plt.ion()
    fig1 = plt.figure(1)
    plt.suptitle('$CL_{s+b}$')
    plt.xlabel(model.poi(0).name)
    plt.ylabel('$CL_{s+b}$')
    plt.fill_between([ hypo[poi.name] for hypo in faster.plr_data ],
                     [ plr_data.pvs['sampling_pv'][0] + plr_data.pvs['sampling_pv'][1] for plr_data in faster.plr_data.values() ],
                     [ plr_data.pvs['sampling_pv'][0] - plr_data.pvs['sampling_pv'][1] for plr_data in faster.plr_data.values() ], facecolor='b', alpha=0.5)
    plt.plot([ hypo[poi.name] for hypo in faster.plr_data ], [ plr_data.pvs['pv']             for plr_data in faster.plr_data.values() ], options.marker + 'r:' , label = 'Asymptotics')
    plt.plot([ hypo[poi.name] for hypo in faster.plr_data ], [ plr_data.pvs['sampling_pv'][0] for plr_data in faster.plr_data.values() ], options.marker + 'b-' , label = 'Sampling')

    plt.legend(loc=1) # 1 -> upper right
    plt.axhline(y=1 - options.cl, color='k', linestyle='dotted')

    fig2 = plt.figure(2)
    plt.suptitle('$CL_s$')
    plt.xlabel(model.poi(0).name)
    plt.ylabel('$CL_s$')
    if options.bands :
      opti_samples.plot_bands(options.bands)
    plt.fill_between([ hypo[poi.name] for hypo in faster.plr_data ],
                     [ plr_data.pvs['sampling_cls'][0] + plr_data.pvs['sampling_cls'][1] for plr_data in faster.plr_data.values() ],
                     [ plr_data.pvs['sampling_cls'][0] - plr_data.pvs['sampling_cls'][1] for plr_data in faster.plr_data.values() ], facecolor='b', alpha=0.5)
    plt.plot([ hypo[poi.name] for hypo in faster.plr_data ], [ plr_data.pvs['cls']          for plr_data in faster.plr_data.values() ], options.marker + 'r:' , label = 'Asymptotics')
    plt.plot([ hypo[poi.name] for hypo in faster.plr_data ], [ plr_data.pvs['sampling_cls'][0] for plr_data in faster.plr_data.values() ], options.marker + 'b-' , label = 'Sampling')
    plt.legend(loc=1) # 1 -> upper right
    plt.axhline(y=1 - options.cl, color='k', linestyle='dotted')
    fig1.savefig(options.output_file + '_clsb.pdf')
    fig2.savefig(options.output_file + '_cls.pdf')
    fig2.savefig(options.output_file + '_cls.png')
    plt.show()

  jdict = {}
  jdict['cl'] = options.cl
  jdict['poi_name'] = model.poi(0).name
  jdict['poi_unit'] = model.poi(0).unit
  jdict['limit_sampling_CLs']    = limit_sampling_cls[0]
  jdict['limit_sampling_CLs_up'] = limit_sampling_cls[1]
  jdict['limit_sampling_CLs_dn'] = limit_sampling_cls[2]
  jdict['limit_asymptotics_CLs'] = limit_asy_full_cls
  jdict['limit_sampling_CLsb']    = limit_sampling_clsb[0]
  jdict['limit_sampling_CLsb_up'] = limit_sampling_clsb[1]
  jdict['limit_sampling_CLsb_dn'] = limit_sampling_clsb[2]
  jdict['limit_asymptotics_CLsb'] = limit_asy_full_clsb

  if options.bands :
    for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
      jdict['limit_sampling_CLs_expected_band_%+d' % band] = limit_sampling_cls_bands[band]

  with open(options.output_file + '_results.json', 'w') as fd:
    json.dump(jdict, fd, ensure_ascii=True, indent=3)

if __name__ == '__main__' : run()
