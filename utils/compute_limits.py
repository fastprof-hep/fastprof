#! /usr/bin/env python

__doc__ = "Compute limits from sampling distributions"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import copy
import json

from fastprof import Model, Data, Samples, CLsSamples, OptiSampler, OptiMinimizer, Raster, QMuCalculator, QMuTildaCalculator, ParBound

####################################################################################################################################
###

def run(argv = None) :
  parser = ArgumentParser("compute_limits.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"    , type=str  , required=True , help="Name of JSON file defining model")
  parser.add_argument("-f", "--fits-file"     , type=str  , required=True , help="Name of JSON file containing full-model fit results")
  parser.add_argument("-n", "--ntoys"         , type=int  , default=10000 , help="Number of pseudo-datasets to produce")
  parser.add_argument("-s", "--seed"          , type=int  , default='0'   , help="Seed to use for random number generation")
  parser.add_argument("-c", "--cl"            , type=float, default=0.95  , help="Confidence level at which to compute the limit")
  parser.add_argument("-o", "--output-file"   , type=str  , required=True , help="Name of output file")
  parser.add_argument("-%", "--print-freq"    , type=int  , default=1000  , help="Verbosity level")
  parser.add_argument("-d", "--data-file"     , type=str  , default=None  , help="Perform checks using the dataset stored in the specified JSON file")
  parser.add_argument("-a", "--asimov"        , type=str  , default=None  , help="Perform checks using an Asimov dataset for the specified POI value")
  parser.add_argument("-i", "--iterations"    , type=int  , default=1     , help="Number of iterations to perform for NP computation")
  parser.add_argument(      "--regularize"    , type=float, default=None  , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
  parser.add_argument(      "--cutoff"        , type=float, default=None  , help="Cutoff to regularize the impact of NPs")
  parser.add_argument(      "--bounds"        , type=str  , default=None  , help="Parameter bounds in the form name1:[min]:[max],name2:[min]:[max],...")
  parser.add_argument(      "--sethypo"       , type=str  , default=''    , help="Change hypo parameter values, in the form par1=val1,par2=val2,...")
  parser.add_argument("-t", "--test-statistic", type=str  , default='q~mu', help="Test statistic to use")
  parser.add_argument(      "--break-locks"   , action='store_true'       , help="Allow breaking locks from other sample production jobs")
  parser.add_argument(      "--debug"         , action='store_true'       , help="Produce debugging output")
  parser.add_argument(      "--bands"         , type=int  , default=None  , help="Number of bands to show")
  parser.add_argument(      "--marker"        , type=str  , default=''    , help="Marker type for plots")
  parser.add_argument("-b", "--batch-mode"    , action='store_true'       , help="Batch mode: no plots shown")
  parser.add_argument(      "--truncate-dist" , type=float, default=None  , help="Truncate high p-values (just below 1) to get reasonable bands")
  parser.add_argument("-v", "--verbosity"     , type=int  , default=1     , help="Verbosity level")
  print(argv)
  options = parser.parse_args(argv)
  if not options :
    parser.print_help()
    sys.exit(0)

  model = Model.create(options.model_file)
  if model is None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
  if not options.regularize is None : model.set_gamma_regularization(options.regularize)
  if not options.cutoff is None : model.cutoff = options.cutoff

  raster = Raster('data', model=model, filename=options.fits_file)

  if options.sethypo != '' :
    try:
      sets = [ v.replace(' ', '').split('=') for v in options.sethypo.split(',') ]
      for plr_data in raster.plr_data.values() :
        for (var, val) in sets :
          if not var in plr_data.hypo : raise ValueError("Cannot find '%s' among hypothesis parameters." % var)
          plr_data.ref_pars[var] = float(val)
          print("INFO : setting %s=%g in reference parameters for hypothesis %s, new generation parameters are" % (var, float(val), plr_data.hypo.dict(pois_only=True)))
          print(plr_data.ref_pars)
    except Exception as inst :
      print(inst)
      raise ValueError("ERROR : invalid hypo assignment string '%s'." % options.sethypo)

  if options.data_file :
    data = Data(model).load(options.data_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    print('Using dataset stored in file %s.' % options.data_file)
  elif options.asimov != None :
    try:
      sets = [ v.replace(' ', '').split('=') for v in options.asimov.split(',') ]
      data = Data(model).set_expected(model.expected_pars(sets))
    except Exception as inst :
      print(inst)
      raise ValueError("Cannot define an Asimov dataset from options '%s'." % options.asimov)
    print('Using Asimov dataset with POIs %s.' % str(sets))
  else :
    data = Data(model).load(options.model_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    print('Using dataset stored in file %s.' % options.model_file)

  bounds = []
  if options.bounds :
    bound_specs = options.bounds.split(',')
    try :
      for spec in bound_specs :
        fields = spec.split(':')
        bounds.append(ParBound(fields[0], float(fields[1]) if fields[1] != '' else None, float(fields[2]) if fields[2] != '' else None))
    except Exception as inst:
      print('ERROR: could not parse parameter bound specification "%s", expected in the form name1:[min]:[max],name2:[min]:[max],...' % options.bounds)
      raise(inst)

  if options.test_statistic == 'q~mu' :
    if len(raster.pois()) > 1 : raise ValueError('Currently not supporting more than 1 POI for this operation')
    poi = raster.pois()[list(raster.pois())[0]]
    calc = QMuTildaCalculator(OptiMinimizer(poi.initial_value, (poi.min_value, poi.max_value)))
  elif options.test_statistic == 'q_mu' :
    if len(raster.pois()) > 1 : raise ValueError('Currently not supporting more than 1 POI for this operation')
    poi = raster.pois()[list(raster.pois())[0]]
    calc = QMuCalculator(OptiMinimizer(poi.initial_value, (poi.min_value, poi.max_value)))
  else :
    raise ValueError('Unknown test statistic %s' % options.test_statistic)

  # Check the fastprof CLs against the ones in the reference: in principle this should match well,
  # otherwise it means what we generate isn't exactly comparable to the observation, which would be a problem...
  print('Check CL computed from fast model against those of the full model (a large difference would require to correct the sampling distributions) :')
  calc.fill_all_pv(raster)
  faster = calc.compute_fast_results(raster, data)
  raster.print(verbosity = options.verbosity, other=faster)

  if options.seed != None : np.random.seed(options.seed)
  niter = options.iterations
  samplers_clsb = []
  samplers_cl_b = []

  print('Running with POI %s, bounds %s, and %d iteration(s).' % (str(poi), str(bounds), niter))

  for plr_data, fast_plr_data in zip(raster.plr_data.values(), faster.plr_data.values()) :
    test_hypo = plr_data.ref_pars
    tmu_0 = fast_plr_data.test_statistics['tmu_0']
    gen0_hypo = copy.deepcopy(test_hypo).set(list(model.pois)[0], 0)
    samplers_clsb.append(OptiSampler(model, test_hypo, poi.initial_value, (poi.min_value, poi.max_value), bounds, print_freq=options.print_freq, debug=options.debug, niter=niter, tmu_A=tmu_0, tmu_0=tmu_0))
    samplers_cl_b.append(OptiSampler(model, test_hypo, poi.initial_value, (poi.min_value, poi.max_value), bounds, print_freq=options.print_freq, debug=options.debug, niter=niter, tmu_A=tmu_0, tmu_0=tmu_0, gen_hypo=gen0_hypo))

  opti_samples = CLsSamples( \
    Samples(samplers_clsb, options.output_file), \
    Samples(samplers_cl_b, options.output_file + '_clb')) \
    .generate_and_save(options.ntoys, break_locks=options.break_locks)

  if options.truncate_dist : opti_samples.cut(None, options.truncate_dist)

  for plr_data in raster.plr_data.values() :
    plr_data.pvs['sampling_pv' ] = opti_samples.clsb.pv(plr_data.hypo[poi.name], plr_data.pvs['pv'], with_error=True)
    plr_data.pvs['sampling_clb'] = opti_samples.cl_b.pv(plr_data.hypo[poi.name], plr_data.pvs['pv'], with_error=True)
    plr_data.pvs['sampling_cls'] = opti_samples.pv     (plr_data.hypo[poi.name], plr_data.pvs['pv'], with_error=True)

  if options.bands :
    sampling_bands = opti_samples.bands(options.bands)
    for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
      for plr_data, band_point in zip(raster.plr_data.values(), sampling_bands[band]) : plr_data.pvs['sampling_cls_%+d' % band] = band_point

  def limit(rast, key, description, with_error=False) :
    limit_result = rast.compute_limit(key, 1 - options.cl, with_error=with_error)
    limit_value = limit_result if not with_error else limit_result[0]
    error_str = ''
    if with_error :
      limit_error = (limit_result[1] - limit_result[2])/2 - limit_result[0] if  not (limit_result[0] is None or limit_result[1] is None or limit_result[2] is None) else None
      error_str = '+/- %g' % limit_error if not limit_error is None else ''
    if not limit_value is None : print(description + ' : UL(%g%%) = %g %s (N = %s)' % (100*options.cl, limit_value, error_str, str(model.n_exp(model.expected_pars(limit_value)).sum(axis=1))) )
    return limit_result

  raster.print(keys=[ 'sampling_pv', 'sampling_cls', 'sampling_clb' ], verbosity=1)

  limit_asy_full_clsb = limit(raster, 'pv'          , 'Asymptotics, full model, CLsb')
  limit_asy_fast_clsb = limit(faster, 'pv'          , 'Asymptotics, fast model, CLsb')
  limit_sampling_clsb = limit(raster, 'sampling_pv' , 'Sampling   , fast model, CLsb', with_error=True)
  limit_asy_full_cls  = limit(raster, 'cls'         , 'Asymptotics, full model, CLs ')
  limit_asy_fast_cls  = limit(faster, 'cls'         , 'Asymptotics, fast model, CLs ')
  limit_sampling_cls  = limit(raster, 'sampling_cls', 'Sampling   , fast model, CLs ', with_error=True)

  if options.bands :
    limit_sampling_cls_bands = {}
    for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
      limit_sampling_cls_bands[band] = limit(raster, 'sampling_cls_%+d' % band, 'Expected limit band, fast model, %+d sigma band' % band)

  # Plot results
  if not options.batch_mode :
    plt.ion()
    fig1 = plt.figure(1)
    plt.suptitle('$CL_{s+b}$')
    plt.xlabel(list(model.pois)[0])
    plt.ylabel('$CL_{s+b}$')
    plt.fill_between([ hypo[poi.name] for hypo in raster.plr_data ],
                     [ plr_data.pvs['sampling_pv'][0] + plr_data.pvs['sampling_pv'][1] for plr_data in raster.plr_data.values() ],
                     [ plr_data.pvs['sampling_pv'][0] - plr_data.pvs['sampling_pv'][1] for plr_data in raster.plr_data.values() ], facecolor='b', alpha=0.5)
    plt.plot([ hypo[poi.name] for hypo in raster.plr_data ], [ plr_data.pvs['pv']             for plr_data in raster.plr_data.values() ], options.marker + 'r:' , label = 'Asymptotics')
    plt.plot([ hypo[poi.name] for hypo in raster.plr_data ], [ plr_data.pvs['sampling_pv'][0] for plr_data in raster.plr_data.values() ], options.marker + 'b-' , label = 'Sampling')

    plt.legend(loc=1) # 1 -> upper right
    plt.axhline(y=1 - options.cl, color='k', linestyle='dotted')

    fig2 = plt.figure(2)
    plt.suptitle('$CL_s$')
    plt.xlabel(list(model.pois)[0])
    plt.ylabel('$CL_s$')
    if options.bands :
      opti_samples.plot_bands(options.bands)
    plt.fill_between([ hypo[poi.name] for hypo in raster.plr_data ],
                     [ plr_data.pvs['sampling_cls'][0] + plr_data.pvs['sampling_cls'][1] for plr_data in raster.plr_data.values() ],
                     [ plr_data.pvs['sampling_cls'][0] - plr_data.pvs['sampling_cls'][1] for plr_data in raster.plr_data.values() ], facecolor='b', alpha=0.5)
    plt.plot([ hypo[poi.name] for hypo in raster.plr_data ], [ plr_data.pvs['cls']          for plr_data in raster.plr_data.values() ], options.marker + 'r:' , label = 'Asymptotics')
    plt.plot([ hypo[poi.name] for hypo in raster.plr_data ], [ plr_data.pvs['sampling_cls'][0] for plr_data in raster.plr_data.values() ], options.marker + 'b-' , label = 'Sampling')
    plt.legend(loc=1) # 1 -> upper right
    plt.axhline(y=1 - options.cl, color='k', linestyle='dotted')
    fig1.savefig(options.output_file + '_clsb.pdf')
    fig2.savefig(options.output_file + '_cls.pdf')
    plt.show()

  jdict = {}
  jdict['cl'] = options.cl
  jdict['poi'] = list(model.pois)[0]
  jdict['limit_sampling_CLs']    = limit_sampling_cls[0]
  jdict['limit_sampling_CLs_up'] = limit_sampling_cls[1]
  jdict['limit_sampling_CLs_dn'] = limit_sampling_cls[2]
  jdict['limit_asymptotics_CLs'] = limit_asy_full_cls
  jdict['limit_asymptotics_CLs_fast'] = limit_asy_fast_cls
  jdict['limit_sampling_CLsb']    = limit_sampling_clsb[0]
  jdict['limit_sampling_CLsb_up'] = limit_sampling_clsb[1]
  jdict['limit_sampling_CLsb_dn'] = limit_sampling_clsb[2]
  jdict['limit_asymptotics_CLsb'] = limit_asy_full_clsb
  jdict['limit_asymptotics_CLsb_fast'] = limit_asy_fast_clsb

  if options.bands :
    for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
      jdict['limit_sampling_CLs_expected_band_%+d' % band] = limit_sampling_cls_bands[band]

  with open(options.output_file + '_results.json', 'w') as fd:
    json.dump(jdict, fd, ensure_ascii=True, indent=3)

if __name__ == '__main__':
  run()
