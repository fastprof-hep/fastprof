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
import time

from fastprof import Model, Data, Parameters, POIHypo, OptiMinimizer, Raster, TMuCalculator, PLRScan1D, PLRScan2D
from fastprof_utils import make_model, make_data, make_hypos, init_calc, try_loading_results


####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("poi_scan.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"    , type=str  , required=True    , help="Name of markup file defining model")
  parser.add_argument("-d", "--data-file"     , type=str  , default=None     , help="Use the dataset stored in the specified markup file")
  parser.add_argument("-a", "--asimov"        , type=str  , default=None     , help="Use an Asimov dataset for the specified POI values (format: 'poi1=xx,poi2=yy'")
  parser.add_argument("-y", "--hypos"         , type=str  , default=None     , help="List of POI hypothesis values (poi1=val1,poi2=val2#...)")
  parser.add_argument("-c", "--cl"            , type=str  , default="1"      , help="Confidence levels at which to compute the limit, either integers (nsigmas) or floats (CL)")
  parser.add_argument("-o", "--output-file"   , type=str  , required=True    , help="Name of output file")
  parser.add_argument("-x", "--overwrite"     , action='store_true'          , help="Force a repeat of the computation, ignoring data in output file if it already exists.")
  parser.add_argument("-b", "--best-fit-mode" , type=str  , default='single' , help="Best-fit computation: at all points (all), at best point (single) or just the best fixed fit (best_fixed)")
  parser.add_argument("-r", "--setrange"      , type=str  , default=None     , help="List of variable range changes, in the form var1=[min1]:[max1],var2=[min2]:[max2],...")
  parser.add_argument(      "--linear-nps"       , action='store_true'         , help="Use linear NP impacts")
  parser.add_argument("-i", "--iterations"    , type=int  , default=1        , help="Number of iterations to perform for NP computation")
  parser.add_argument(      "--regularize"    , type=float, default=None     , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
  parser.add_argument(      "--cutoff"        , type=float, default=None     , help="Cutoff to regularize the impact of NPs")
  parser.add_argument(      "--marker"        , type=str  , default='+'      , help="Marker type for plots")
  parser.add_argument("-l", "--linestyle"     , type=str  , default='-,--,:' , help="Line style for plots")
  parser.add_argument(      "--smoothing"     , type=int  , default=None     , help="Smoothing for contours (0=no smoothing)")
  parser.add_argument(      "--batch-mode"    , action='store_true'          , help="Batch mode: no plots shown")
  parser.add_argument("-v", "--verbosity"     , type=int  , default=0        , help="Verbosity level")
  parser.add_argument("-t", "--show-timing"   , action='store_true'          , help="Enables printout of timing information")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args(argv)
  if options is None :
    parser.print_help()
    sys.exit(0)

  if options.show_timing : start_time = time.time()

  results_file = options.output_file + '_results.json'
  raster_file = options.output_file + '_raster.json'

  model = make_model(options)
  data = make_data(model, options)
  hypos = make_hypos(model, options)
  
  raster = try_loading_results(model, raster_file, options, hypos)
  if options.hypos is None : hypos = [ hypo for hypo in raster.plr_data ]
  pois = Raster.have_compatible_pois(hypos)
  if pois is None : raise ValueError("Hypotheses '%s' are not compatible (different POIs or different POI ordering)." % ', '.join([str(hypo) for hypo in hypos]))
  if options.verbosity > 1 : 
    print('Will scan the following hypotheses : \n- %s' % '\n- '.join([str(hypo) for hypo in hypos]))

  if raster is None :
    calc = TMuCalculator(OptiMinimizer(niter=options.iterations, verbosity=options.verbosity).set_pois(model), verbosity=options.verbosity)
    if options.verbosity > 1 : 
      prof_pois = set(calc.minimizer.free_pois()) - set(pois)
      if len(prof_pois) > 0 : print('Will profile over POI(s) %s.' % ','.join(prof_pois)) 
    if len(pois) > 2 : raise ValueError('Currently not supporting more than 2 POIs for this operation')
    print('Producing PLR scan with POI(s) %s, bounds %s.' % (str(pois), str(calc.minimizer.bounds)))

  if options.show_timing : comp_start_time = time.time()
  if raster is None :
    raster = calc.compute_fast_results(hypos, data, free_fit=options.best_fit_mode)
    raster.save(raster_file)
  if options.show_timing : comp_stop_time = time.time()

  if not options.batch_mode : raster.print(keys=[ 'tmu' ], verbosity=options.verbosity + 1)
  sdict = {}

  try :
    cl_values = [ float(cl) for cl in options.cl.split(',') ]
  except Exception as inst :
    print(inst)
    print("Could not parse CL/nsigmas specification, expected comma-separated list of float values, got '%s'." % options.cl)
    return
  
  linestyles = options.linestyle.split(',')
  if len(linestyles) == 0 : linestyles = [ 'solid' ]
  if len(linestyles) < len(cl_values) : linestyles.extend([linestyles[-1]]*(len(cl_values) - len(linestyles)))
  if len(linestyles) > len(cl_values) : linestyles = linestyles[:len(cl_values)]
  first = True
  
  if len(pois) == 1 :
    poi_name = pois[0]
    if not options.batch_mode :
      plt.ion()
      fig1, ax1 = plt.subplots(constrained_layout=True)
    for cl, linestyle in zip(cl_values, linestyles) :
      poi_scan = PLRScan1D(raster, 'tmu', name='PLR Scan for %s' % poi_name, ts_name='t_{\\mu}', nsigmas=int(cl) if cl.is_integer() else None, cl=cl if not cl.is_integer() else None)
      interval = poi_scan.interval(print_result=True)
      # Plot results
      if not options.batch_mode :
        #poi_scan.plot(plt, marker='b', linestyle=linestyle, label='PRL_smooth', smooth=100)
        poi_scan.plot(fig1, linestyle=linestyle, label='tmu interpolation', smooth=100)
        poi_scan.plot(fig1, marker=options.marker, linestyle='None')
        plt.ylim(0, None)
        plt.axhline(y=poi_scan.ts_level, color='k', linestyle='dotted')
        plt.show()
        plt.legend()
        plt.savefig(options.output_file + '%.0f%%CL.pdf' % (100*poi_scan.cl()))
        plt.savefig(options.output_file + '%.0f%%CL.png' % (100*poi_scan.cl()))
      cl_dict = {}
      cl_dict['cl'] = poi_scan.cl()
      cl_dict['poi_name'] = poi_name
      cl_dict['poi_unit'] = model.pois[poi_name].unit
      cl_dict['central_value']  = interval[0] if interval is not None else None
      cl_dict['uncertainty_up'] = interval[1] if interval is not None else None
      cl_dict['uncertainty_dn'] = interval[2] if interval is not None else None
      sdict[cl] = cl_dict
      first = False
  else :
    poi1_name = pois[0]
    poi2_name = pois[1]
    if not options.batch_mode :
      plt.ion()
      fig1, ax1 = plt.subplots(constrained_layout=True)
    for cl, linestyle in zip(cl_values, linestyles) :
      poi_scan = PLRScan2D(raster, 'tmu', name='PLR Scan for (%s,%s)' % (poi1_name, poi2_name), ts_name='t_{\\mu}', nsigmas=int(cl) if cl.is_integer() else None, cl=cl if not cl.is_integer() else None)
      if first : best_fit = poi_scan.best_fit(print_result=True)
      if not options.batch_mode :
        poi_scan.plot(fig1, label='%3.1f%% CL' % (poi_scan.cl()*100), best_fit=first,
                      marker=options.marker, linestyle=linestyle, smoothing=options.smoothing)
        plt.show()
        plt.legend()
      first = False

  if options.show_timing :
    stop_time = time.time()
    print("##           Setup time : %g s" % (comp_start_time - start_time))
    print("##     Computation time : %g s" % (comp_stop_time - comp_start_time))
    print("## Post-processing time : %g s" % (stop_time - comp_stop_time))

  with open(results_file, 'w') as fd:
    json.dump(sdict, fd, ensure_ascii=True, indent=3)

if __name__ == '__main__' : run()
