#! /usr/bin/env python

__doc__ = """
*Perform a POI scan*

"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fastprof import Model, Data, OptiMinimizer, NPMinimizer, TMuCalculator, process_setvals, process_setranges
import numpy as np
import matplotlib.pyplot as plt

####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("fit_fast.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"       , type=str  , required=True , help="Name of markup file defining model")
  parser.add_argument("-d", "--data-file"        , type=str  , default=''    , help="Name of markup file defining the dataset (optional, otherwise taken from model file)")
  parser.add_argument("-y", "--hypos"            , type=str  , required=True , help="Parameter hypothesis to test")
  parser.add_argument("-a", "--asimov"           , type=str  , default=None  , help="Use an Asimov dataset for the specified POI values (format: 'poi1=xx,poi2=yy'")
  parser.add_argument("-r", "--setrange"         , type=str  , default=None  , help="List of variable range changes, in the form var1:[min1]:[max1],var2:[min2]:[max2],...")
  parser.add_argument("-i", "--iterations"       , type=int  , default=1     , help="Number of iterations to perform for NP computation")
  parser.add_argument(      "--regularize"       , type=float, default=None  , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
  parser.add_argument(      "--cutoff"           , type=float, default=None  , help="Cutoff to regularize the impact of NPs")
  parser.add_argument("-l", "--log-scale"        , action='store_true'       , help="Use log scale for plotting")
  parser.add_argument("-v", "--verbosity"        , type = int, default=0     , help="Verbosity level")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args()
  if not options :
    parser.print_help()
    return

  # Define the model
  model = Model.create(options.model_file)
  if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
  if options.regularize is not None : model.set_gamma_regularization(options.regularize)
  if options.cutoff is not None : model.cutoff = options.cutoff
  if options.setrange is not None : process_setranges(options.setrange, model)

  # Define the data
  if options.data_file :
    data = Data(model).load(options.data_file)
    if data is None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    print('Using dataset stored in file %s.' % options.data_file)
  elif options.asimov is not None :
    try :
      sets = process_setvals(options.asimov, model)
    except Exception as inst :
      print(inst)
      raise ValueError("ERROR : invalid POI specification string '%s'." % options.asimov)
    data = model.generate_expected(sets)
    print('Using Asimov dataset with parameters %s' % str(sets))
  else :
    data = Data(model).load(options.model_file)

  # Parse the hypothesis values
  if options.hypos.find(':') :
    try :
      hypo_specs = options.hypos.split(':')
      poi_name = None
      if hypo_specs[-1] == 'log' :
        hypos = np.logspace(1, math.log(float(hypo_specs[-3]))/math.log(float(hypo_specs[-4])), int(hypo_specs[-2]) + 1, True, float(hypo_specs[0]))
        if len(hypo_specs) == 5 : poi_name = hypo_specs[0] 
      else :
        hypos = np.linspace(float(hypo_specs[-3]), float(hypo_specs[-2]), int(hypo_specs[-1]) + 1)
        if len(hypo_specs) == 4 : poi_name = hypo_specs[0]
    except Exception as inst :
      print(inst)
      raise ValueError("Could not parse list of hypothesis values '%s' : expected min:max:num[:log] format" % options.hypos)
    if poi_name is not None :
      if not poi_name in model.pois : raise ValueError("Unknown POI '%s' in hypothesis definitions" % poi_name)
    else :
      poi_name = model.poi(0).name
    hypo_sets = [ { poi_name : hypo } for hypo in hypos ] 
  else :
    try :
      hypo_sets = [ process_setvals(spec, model, match_nps=False) for spec in options.hypos.split('/') ]
    except Exception as inst :
      print(inst)
      raise ValueError("Could not parse list of hypothesis values '%s' : expected /-separated list of POI assignments" % options.hypos)
  hypos = [ model.expected_pars(sets) for sets in hypo_sets ]

  # Compute the tmu values
  calc = TMuCalculator(OptiMinimizer(niter=options.iterations).set_pois_from_model(model))
  raster = calc.compute_fast_results(hypos, data)
  hypos = [ hypo[poi_name] for hypo in raster.plr_data.keys() ]
  tmus  = [ plr_data.test_statistics['tmu'] for plr_data in raster.plr_data.values() ]
  #print(raster)
  
  # Find the minimal tmu
  min_index = np.argmin(tmus)
  if min_index == 0 :
    print('Found minimum at the lower edge of the scan, returning this value')
    min_hypo = hypos[min_index]
  elif min_index == len(tmus) :
    print('Found minimum at the upper edge of the scan, returning this value')
    min_hypo = hypos[min_index]
  else :
    calc.minimizer.minimize(data, list(raster.plr_data.keys())[min_index])
    min_hypo = calc.minimizer.min_pars[poi_name]
    
  # Compute the tmu=1 crossings and uncertainties
  crossings = raster.interpolate(hypos, tmus, 1)
  if len(crossings) == 2 :
    print('1-sigma interval : %g + %g - %g' % (min_hypo, crossings[1] - min_hypo, min_hypo - crossings[0]))
  
  # Plot the result
  plt.ion()
  plt.figure(1)
  plt.plot(hypos, tmus)
  plt.ylim(0, None)
  plt.xlabel(poi_name)
  plt.ylabel('t_mu(%s)' % poi_name)

if __name__ == '__main__' : run()
