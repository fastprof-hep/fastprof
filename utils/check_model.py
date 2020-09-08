#! /usr/bin/env python

__doc__ = "Check asymptotic results of the fast model against those of the full model"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import json

from fastprof import Model, Data, Raster, QMuCalculator, QMuTildaCalculator, OptiMinimizer

####################################################################################################################################
###

parser = ArgumentParser("check_model.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-m", "--model-file"    , type=str  , default=''     , help="Name of JSON file defining model")
parser.add_argument("-d", "--data-file"     , type=str  , default=''     , help="Name of JSON file defining the dataset (optional, otherwise taken from model file)")
parser.add_argument("-a", "--asimov"        , type=float, default=None   , help="Fit an Asimov dataset for the specified POI value")
parser.add_argument("-f", "--fits-file"     , type=str  , default=''     , help="Name of JSON file containing full-model fit results")
parser.add_argument(      "--regularize"    , type=float, default=None   , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
parser.add_argument("-t", "--test-statistic", type=str  , default='q~mu' , help="Test statistic to use in the check")
parser.add_argument(      "--marker"        , type=str  , default=''     , help="Marker type for plots")
parser.add_argument("-b", "--batch-mode"    , action='store_true'        , help="Batch mode: no plots shown")
parser.add_argument(      "--sethypo"       , type=str  , default=''     , help="Change hypo parameter values, in the form par1=val1,par2=val2,...")
parser.add_argument("-v", "--verbosity"     , type=int  , default=0      , help="Verbosity level")
parser.add_argument("-o", "--output-file"   , type=str  , default='check', help="Output file name")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

model = Model.create(options.model_file)
if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
if options.regularize != None : model.set_gamma_regularization(options.regularize)

raster = Raster('data', model=model, filename=options.fits_file)

if options.sethypo != '' :
  try:
    sets = [ v.replace(' ', '').split('=') for v in options.sethypo.split(',') ]
    for plr_data in raster.plr_data.values() :
      for (var, val) in sets :
        if not var in plr_data.hypo : raise ValueError("Cannot find '%s' among hypothesis parameters." % var)
        plr_data.ref_pars[var] = float(val)
        print("INFO : setting %s=%g in reference parameters for %s" % (var, float(val), model.poi_name, plr_data.hypoi))
  except Exception as inst :
    print(inst)
    raise ValueError("ERROR : invalid hypo assignment string '%s'." % options.sethypo)

if options.data_file :
  data = Data(model).load(options.data_file)
  if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
  print('Using dataset stored in file %s.' % options.data_file)
elif options.asimov != None :
  try:
    pars = model.expected_pars()
    sets = [ v.replace(' ', '').split('=') for v in options.asimov.split(',') ]
    data = Data(model).set_expected(model.expected_pars(sets))
  except Exception as inst :
    print(inst)
    raise ValueError("Cannot define an Asimov dataset from options '%s'." % options.asimov)
  print('Using Asimov dataset with POIs %s.' % str(sets))
else :
  print('Using dataset stored in file %s.' % options.model_file)
  data = Data(model).load(options.model_file)

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
calc.fill_all_pv(raster)
faster = calc.compute_fast_results(raster, data)
raster.print(verbosity = options.verbosity, other=faster)

# Plot results
if not options.batch_mode :
  plt.ion()
  fig1 = plt.figure(1)
  plt.suptitle('$CL_{s+b}$')
  plt.xlabel(list(model.pois)[0])
  plt.ylabel('$CL_{s+b}$')
  plt.plot([ hypo[poi.name] for hypo in raster.plr_data ], [ full.pvs['pv'] for full in raster.plr_data.values() ], options.marker + 'r:' , label = 'Full model')
  plt.plot([ hypo[poi.name] for hypo in raster.plr_data ], [ fast.pvs['pv'] for fast in faster.plr_data.values() ], options.marker + 'g-' , label = 'Fast model')
  plt.legend()

  fig2 = plt.figure(2)
  plt.suptitle('$CL_s$')
  plt.xlabel(list(model.pois)[0])
  plt.ylabel('$CL_s$')
  plt.plot([ hypo[poi.name] for hypo in raster.plr_data ], [ full.pvs['cls'] for full in raster.plr_data.values() ], options.marker + 'r:' , label = 'Full model')
  plt.plot([ hypo[poi.name] for hypo in raster.plr_data ], [ fast.pvs['cls'] for fast in faster.plr_data.values() ], options.marker + 'g-' , label = 'Fast model')
  plt.legend()
  fig1.savefig(options.output_file + '_clsb.pdf')
  fig2.savefig(options.output_file + '_cls.pdf')
  plt.show()
