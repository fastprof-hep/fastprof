#! /usr/bin/env python

__doc__ = """
*Check the asymptotic behavior of the model*

Performs p-value and cls computations on the specified model
(using the `--model-file` argument) and compares the results
to those provided in the *fits* file (specified using the
`--fits-file` argument).

The two sets of results should match, if the model is a good
approximation of the full model used to produce the fits file
results.

The dataset used for the computation is supplied either as
observed data (`--data-file` argument) or an Asimov dataset
(`--asimov`).

Several options can be specified to account for non linear
effects (`--iterations`) or regularize the model (`--regularize`,
`--cutoff`), as described in the package documentation.
"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt

from fastprof import Model, Data, Raster, QMuCalculator, QMuTildaCalculator, OptiMinimizer
from fastprof_utils import process_setvals


####################################################################################################################################
def make_parser() :
  parser = ArgumentParser("check_model.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"    , type=str  , default=''     , help="Name of markup file defining model")
  parser.add_argument("-d", "--data-file"     , type=str  , default=''     , help="Name of markup file defining the dataset (optional, otherwise taken from model file)")
  parser.add_argument("-a", "--asimov"        , type=str  , default=None   , help="Fit an Asimov dataset for the specified POI value")
  parser.add_argument("-f", "--fits-file"     , type=str  , default=''     , help="Name of markup file containing full-model fit results")
  parser.add_argument("-r", "--setrange"      , type=str  , default=None   , help="List of variable range changes, in the form var1:[min1]:[max1],var2:[min2]:[max2],...")
  parser.add_argument("-i", "--iterations"    , type=int  , default=1      , help="Number of iterations to perform for NP computation")
  parser.add_argument(      "--regularize"    , type=float, default=None   , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
  parser.add_argument(      "--cutoff"        , type=float, default=None   , help="Cutoff to regularize the impact of NPs")
  parser.add_argument("-t", "--test-statistic", type=str  , default='q~mu' , help="Test statistic to use in the check")
  parser.add_argument(      "--marker"        , type=str  , default=''     , help="Marker type for plots")
  parser.add_argument("-b", "--batch-mode"    , action='store_true'        , help="Batch mode: no plots shown")
  parser.add_argument("-v", "--verbosity"     , type=int  , default=1      , help="Verbosity level")
  parser.add_argument("-o", "--output-file"   , type=str  , default='check', help="Output file name")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args()
  if not options :
    parser.print_help()
    return

  model = Model.create(options.model_file)
  if model is None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
  if options.regularize is not None : model.set_gamma_regularization(options.regularize)
  if options.cutoff is not None : model.cutoff = options.cutoff
  if options.setrange is not None : process_setranges(options.setrange, model)

  raster = Raster('data', model=model, filename=options.fits_file)

  if options.data_file :
    data = Data(model).load(options.data_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
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
    print('Using dataset stored in file %s.' % options.model_file)
    data = Data(model).load(options.model_file)

  if options.test_statistic == 'q~mu' :
    if len(raster.pois()) > 1 : raise ValueError('Currently not supporting more than 1 POI for this operation')
    calc = QMuTildaCalculator(OptiMinimizer(niter=options.iterations).set_pois_from_model(model))
  elif options.test_statistic == 'q_mu' :
    if len(raster.pois()) > 1 : raise ValueError('Currently not supporting more than 1 POI for this operation')
    calc = QMuCalculator(OptiMinimizer(niter=options.iterations).set_pois_from_model(model))
  else :
    raise ValueError('Unknown test statistic %s' % options.test_statistic)
  calc.fill_all_pv(raster)
  faster = calc.recompute_raster(raster, data)
  raster.print(verbosity = options.verbosity, other=faster)
  if options.verbosity > 2 : print(str(faster))
  # Plot results
  if not options.batch_mode :
    poi = raster.pois()[list(raster.pois())[0]]
    plt.ion()
    fig1 = plt.figure(1)
    plt.suptitle('$CL_{s+b}$')
    plt.xlabel(model.poi(0).name)
    plt.ylabel('$CL_{s+b}$')
    plt.plot([ hypo[poi.name] for hypo in raster.plr_data ], [ full.pvs['pv'] for full in raster.plr_data.values() ], options.marker + 'r:' , label = 'Full model')
    plt.plot([ hypo[poi.name] for hypo in faster.plr_data ], [ fast.pvs['pv'] for fast in faster.plr_data.values() ], options.marker + 'g-' , label = 'Fast model')
    plt.legend()

    fig2 = plt.figure(2)
    plt.suptitle('$CL_s$')
    plt.xlabel(model.poi(0).name)
    plt.ylabel('$CL_s$')
    plt.plot([ hypo[poi.name] for hypo in raster.plr_data ], [ full.pvs['cls'] for full in raster.plr_data.values() ], options.marker + 'r:' , label = 'Full model')
    plt.plot([ hypo[poi.name] for hypo in faster.plr_data ], [ fast.pvs['cls'] for fast in faster.plr_data.values() ], options.marker + 'g-' , label = 'Fast model')
    plt.legend()
    fig1.savefig(options.output_file + '_clsb.pdf')
    fig2.savefig(options.output_file + '_cls.pdf')
    plt.show()

if __name__ == '__main__' : run()
