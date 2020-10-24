#! /usr/bin/env python

__doc__ = """
*Fit a model to a dataset*

Performs the fit of a model (defined using the `--model-file` argument)
to a dataset, supplied either as observed data (`--data-file` argument)
or an Asimov dataset (`--asimov`).

If a parameter hypothesis is specified (`--hypo` argument), the results
of the test of the hypothesis in the data is also shown.

Several options can be specified to account for non linear effects (`--iterations`)
or regularize the model (`--regularize`, `--cutoff`),
as described in the package documentation.
"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fastprof import Model, Data, OptiMinimizer, NPMinimizer, QMuTildaCalculator, process_setvals, process_setranges
import matplotlib.pyplot as plt

####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("fit_fast.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"       , type=str  , required=True , help="Name of JSON file defining model")
  parser.add_argument("-d", "--data-file"        , type=str  , default=''    , help="Name of JSON file defining the dataset (optional, otherwise taken from model file)")
  parser.add_argument("-y", "--hypo"             , type=str  , default=None  , help="Parameter hypothesis to test")
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

  model = Model.create(options.model_file)
  if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
  if options.regularize is not None : model.set_gamma_regularization(options.regularize)
  if options.cutoff is not None : model.cutoff = options.cutoff
  if options.setrange is not None : process_setranges(options.setrange, model)

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

  if options.hypo is not None :
    try :
      sets = process_setvals(options.hypo, model)
    except Exception as inst :
      print(inst)
      raise ValueError("ERROR : invalid POI specification string '%s'." % options.hypo)
    hypo_pars = model.expected_pars(sets)

  opti = OptiMinimizer().set_pois_from_model(model)
  min_nll = opti.minimize(data)
  min_pars = opti.min_pars
  print('\n== Best-fit: nll = %g @ at parameter values =' % min_nll)
  print(min_pars)

  if options.hypo is not None :
    tmu = opti.tmu(hypo_pars, data, hypo_pars)
    print('\n== Profile-likelihood ratio tmu = %g for hypothesis' % tmu, hypo_pars.dict(pois_only=True))
    print('-- Profiled NP values :\n' + str(opti.hypo_pars))
    if len(model.pois) == 1 :
      print('\n== Computing the q~mu test statistic')
      asimov = model.generate_expected(0, NPMinimizer(data))
      calc = QMuTildaCalculator(opti)
      plr_data = calc.compute_fast_q(hypo_pars, data)
      print('best-fit %s = % g' % (model.poi(0).name, opti.free_pars.pois[0]))
      print('tmu         = % g' % plr_data.test_statistics['tmu'])
      print('q~mu        = % g' % plr_data.test_statistics['q~mu'])
      print('pv          = % g' % plr_data.pvs['pv'])
      print('cls         = % g' % plr_data.pvs['cls'])
  plt.ion()
  plt.figure(1)
  model.plot(min_pars, data=data)
  if options.log_scale : plt.yscale('log')

if __name__ == '__main__' : run()
