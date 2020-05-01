#! /usr/bin/env python

__doc__ = "Convert a ROOT workspace into fastprof JSON format"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser
import ROOT
import numpy as np
import json
import collections

# TODO: 
# - add fit options support
# - other test statistics than q_mu

####################################################################################################################################
###

def fit_ws() :
  """fit_ws """
  
  parser = ArgumentParser("convert_ws.py")
  parser.description = __doc__
  parser.add_argument("-f", "--ws-file",           required=True,     help="Name of file containing the workspace", type=str)
  parser.add_argument("-w", "--ws-name",           default='modelWS', help="Name workspace object inside the specified file", type=str)
  parser.add_argument("-m", "--model-config-name", default='mconfig', help="Name of model config within the specified workspace", type=str)
  parser.add_argument("-d", "--data-name",         default='',        help="Name of dataset object within the input workspace", type=str)
  parser.add_argument("-a", "--asimov",        action="store_true",   help="Fit an Asimov dataset")
  parser.add_argument("-y", "--hypos",             default='',        help="Comma-separated list of POI hypothesis values", type=str)
  parser.add_argument(      "--fit-options",       default='',        help="RooFit fit options to use", type=str)
  parser.add_argument("-=", "--setval",            default='',        help="Variables to set, in the form var1=val1,var2=val2,...", type=str)
  parser.add_argument("-i", "--poi-initial-value", default=None,      help="POI allowed range, in the form min,max", type=float)
  parser.add_argument("-r", "--poi-range",         default='',        help="POI allowed range, in the form min,max", type=str)
  parser.add_argument("-o", "--output-file",       required=True,     help="Name of output file", type=str)
  parser.add_argument("-v", "--verbosity",         default=0,         help="Verbosity level", type=int)
  
  options = parser.parse_args()
  if not options : 
    parser.print_help()
    return

  f = ROOT.TFile(options.ws_file)
  if not f or not f.IsOpen() :
    raise FileNotFoundError('Cannot open file %s' % options.ws_file)
  
  ws = f.Get(options.ws_name)
  if not ws :
    raise KeyError('Workspace %s not found in file %s.' % (options.ws_name, options.ws_file))
  
  mconfig = ws.obj(options.model_config_name)
  if not mconfig :
    raise KeyError('Model config %s not found in workspace.' % options.model_config_name)

  main_pdf = mconfig.GetPdf()
  nuis_pars = mconfig.GetNuisanceParameters()
  poi = ROOT.RooArgList(mconfig.GetParametersOfInterest()).at(0) # make safer!

  if options.setval != '' :
    try: 
      sets = [ v.replace(' ', '').split('=') for v in options.setval.split(',') ]
      for (var, val) in sets :
        if not ws.var(var) :
          raise ValueError("Cannot find variable '%s' in workspace" % var)          
        ws.var(var).setVal(float(val))
        print "INFO : setting %s=%g" % (var, float(val))
    except Exception as inst :
      print(inst)
      raise ValueError("ERROR : invalid variable assignment string '%s'." % options.setval)

  if options.poi_initial_value != None :
    poi.setVal(options.poi_initial_value)
    
  if options.poi_range != '' :
    try:
      poi_min, poi_max = [ float(p) for p in options.poi_range.split(',') ]
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid POI range specification %s, expected poi_min,poi_max' % options.poi_range)
    if poi_min > poi_max : poi_min, poi_max = poi_max, poi_min
    poi.setRange(poi_min, poi_max)

  # Asimov dataset -- used later for qA computation
  asimov = ROOT.RooStats.AsymptoticCalculator.MakeAsimovData(mconfig, ROOT.RooArgSet(), ROOT.RooArgSet())

  if options.data_name != '' :
    data = ws.data(options.data_name)
    if data == None :
      ds = [ d.GetName() for d in ws.allData() ]
      raise KeyError('Dataset %s not found in workspace. Available datasets are: %s' % (options.data_name, ', '.join(ds)))
  elif options.asimov :
    data = asimov
  else :
    raise ValueError('Should specify an input dataset, using either the --data-name or --asimov argument.')

  try :
    hypos = [ float(h) for h in options.hypos.split(',') ]
  except Exception as inst :
    print(inst)
    raise ValueError("Could not parse list of hypothesis values '%s' : expected comma-separated list of real values" % options.hypos)

  ws.saveSnapshot('init', nuis_pars)
  poi_init_val = poi.getVal()
  jdict = collections.OrderedDict()
  fit_results = []
  
  nll = main_pdf.createNLL(data, ROOT.RooFit.SumW2Error(False))
  asimov_nll = main_pdf.createNLL(asimov, ROOT.RooFit.SumW2Error(False))

  for hypo in hypos :
    # Set the hypothesis
    ws.loadSnapshot('init')
    poi.setVal(hypo)
    result = collections.OrderedDict()
    result[poi.GetName()] = hypo
    # Fixed-mu fit
    poi.setConstant(True)
    result_hypo = main_pdf.fitTo(data, ROOT.RooFit.Save(), ROOT.RooFit.Offset(), ROOT.RooFit.SumW2Error(False), ROOT.RooFit.Minimizer('Minuit2', 'migrad'))
    result['nll_hypo'] = nll.getVal()
    # Free-mu fit
    poi.setConstant(False)
    result_free = main_pdf.fitTo(data, ROOT.RooFit.Save(), ROOT.RooFit.Offset(), ROOT.RooFit.SumW2Error(False), ROOT.RooFit.Minimizer('Minuit2', 'migrad'))
    result_free.floatParsFinal().Print("V")
    result['nll_free'] = nll.getVal()
    result['fit_val'] = poi.getVal()
    result['fit_err'] = poi.getError()
    # Repeat for Asimov
    ws.loadSnapshot('init')
    poi.setVal(hypo)
    poi.setConstant(True)
    result_asimov_hypo = main_pdf.fitTo(asimov, ROOT.RooFit.Save(), ROOT.RooFit.Offset(), ROOT.RooFit.SumW2Error(False), ROOT.RooFit.Minimizer('Minuit2', 'migrad'))
    result['asimov_nll_hypo'] = asimov_nll.getVal()
    # Free-mu fit
    poi.setConstant(False)
    result_asimov_free = main_pdf.fitTo(asimov, ROOT.RooFit.Save(), ROOT.RooFit.Offset(), ROOT.RooFit.SumW2Error(False), ROOT.RooFit.Minimizer('Minuit2', 'migrad'))
    result['asimov_nll_free'] = asimov_nll.getVal()
    result_asimov_free.floatParsFinal().Print("V")
    # Store results
    fit_results.append(result)

  nlls = np.array([ result['nll_free'] for result in fit_results ])
  nll_best = np.amin(nlls)
  asimov_nlls = np.array([ result['asimov_nll_free'] for result in fit_results ])
  asimov_nll_best = np.amin(asimov_nlls)
  best_fit_val = fit_results[np.argmin(nlls)]['fit_val']
  best_fit_err = fit_results[np.argmin(nlls)]['fit_err']
  for result in fit_results :
    result['nll_best'] = nll_best
    result['best_fit_val'] = best_fit_val
    result['best_fit_err'] = best_fit_err
    result['qmu'] = 2*(result['nll_hypo'] - nll_best) if best_fit_val < result[poi.GetName()] else 0 # q_mu case only
    result['asimov_nll_best'] = asimov_nll_best
    result['qmu_A'] = 2*(result['asimov_nll_hypo'] - asimov_nll_best)
  
  jdict['POI_name'] = poi.GetName()
  jdict['POI_initial_value'] = poi_init_val
  jdict['POI_range'] = poi.getMin(), poi.getMax()
  jdict['test_statistic'] = 'qmu'
  jdict['fit_results'] = fit_results
  with open(options.output_file, 'w') as fd:
    json.dump(jdict, fd, ensure_ascii=True, indent=3)

  return 0

fit_ws()
