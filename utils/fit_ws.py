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
# - Ass Asimov fit to better compute fit_err from qA

####################################################################################################################################
###

def fit_ws() :
  """convert """
  
  parser = ArgumentParser("convert_ws.py")
  parser.description = __doc__
  parser.add_argument("-f", "--ws-file",           default='',        help="Name of file containing the workspace", type=str)
  parser.add_argument("-w", "--ws-name",           default='modelWS', help="Name workspace object inside the specified file", type=str)
  parser.add_argument("-m", "--model-config-name", default='mconfig', help="Name of model config within the specified workspace", type=str)
  parser.add_argument("-d", "--data-name",         default='',        help="Name of dataset object within the input workspace", type=str)
  parser.add_argument("-y", "--hypos",             default='',        help="Comma-separated list of POI hypothesis values", type=str)
  parser.add_argument(      "--fit-options",       default='',        help="RooFit fit options to use", type=str)
  parser.add_argument("-=", "--setval",            default='',        help="Variables to set, in the form var1=val1,var2=val2,...", type=str)
  parser.add_argument("-o", "--output-file",       default='',        help="Name of output file", type=str)
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
  
  data = ws.data(options.data_name)
  if data == None :
    ds = [ d.GetName() for d in ws.allData() ]
    raise KeyError('Dataset %s not found in workspace. Available datasets are: %s' % (options.data_name, ', '.join(ds)))

  if options.setval != '' :
    try: 
      sets = [ v.replace(' ', '').split('=') for v in options.setval.split(',') ]
      for (var, val) in sets :
        if not ws.var(var) :
          raise ValueError("Cannot find variable '%s' in workspace" % var)          
        ws.var(var).setVal(float(val))
        print "INFO : setting %s=%g" % (var, float(val))
    except:
      raise ValueError("ERROR : invalid variable assignment string '%s'." % options.setval)

  try :
    hypos = [ float(h) for h in options.hypos.split(',') ]
  except:
    raise ValueError("Could not parse list of hypothesis values '%s' : expected comma-separated list of real values" % options.hypos)

  ws.saveSnapshot('init', nuis_pars)
  jdict = []
  
  for hypo in hypos :
    ws.loadSnapshot('init')
    poi.setVal(hypo)
    hypo_dict = collections.OrderedDict()
    hypo_dict[poi.GetName()] = hypo
    poi.setConstant(True)
    result_hypo = main_pdf.fitTo(data, ROOT.RooFit.Save(), ROOT.RooFit.Minimizer('Minuit2', 'migrad'))
    hypo_dict['nll_hypo'] = result_hypo.minNll()
    poi.setConstant(False)
    result_free = main_pdf.fitTo(data, ROOT.RooFit.Save(), ROOT.RooFit.Minimizer('Minuit2', 'migrad'))
    hypo_dict['nll_free'] = result_free.minNll()
    hypo_dict['fit_val'] = poi.getVal()
    hypo_dict['fit_err'] = poi.getError()
    jdict.append(hypo_dict)

  nlls = np.array([ hd['nll_free'] for hd in jdict ])
  nll_best = np.amin(nlls)
  best_fit_val = jdict[np.argmin(nlls)]['fit_val']
  best_fit_err = jdict[np.argmin(nlls)]['fit_err']
  for hd in jdict :
    hd['nll_best'] = nll_best
    hd['best_fit_val'] = best_fit_val
    hd['best_fit_err'] = best_fit_err
    hd['qmu'] = 2*(hd['nll_hypo'] - nll_best) if fitval_best < hd[poi.GetName()] else 0 # q_mu case only
  
  with open(options.output_file, 'w') as fd:
    json.dump(jdict, fd, ensure_ascii=True, indent=3)
  
  return 0

fit_ws()
