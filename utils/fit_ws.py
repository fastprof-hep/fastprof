#! /usr/bin/env python

__doc__ = "Convert a ROOT workspace into fastprof JSON format"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import ROOT
import numpy as np
import json
from scipy.stats import norm
import collections

# TODO: 
# - add fit options support
# - other test statistics than q_mu

####################################################################################################################################
###

parser = ArgumentParser("convert_ws.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-f", "--ws-file"          , type=str   , required=True,       help="Name of file containing the workspace")
parser.add_argument("-w", "--ws-name"          , type=str   , default='modelWS',   help="Name workspace object inside the specified file")
parser.add_argument("-m", "--model-config-name", type=str   , default='mconfig',   help="Name of model config within the specified workspace")
parser.add_argument("-d", "--data-name"        , type=str   , default='',          help="Name of dataset object within the input workspace")
parser.add_argument("-a", "--asimov"           , action="store_true",              help="Fit an Asimov dataset")
parser.add_argument("-y", "--hypos"            , type=str   , default='',          help="Comma-separated list of POI hypothesis values")
parser.add_argument(      "--fit-options"      , type=str   , default='',          help="RooFit fit options to use")
parser.add_argument("-=", "--setval"           , type=str   , default='',          help="Variables to set, in the form var1=val1,var2=val2,...")
parser.add_argument("-k", "--setconst"         , type=str   , default='',          help="Variables to set constant")
parser.add_argument("-i", "--poi-initial-value", type=float , default=None,        help="POI allowed range, in the form min,max")
parser.add_argument("-r", "--poi-range"        , type=str   , default='',          help="POI allowed range, in the form min,max")
parser.add_argument("-n", "--signal-yield"     , type=str   , default='nSignal'  , help="Name of signal yield variable")
parser.add_argument("-o", "--output-file"      , type=str   , required=True,       help="Name of output file")
parser.add_argument("-v", "--verbosity"        , type=int   , default=0,           help="Verbosity level")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

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
      print("INFO : setting %s=%g" % (var, float(val)))
  except Exception as inst :
    print(inst)
    raise ValueError("ERROR : invalid variable assignment string '%s'." % options.setval)

if options.setconst != '' :
  varlist = options.setconst.split(',')
  for var in varlist :
    matching_vars = ROOT.RooArgList(ws.allVars().selectByName(var))
    if matching_vars.getSize() == 0 :
      print("ERROR : no variables matching '%s' in model" % var)
      raise ValueError
    for i in range(0, matching_vars.getSize()) :
      thisvar =  matching_vars.at(i)
      thisvar.setConstant()
      print("INFO : setting variable '%s' constant (current value: %g)" % (thisvar.GetName(), thisvar.getVal()))

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
  nhypos = int(options.hypos)
  n1 = (nhypos + 1) // 4
  pos = np.concatenate((np.linspace(0, 2, n1+2)[1:-1], np.linspace(2, 5, n1))) # same number of points between [0,2] and [2,5]
  hypo_zs = np.concatenate((np.flip(-pos), np.zeros(1), pos))
  hypos = None
except:
  try :
    hypos = [ float(h) for h in options.hypos.split(',') ]
  except Exception as inst :
    print(inst)
    raise ValueError("Could not parse list of hypothesis values '%s' : expected comma-separated list of real values" % options.hypos)

ws.saveSnapshot('init', nuis_pars)
nuis_pars.Print("V")
poi_init_val = poi.getVal()
jdict = collections.OrderedDict()
fit_results = []

nll = main_pdf.createNLL(data, ROOT.RooFit.SumW2Error(False))
asimov_nll = main_pdf.createNLL(asimov, ROOT.RooFit.SumW2Error(False))

if hypos == None : # we need to auto-define them based on the POI uncertainty
  nSignal = ws.var(options.signal_yield)
  if nSignal == None :
    nSignal = ws.function(options.signal_yield)
    if nSignal == None :
      raise ValueError('Could not locate signal yield variable %s')
  def hypo_guess(i, unc) :
    cl = 0.05
    return (3 + 0.5*i)*np.exp(-unc**2/3) + (1 - np.exp(-unc**2/3))*(i + norm.isf(cl*norm.cdf(i)))*np.sqrt(9 + unc**2)
  poi.setConstant(True)
  main_pdf.fitTo(asimov, ROOT.RooFit.Save(), ROOT.RooFit.Offset(), ROOT.RooFit.SumW2Error(False), ROOT.RooFit.Minimizer('Minuit2', 'migrad'))
  poi.setConstant(False)
  main_pdf.fitTo(asimov, ROOT.RooFit.Save(), ROOT.RooFit.Offset(), ROOT.RooFit.SumW2Error(False), ROOT.RooFit.Minimizer('Minuit2', 'migrad'))
  poi.setVal(1)
  hypos_nS = np.array([ hypo_guess(i, poi.getError()/poi.getVal()*nSignal.getVal()) for i in hypo_zs ])
  hypos = hypos_nS/nSignal.getVal()*poi.getVal()
  print('Auto-defined the following hypotheses :')
  print('  ' + '\n  '.join([ '%5g : Nsig = %10g, POI = %10g' % (h_z, h_n, h_p) for h_z, h_n, h_p in zip(hypo_zs, hypos_nS, hypos) ] ))
  if options.poi_range == '' :
    # Set range up to the 10sigma hypothesis, should be enough...
    poi.setRange(0, hypo_guess(10, poi.getError()/poi.getVal()*nSignal.getVal())/nSignal.getVal()*poi.getVal())
    print('Auto-set POI range to [%g, %g]' % (poi.getMin(), poi.getMax()))

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
  result['nll_best']        = nll_best
  result['best_fit_val']    = best_fit_val
  result['best_fit_err']    = best_fit_err
  result['tmu']             = 2*(result['nll_hypo'] - nll_best)
  result['asimov_nll_best'] = asimov_nll_best
  result['tmu_A']           = 2*(result['asimov_nll_hypo'] - asimov_nll_best)
  result['tmu_0']           = result['tmu_A'] # Since the Asimov is produced with mu=0 (A=0...)

jdict['POI_name'] = poi.GetName()
jdict['POI_initial_value'] = poi_init_val
jdict['POI_range'] = poi.getMin(), poi.getMax()
jdict['fit_results'] = fit_results
with open(options.output_file, 'w') as fd:
  json.dump(jdict, fd, ensure_ascii=True, indent=3)
