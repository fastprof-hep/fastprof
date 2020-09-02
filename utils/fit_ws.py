#! /usr/bin/env python

__doc__ = "Convert a ROOT workspace into fastprof JSON format"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import math
import json
from scipy.stats import norm
import collections
import ROOT

from workspace_tools import process_setvals, process_setranges, process_setconsts, fit, make_asimov, make_binned

####################################################################################################################################
###

parser = ArgumentParser("convert_ws.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-f", "--ws-file"          , type=str  , required=True    , help="Name of file containing the workspace")
parser.add_argument("-w", "--ws-name"          , type=str  , default='modelWS', help="Name workspace object inside the specified file")
parser.add_argument("-m", "--model-config-name", type=str  , default='mconfig', help="Name of model config within the specified workspace")
parser.add_argument("-d", "--data-name"        , type=str  , default=''       , help="Name of dataset object within the input workspace")
parser.add_argument("-a", "--asimov"           , action="store_true"          , help="Fit an Asimov dataset")
parser.add_argument("-y", "--hypos"            , type=str  , default=''       , help="Colon-separated list of POI hypothesis values")
parser.add_argument(      "--fit-options"      , type=str  , default=''       , help="RooFit fit options to use")
parser.add_argument(      "--binned"           , action="store_true"          , help="Use binned data")
parser.add_argument(      "--input_bins"       , type=int  , default=0        , help="Number of bins to use when binning the input dataset")
parser.add_argument("-=", "--setval"           , type=str  , default=''       , help="Variables to set, in the form var1=val1,var2=val2,...")
parser.add_argument("-k", "--setconst"         , type=str  , default=''       , help="Variables to set constant")
parser.add_argument("-r", "--setrange"         , type=str  , default=''       , help="List of variable range changes, in the form var1:[min1]:[max1],var2:[min2]:[max2],...")
parser.add_argument(      "--poi-min"          , type=float, default=0        , help="POI range minimum")
parser.add_argument(      "--poi-max"          , type=float, default=None     , help="POI range maximum")
parser.add_argument("-n", "--signal-yield"     , type=str  , default='nSignal', help="Name of signal yield variable")
parser.add_argument(      "--nps"              , type=str  , default=''       , help="Constant parameters to include as NPs")
parser.add_argument("-o", "--output-file"      , type=str  , required=True    , help="Name of output file")
parser.add_argument("-v", "--verbosity"        , type=int  , default=0        , help="Verbosity level")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

# 1. Load workspace and modify as specified
# =========================================

f = ROOT.TFile(options.ws_file)
if not f or not f.IsOpen() :
  raise FileNotFoundError('Cannot open file %s' % options.ws_file)

ws = f.Get(options.ws_name)
if not ws : raise KeyError('Workspace %s not found in file %s.' % (options.ws_name, options.ws_file))

if options.setval   != '' : process_setvals  (options.setval  , ws)
if options.setconst != '' : process_setconsts(options.setconst, ws)
if options.setrange != '' : process_setranges(options.setrange, ws)

mconfig = ws.obj(options.model_config_name)
if not mconfig : raise KeyError('Model config %s not found in workspace.' % options.model_config_name)

main_pdf = mconfig.GetPdf()
poi_set = mconfig.GetParametersOfInterest()

if options.poi_min != None : 
  for poi in poi_set : poi.setMin(options.poi_min)
if options.poi_max != None :
  for poi in poi_set : poi.setMax(options.poi_max)

np_set = mconfig.GetNuisanceParameters().selectByAttrib('Constant', False)
extra_nps = ROOT.RooArgSet()

if options.nps != '' :
  varlist = options.nps.split(',')
  for var in varlist :
    matching_vars = ROOT.RooArgList(ws.allVars().selectByName(var))
    if matching_vars.getSize() == 0 :
      print("ERROR : no variables matching '%s' in model" % var)
      raise ValueError
    for i in range(0, matching_vars.getSize()) :
      extra_nps.add(matching_vars.at(i))

ws.saveSnapshot('init', np_set)
poi_init_vals = [ poi.getVal() for poi in poi_set ]


# 2. Get the data and refit as needed
# ===================================

data = None
if options.data_name != '' :
  data = ws.data(options.data_name)
  if data == None :
    ds = [ d.GetName() for d in ws.allData() ]
    raise KeyError('Dataset %s not found in workspace. Available datasets are: %s' % (options.data_name, ', '.join(ds)))
elif options.asimov :
  data = make_asimov(options.asimov, mconfig)

if data == None :
  raise ValueError('Should specify an input dataset, using either the --data-name or --asimov argument.')

# If we specified both, then it means an Asimov with NP values profiled on the observed
if options.data_name != '' and options.asimov : 
  print('=== Generating the main dataset as an Asimov, fitted as below')
  data = make_asimov(options.asimov, mconfig, main_pdf, data)
    
poi.setVal(0)
poi.setConstant(True)
result_bkg_only = fit(main_pdf, data, robust=True)
print('=== Generating an Asimov dataset with POI = 0 and NP values below:')
np_set.Print('V')
asimov0 = ROOT.RooStats.AsymptoticCalculator.MakeAsimovData(mconfig, ROOT.RooArgSet(), ROOT.RooArgSet())

nll  = main_pdf.createNLL(data)
nll0 = main_pdf.createNLL(asimov0)


# 3. Define hypothesis values
# ===========================

try :
  fits_file = open(options.hypos)
  fits_dict = json.load(fits_file)
  hypos = [ fr[poi.GetName()] for fr in fits_dict['fit_results'] ]
except :
  try :
    nhypos = int(options.hypos)
    n1 = (nhypos + 1) // 6
    pos = np.concatenate((np.linspace(0, 3, 2*n1 + 1)[1:-1], np.linspace(3, 8, n1))) # twice as many points in ]0,3[ as in [3,8]
    hypo_zs = np.concatenate((np.flip(-pos), np.zeros(1), pos))
    hypos = None
  except:
    try :
      hypos = [ float(h) for h in options.hypos.split(',') ]
    except Exception as inst :
      print(inst)
      raise ValueError("Could not parse list of hypothesis values '%s' : expected comma-separated list of real values" % options.hypos)

if hypos == None : # we need to auto-define them based on the POI uncertainty
  if len(poi_set) > 1 : raise('Cannot auto-set hypotheses for more than 1 POI')
  ws.loadSnapshot('init')
  nSignal = ws.var(options.signal_yield)
  if nSignal == None :
    nSignal = ws.function(options.signal_yield)
    if nSignal == None :
      raise ValueError('Could not locate signal yield variable %s')
  def hypo_guess(i, unc) :
    pv = 0.05
    return (3*math.exp(0.5/3*i))*math.exp(-unc**2/3) + (1 -math.exp(-unc**2/3))*(i + norm.isf(pv*norm.cdf(i)))*np.sqrt(9 + unc**2)
  poi.setConstant(False)
  poi.setVal(poi_init_val)
  fit(main_pdf, asimov0, robust=True)
  free_nll = nll0.getVal()
  free_val = poi.getVal()
  hypo_val = free_val + poi.getError()/10 # In principle shouldn't have the factor 10, but helps to protect against bad estimations of poi uncertainty
  poi.setVal(hypo_val)
  poi.setConstant(True)
  print('=== Computing qA for poi = %g, computed from val = %g and error = %g' % (poi.getVal(), free_val, poi.getError()))
  fit(main_pdf, asimov0, robust=True)
  hypo_nll = nll0.getVal()
  dll = 2*(hypo_nll - free_nll)
  print('=== Hypo_nll = %10.2f, free_nll = %10.2f => t = %10.2f' % (hypo_nll, free_nll, dll))
  sigma_A = (poi.getVal() - free_val)/math.sqrt(dll) if dll > 0 else poi.getError()
  poi2sig = nSignal.getVal()/poi.getVal()
  print('=== Asimov qA uncertainty = %g (fit uncertainty = %g) evaluated at POI hypo = %g (nSignal = %g)' % (sigma_A, poi.getError(), poi.getVal(), nSignal.getVal()))
  hypos_nS = np.array([ hypo_guess(i, sigma_A*poi2sig) for i in hypo_zs ])
  hypos = hypos_nS/poi2sig
  print('=== Auto-defined the following hypotheses :')
  print('  ' + '\n  '.join([ '%5g : Nsig = %10g, POI = %10g' % (h_z, h_n, h_p) for h_z, h_n, h_p in zip(hypo_zs, hypos_nS, hypos) ] ))

  if options.poi_min == None : 
    # Set min to the -5sigma hypothesis, should be enough...
    poi.setMin(-hypo_guess(5, sigma_A*poi2sig)/poi2sig)
    print('=== Auto-set POI min to %g' % poi.getMin())

  if options.poi_max == None :
    # Set max to the 20sigma hypothesis, should be enough...
    poi.setMax(hypo_guess(20, sigma_A*poi2sig)/poi2sig)
    print('=== Auto-set POI max to %g' % poi.getMax())

jdict = collections.OrderedDict()
fit_results = []

for hypo in hypos :
  result = collections.OrderedDict()
  result[poi.GetName()] = hypo
  # Fit the data first
  print('=== Fitting data to hypothesis %g' % hypo)
  ws.loadSnapshot('init')
  poi.setVal(hypo)
  # Fixed-mu fit
  poi.setConstant(True)
  result_hypo = fit(main_pdf, data, robust=True)
  result['nll_hypo'] = nll.getVal()
  for p in np_set    : result['hypo_' + p.GetName()] = result_hypo.floatParsFinal().find(p.GetName()).getVal()
  for p in extra_nps : result['hypo_' + p.GetName()] = ws.var(p.GetName()).getVal()
  # Free-mu fit
  poi.setConstant(False)
  result_free = fit(main_pdf, data, robust=True)
  result_free.floatParsFinal().Print("V")
  result['nll_free'] = nll.getVal()
  result['fit_val'] = poi.getVal()
  result['fit_err'] = poi.getError()
  for p in np_set    : result['free_' + p.GetName()] = result_free.floatParsFinal().find(p.GetName()).getVal()
  for p in extra_nps : result['free_' + p.GetName()] = ws.var(p.GetName()).getVal()
  # Repeat for Asimov0
  print('=== Fitting Asimov to hypothesis %g' % hypo)
  ws.loadSnapshot('init')
  poi.setVal(hypo)
  # Fixed-mu fit
  poi.setConstant(True)
  result0_hypo = fit(main_pdf, asimov0, robust=True)
  result['nll0_hypo'] = nll0.getVal()
  # Free-mu fit
  poi.setConstant(False)
  result0_free = fit(main_pdf, asimov0, robust=True)
  result['nll0_free'] = nll0.getVal()
  result0_free.floatParsFinal().Print("V")
  # Store results
  fit_results.append(result)

nlls = np.array([ result['nll_free'] for result in fit_results ])
nll_best = np.amin(nlls)
nll0s = np.array([ result['nll0_free'] for result in fit_results ])
nll0_best = np.amin(nll0s)
best_fit_val = fit_results[np.argmin(nlls)]['fit_val']
best_fit_err = fit_results[np.argmin(nlls)]['fit_err']
for result in fit_results :
  result['nll_best']        = nll_best
  result['best_fit_val']    = best_fit_val
  result['best_fit_err']    = best_fit_err
  result['tmu']             = 2*(result['nll_hypo'] - nll_best)
  result['nll0_best']       = nll0_best
  result['tmu_0']           = 2*(result['nll0_hypo'] - nll0_best)

jdict['poi_name'] = poi.GetName()
jdict['poi_initial_value'] = poi_init_val
jdict['poi_range'] = poi.getMin(), poi.getMax()
jdict['fit_results'] = fit_results
with open(options.output_file, 'w') as fd:
  json.dump(jdict, fd, ensure_ascii=True, indent=3)
