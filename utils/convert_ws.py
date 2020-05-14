#! /usr/bin/env python

__doc__ = "Convert a ROOT workspace into fastprof JSON format"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import ROOT
import numpy as np
import json
import collections
import array
import math

####################################################################################################################################
###

parser = ArgumentParser("convert_ws.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-f", "--ws-file"            , type=str  , required=True    , help="Name of file containing the workspace")
parser.add_argument("-w", "--ws-name"            , type=str  , default='modelWS', help="Name workspace object inside the specified file")
parser.add_argument("-m", "--model-config-name"  , type=str  , default='mconfig', help="Name of model config within the specified workspace")
parser.add_argument("-s", "--signal-pdf"         , type=str  , default='Signal' , help="Name of signal component PDF")
parser.add_argument("-n", "--signal-yield"       , type=str  , default='nSignal', help="Name of signal yield variable")
parser.add_argument("-b", "--binning"            , type=str  , required=True    , help="Binning used, in the form xmin:xmax:nbins[:log]")
parser.add_argument("-p", "--nps"                , type=str  , default=''       , help="List of constrained nuisance parameters")
parser.add_argument("-e", "--epsilon"            , type=float, default=1        , help="Scale factor applied to uncertainties for impact computations")
parser.add_argument("-=", "--setval"             , type=str  , default=''       , help="Variables to set, in the form var1=val1,var2=val2,...")
parser.add_argument("-k", "--setconst"           , type=str  , default=''       , help="Variables to set constant")
parser.add_argument("-r", "--refit"              , action="store_true"          , help="Perform a fit to the dataset (specified by --data-name) before conversion")
parser.add_argument("-u", "--refit-uncertainties", action="store_true"          , help="Update uncertainties (but not central values) from a fit to the specified dataset")
parser.add_argument("-a", "--asimov"             , action="store_true"          , help="Perform an Asimov fit before conversion")
parser.add_argument("-x", "--data-only"          , action="store_true"          , help="Only dump the specified dataset, not the model")
parser.add_argument("-d", "--data-name"          , type=str  , default=''       , help="Name of dataset object within the input workspace")
parser.add_argument("-o", "--output-file"        , type=str  , required=True    , help="Name of output file")
parser.add_argument(      "--output-name"        , type=str  , default=''       , help="Name of the output model")
parser.add_argument(      "--validation-data"    , type=str  , default=''       , help="Name of output file for validation data")
parser.add_argument("-v", "--verbosity"          , type=int  , default=0        , help="Verbosity level")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

try:
  binspec = options.binning.split(':')
  if len(binspec) == 4 and binspec[3] == 'log' :
    bins = np.logspace(1, math.log(float(binspec[1]))/math.log(float(binspec[0])), int(binspec[2]) + 1, True, float(binspec[0]))
    print(bins)
  else :
    bins = np.linspace(float(binspec[0]), float(binspec[1]), int(binspec[2]) + 1)
except Exception as inst :
  print(inst)
  raise ValueError('Invalid bin specification %s : the format should be xmin:xmax:nbins[:log]' % options.binning)
nbins = len(bins) - 1

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
signal_pdf = ws.pdf(options.signal_pdf)
nSignal = ws.var(options.signal_yield)
if nSignal == None :
  nSignal = ws.function(options.signal_yield)
  if nSignal == None :
    raise ValueError('Could not locate signal yield variable %s')

try :
  obs = ROOT.RooArgList(mconfig.GetObservables()).at(0)
except Exception as inst :
  print(inst)
  ValueError('Could not locate observable')

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

if options.setconst != '' :
  varlist = options.setconst.split(',')
  for var in varlist :
    matching_vars = ROOT.RooArgList(ws.allVars().selectByName(var))
    if matching_vars.getSize() == 0 :
      print "ERROR : no variables matching '%s' in model" % var
      raise ValueError
    for i in range(0, matching_vars.getSize()) :
      thisvar =  matching_vars.at(i)
      thisvar.setConstant()
      print "INFO : setting variable '%s' constant (current value: %g)" % (thisvar.GetName(), thisvar.getVal())

data = None
if options.data_name != '' :
  data = ws.data(options.data_name)
  if data == None :
    ds = [ d.GetName() for d in ws.allData() ]
    raise KeyError('Dataset %s not found in workspace. Available datasets are: %s' % (options.data_name, ', '.join(ds)))

if options.asimov :
  data = ROOT.RooStats.AsymptoticCalculator.MakeAsimovData(mconfig, ROOT.RooArgSet(), ROOT.RooArgSet())

if options.data_only and not data :
  raise ValueError('Requested to dump only the data (--data-only) but not dataset was specified either using --data-name or --asimov')


if not options.data_only and (options.refit or options.refit_uncertainties) :
  if data == None :
    raise ValueError('Should specify a dataset on which to perform the fit, using either the --data-name or --asimov argument.')
  allVars = ROOT.RooArgList(ws.allVars())
  if options.refit_uncertainties :
    save_vals = {}
    for i in range(0, allVars.getSize()) :
      v = allVars.at(i)
      if not v.isConstant() : save_vals[v.GetName()] = v.getVal()
  main_pdf.fitTo(data, ROOT.RooFit.SumW2Error(False), ROOT.RooFit.Offset())
  if options.refit_uncertainties :
    for i in range(0, allVars.getSize()) :
      v = allVars.at(i)
      if not v.isConstant() : v.setVal(save_vals[v.GetName()])

aux_alphas = []
aux_betas  = []
alphas = []
betas = []
gammas = []

aux_obs = ROOT.RooArgList(mconfig.GetGlobalObservables())
nuis_pars = mconfig.GetNuisanceParameters().selectByAttrib('Constant', False)
pdfs = main_pdf.pdfList()
try:
  for o in range(0, len(aux_obs)) :
    aux = aux_obs.at(o)
    for p in range(0, len(pdfs)) :
      pdf = pdfs.at(p)
      if len(pdf.getDependents(ROOT.RooArgSet(aux))) > 0 :
        matching_pars = pdf.getDependents(nuis_pars)
        if len(matching_pars) == 1 :
          mpar = ROOT.RooArgList(matching_pars).at(0)
          print('Matching aux %s to NP %s' % (aux.GetName(), mpar.GetName()))
          if len(signal_pdf.getDependents(matching_pars)) > 0 or len(nSignal.getDependents(matching_pars)) > 0 :
            alphas.append(mpar)
            aux_alphas.append(aux)
          else :
            betas.append(mpar)
            aux_betas.append(aux)
except Exception as inst :
  print(inst)
  ValueError('Could not identify nuisance parameters')
  
bin_data = []
for b in range(0, nbins) :
  bin_datum = collections.OrderedDict()
  bin_datum['lo_edge'] = bins[b]
  bin_datum['hi_edge'] = bins[b+1]
  bin_data.append(bin_datum)

jdict = collections.OrderedDict()
jdict['model_name' ] = options.output_name
jdict['obs_name' ] = obs.GetTitle().replace('#','\\')
jdict['obs_unit' ] = obs.getUnit()
jdict['bins'] = bin_data
if not options.data_only :
  np_list = ROOT.RooArgList(nuis_pars)
  for p in range(0, len(np_list)) :
    par = np_list.at(p)
    if not par in alphas and not par in betas :
      gammas.append(par)

  if options.validation_data :
    validation_points = np.linspace(-3, 3, 13)
    valid_data = collections.OrderedDict()
    valid_data['points'] = np.array(validation_points)
    for par in alphas + betas + gammas :
      valid_data[par.GetName()] = np.ndarray((nbins, len(validation_points)))

  impacts_s = np.ndarray((nbins, len(np_list)))
  impacts_b = np.ndarray((nbins, len(np_list)))
  nom_sig = np.zeros(nbins)
  nom_bkg = np.zeros(nbins)
  
  for i in range(0, nbins) :
    xmin = bins[i]
    xmax = bins[i + 1]
    obs.setRange('bin_%d' % i, xmin, xmax)
    totint =   main_pdf.createIntegral(ROOT.RooArgSet(obs), ROOT.RooArgSet(obs), 'bin_%d' % i)
    sigint = signal_pdf.createIntegral(ROOT.RooArgSet(obs), ROOT.RooArgSet(obs), 'bin_%d' % i)
    ntot = main_pdf.expectedEvents(ROOT.RooArgSet(obs))
    sig0 = nSignal.getVal()*sigint.getVal()
    bkg0 = ntot*totint.getVal() - sig0
    print('-- Nominal sig = %g' % sig0)
    print('-- Nominal bkg = %g' % bkg0)
    nom_sig[i] = sig0
    nom_bkg[i] = bkg0
    for p in range(0, len(np_list)) :
      par = np_list.at(p)
      par0 = par.getVal()
      if par in gammas :
        error = par.getError() if par.getError() > 0 else (par.getMax() - par0)/10
      else :
        error = 1
      if i == 0 : print('Parameter %s : using deviation %g from nominal value %g for impact computation (x%g)' % (par.GetName(), error, par0, options.epsilon))
      delta = error*options.epsilon
      par.setVal(par0 + delta)
      ntot = main_pdf.expectedEvents(ROOT.RooArgSet(obs))
      sig_pos = nSignal.getVal()*sigint.getVal()
      bkg_pos = ntot*totint.getVal() - sig_pos
      impact_s_pos = ((sig_pos/sig0)**(1/options.epsilon) - 1) if sig0 != 0 else 0
      impact_b_pos = ((bkg_pos/bkg0)**(1/options.epsilon) - 1) if bkg0 != 0 else 0
      par.setVal(par0 - delta)
      ntot = main_pdf.expectedEvents(ROOT.RooArgSet(obs))
      sig_neg = nSignal.getVal()*sigint.getVal()
      bkg_neg = ntot*totint.getVal() - sig_neg
      impact_s_neg = ((sig0/sig_neg)**(1/options.epsilon) - 1) if sig_neg != 0 else 0
      impact_b_neg = ((bkg0/bkg_neg)**(1/options.epsilon) - 1) if bkg_neg != 0 else 0
      if par in alphas :
        #impacts_s[i,p] = impact_s_pos if abs(impact_s_pos) < abs(impact_s_neg) else impact_s_neg
        impacts_s[i,p] = math.sqrt((1 + impact_s_pos)*(1 + impact_s_neg)) - 1
        print('-- parameter %-10s : +1 sigma sig impact = %g' % (par.GetName(), impact_s_pos))
        print('-- parameter %-10s : -1 sigma sig impact = %g' % (''           , impact_s_neg))
        print('-- parameter %-10s : selected sig impact = %g' % (''           , impacts_s[i,p]))
      else :
        #impacts_b[i,p] = impact_b_pos if abs(impact_b_pos) < abs(impact_b_neg) else impact_b_neg
        impacts_b[i,p] = math.sqrt((1 + impact_b_pos)*(1 + impact_b_neg)) - 1
        print('-- parameter %-10s : +1 sigma bkg impact = %g' % (par.GetName(), impact_b_pos))
        print('-- parameter %-10s : -1 sigma bkg impact = %g' % (''           , impact_b_neg))
        print('-- parameter %-10s : selected bkg impact = %g' % (''           , impacts_b[i,p]))
      par.setVal(par0)
      if options.validation_data :
        par_data = valid_data[par.GetName()]
        for k, val in enumerate(validation_points) :
          par.setVal(par0 + val*delta)
          ntot = main_pdf.expectedEvents(ROOT.RooArgSet(obs))
          sig_var = nSignal.getVal()*sigint.getVal()
          bkg_var = ntot*totint.getVal() - sig_var
          print('== validation %-10s: %+6g sigma sig yield = %g' % ('', val, sig_var))
          print('== validation %-10s: %+6g sigma bkg yield = %g' % ('', val, bkg_var))
          par_data[i,k] = (sig_var + bkg_var)/(sig0 + bkg0) if sig0 + bkg0 != 0 else 0
          print('== validation %-10s: %+6g variation = %g' % ('', val, par_data[i,k]))
        par.setVal(par0)

  impacts = {}
  for p in range(0, len(np_list)) :
    par = np_list.at(p)
    if par in alphas :
      impacts[par] = impacts_s[:,p]
    else :
      impacts[par] = impacts_b[:,p]
  
  jdict['signal'] = nom_sig.tolist()
  jdict['background'] = nom_bkg.tolist()
  
  alpha_specs = [ ]
  for i, alpha in enumerate(alphas) :
    od = collections.OrderedDict()
    od['name'] = alpha.GetName()
    od['impact'] = impacts[alpha].tolist()
    alpha_specs.append(od)
  jdict['alphas'] = alpha_specs

  beta_specs = []
  for i, beta in enumerate(betas) :
    od = collections.OrderedDict()
    od['name'] = beta.GetName()
    od['impact'] = impacts[beta].tolist()
    beta_specs.append(od)
  jdict['betas'] = beta_specs

  gamma_specs= []
  for i, gamma in enumerate(gammas) :
    od = collections.OrderedDict()
    od['name'] = gamma.GetName()
    od['impact'] = impacts[gamma].tolist()
    gamma_specs.append(od)
  jdict['gammas'] = gamma_specs

if data :
  bin_array = array.array('d', bins)
  hist = ROOT.TH1D('h', 'histogram', nbins, bin_array)
  data.fillHistogram(hist, ROOT.RooArgList(obs))
  bin_counts = [ hist.GetBinContent(i+1) for i in range(0, nbins) ]
  data_dict = collections.OrderedDict()
  data_dict['bin_counts'] = bin_counts
  data_dict['aux_alphas'] = [ aux_alpha.getVal() for aux_alpha in aux_alphas ]
  data_dict['aux_betas' ] = [ aux_beta.getVal() for aux_beta in aux_betas ]
  jdict['data'] = data_dict

with open(options.output_file, 'w') as fd:
  json.dump(jdict, fd, ensure_ascii=True, indent=3)

if options.validation_data :
  valid_lists = collections.OrderedDict()
  valid_lists['points'] = valid_data['points'].tolist()
  for par in alphas + betas + gammas :
    valid_lists[par.GetName()] = valid_data[par.GetName()].tolist()
  with open(options.validation_data, 'w') as fd:
    json.dump(valid_lists, fd, ensure_ascii=True, indent=3)
