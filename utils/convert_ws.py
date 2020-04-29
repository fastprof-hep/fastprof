#! /usr/bin/env python

__doc__ = "Convert a ROOT workspace into fastprof JSON format"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser
import ROOT
import numpy as np
import json
import collections
import array

####################################################################################################################################
###

def convert_ws() :
  """convert """
  
  parser = ArgumentParser("convert_ws.py")
  parser.description = __doc__
  parser.add_argument("-f", "--ws-file",                 default='',        help="Name of file containing the workspace", type=str)
  parser.add_argument("-w", "--ws-name",                 default='modelWS', help="Name workspace object inside the specified file", type=str)
  parser.add_argument("-m", "--model-config-name",       default='mconfig', help="Name of model config within the specified workspace", type=str)
  parser.add_argument("-s", "--signal-pdf",              default='Signal',  help="Name of signal component PDF", type=str)
  parser.add_argument("-n", "--signal-yield",            default='nSignal', help="Name of signal yield variable", type=str)
  parser.add_argument("-b", "--binning",                 default='',        help="Name of output file", type=str)
  parser.add_argument("-p", "--nps",                     default='',        help="List of constrained nuisance parameters", type=str)
  parser.add_argument("-e", "--epsilon",                 default=1,         help="Scale factor applied to uncertainties for impact computations", type=str)
  parser.add_argument("-=", "--setval",                  default='',        help="Variables to set, in the form var1=val1,var2=val2,...", type=str)
  parser.add_argument("-r", "--refit",               action="store_true",   help="Perform a fit to the dataset (specified by --data-name) before conversion")
  parser.add_argument("-u", "--refit-uncertainties", action="store_true",   help="Update uncertainties (but not central values) from a fit to the specified dataset")
  parser.add_argument("-a", "--asimov",              action="store_true",   help="Perform an Asimov fit before conversion")
  parser.add_argument("-x", "--data-only",           action="store_true",   help="Only dump the specified dataset, not the model")
  parser.add_argument("-d", "--data-name",               default='',        help="Name of dataset object within the input workspace", type=str)
  parser.add_argument("-o", "--output-file",             default='',        help="Name of output file", type=str)
  parser.add_argument("-v", "--verbosity",               default=0,         help="Verbosity level", type=int)
  
  options = parser.parse_args()
  if not options : 
    parser.print_help()
    return

  try:
    binspec = options.binning.split(':')
    if len(binspec) == 4 and binspec[3] == 'log' : 
      bins = np.logspace(float(binspec[0]), float(binspec[1]), int(binspec[2]) + 1)
    else :
      bins = np.linspace(float(binspec[0]), float(binspec[1]), int(binspec[2]) + 1)
  except:
    raise ValueError('Invalid bin specification %s : the format should be xmin:xmax:bins[:log]' % options.binning)
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
  except:
    ValueError('Could not locate observable')

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
  nuis_pars = mconfig.GetNuisanceParameters()
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
  except :
    ValueError('Could not identify nuisance parameters')
    
  jdict = collections.OrderedDict()

  if not options.data_only :
    np_list = ROOT.RooArgList(nuis_pars)
    for p in range(0, len(np_list)) :
      par = np_list.at(p)
      if not par in alphas and not par in betas :
        gammas.append(par)
  
    impacts_s = np.ndarray((nbins, len(np_list)))
    impacts_b = np.ndarray((nbins, len(np_list)))
    nom_sig = np.zeros(nbins)
    nom_bkg = np.zeros(nbins)
    
    for i in range(0, nbins) :
      xmin = bins[i]
      xmax = bins[i + 1]
      obs.setRange('bin_%d' % (i+1), xmin, xmax)
      totint = main_pdf.createIntegral(ROOT.RooArgSet(obs), ROOT.RooArgSet(obs), 'bin_%d' % (i+1))
      sigint = signal_pdf.createIntegral(ROOT.RooArgSet(obs), ROOT.RooArgSet(obs), 'bin_%d' % (i+1))
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
        error = par.getError() if par.getError() > 0 else (par.getMax() - par0)/10
        if i == 0 : print('Parameter %s : using deviation %g from nominal value %g for impact computation (x%g)' % (par.GetName(), error, par0, options.epsilon))
        delta = error*options.epsilon
        par.setVal(par0 + delta)
        ntot = main_pdf.expectedEvents(ROOT.RooArgSet(obs))
        sig_pos = nSignal.getVal()*sigint.getVal()
        bkg_pos = ntot*totint.getVal() - sig_pos
        impact_s_pos = (sig_pos/sig0 - 1)/options.epsilon if sig0 != 0 else 0
        impact_b_pos = (bkg_pos/bkg0 - 1)/options.epsilon if bkg0 != 0 else 0
        par.setVal(par0 - delta)
        ntot = main_pdf.expectedEvents(ROOT.RooArgSet(obs))
        sig_neg = nSignal.getVal()*sigint.getVal()
        bkg_neg = ntot*totint.getVal() - sig_neg
        impact_s_neg = (sig0/sig_neg - 1)/options.epsilon if sig_neg != 0 else 0
        impact_b_neg = (bkg0/bkg_neg - 1)/options.epsilon if bkg_neg != 0 else 0
        # take the minimal value in case the negative variation (on bin yields) would go below n=0
        if par in alphas :
          impacts_s[i,p] = impact_s_pos if abs(impact_s_pos) < abs(impact_s_neg) else impact_s_neg
          print('-- parameter %-10s : +1 sigma sig impact = %g' % (par.GetName(), impact_s_pos))
          print('-- parameter %-10s : -1 sigma sig impact = %g' % (''           , impact_s_neg))
          print('-- parameter %-10s : selected sig impact = %g' % (''           , impacts_s[i,p]))
        else :
          impacts_b[i,p] = impact_b_pos if abs(impact_b_pos) < abs(impact_b_neg) else impact_b_neg
          print('-- parameter %-10s : +1 sigma bkg impact = %g' % (par.GetName(), impact_b_pos))
          print('-- parameter %-10s : -1 sigma bkg impact = %g' % (''           , impact_b_neg))
          print('-- parameter %-10s : selected bkg impact = %g' % (''           , impacts_b[i,p]))
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
  
  return 0

convert_ws()
