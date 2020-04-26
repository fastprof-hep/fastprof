#! /usr/bin/env python

__doc__ = "Convert a ROOT workspace into fastprof JSON format"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser
import ROOT
import numpy as np
import json
import collections

####################################################################################################################################
###

def convert_ws() :
  """convert """
  
  parser = ArgumentParser("convert_ws.py")
  parser.description = __doc__
  parser.add_argument("-f", "--ws-file",           default='',        help="Name of file containing the workspace", type=str)
  parser.add_argument("-w", "--ws-name",           default='modelWS', help="Name workspace object inside the specified file", type=str)
  parser.add_argument("-m", "--model-config-name", default='mconfig', help="Name of model config within the specified workspace", type=str)
  parser.add_argument("-s", "--signal-pdf",        default='Signal',  help="Name of signal component PDF", type=str)
  parser.add_argument("-n", "--signal-yield",      default='nSignal', help="Name of signal yield variable", type=str)
  parser.add_argument("-b", "--binning",           default='',        help="Name of output file", type=str)
  parser.add_argument("-p", "--nps",               default='',        help="List of constrained nuisance parameters", type=str)
  parser.add_argument("-d", "--data-name",         default='obsData', help="Name of dataset object within the input workspace", type=str)
  parser.add_argument("-a", "--asimov",        action="store_true",   help="Use an Asimov dataset as the data")
  parser.add_argument("-o", "--output-file",       default='',        help="Name of output file", type=str)
  parser.add_argument("-v", "--verbosity",         default=0,         help="Verbosity level", type=int)
  
  options = parser.parse_args()
  if not options : 
    parser.print_help()
    return

  try:
    binspec = options.binning.split(':')
    if len(binspec) == 4 and binspec[3] == 'log' : 
      bins = np.logspace(float(binspec[0]), float(binspec[1]), int(binspec[2]))
    else :
      bins = np.linspace(float(binspec[0]), float(binspec[1]), int(binspec[2]))
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
  
  eps = 0.1
  
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
  
  aux_alphas = []
  aux_betas  = []
  alphas = []
  betas = []
  gammas = []
  
  aux_obs = ROOT.RooArgList(mconfig.GetGlobalObservables())
  nuis_pars = mconfig.GetNuisanceParameters()
  pdfs = main_pdf.pdfList()
  print len(nuis_pars)
  try:
    for o in range(0, len(aux_obs)) :
      aux = aux_obs.at(o)
      print aux
      for p in range(0, len(pdfs)) :
        pdf = pdfs.at(p)
        print ' - ', pdf
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
  
  np_list = ROOT.RooArgList(nuis_pars)
  for p in range(0, len(np_list)) :
    par = np_list.at(p)
    if not par in alphas and not par in betas :
      gammas.append(par)

  print alphas
  print betas
  print gammas

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
    nom_sig[i] = sig0
    nom_bkg[i] = bkg0
    for p in range(0, len(np_list)) :
      par = np_list.at(p)
      par0 = par.getVal()
      delta = (par.getMax() - par.getVal())*eps
      par.setVal(par0 + delta)
      sig1 = nSignal.getVal()*sigint.getVal()
      bkg1 = ntot*totint.getVal() - sig0
      impacts_s[i,p] = (sig1/sig0 - 1)/delta if sig0 != 0 else 0
      impacts_b[i,p] = (bkg1/bkg0 - 1)/delta if bkg0 != 0 else 0
  
  impacts = {}
  
  for p in range(0, len(np_list)) :
    par = np_list.at(p)
    if p in alphas :
      impacts[par] = impacts_s[:,p]
    else :
      impacts[par] = impacts_b[:,p]
  
  jdict = collections.OrderedDict()
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
  print jdict
  with open(options.output_file, 'w') as fd:
    json.dump(jdict, fd, ensure_ascii=True, indent=3)
  
  return 0

convert_ws()
