#! /usr/bin/env python

__doc__ = """
*Compute fit results in a ROOT workspace*

The script takes as input a ROOT workspace, and computes the information
needed to evaluate the q_mu and q_mu~ test statistics (see
[arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>]_) at a number of POI
hypothesis points.

These values and hypotheses can then be used to compute sampling distributions
that allow the estimation of p-values corresponding to the test statistics,
without the use of the asymptotic approximation of 
[arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>]_. An example of this
procedure can be found in the utils/compute_limits.py script.

* The input dataset is specified using the `--ws-file` and `--ws-name`
  options. The input dataset can be either observed data (`--data-file` and
  `--data-name`) or an Asimov dataset (`--asimov`). For the latter, the POI
  values are given as argument, and the NP values are taken from a fit to 
  an observed dataset, if provided (and kept at their nominal values otherwise).

* Prior to computations, the model can be adjusted using the `--setval`,
  `--setconst` and `--setrange` options, which set respectively the value,
  constness and range of model parameters.

* The hypotheses can be set using the `--hypos` option, providing as argument
  a colon-separated list of definitions. Each definition consists in a comma-
  separated list of POI assignments. 
  For the case of a single POI, a single integer can provided as argument. In
  this case, a set of hypotheses appropriate for the setting of an upper limit
  on the POI are defined automatically. This set is defined as follows:
  
  * 2/3 of the specified points as used to define a fine grid in the range
    from :math:`-3\sigma` to :math:`+3\sigma` around the expected limit,
    where both the expected limit and the value of :math:`\sigma` are
    estimated from the uncertainty on the POI
    
  * 1/3 of the specified points define a looser grid in the range from
    :math:`+3\sigma` and :math:`+8\sigma`, and the corresponding
    negative range.
  
  A good balance is the default `hypos=17`, which defines 6 hypothesis
  between 0 and :math:`+3\sigma` (0,0.5,1,1.5,2,2.5), another 3 above
  :math:`+3\sigma` (3, 5.5, 8), and the corresponding negative values.

* At each hypothesis, the data is fit twice : once with the POIs set to their
  hypothesis values, and once free to vary in the fit. The same procedure is
  performed for an Asimov dataset, with a POI value set to zero. This provides
  all the information needed to compute q_mu and q_mu~.

The output is a single JSON file containing the result of the fits at each hypothesis,
including the best-fit values of all parameters and the NLL values at minimum.
"""

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import math
import json
from scipy.stats import norm
import ROOT

from fastprof_import.tools import process_setvals, process_setranges, process_setconsts, fit, make_asimov, make_binned

def make_parser() :
  parser = ArgumentParser("fit_ws.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-f", "--ws-file"          , type=str  , required=True    , help="Name of file containing the workspace")
  parser.add_argument("-w", "--ws-name"          , type=str  , default='modelWS', help="Name workspace object inside the specified file")
  parser.add_argument("-m", "--model-config-name", type=str  , default='mconfig', help="Name of model config within the specified workspace")
  parser.add_argument("-d", "--data-name"        , type=str  , default=''       , help="Name of dataset object within the input workspace")
  parser.add_argument("-a", "--asimov"           , action="store_true"          , help="Fit an Asimov dataset")
  parser.add_argument("-y", "--hypos"            , type=str  , default='17'     , help="List of POI hypothesis values (poi1=val1,poi2=val2:...)")
  parser.add_argument(      "--hypos-file"       , type=str  , default=None     , help="File containing POI hypothesis values")
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
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args(argv)
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
  pois = mconfig.GetParametersOfInterest()

  if options.poi_min != None :
    for poi in pois : poi.setMin(options.poi_min)
  if options.poi_max != None :
    for poi in pois : poi.setMax(options.poi_max)

  nps = mconfig.GetNuisanceParameters().selectByAttrib('Constant', False)
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

  ws.saveSnapshot('init_nps', nps)
  ws.saveSnapshot('init_poi', pois)


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

  if options.binned :
    unbinned_data = data
    rebinnings = {}
    if options.input_bins > 0 :
      obs_list = ROOT.RooArgList(mconfig.GetObservables())
      for i in range(0, obs_list.getSize()) : rebinnings[obs_list.at(i)] = options.input_bins
    data = make_binned(data, rebinnings)
  else :
    unbinned_data = data

  poi.setVal(0)
  poi.setConstant(True)
  result_bkg_only = fit(main_pdf, data, robust=True)
  print('=== Generating an Asimov dataset with POI = 0 and NP values below:')
  nps.Print('V')
  asimov0 = ROOT.RooStats.AsymptoticCalculator.MakeAsimovData(mconfig, ROOT.RooArgSet(), ROOT.RooArgSet())


  # 3. Define hypothesis values
  # ===========================

  hypos = None
  if options.hypos :
    try :
      nhypos = int(options.hypos)
      n1 = (nhypos + 1) // 6
      pos = np.concatenate((np.linspace(0, 3, 2*n1 + 1)[1:-1], np.linspace(3, 8, n1))) # twice as many points in ]0,3[ as in [3,8]
      hypo_zs = np.concatenate((np.flip(-pos), np.zeros(1), pos))
      hypos = None
    except:
      try :
        hypo_specs = [ process_setval(spec, parse_only=True) for spec in options.hypos.split('/') ]
        hypos = [ { var : val for var, val, save in hypo_spec } for hypo_spec in hypo_specs ]
      except Exception as inst :
        print(inst)
        raise ValueError("Could not parse list of hypothesis values '%s' : expected /-separated list of variable assignments" % options.hypos)
  elif option.hypos_file :
    fits_file = open(options.hypos)
    fit_results_dict = json.load(fits_file)
    hypos = [ results_dict['hypo'] for results_dict in fit_results_dict['results'] ]

  if hypos == None : # we need to auto-define them based on the POI uncertainty
    if len(pois) > 1 : raise('Cannot auto-set hypotheses for more than 1 POI')
    ws.loadSnapshot('init_nps')
    ws.loadSnapshot('init_pois')
    nll0  = main_pdf.createNLL(asimov0)
    nSignal = ws.var(options.signal_yield)
    if nSignal == None :
      nSignal = ws.function(options.signal_yield)
      if nSignal == None :
        raise ValueError('Could not locate signal yield variable %s')
    def hypo_guess(i, unc) :
      pv = 0.05
      return (3*math.exp(0.5/3*i))*math.exp(-unc**2/3) + (1 -math.exp(-unc**2/3))*(i + norm.isf(pv*norm.cdf(i)))*np.sqrt(9 + unc**2)
    poi.setConstant(False)
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
    hypo_vals = hypos_nS/poi2sig
    hypos = [ { poi : hypo_val } for hypo_val in hypo_vals ]
    print('=== Auto-defined the following hypotheses :')
    print('  ' + '\n  '.join([ '%5g : Nsig = %10g, POI = %10g' % (h_z, h_n, h_p) for h_z, h_n, h_p in zip(hypo_zs, hypos_nS, hypo_vals) ] ))

    if options.poi_min == None :
      # Set min to the -5sigma hypothesis, should be enough...
      poi.setMin(-hypo_guess(5, sigma_A*poi2sig)/poi2sig)
      print('=== Auto-set POI min to %g' % poi.getMin())

    if options.poi_max == None :
      # Set max to the 20sigma hypothesis, should be enough...
      poi.setMax(hypo_guess(20, sigma_A*poi2sig)/poi2sig)
      print('=== Auto-set POI max to %g' % poi.getMax())

  jdict = {}

  def fitpar_dict(par, result) :
    dic = {}
    dic['value'] = result.floatParsFinal().find(par.GetName()).getVal()
    dic['error'] = result.floatParsFinal().find(par.GetName()).getError()
    dic['min_value'] = par.getMin()
    dic['max_value'] = par.getMax()
    dic['initial_value'] = result.floatParsInit().find(par.GetName()).getVal()
    return dic

  def plr_spec(hypo, dataset) :
    spec = {}
    # Hypo
    hypo_spec = { poi.GetName() : val for poi, val in hypo.items() }
    spec['hypo'] = hypo
    print('=== Fitting dataset %s to hypothesis %s' % (dataset.GetName(), str(hypo_spec)))
    nll  = main_pdf.createNLL(dataset)
    ws.loadSnapshot('init_nps')
    ws.loadSnapshot('init_pois')
    # Hypo
    spec['hypo'] = hypo_spec
    # Initial valueshypos
    init_vals = {}
    for p in pois : init_vals[p.GetName()] = p.getVal()
    for p in nps  : init_vals[p.GetName()] = p.getVal()
    # Fixed-POI fit
    hypo_fit_spec = {}
    for poi in hypo :
      poi.setVal(hypo[poi])
      poi.setConstant(True)
    result_hypo = fit(main_pdf, dataset, robust=True)
    hypo_fit_spec['nll'] = nll.getVal()
    hypo_fit_spec['fit_pars'] = {}
    for p in nps       : hypo_fit_spec['fit_pars'][p.GetName()] = fitpar_dict(p, result_hypo)
    for p in extra_nps : hypo_fit_spec['fit_pars'][p.GetName()] = { 'value' : ws.var(p.GetName()).getVal() }
    spec['hypo_fit'] = hypo_fit_spec
    # Free-mu fit
    free_fit_spec = {}
    for poi in hypo : poi.setConstant(False)
    result_free = fit(main_pdf, dataset, robust=True)
    result_free.floatParsFinal().Print("V")
    free_fit_spec['nll'] = nll.getVal()
    free_fit_spec['fit_pars'] = {}
    for p in pois      : free_fit_spec['fit_pars'][p.GetName()] = fitpar_dict(p, result_free)
    for p in nps       : free_fit_spec['fit_pars'][p.GetName()] = fitpar_dict(p, result_free)
    for p in extra_nps : free_fit_spec['fit_pars'][p.GetName()] = { 'value' : ws.var(p.GetName()).getVal() }
    spec['free_fit'] = free_fit_spec
    return spec

  data_specs = []
  asimov_specs = []
  for i, hypo in enumerate(hypos) :
    sys.stderr.write('\rProcessing hypothesis %d of %d' % (i+1, len(hypos)))
    sys.stderr.flush()
    data_specs.append(plr_spec(hypo, data))
    asimov_specs.append(plr_spec(hypo, asimov0))
  sys.stderr.write('\n')

  jdict['data'] = data_specs
  jdict['asimov'] = asimov_specs
  with open(options.output_file, 'w') as fd:
    json.dump(jdict, fd, ensure_ascii=True, indent=3)

if __name__ == '__main__':
  run(sys.argv[1:])
