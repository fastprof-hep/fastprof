#! /usr/bin/env python

__doc__ = "Convert a ROOT workspace into fastprof JSON format"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import json
import array
import math
import ROOT

####################################################################################################################################
###

parser = ArgumentParser("convert_ws.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-f", "--ws-file"          , type=str  , required=True    , help="Name of file containing the workspace")
parser.add_argument("-w", "--ws-name"          , type=str  , default='modelWS', help="Name workspace object inside the specified file")
parser.add_argument("-m", "--model-config-name", type=str  , default='mconfig', help="Name of model config within the specified workspace")
parser.add_argument("-c", "--channel-names"    , type=str  , default=None     , help="Names of the model channels, in the form c1,c2,..., in the same order as the RooSimPdf components")
parser.add_argument("-s", "--sample-names"     , type=str  , default=None     , help="Names of the model samples, in the form s1,s2,..., in the same order as the RooAddPdf components")
parser.add_argument(      "--default-sample"   , type=str  , default=None     , help="Names of the model samples, in the form s1,s2,..., in the same order as the RooAddPdf components")
parser.add_argument("-n", "--normpar-names"    , type=str  , default=None     , help="Names of the norm pars for each sample, in the form p1,p2,..., in the same order as the samples")
parser.add_argument("-b", "--binning"          , type=str  , required=True    , help="Binning used, in the form xmin:xmax:nbins[:log]")
parser.add_argument("-e", "--epsilon"          , type=float, default=1        , help="Scale factor applied to uncertainties for impact computations")
parser.add_argument("-=", "--setval"           , type=str  , default=''       , help="List of variable value changes, in the form var1=val1,var2=val2,...")
parser.add_argument("-k", "--setconst"         , type=str  , default=''       , help="List of variables to set constant")
parser.add_argument("-r", "--setrange"         , type=str  , default=''       , help="List of variable range changes, in the form var1:[min1]:[max1],var2:[min2]:[max2],...")
parser.add_argument("-d", "--data-name"        , type=str  , default=''       , help="Name of dataset object within the input workspace")
parser.add_argument("-a", "--asimov"           , type=str  , default=None     , help="Perform an Asimov fit before conversion")
parser.add_argument("-x", "--data-only"        , action="store_true"          , help="Only dump the specified dataset, not the model")
parser.add_argument(      "--refit"            , type=str  , default=None     , help="Fit the model to the specified dataset before conversion")
parser.add_argument(      "--binned"           , action="store_true"          , help="Use binned data")
parser.add_argument(      "--input_bins"       , type=int  , default=0        , help="Number of bins to use when binning the input dataset")
parser.add_argument(      "--regularize"       , type=float, default=0        , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
parser.add_argument("-o", "--output-file"      , type=str  , required=True    , help="Name of output file")
parser.add_argument(      "--output-name"      , type=str  , default=''       , help="Name of the output model")
parser.add_argument(      "--validation-data"  , type=str  , default=''       , help="Name of output file for validation data")
parser.add_argument("-v", "--verbosity"        , type=int  , default=0        , help="Verbosity level")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

# 1 - Parse bin specifications, retrieve workspace contents
# ---------------------------------------------------------
try:
  binspec = options.binning.split(':')
  if len(binspec) == 4 and binspec[3] == 'log' :
    bins = np.logspace(1, math.log(float(binspec[1]))/math.log(float(binspec[0])), int(binspec[2]) + 1, True, float(binspec[0]))
    print('bins = ', bins)
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

# 2 - Update parameter values and constness as specified in options
# -----------------------------------------------------------------

def process_setval(opt) :
  try:
    sets = [ v.replace(' ', '').split('=') for v in opt.split(',') ]
    output = []
    for (var, val) in sets :
      if not ws.var(var) :
        raise ValueError("ERROR: Cannot find variable '%s' in workspace" % var)
      save_val = ws.var(var).getVal()
      ws.var(var).setVal(float(val))
      print("INFO : setting %s=%g for Asimov generation" % (var, float(val)))
      output.append((ws.var(var), float(val), save_val))
  except Exception as inst :
    print(inst)
    raise ValueError("ERROR : invalid variable assignment string '%s'." % opt)
  return output

if options.setval != '' :
  process_setval(options.setval)

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

if options.setrange != '' :
  try:
    sets = [ v.replace(' ', '').split(':') for v in options.setrange.split(',') ]
    for (var, minval, maxval) in sets :
      if not ws.var(var) :
        raise ValueError("Cannot find variable '%s' in workspace" % var)
      if minval == '' : 
        ws.var(var).setMax(float(maxval))
        print("INFO : setting upper bound of %s to %g" % (var, float(maxval)))
      elif maxval == '' :
        ws.var(var).setMin(float(minval))
        print("INFO : setting lower bound of %s to %g" % (var, float(minval)))
      else :
        ws.var(var).setRange(float(minval), float(maxval))
        print("INFO : setting range of %s to [%g, %g]" % (var, float(minval), float(maxval)))
  except Exception as inst :
    print(inst)
    raise ValueError("ERROR : invalid variable range specification '%s'." % options.setrange)

# 3 - Define the primary dataset
# ------------------------------
data = None
if options.data_name != '' :
  data = ws.data(options.data_name)
  if data == None :
    ds = [ d.GetName() for d in ws.allData() ]
    raise KeyError('Dataset %s not found in workspace. Available datasets are: %s' % (options.data_name, ', '.join(ds)))
elif options.asimov != None :
  saves = process_setval(options.asimov)
  data = ROOT.RooStats.AsymptoticCalculator.MakeAsimovData(mconfig, ROOT.RooArgSet(), ROOT.RooArgSet())
  print("INFO : generating Asimov for parameter values :")
  for (var, val, save_val) in saves :
    print("INFO :   %s=%g" % (var.GetName(), val))
    var.setVal(save_val)
else:
  raise ValueError('ERROR: no dataset was specified either using --data-name or --asimov')

# 4 - Identify the model parameters and main PDF
# ----------------------------------------------

main_pdf = mconfig.GetPdf()
pois = mconfig.GetParametersOfInterest()

cons_aux = {}
cons_nps = []
free_nps = []
aux_obs = ROOT.RooArgList(mconfig.GetGlobalObservables())
nuis_par_set = mconfig.GetNuisanceParameters().selectByAttrib('Constant', False)

pdfs = main_pdf.pdfList()
try:
  for o in range(0, len(aux_obs)) :
    aux = aux_obs.at(o)
    for p in range(0, len(pdfs)) :
      pdf = pdfs.at(p)
      if len(pdf.getDependents(ROOT.RooArgSet(aux))) > 0 :
        matching_pars = pdf.getDependents(nuis_par_set)
        if len(matching_pars) == 1 :
          mpar = ROOT.RooArgList(matching_pars).at(0)
          print('INFO: Matching aux %s to NP %s' % (aux.GetName(), mpar.GetName()))
          cons_nps.append(mpar)
          cons_aux[mpar.GetName()] = aux
except Exception as inst :
  print(inst)
  ValueError('Could not identify nuisance parameters')

nuis_pars = []
class NuisancePar : pass

np_list = ROOT.RooArgList(nuis_par_set)
for p in range(0, len(np_list)) :
  par = np_list.at(p)
  nuis_par = NuisancePar()
  nuis_par.name = par.GetName()
  nuis_par.obj = par
  nuis_par.is_free = not par in cons_nps
  if nuis_par.is_free : free_nps.append(par)
  nuis_pars.append(nuis_par)
  
# 5. Identify the model channels
# ---------------------------------------------------------

for c in main_pdf.getComponents() :
  if isinstance(c, ROOT.RooSimultaneous) :
    print('Building multi-channel model from RooSimultaneous not supported yet -- coming soon!')
    sys.exit(0)
  if isinstance(c, ROOT.RooAddPdf) :
    channel_pdf = c
    break

# TODO: iterate this next part to support multi-channel models

channels = []
class Channel() : pass

if options.channel_names == None :
  channel_names = []
else :
  try:
    channel_names = options.channel_names.split(',')
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid channel name specification %s : should be of the form name1,name2,...' % options.channel_names)
    
# 6. Identify the samples for this channel
# ---------------------------------------------------------

channel = Channel()
channel.type = 'binned_range'
channel.name = channel_names[0] if 0 < len(channel_names) else channel_pdf.GetName()
channel.pdf = channel_pdf

channel_obs = mconfig.GetObservables().selectCommon(channel_pdf.getVariables())
if channel_obs.getSize() == 0 :
  raise ValueError('Cannot identify observables for channel %s.')
if channel_obs.getSize() > 1 :
  raise ValueError('Channel %s has %d observables -- multiple observables not supported yet.')
channel.obs = ROOT.RooArgList(channel_obs).at(0)

if options.sample_names == None :
  sample_names = []
else :
  try:
    sample_names = options.sample_names.split(',')
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid sample name specification %s : should be of the form name1,name2,...' % options.sample_names)

if options.normpar_names == None :
  normpar_names = []
else :
  try:
    normpar_names = options.normpar_names.split(',')
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid sample name %s : should be of the form par1,par2,...' % options.normpar_names)
normpars = []
for normpar_name in normpar_names :
  if normpar_name == '' :
    normpars.append(None)
    continue
  normpar = ws.var(normpar_name)
  if normpar != None :
    normpars.append(normpar)
  else :
    raise ValueError('Normalization parameter %s not found in workspace' % normpar_name)

channel.samples = []
default_sample = None # the sample to which unassigned variations will be associated (e.g. spurious signal, not scaled by any sample normpars)
class Sample() : pass

for i in range(0, channel_pdf.pdfList().getSize()) :
  sample = Sample()
  sample.pdf = channel_pdf.pdfList().at(i)
  sample.name = sample_names[i] if i < len(sample_names) else sample.pdf.GetName()
  sample.normvar = channel_pdf.coefList().at(i)
  sample.normpar = normpars[i] if i < len(normpars) else None
  if sample.normpar == None :
    if isinstance(sample.normvar, ROOT.RooRealVar) :
      sample.normpar = sample.normvar
    else :
      poi_candidates = sample.normvar.getVariables().selectCommon(pois)
      if poi_candidates.getSize() == 1 : sample.normpar = ROOT.RooArgList(poi_candidates).at(0)
  if sample.normpar == None :
    raise ValueError('Cannot identify normalization variable for sample %s, please specify manually.' % sample.name)
  channel.samples.append(sample)
  if sample.name == options.default_sample : default_sample = sample

if default_sample == None : default_sample = channel.samples[-1] # if unspecified, take the last one


# 7 - Fill the model information
# ------------------------------

def fit(dataset, robust = False, n_max = 3, ref_nll = 0) :
   if options.binned :
     if options.input_bins > 0 : obs.setBins(options.input_bins)
     fit_data = dataset.binnedClone()
   else :
     fit_data = dataset
   result = main_pdf.fitTo(fit_data, ROOT.RooFit.Offset(), ROOT.RooFit.SumW2Error(False), ROOT.RooFit.Minimizer('Minuit2', 'migrad'), ROOT.RooFit.Hesse(True), ROOT.RooFit.Save())
   if robust and (result.status() != 0 or abs(result.minNll() - ref_nll) > 1) :
     return fit(dataset, robust, n_max - 1, result.minNll())
   else :
     return result

if options.refit != None :
  saves = process_setval(options.refit)
  data = ROOT.RooStats.AsymptoticCalculator.MakeAsimovData(mconfig, ROOT.RooArgSet(), ROOT.RooArgSet())
  print('=== Refitting PDF to specified dataset with under the hypothesis :')
  for (var, val, save_val) in saves :
    print("INFO :   %s=%g" % (var.GetName(), val))
    var.setConstant()
  fit(data, robust=True)

# If we specified both, then it means an Asimov with NP values profiled on the observed
if options.data_name != '' and options.asimov != None :
  saves = process_setval(options.asimov)
  print('=== Generating the main dataset as an Asimov with parameter values :')
  for (var, val, save_val) in saves : print("INFO :   %s=%g" % (var.GetName(), val))  
  fit(data, robust=True)
  nuis_par_set.Print('V')
  data = ROOT.RooStats.AsymptoticCalculator.MakeAsimovData(mconfig, ROOT.RooArgSet(), ROOT.RooArgSet())
  for (var, val, save_val) in saves : var.setVal(save_val)

if not options.data_only :
  for sample in channel.samples :
    if sample.normpar.getMin() > 0 : sample.normpar.setMin(0) # allow setting variable to 0

  # If a normpar is zero, we cannot get the expected yields for this component. In this case, fit an Asimov and set the parameter at the +2sigma level
  zero_normpars = []
  for sample in channel.samples : zero_normpars.append(sample.normpar.getVal() == 0)
  if any(zero_normpars) :
    ws.saveSnapshot('nominalNPs', nuis_par_set)
    asimov = ROOT.RooStats.AsymptoticCalculator.MakeAsimovData(mconfig, ROOT.RooArgSet(), ROOT.RooArgSet())
    print('=== Determining POI uncertainty using an Asimov dataset with parameters :')
    for sample in channel.samples :
      sample.normpar.setConstant(False)
      print('===   %s=%g' % (sample.normpar.GetName(), sample.normpar.getVal()))
    nuis_par_set.Print('V')
    fit(asimov, robust=True)
    # The S/B should be adjusted to the expected sensitivity value to get
    # reliable uncertainties on signal NPs. Choose POI = 2*uncertainty or this.
    ws.loadSnapshot('nominalNPs')
    for sample, z in zip(channel.samples, zero_normpars) : 
      if z :
        sample.normpar.setVal(2*sample.normpar.getError())
        if sample.normpar.getVal() == 0 :
          raise ValueError('ERROR : normalization parameter %s is exactly 0, cannot extract sample yields' % sample.normpar.GetName())

  if options.validation_data :
    validation_points = np.linspace(-3, 3, 13)
    valid_data = collections.OrderedDict()
    valid_data['points'] = np.array(validation_points)
    for par in nuis_pars :
      valid_data[par.GetName()] = np.ndarray((nbins, len(validation_points)))
  
  for par in nuis_pars :
    par.nominal = par.obj.getVal()
    if par.is_free :
      par.error = par.obj.getError()
      if par.error <= 0 :
        raise ValueError('Parameter %s has an uncertainty %g which is <= 0' % (par.name, par.error))
    else :
      par.error = 1
    print('=== Parameter %s : using deviation %g from nominal value %g for impact computation (x%g)' % (par.name, par.error, par.nominal, options.epsilon))

  for sample in channel.samples :
    print('=== Sample %s normalized to POI %s = %g -> n_events = %g' % (sample.name, sample.normpar.GetName(), sample.normpar.getVal(), sample.normvar.getVal()))
    sample.nom_norm = sample.normpar.getVal()
    sample.yields = np.zeros(nbins) 
    sample.impacts = {}
    for par in nuis_pars : sample.impacts[par.name] = np.zeros(nbins)
  print('=== Nominal NP values :')
  nuis_par_set.Print("V")
  for i in range(0, nbins) :
    xmin = bins[i]
    xmax = bins[i + 1]
    channel.obs.setRange('bin_%d' % i, xmin, xmax)
    bin_integral = channel.pdf.createIntegral(ROOT.RooArgSet(channel.obs), ROOT.RooArgSet(channel.obs), 'bin_%d' % i)
    for sample in channel.samples : sample.normpar.setVal(0)
    #  sample.integral = sample.pdf.createIntegral(ROOT.RooArgSet(channel.obs), ROOT.RooArgSet(channel.obs), 'bin_%d' % i)
    n_unassigned = main_pdf.expectedEvents(ROOT.RooArgSet(channel.obs))*bin_integral.getVal() # event yield which is not assigned to any sample (->goes into the default sample)
    for sample in channel.samples :
      sample.normpar.setVal(sample.nom_norm)
      sample.yields[i] = main_pdf.expectedEvents(ROOT.RooArgSet(channel.obs))*bin_integral.getVal() - n_unassigned
      sample.normpar.setVal(0)
    default_sample.yields[i] += n_unassigned # assign the unassigned to the default sample
    for sample in channel.samples :
      print('-- Nominal %s = %g' % (sample.name, sample.yields[i]))
    for par in nuis_pars :
      delta = par.error*options.epsilon
      par.obj.setVal(par.nominal + delta)
      n_unassigned_pos = main_pdf.expectedEvents(ROOT.RooArgSet(channel.obs))*bin_integral.getVal()
      for sample in channel.samples :
        sample.normpar.setVal(sample.nom_norm)
        sample.n_pos = main_pdf.expectedEvents(ROOT.RooArgSet(channel.obs))*bin_integral.getVal() - n_unassigned_pos
        sample.normpar.setVal(0)
      default_sample.n_pos += n_unassigned_pos
      par.obj.setVal(par.nominal - delta)
      n_unassigned_neg = main_pdf.expectedEvents(ROOT.RooArgSet(channel.obs))*bin_integral.getVal()
      for sample in channel.samples :
        sample.normpar.setVal(sample.nom_norm)
        sample.n_neg = main_pdf.expectedEvents(ROOT.RooArgSet(channel.obs))*bin_integral.getVal() - n_unassigned_neg
        sample.normpar.setVal(0)
      default_sample.n_neg += n_unassigned_neg
      for sample in channel.samples :
        impact_pos = ((sample.n_pos/sample.yields[i])**(1/options.epsilon) - 1) if sample.yields[i] != 0 else 0
        impact_neg = ((sample.yields[i]/sample.n_neg)**(1/options.epsilon) - 1) if sample.n_neg     != 0 else 0
        sample.impacts[par.name][i] = math.sqrt((1 + impact_pos)*(1 + impact_neg)) - 1
        print('-- sample %s, parameter %-10s : +1 sigma sig impact = %g' % (sample.name, par.name, impact_pos))
        print('-- sample %s, parameter %-10s : -1 sigma sig impact = %g' % (sample.name, ''      , impact_neg))
        print('-- sample %s, parameter %-10s : selected sig impact = %g' % (sample.name, ''      , sample.impacts[par.name][i]))
      par.obj.setVal(par.nominal)
      if options.validation_data :
        par_data = valid_data[par.name]
        nref = main_pdf.expectedEvents(channel.obs)*total_integral.getVal()
        for k, val in enumerate(validation_points) :
          par.obj.setVal(par.nominal + val*delta)
          nvar = main_pdf.expectedEvents(channel.obs)*total_integral.getVal()
          par_data[i,k] = nvar/nref if nref != 0 else 0
          print('== validation %-10s: %+6g variation = %g' % ('', val, par_data[i,k]))
        par.obj.setVal(par.nominal)


# 8 - Fill model JSON
# --------------------------------

jdict = {}
jdict['model_name'] = options.output_name

poi_specs = []
for poi in pois :
  poi_spec = {}
  poi_spec['name'] = poi.GetName()
  poi_spec['min'] = poi.getMin()
  poi_spec['max'] = poi.getMax()
  poi_specs.append(poi_spec)
jdict['pois'] = poi_specs

np_specs = []
for par in nuis_pars :
  np_spec = {}
  np_spec['name'] = par.name
  np_spec['nominal_val'] = par.nominal
  np_spec['variation'] = par.error
  np_spec['constraint'] = None if par.is_free else 1
  np_spec['aux_obs'] = None if par.is_free else cons_aux[par.name].GetName()
  np_specs.append(np_spec)
jdict['nps'] = np_specs

channel_specs = []

# TODO :iterate the block below for multiple channels
channel_spec = {}
channel_spec['name'] = channel.name
channel_spec['type'] = channel.type

bin_data = []
for b in range(0, nbins) :
  bin_datum = {}
  bin_datum['lo_edge'] = bins[b]
  bin_datum['hi_edge'] = bins[b+1]
  bin_data.append(bin_datum)
channel_spec['obs_name'] = channel.obs.GetTitle().replace('#','\\')
channel_spec['obs_unit'] = channel.obs.getUnit()
channel_spec['bins'] = bin_data
sample_specs = []
for sample in channel.samples :
  sample_spec = {}
  sample_spec['name'] = sample.name
  sample_spec['normalization'] = sample.normpar.GetName()
  sample_spec['yields'] = sample.yields.tolist()
  sample_spec['impacts'] = { par : sample.impacts[par].tolist() for par in sample.impacts }
  sample_specs.append(sample_spec)
channel_spec['samples'] = sample_specs

channel_specs.append(channel_spec)
jdict['channels'] = channel_specs

# 9 - Fill the dataset information
# --------------------------------
bin_array = array.array('d', bins)
hist = ROOT.TH1D('h', 'histogram', nbins, bin_array)
data.fillHistogram(hist, ROOT.RooArgList(channel.obs))
bin_counts = [ hist.GetBinContent(i+1) for i in range(0, nbins) ]
data_dict = {}
data_dict['bin_counts'] = bin_counts
aux_specs = []
for par in cons_nps :
  aux_spec = {}
  aux = cons_aux[par.GetName()]
  aux_spec['name']  = aux.GetName()
  aux_spec['value'] = aux.getVal()
  aux_spec['min']   = aux.getMin()
  aux_spec['max']   = aux.getMax()
  aux_specs.append(aux_spec)
data_dict['aux_obs'] = aux_specs
jdict['data'] = data_dict

# 10 - Write everything to file
# ----------------------------
with open(options.output_file, 'w') as fd:
  json.dump(jdict, fd, ensure_ascii=True, indent=3)

# 11 - If requested, also dump validation information
# --------------------------------------------------
if options.validation_data :
  valid_lists = {}
  for poi in pois : valid_lists[poi.GetName()] = poi.getVal()
  valid_lists['points'] = valid_data['points'].tolist()
  for par in nuis_pars : valid_lists[par.name] = valid_data[par.name].tolist()
  with open(options.validation_data, 'w') as fd:
    json.dump(valid_lists, fd, ensure_ascii=True, indent=3)
