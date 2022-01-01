#! /usr/bin/env python

__doc__ = """
*Convert a ROOT workspace into fastprof markup format*

The script takes as input a ROOT workspace, and converts the contents
into the definition file of a linear model, as follows:

* The model POIs and NPs are taken from the ModelConfig file.

* The model PDF is also taken from the ModelConfig. Two cases are currently
  implemented:
  
  * *The PDF is a `RooAddPdf`*: the components of the sum are then 
    used to define the samples of a single channel. The channel
    observable is taken from the PDF, and the binning from the `-b` option.
      
  * *The PDF is a `RooSimultaneous`* : the states of the PDF are taken
    to correspond each to a separate channel. Each channel must have a 
    PDF of *RooAddPdf* type, which is then treated as above.

* Nominal yields in each bin are computed from integrating the PDF of each
  sample in each channel.

* Linear impacts are computed by changing the values of the NPs as 
  specified by the `--variations` option. By default :math:`\pm 1 \sigma`
  variations are used. The impact on each sample are separated by setting
  the normalizations of all but one to zero for each in turn. Variations
  which are present when all normalizations are set to 0 are assigned to
  the default sample, specified by the `--default-sample` option.

* NP central values and uncertainties taken from directly from the workspace,
  or from a fit to data (`--data-file` option) or to an Asimov dataset
  (`--asimov` option). The same applied to the POI value used to define
  the nominal yields. If the POI value leads to a normalization of 0, the POI
  is instead set to twice its uncertainty.
  
* Prior to computations, the model can be adjusted using the `--setval`,
  `--setconst` and `--setrange` options, which set respectively the value,
  constness and range of model parameters.

The output is a single markup file which defines the model as well as the 
dataset if one was specified. A validation file is also produced if the 
`--validation-output` option was set: this contains information that can be
used to assess if the linearity assumption is valid for the model.
"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import json,yaml
import array
import math
import ROOT
from fastprof_import.tools import process_setvals, process_setranges, process_setconsts, fit, make_asimov, make_binned, trim_float

class WSChannel() : pass
class WSSample() : pass

####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("convert_ws.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-f", "--ws-file"          , type=str  , required=True    , help="Name of file containing the workspace")
  parser.add_argument("-w", "--ws-name"          , type=str  , default='modelWS', help="Name workspace object inside the specified file")
  parser.add_argument("-m", "--model-config-name", type=str  , default='mconfig', help="Name of model config within the specified workspace")
  parser.add_argument("-c", "--channels"         , type=str  , default=None     , help="Names of the model channels, in the form c1,c2,..., in the same order as the RooSimPdf components")
  parser.add_argument("-s", "--samples"          , type=str  , default=None     , help="Names of the model samples, in the form c1:s1:p1,..., with c1 a channel name, s1 the sample name, and p1 its normpar name")
  parser.add_argument(      "--default-sample"   , type=str  , default=None     , help="Names of the model samples, in the form s1,s2,..., in the same order as the RooAddPdf components")
  parser.add_argument("-b", "--binning"          , type=str  , required=True    , help="Binning used, in the form xmin:xmax:nbins[:log]~...")
  parser.add_argument("-e", "--epsilon"          , type=float, default=1        , help="Scale factor applied to uncertainties for impact computations")
  parser.add_argument("-=", "--setval"           , type=str  , default=''       , help="List of variable value changes, in the form var1=val1,var2=val2,...")
  parser.add_argument("-k", "--setconst"         , type=str  , default=''       , help="List of variables to set constant")
  parser.add_argument(      "--setfree"          , type=str  , default=''       , help="List of variables to set free")
  parser.add_argument("-r", "--setrange"         , type=str  , default=''       , help="List of variable range changes, in the form var1=[min1]:[max1],var2=[min2]:[max2],...")
  parser.add_argument("-d", "--data-name"        , type=str  , default=''       , help="Name of dataset object within the input workspace")
  parser.add_argument("-a", "--asimov"           , type=str  , default=None     , help="Perform an Asimov fit before conversion")
  parser.add_argument("-x", "--data-only"        , action="store_true"          , help="Only dump the specified dataset, not the model")
  parser.add_argument(      "--refit"            , type=str  , default=None     , help="Fit the model to the specified dataset before conversion")
  parser.add_argument(      "--binned"           , action="store_true"          , help="Use binned data")
  parser.add_argument(      "--input_bins"       , type=int  , default=0        , help="Number of bins to use when binning the input dataset")
  parser.add_argument(      "--variations"       , type=str  , default='1'      , help="Comma-separated list of NP variations to tabulate")
  parser.add_argument(      "--regularize"       , type=float, default=0        , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
  parser.add_argument("-o", "--output-file"      , type=str  , required=True    , help="Name of output file")
  parser.add_argument(      "--output-name"      , type=str  , default=''       , help="Name of the output model")
  parser.add_argument(      "--digits"           , type=int  , default=7        , help="Number of significant digits in float values")
  parser.add_argument(      "--markup"           , type=str  , default='json'   , help="Output markup flavor (supported : 'json', 'yaml')")
  parser.add_argument("-t", "--packing-tolerance", type=float, default=None     , help="Level of precision for considering two impact values to be equal")
  parser.add_argument("-l", "--validation-output", type=str  , default=None     , help="Name of output file for validation data")
  parser.add_argument("-v", "--verbosity"        , type=int  , default=0        , help="Verbosity level")
  return parser

def run(argv = None) :
  """
  Main method in the script
  """
  parser = make_parser()
  options = parser.parse_args(argv)
  if not options :
    parser.print_help()
    sys.exit(0)

  # 1 - Parse bin specifications, retrieve workspace contents
  # ---------------------------------------------------------

  bins = []
  try:
    binspecs = options.binning.split('~')
    for binspec in binspecs :
      add_last = False
      binspec_fields = binspec.split(':')
      if binspec_fields[2][-1] == '+' :
        add_last = True
        binspec_fields[2] = binspec_fields[2][:-1]
      nbins = int(binspec_fields[2])
      if len(binspec_fields) == 4 and binspec_fields[3] == 'log' :
        bins.extend(np.logspace(1, math.log(float(binspec_fields[1]))/math.log(float(binspec_fields[0])), nbins, add_last, float(binspec_fields[0])))
      else :
        bins.extend(np.linspace(float(binspec_fields[0]), float(binspec_fields[1]), nbins, add_last))
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


  # 2 - Update parameter values and constness as specified in options
  # -----------------------------------------------------------------

  if options.setval   != '' : process_setvals  (options.setval  , ws)
  if options.setconst != '' : process_setconsts(options.setconst, ws, const=True)
  if options.setfree  != '' : process_setconsts(options.setfree , ws, const=False)
  if options.setrange != '' : process_setranges(options.setrange, ws)


  # 3 - Define the primary dataset
  # ------------------------------

  mconfig = ws.obj(options.model_config_name)
  if not mconfig : raise KeyError('Model config %s not found in workspace.' % options.model_config_name)

  main_pdf = mconfig.GetPdf()
  pois = mconfig.GetParametersOfInterest().selectByAttrib('Constant', False)
  nps = mconfig.GetNuisanceParameters().selectByAttrib('Constant', False)
  aux_obs = mconfig.GetGlobalObservables()

  data = None
  if options.data_name != '' :
    data = ws.data(options.data_name)
    if data == None :
      ds = [ d.GetName() for d in ws.allData() ]
      raise KeyError('Dataset %s not found in workspace. Available datasets are: %s' % (options.data_name, ', '.join(ds)))
  elif options.asimov != None :
    data = make_asimov(mconfig, options.asimov)
  else:
    raise ValueError('ERROR: no dataset was specified either using --data-name or --asimov')


  # 4 - Identify the model parameters and main PDF
  # ----------------------------------------------

  cons_aux = {}
  cons_nps = []
  free_nps = []
  
  try:
    for pdf in ws.allPdfs() :
      matching_auxs = pdf.getDependents(aux_obs)
      if len(matching_auxs) == 1 :
        matching_pars = pdf.getDependents(nps)
        maux = ROOT.RooArgList(matching_auxs).at(0)
        if len(matching_pars) == 1 :
          mpar = ROOT.RooArgList(matching_pars).at(0)
          print('INFO: Matching aux %s to NP %s' % (maux.GetName(), mpar.GetName()))
          cons_nps.append(mpar)
          cons_aux[mpar.GetName()] = maux
  except Exception as inst :
    print(inst)
    ValueError('Could not identify nuisance parameters')
  
  nuis_pars = []
  class NuisancePar : pass
  
  for par in nps :
    nuis_par = NuisancePar()
    nuis_par.name = par.GetName()
    nuis_par.unit = par.getUnit()
    nuis_par.obj = par
    nuis_par.is_free = not par in cons_nps
    if nuis_par.is_free : free_nps.append(par)
    nuis_pars.append(nuis_par)


  # 5. Identify the model channels
  # ---------------------------------------------------------
  
  user_channels = {}
  if options.channels is not None :
    try:
      channel_specs = options.channels.split(',')
      for channel_spec in channel_specs :
        fields = channel_spec.split(':')
        user_channels[fields[0]] = fields[1] if len(fields) == 2 else ''  
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid channel name specification %s : should be of the form index1:name1,index2:name2,...' % options.channels)

  channels = []

  for component in main_pdf.getComponents() :
    if isinstance(component, ROOT.RooSimultaneous) :
      cat = component.indexCat()
      for i in range(0, cat.size()) :
        channel = WSChannel()
        channel.type = 'binned_range'
        channel_name = cat.lookupType(i).GetName()
        if len(user_channels) > 0 and not str(i) in user_channels and not channel_name in user_channels : continue
        channel.pdf = component.getPdf(channel_name)
        channel.name = user_channels[str(i)] if str(i) in user_channels and user_channels[str(i)] != '' else channel_name
        channel.index = i
        channel.cat = cat
        channels.append(channel)
      break
    elif isinstance(component, ROOT.RooAddPdf) :
      channel = WSChannel()
      channel.type = 'binned_range'
      channel.pdf = component
      channel.name = user_channels[0] if 0 in user_channels else component.GetName()
      channel.cat = None
      channels.append(channel)
      break

  # 6. Make the channel objects (and identify samples)
  # ---------------------------------------------------------
  
  normpars = {}
  if options.samples is not None :
    try:
      sample_specs = options.samples.replace(' ', '').replace('\n', '').split(',')
      normpar_names = {}
      for spec in sample_specs :
        if spec == '' : continue
        spec_names = spec.split(':')
        if len(spec_names) == 1 : spec_names = [ '' ] + spec_names
        if len(spec_names) == 2 : spec_names = [ '' ] + spec_names
        normpar_names[(spec_names[0], spec_names[1])] = spec_names[2]
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid sample normpar name specification %s : should be of the form chan1:samp1:par1,chan2:samp2:par2,...' % options.samples)
    for (channel_name, sample_name), normpar_name in normpar_names.items() :
      normpar = ws.var(normpar_name)
      if normpar == None : raise ValueError("Normalization parameter '%s' not found in workspace (was for sample '%s' of channel '%s')." % (normpar_name, sample_name, channel_name))
      if not channel_name in normpars : normpars[channel_name] = {}
      normpars[channel_name][sample_name] = normpar
  
  for channel in channels :
    fill_channel(channel, pois, aux_obs, mconfig, normpars, options)

  # 7 - Fill the model information
  # ------------------------------

  if options.binned : 
    unbinned_data = data
    rebinnings = {}
    if options.input_bins > 0 :
      for channel in channels : rebinnings[channel.obs] = options.input_bins
    data = make_binned(data, rebinnings)
  else :
    unbinned_data = data

  if options.refit != None :
    saves = process_setvals(options.refit, ws)
    print('=== Refitting PDF to specified dataset with under the hypothesis :')
    for (var, val, save_val) in saves :
      print("INFO :   %s=%g" % (var.GetName(), val))
      var.setConstant()
    fit(main_pdf, data, robust=True)

  # If we specified both, then it means an Asimov with NP values profiled on the observed
  if options.data_name != '' and options.asimov != None :
    print('=== Generating the main dataset as an Asimov, fitted as below')
    data = make_asimov(option.asimov, mconfig, main_pdf, data)

  if not options.data_only :

    # Check if the normalization terms are OK
    zero_normpars = []
    for channel in channels :
      for sample in channel.samples :
        if sample.normpar is not None and sample.normpar.getMin() > 0 : sample.normpar.setMin(0) # allow setting variable to 0
        # If a normpar is zero, we cannot get the expected nominal_yields for this component. In this case, fit an Asimov and set the parameter at the +2sigma level
      for sample in channel.samples : 
        if sample.normpar is not None and sample.normpar.getVal() == 0 : zero_normpars.append(sample.normpar)
    if len(zero_normpars) > 0 :
      ws.saveSnapshot('nominalNPs', nps)
      asimov = ROOT.RooStats.AsymptoticCalculator.MakeAsimovData(mconfig, ROOT.RooArgSet(), ROOT.RooArgSet())
      print('=== Determining POI uncertainty using an Asimov dataset with parameters :')
      for par in zero_normpars :
        par.setConstant(False)
        print('===   %s=%g' % (par.GetName(), par.getVal()))
      nps.Print('V')
      fit(main_pdf, asimov, robust=True)
      # The S/B should be adjusted to the expected sensitivity value to get
      # reliable uncertainties on signal NPs. Choose POI = 2*uncertainty or this.
      ws.loadSnapshot('nominalNPs')
      for par in zero_normpars : 
        par.setVal(2*par.getError())
        if par.getVal() == 0 :
          raise ValueError('ERROR : normalization parameter %s is exactly 0, cannot extract sample nominal_yields' % par.GetName())
        print('=== Set POI nominal value to %s = %g' % (par.GetName(), par.getVal()))

    for par in nuis_pars :
      par.nominal = par.obj.getVal()
      if par.is_free :
        par.error = par.obj.getError()
        if par.error <= 0 :
          raise ValueError('Parameter %s has an uncertainty %g which is <= 0' % (par.name, par.error))
      else :
        par.error = 1
      print('=== Parameter %s : using deviation %g from nominal value %g for impact computation (x%g)' % (par.name, par.error, par.nominal, options.epsilon))

    variations = []
    try :
      for v in options.variations.split(',') : variations.extend([ +float(v), -float(v) ])
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid NP variations specification %s : should be of the form v1,v2,...' % options.variations)
    if len(variations) == 0 :
      raise ValueError('Should have at least 1 NP variation implemented for a valid model')

    validation_points = np.linspace(-5, 5, 21) if options.validation_output is not None else []

    # Fill the channel information
    for i, channel in enumerate(channels) :
      fill_channel_yields(channel, i, len(channels), bins, nuis_pars, variations, options, validation_points)

    if options.packing_tolerance is not None :
      for channel in channels :
        for sample in channel.samples :
          for par_name in sample.impacts :
            sample.impacts[par_name] = pack(sample.impacts[par_name], options.packing_tolerance, options.digits)


  # 8 - Fill model markup
  # --------------------------------

  sdict = {}

  if not options.data_only :
    model_dict = {}
    model_dict['name'] = options.output_name
    # POIs
    poi_specs = []
    for poi in pois :
      poi_spec = {}
      poi_spec['name'] = poi.GetName()
      poi_spec['unit'] = poi.getUnit()
      poi_spec['min_value'] = poi.getMin()
      poi_spec['max_value'] = poi.getMax()
      poi_specs.append(poi_spec)
    model_dict['POIs'] = poi_specs
    # NPs
    np_specs = []
    for par in nuis_pars :
      np_spec = {}
      np_spec['name'] = par.name
      np_spec['unit'] = par.unit
      np_spec['nominal_value'] = par.nominal
      np_spec['variation'] = par.error
      np_spec['constraint'] = None if par.is_free else par.error
      np_spec['aux_obs'] = None if par.is_free else cons_aux[par.name].GetName()
      np_specs.append(np_spec)
    model_dict['NPs'] = np_specs
    # Aux obs
    aux_specs = []
    for par in cons_nps :
      aux_spec = {}
      aux = cons_aux[par.GetName()]
      aux_spec['name']  = aux.GetName()
      aux_spec['unit']  = aux.getUnit()
      aux_spec['min_value'] = aux.getMin()
      aux_spec['max_value'] = aux.getMax()
      aux_specs.append(aux_spec)
    model_dict['aux_obs'] = aux_specs
    # Channels
    channel_specs = []
    for channel in channels :
      channel_spec = {}
      channel_spec['name'] = channel.name
      channel_spec['type'] = channel.type
      bin_specs = []
      for b in range(0, nbins) :
        bin_spec = {}
        bin_spec['lo_edge'] = float(bins[b])
        bin_spec['hi_edge'] = float(bins[b+1])
        bin_specs.append(bin_spec)
      channel_spec['obs_name'] = channel.obs.GetTitle().replace('#','\\')
      channel_spec['obs_unit'] = channel.obs.getUnit()
      channel_spec['bins'] = bin_specs
      sample_specs = []
      for sample in channel.samples :
        sample_spec = {}
        sample_spec['name'] = sample.name
        if sample.normpar is not None and (pois.find(sample.normpar.GetName()) or nps.find(sample.normpar.GetName())) :
          sample_spec['norm'] = sample.normpar.GetName()
          sample_spec['nominal_norm'] = sample.nominal_norm
        else :
          sample_spec['norm'] = ''
        sample_spec['nominal_yields'] = sample.nominal_yields.tolist()
        sample_spec['impacts'] = sample.impacts
        sample_specs.append(sample_spec)
      channel_spec['samples'] = sample_specs
      channel_specs.append(channel_spec)
    model_dict['channels'] = channel_specs
    sdict['model'] = model_dict

  # 9 - Fill the dataset information
  # --------------------------------

  data_dict = {}
  # Channels
  channel_data = []
  for channel in channels :
    channel_datum = {}
    channel_datum['name'] = channel.name
    channel_datum['type'] = channel.type
    channel_datum['obs_name'] = channel.obs.GetTitle().replace('#','\\')
    channel_datum['obs_unit'] = channel.obs.getUnit()
    bin_array = array.array('d', bins)
    hist = ROOT.TH1D('%s_hist' % channel.name, 'Data histogram for channel %s' % channel.name, nbins, bin_array)
    unbinned_data.fillHistogram(hist, ROOT.RooArgList(channel.obs), '' if channel.cat is None else '%s==%d' % (channel.cat.GetName(), channel.index))
    bin_specs = []
    for b in range(0, nbins) :
      bin_spec = {}
      bin_spec['lo_edge'] = float(bins[b])
      bin_spec['hi_edge'] = float(bins[b+1])
      bin_spec['counts'] = hist.GetBinContent(b+1)
      bin_specs.append(bin_spec)
    channel_datum['bins'] = bin_specs
    channel_data.append(channel_datum)
  data_dict['channels'] = channel_data
  # Aux obs
  aux_specs = []
  for par in cons_nps :
    aux_spec = {}
    aux = cons_aux[par.GetName()]
    aux_spec['name']  = aux.GetName()
    aux_spec['value'] = aux.getVal()
    aux_specs.append(aux_spec)
  data_dict['aux_obs'] = aux_specs

  sdict['data'] = data_dict


  # 10 - Write everything to file
  # ----------------------------

  with open(options.output_file, 'w') as fd:
    if options.markup == 'json' : 
      json.dump(sdict, fd, ensure_ascii=True, indent=3)
    else :
      yaml.dump(sdict, fd, sort_keys=False, default_flow_style=None, width=10000)


  # 11 - Also dump validation information unless deactivated
  # --------------------------------------------------------

  if options.validation_output is not None :
    valid_lists = {}
    for poi in pois : valid_lists[poi.GetName()] = poi.getVal()
    valid_lists['points'] = validation_points.tolist()
    for channel in channels :
      channel_valid = {}
      for par in nuis_pars : 
        channel_valid[par.name] = channel.valid_data[par.name].tolist()
      valid_lists[channel.name] = channel_valid
    if options.validation_output != '' :
      validation_filename = options.validation_output
    else :
      split_name = os.path.splitext(options.output_file)
      validation_filename = split_name[0] + '_validation' + split_name[1]
    with open(validation_filename, 'w') as fd:
      json.dump(valid_lists, fd, ensure_ascii=True, indent=3)


# ---------------------------------------------------------------------
def fill_channel(channel, pois, aux_obs, mconfig, normpars, options) :
  
  if options.verbosity > 0 : print("Creating channel '%' from PDF '%'" % (channel.name, channel.pdf.GetName()))
  if isinstance(channel.pdf, ROOT.RooProdPdf) :
    for i in range(0, channel.pdf.pdfList().getSize()) :
      pdf = channel.pdf.pdfList().at(i)
      if len(pdf.getDependents(aux_obs)) > 0 : continue
      if not isinstance(pdf, ROOT.RooAbsPdf) :
        print("Got unexpected PDF of class '%' in PDF '%s' for channel '%s', skipping it." % (pdf.Class().GetName(), channel.pdf.GetName(), channel.name))
        continue
      channel.pdf = pdf
      break

  channel_obs = mconfig.GetObservables().selectCommon(channel.pdf.getVariables())
  if channel_obs.getSize() == 0 :
    raise ValueError('Cannot identify observables for channel %s.')
  if channel_obs.getSize() > 1 :
    raise ValueError('Channel %s has %d observables -- multiple observables not supported yet.')
  channel.obs = ROOT.RooArgList(channel_obs).at(0)
  if options.data_only : return channel

  channel.samples = []
  channel.default_sample = None # the sample to which unassigned variations will be associated (e.g. spurious signal, not scaled by any sample normpars)

  user_samples = normpars[''] if '' in normpars else {}
  if channel.name in normpars : user_samples.update(normpars[channel.name])
  if len(user_samples) > 0 :
    for user_sample in user_samples :
      sample = WSSample()
      sample.name = user_sample
      sample.normpar = user_samples[user_sample]
      if sample.name == options.default_sample : channel.default_sample = sample
      channel.samples.append(sample)
  else :
    for i in range(0, channel.pdf.pdfList().getSize()) :
      sample = WSSample()
      sample.name = channel.pdf.pdfList().at(i).GetName()
      sample.normpar = None
      normvar = channel.pdf.coefList().at(i)
      if isinstance(normvar, ROOT.RooRealVar) :
        sample.normpar = normvar
      else :
        poi_candidates = normvar.getVariables().selectCommon(pois)
        if poi_candidates.getSize() == 1 :
          sample.normpar = ROOT.RooArgList(poi_candidates).at(0)
        else :
          print("Normalization of sample '%s' depends on multiple POIs:" % sample.name)
          normvar.Print()
          poi_candidates.Print()
      if sample.normpar == None :
        raise ValueError('Cannot identify normalization variable for sample %s, please specify manually. Known specifications are : \n%s' % (sample.name, str(normpars)))
      if sample.name == options.default_sample : channel.default_sample = sample
      channel.samples.append(sample)  
  if channel.default_sample is None : 
    default_sample = WSSample()
    default_sample.name = 'unassigned_background'
    default_sample.normpar = None
    channel.samples.append(default_sample)
    channel.default_sample = default_sample
  return channel

def integral(channel) :
  return (channel.pdf.expectedEvents(0) if channel.normalized_integral else 1)*channel.bin_integral.getVal()

def save_normpars(channel) :
  for sample in channel.samples :
    if sample.normpar is not None :
      sample.normpar_save = sample.normpar.getVal()

def save_nominal(channel) :
  for sample in channel.samples :
    if sample.normpar is not None :
      sample.nominal_norm = sample.normpar.getVal()

def set_normpars(channel, value = None) :
  for sample in channel.samples : 
    if sample.normpar is not None :
      sample.normpar.setVal(sample.nominal_norm if value is None else value)


# ---------------------------------------------------------------------
def fill_yields(channel, key) :
  save_normpars(channel)
  set_normpars(channel, 0)
  n_unassigned = integral(channel)
  for sample in channel.samples :
    if sample.normpar is None : 
      sample.yields[key] = 0
      continue
    sample.normpar.setVal(sample.normpar_save)
    sample.yields[key] = integral(channel) - n_unassigned
    sample.normpar.setVal(0)
  channel.default_sample.yields[key] += n_unassigned
  #for sample in channel.samples : print('yield', key, sample.name, sample.yields[key], n_unassigned)

def fill_channel_yields(channel, channel_index, nchannels, bins, nuis_pars, variations, options, validation_points) :
  if options.verbosity > 0 : print('=== Processing channel %s (%d of %d)' % (channel.name, channel_index, nchannels))
  nbins = len(bins) - 1
  if options.validation_output is not None :
    channel.valid_data = {}
    for par in nuis_pars :
      channel.valid_data[par.name] = np.ndarray((len(channel.samples), nbins, len(validation_points)))
  save_nominal(channel)
  set_normpars(channel, 0)
  unassigned_nevents = channel.pdf.expectedEvents(ROOT.RooArgSet(channel.obs))
  for sample in channel.samples :
    if sample.normpar is not None :
      sample.normpar.setVal(sample.nominal_norm)
      if options.verbosity > 0 : print('=== Sample %s normalized to normalization parameter %s = %g -> n_events = %g' % (sample.name, sample.normpar.GetName(), sample.normpar.getVal(), channel.pdf.expectedEvents(ROOT.RooArgSet(channel.obs)) - unassigned_nevents))
      sample.normpar.setVal(0)
    else :
      if options.verbosity > 0 : print('=== Sample %s (unnormalized)' % sample.name)    
    sample.nominal_yields = np.zeros(nbins) 
    sample.yields = {}
    sample.impacts = {}
    for par in nuis_pars : sample.impacts[par.name] = []
  if options.verbosity > 0 : print('=== Unassigned n_events = %g' % unassigned_nevents)    
  if options.verbosity > 0 : print('=== Nominal NP values :')
  for par in nuis_pars : par.obj.Print()
  for i in range(0, nbins) :
    print('=== Bin [%g, %g] (%d of %d)' % (bins[i], bins[i+1], i, nbins))
    #sys.stderr.write("\rProcessing bin %d of %d in channel '%s' (%d of %d)" % (i+1, nbins, channel.name, channel_index + 1, nchannels))
    #sys.stderr.flush()
    xmin = bins[i]
    xmax = bins[i + 1]
    channel.obs.setRange('bin_%d' % i, xmin, xmax)
    create_bin_integral(channel, i)
    #ROOT.SetOwnership(channel.bin_integral, True) # needed ?
    set_normpars(channel)
    #print('Nominal pars:')
    #print('-------------')
    #for s in channel.samples :
      #if s.normpar is not None : print(s.normpar.GetName(), '=', s.normpar.getVal())
    #for p in nuis_pars : print(p.obj.GetName(), '=', p.obj.getVal())
    fill_yields(channel, 'nominal')
    for sample in channel.samples :
      sample.nominal_yields[i] = trim_float(sample.yields['nominal'], options.digits)
      if sample.normpar is not None :
        if options.verbosity > 0 : print('-- Nominal %s = %g (%s = %g)' % (sample.name, sample.nominal_yields[i], sample.normpar.GetName(), sample.nominal_norm))
      else :
        if options.verbosity > 0 : print('-- Nominal %s = %g' % (sample.name, sample.nominal_yields[i]))
    for p, par in enumerate(nuis_pars) :
      sys.stderr.write("\rProcessing channel '%s' (%3d of %3d), bin %3d of %3d, NP %4d of %4d [%30s]" % (channel.name, channel_index + 1, nchannels, i+1, nbins, p, len(nuis_pars), par.name[:30]))
      sys.stderr.flush()
      delta = par.error*options.epsilon
      for sample in channel.samples : sample.impacts[par.name].append({})
      for variation in variations :
        set_normpars(channel)
        par.obj.setVal(par.nominal + variation*delta)
        #print('%s %g impact pars:' % (par.name, variation))
        #print('-------------')
        #for s in channel.samples :
          #if s.normpar is not None : print(s.normpar.GetName(), '=', s.normpar.getVal())
        #for p in nuis_pars : print(p.obj.GetName(), '=', p.obj.getVal())
        fill_yields(channel, 'var')
        par.obj.setVal(par.nominal)
        for sample in channel.samples : 
          sample.impact = (sample.yields['var']/sample.yields['nominal'])**(1/options.epsilon) - 1 if sample.yields['nominal'] != 0 else 0
          sample.impacts[par.name][-1]['%+g' % variation] = sample.impact
          if options.verbosity > 1 : print('-- sample %10s, parameter %-10s : %+g sigma impact = %g' % (sample.name, par.name, variation, sample.impact))
      if options.validation_output is not None :
        par_data = channel.valid_data[par.name]
        for k, val in enumerate(validation_points) :
          set_normpars(channel)
          par.obj.setVal(par.nominal + val*par.error)
          fill_yields(channel, 'valid')
          par.obj.setVal(par.nominal)
          for s, sample in enumerate(channel.samples) : 
            par_data[s,i,k] = trim_float(sample.yields['valid']/max(sample.yields['nominal'], 1E-5), options.digits)
            print('== validation %-10s: %+6g variation = %s' % (par.name, val, str(par_data[s,i,k])))
  set_normpars(channel)
  sys.stderr.write('\n')

def create_bin_integral(channel, bin_index) :
  if isinstance(channel.pdf, ROOT.RooAddPdf) :
    channel.terms = ROOT.RooArgList()
    channel.norm_factors = []
    channel.pdf_ints = []
    channel.products = []
    for i in range(0, channel.pdf.pdfList().getSize()) :
      norm_factor = channel.pdf.coefList().at(i)
      channel.norm_factors.append(norm_factor)
      pdf = channel.pdf.pdfList().at(i)
      pdf_int = pdf.createIntegral(ROOT.RooArgSet(channel.obs), ROOT.RooArgSet(channel.obs), 'bin_%d' % bin_index)
      channel.pdf_ints.append(pdf_int)
      product = ROOT.RooProduct('prod_term_%d_bin%d' % (i, bin_index), '', ROOT.RooArgList(norm_factor, pdf_int))
      channel.terms.add(product)
      channel.products.append(product)
    channel.bin_integral = ROOT.RooAddition('fast_integral_bin%d' % bin_index, '', channel.terms)
    channel.normalized_integral = False
  else :
    channel.bin_integral = channel.pdf.createIntegral(ROOT.RooArgSet(channel.obs), ROOT.RooArgSet(channel.obs), 'bin_%d' % bin_index)
    channel.normalized_integral = True

# Check if all the impacts are the same (within tolerance), if so return the average
def pack(impacts, tolerance=1E-5, num_digits=7) :
  n = len(impacts)
  pack_ok = True
  if n == 0 : return []
  average = { key : 0 for key in impacts[0] }
  for impact in impacts :
    for key in average : average[key] += impact[key]/n
  for impact in impacts :
    for key in average :
      if abs(impact[key] - average[key]) > tolerance :
        if num_digits is None :
          return impacts
        else :
          pack_ok = False
          impact[key] = trim_float(impact[key], num_digits)
  if num_digits is not None :
    for key in average :
      average[key] = trim_float(average[key], num_digits)
  return average if pack_ok else impacts

if __name__ == '__main__' : run()
