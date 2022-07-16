#! /usr/bin/env python

__doc__ = """
*Convert a pyhf workspace into fastprof markup format*

The script takes as input a pyhf workspace, and converts the contents
into the definition file of a linear model.

WIP warning: this script is still preliminary and untested
"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import json,yaml
import math
#import pyhf
from fastprof_import.tools import process_setvals, process_setranges, process_setconsts, fit, make_asimov, make_binned, trim_float

####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("convert_ws.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-f", "--ws-file"          , type=str  , required=True    , help="Name of file containing the workspace")
  parser.add_argument("-s", "--samples"          , type=str  , default=None     , help="Names of the model samples, in the form c1:s1:p1,..., with c1 a channel name, s1 the sample name, and p1 its normpar name")
  parser.add_argument("-e", "--epsilon"          , type=float, default=1        , help="Scale factor applied to uncertainties for impact computations")
  parser.add_argument("-=", "--setval"           , type=str  , default=''       , help="List of variable value changes, in the form var1=val1,var2=val2,...")
  parser.add_argument("-k", "--setconst"         , type=str  , default=''       , help="List of variables to set constant")
  parser.add_argument(      "--setfree"          , type=str  , default=''       , help="List of variables to set free")
  parser.add_argument("-r", "--setrange"         , type=str  , default=''       , help="List of variable range changes, in the form var1=[min1]:[max1],var2=[min2]:[max2],...")
  parser.add_argument("-d", "--load-data"        , action="store_true"          , help="Load data from workspace")
  parser.add_argument("-x", "--data-only"        , action="store_true"          , help="Only dump the specified dataset, not the model")
  parser.add_argument(      "--fit-result"       , type=str  , default=None     , help="File containing a pyhf fit result with parameter uncertainties")
  parser.add_argument("-a", "--asimov-fit"       , type=str  , default=None     , help="Asimov fit from which to get parameter uncertainties, specified as poi=val")
  parser.add_argument(      "--variations"       , type=str  , default='1'      , help="Comma-separated list of NP variations to tabulate")
  parser.add_argument(      "--regularize"       , type=float, default=0        , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
  parser.add_argument("-o", "--output-file"      , type=str  , required=True    , help="Name of output file")
  parser.add_argument(      "--output-name"      , type=str  , default=''       , help="Name of the output model")
  parser.add_argument(      "--digits"           , type=int  , default=7        , help="Number of significant digits in float values")
  parser.add_argument(      "--markup"           , type=str  , default='json'   , help="Output markup flavor (supported : 'json', 'yaml')")
  parser.add_argument("-t", "--packing-tolerance", type=float, default=None     , help="Level of precision for considering two impact values to be equal")
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

  # 1. retrieve workspace contents
  # ---------------------------------------------------------

  try:
    with open(options.ws_file) as input_file:
      spec = json.load(input_file)
  except Exception as inst :
    print(inst)
    raise FileNotFoundError('Cannot open model file %s' % options.ws_file)

  workspace = pyhf.Workspace(spec)
  model = workspace.model()


  # 2 - Update parameter values and constness as specified in options
  # -----------------------------------------------------------------

  init_pars = [*(model.config.suggested_init())]
  fixed_pars = [*(model.config.suggested_fixed())]
  par_ranges = [*(model.config.suggested_bounds())]

  if options.setval   != '' : process_setvals  (options.setval  , model=model, init_pars=init_pars)
  if options.setconst != '' : process_setconsts(options.setconst, model=model, fixed_pars=fixed_pars, init_pars=init_pars, const=True)
  if options.setfree  != '' : process_setconsts(options.setfree , model=model, fixed_pars=fixed_pars, init_pars=init_pars, const=False)
  if options.setrange != '' : process_setranges(options.setrange, model=model, par_ranges=par_ranges)

  variations = []
  try :
    for v in options.variations.split(',') : variations.extend([ +float(v), -float(v) ])
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid NP variations specification %s : should be of the form v1,v2,...' % options.variations)
  if len(variations) == 0 :
    raise ValueError('Should have at least 1 NP variation implemented for a valid model')


  # 3 - Get the fit result from which one can obtain NP uncertainties
  # -----------------------------------------------------------------

  if options.fit_result is not None :
    try:
      fit_result = json.load(options.fit_result)
    except Exception as inst :
      print(inst)
      raise FileNotFoundError('Cannot open fit result file %s' % options.fit_result)
  elif options.asimov_fit is not None :
    poi_val = process_setvals(options.asimov_fit)
    if len(poi_val) != 1 or poi_val[0][0] != model.config.poi_name :
      raise ValueError("Invalid POI value specification '%s', expecting a value for the POI '%s'." % (options.asimov_fit, model.config.poi_name))
    asimov_init  = [*init_pars]
    asimov_init[model.config.poi_index] = poi_val[0][1]
    asimov_fixed = [*fixed_pars]
    asimov_fixed[model.config.poi_index] = True
    asimov_data = model.expected_data(asimov_init)
    fit_result = pyhf.infer.mle.fit(asimov_data, model, init_pars=asimov_init, fixed_params=asimov_fixed, return_uncertainties=True)[0]


  # 4 - Define the primary dataset
  # ------------------------------

  if options.load_data or options.asimov_fit is None :
    data = workspace.data(model)
  elif options.asimov_fit is not None :
    data = asimov_data
  else:
    raise ValueError('ERROR: could not locate a dataset. None found in the model file and none specified using --asimov')


  # 5 - Identify the model parameters
  # ----------------------------------------------

  class POI : pass
  pois = []
  poi = POI()
  poi.name = model.config.poi_name
  poi.unit = ''
  poi.min_value = par_ranges[model.config.poi_index][0]
  poi.max_value = par_ranges[model.config.poi_index][1]
  poi.initial_value = init_pars[model.config.poi_index]
  pois.append(poi)

  nuis_pars = []
  free_nps = []
  cons_nps = []
  class NuisancePar : pass

  try :
    indices = [ k for k in range(0, len(init_pars)) ]
    for par in model.config.parameters :
      if par == model.config.poi_name : continue # for now pyhf only supports 1 POI 
      par_indices = indices[model.config.par_slice(par)]
      for k in par_indices :
        is_const = fixed_pars[k]
        if fixed_pars[par_indices] : continue # ignore constant NPs
        nuis_par = NuisancePar()
        nuis_par.name = par if len(par_indices) == 1 else '%s_%d' % (par, k)
        nuis_par.unit = ''
        nuis_par.index = k
        nuis_par.is_free = not par in model.config.auxdata_order 
        nuis_par.nominal = fit_result[par.index][0]
        nuis_par.error = fit_result[par.index][1]
        if nuis_par.error <= 0 :
          raise ValueError('Parameter %s has an uncertainty %g which is <= 0' % (nuis_par.name, nuis_par.error))
        if nuis_par.is_free :
          free_nps.append(par)
        else :
          cons_nps.append(par)      
        print('=== Parameter %s [%s]: using deviation %g from nominal value %g for impact computation (x%g)' 
              % ('free       ' if nuis_par.is_free else 'constrained', nuis_par.name, nuis_par.error, nuis_par.nominal, options.epsilon))
      nuis_pars.append(nuis_par)
  except Exception as inst :
    print(inst)
    ValueError('Could not identify nuisance parameters')


  # 5. Fill yield information
  # ---------------------------------------------------------
  
  class Channel : pass
  class Sample : pass
  nominal_rates = {}

  pars = np.array(init_pars)
  if options.verbosity > 0 :
    print('=== Nominal NP values :')
    for par in nuis_pars : print('%s = %g' % par.name, init_pars[par.index])
  channels = {}
  for channel_name in model.config.channels :
    channels[channel_name] = Channel()
    channel = channels[channel_name]
    channel.name = channel_name
    channel.bin_slice = model.config.channel_slices[channel_name]
    channel.nbins = model.config.channel_nbins[channel_name]
    if channel.nbins > 1 :
      channel.type = 'multi_bin' 
      channel.bins = [ 'bin%d' % i for i in range(0, channel.nbins) ] 
    else :
      channel.type = 'single_bin'
    channel.samples = {}
    spec_channel = next(chan for chan in model.spec['channels'] if chan['name'] == channel_name)
    for i, sample_name in enumerate(model.config.samples) :
      channel.samples[sample_name] = Sample()
      sample = channel.samples[sample_name]
      sample.name = sample_name
      sample.index = i
      sample.normpar = None
      spec_sample = next(samp for samp in spec_channel['samples'] if samp['name'] == sample_name)
      for mod in spec_sample['modifiers'] :
        if mod['type'] == 'normfactor' :
          sample.normpar = mod['name']
          sample.nominal_norm = init_pars[model.config.par_slice(sample.normpar)][0]
          break
      if sample.normpar is not None :
        print('=== Sample %s of channel %s normalized using parameter %s = %g' % (sample.name, channel.name, sample.normpar, sample.nominal_norm))
      else :
        print('=== Sample %s of channel %s normalized to 1' % (sample.name, channel.name))      
      sample.impacts = {}
      for par in nuis_pars : sample.impacts[par.name] = []
  nsamples = len(model.config.samples)
  nbins = model.nominal_rates.shape[3]
  for i, sample_name in enumerate(model.config.samples) :
    nominal_rates[sample_name] = model.nominal_rates[0,i,0,:]
    model.nominal_rates[0,i,0,:] = np.zeros(nbins)
  for i, sample_name in enumerate(model.config.samples) :
    model.nominal_rates[0,i,0,:] = nominal_rates[sample_name]
    yields = model.expected_actualdata(pars)
    nominal_yields = {}
    for channel in channels.values() :
      channel.samples[sample_name].nominal_yields = yields[channel.bin_slice]
    for p, par in enumerate(nuis_pars) :
      sys.stderr.write("\rProcessing sample '%s' (%3d of %3d), NP %4d of %4d [%30s]" % (sample_name, sample.index + 1, nsamples, p, len(nuis_pars), par.name[:30]))
      sys.stderr.flush()
      delta = par.error*options.epsilon
      for channel in channels.values() :
        channel.samples[sample_name].impacts[par.name].append({}) # add an entry for this parameter
      for variation in variations :
        pars[par.index] = par.nominal + variation*delta
        yields = model.expected_actualdata(pars)
        pars[par.index] = par.nominal
        for channel in channels.values() :
          var_yields = yields[channel.bin_slice]
          impact = (var_yields/channel.samples[sample_name].nominal_yields)**(1/options.epsilon) - 1 if channel.samples[sample_name].nominal_yields != 0 else 0
          channel.samples[sample_name].impacts[par.name][-1]['%+g' % variation] = impact
        if options.verbosity > 1 : print('-- sample %10s, parameter %-10s : %+g sigma impact = %g' % (sample_name, par.name, variation, impact))
    model.nominal_rates[0,i,0,:] = np.zeros(nbins)
    sys.stderr.write('\n')

  # 7 - Pack the data, if requested
  # ------------------------------

  if options.packing_tolerance is not None :
    for channel in channels.values() :
      for sample in channel.samples.values() :
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
      poi_spec['name'] = poi.name
      poi_spec['unit'] = poi.unit
      poi_spec['min_value'] = poi.min_value
      poi_spec['max_value'] = poi.max_value
      poi_spec['initial_value'] = poi.initial_value      
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
    for channel in channels.values() :
      channel_spec = {}
      channel_spec['name'] = channel.name
      channel_spec['type'] = channel.type
      if channel.type == 'multi_bin' : channel.bins = [ 'bin%d' % i for i in range(0, channel.nbins) ] 
      sample_specs = []
      for sample in channel.samples.values() :
        sample_spec = {}
        sample_spec['name'] = sample.name
        if sample.normpar is not None :
          sample_spec['norm'] = sample.normpar
          sample_spec['nominal_norm'] = sample.nominal_norm
        else :
          sample_spec['norm'] = 1
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
  for channel in channels.values() :
    channel_datum = {}
    channel_datum['name'] = channel.name
    channel_datum['type'] = channel.type
    bin_specs = []
    channel_datum['bins'] = channel.bins
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
