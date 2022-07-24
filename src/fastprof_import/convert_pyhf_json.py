#! /usr/bin/env python

__doc__ = """
*Convert a pyhf json file into fastprof markup format*

The script takes as input a pyhf JSON file, and converts the contents
into the definition file of a linear model.
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

####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("convert_ws.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  #parser.add_argument("-f", "--input-file"       , type=str  , required=True    , help="Name of file containing the pyhf model")
  parser.add_argument("input_files"              , type=str  , nargs=1          , help="Name of file containing the pyhf model")
  parser.add_argument("-o", "--output-file"      , type=str  , required=True    , help="Name of output file")
  parser.add_argument("-m", "--measurement"      , type=int  , default=0        , help="Index of measurement config to use [default: 0, the first one]")
  parser.add_argument(      "--markup"           , type=str  , default='json'   , help="Output markup flavor (supported : 'json', 'yaml')")
  parser.add_argument("-k", "--keep-same-sign"   , action="store_true"          , help="Keep same-sign impacts")
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

  try :
    with open(options.input_files[0]) as fd :
      pyhf_model = json.load(fd)
  except Exception as inst :
    print(inst)
    raise ValueError("Could not load input file '%s', exiting.")
    sys.exit(1)

  fastprof_model = {}
  fastprof_data = {}

  fastprof_model['POIs'] = {}
  fastprof_model['NPs'] = {}
  #fastprof_model['aux_obs'] = {}
  fastprof_model['channels'] = []

  central_value = {}

  try :
    measurement = pyhf_model['measurements'][options.measurement]
    config = measurement['config']
  except Exception as inst :
    print(inst)
    raise("Valid measurement not found in model at index %d" % options.measurement)
  try :
    fastprof_model['POIs'][config['poi']] = { 'name' : config['poi'], 'min_value' : 0, 'max_value' : 1 } # POI range is not specified in pyhf ?
  except Exception as inst :
    print(inst)
    raise KeyError('ERROR: could not read parameter of interest information from measurements section.')
  try :
    for par_spec in config['parameters'] :
      if par_spec['name'] in fastprof_model['POIs'] : continue
      constraint = par_spec['sigmas'][0] if 'sigmas' in par_spec else None
      variation = constraint if constraint is not None else 1
      nominal_value = 0
      if 'aux_data' in par_spec :
        nominal_value = par_spec['aux_data'][0] 
      elif 'inits' in par_spec :
        nominal_value = par_spec['inits'][0] 
      np_spec = { 'name' : par_spec['name'], 'nominal_value' : nominal_value }
      if constraint is not None :
        np_spec['variation'] = variation
        np_spec['constraint'] = constraint
        np_spec['aux_obs'] = 'aux_' + par_spec['name']
      fastprof_model['NPs'][par_spec['name']] = np_spec
  except Exception as inst :
    print(inst)
    raise KeyError('ERROR: could not read nuisance parameters information from measurements section.')
  
  if not 'channels' in pyhf_model :
    raise KeyError('ERROR: could not read channel information from model.')

  pyhf_meas = { entry['name'] : entry for entry in pyhf_model['measurements'][0]['config']['parameters'] }

  def add_impact(sample, modifier_name, impact) :
    if not modifier_name in fastprof_model['NPs'] and not modifier_name in fastprof_model['POIs'] :
      fastprof_model['NPs'][modifier_name] = { 'name' : modifier_name, 'nominal_value' : 0, 'constraint' : 1, 'aux_obs' : 'aux_' + modifier_name }
    if np.any([ imp["+1"]*imp["-1"] > 0 for imp in impact ]) : 
      if options.keep_same_sign :
        impact = [ { "+1" : imp["+1"], "-1" : -imp["-1"] } if abs(imp["+1"]) > abs(imp["-1"]) else { "+1" : -imp["+1"], "-1" : imp["-1"] } for imp in impact ]
      else :
        print("WARNING: +1σ and -1σ impacts for '%s' in sample '%s' of channel '%s' (%s) have the same sign, setting to default." % (modifier_name, sample['name'], channel['name'], str(impact)))
        impact = [ { "+1" : 0.0, "-1" : 0.0 } ]
    if len(impact) == 1 : impact = impact[0]
    if modifier_name in sample['impacts'] :
      prev_impact = sample['impacts'][modifier_name]
      if isinstance(prev_impact, dict) and not isinstance(impact, dict) : prev_impact = [ prev_impact ] * len(impact)
      if isinstance(impact, dict) and not isinstance(prev_impact, dict) : impact = [ impact ] * len(impact)
      if isinstance(prev_impact, dict) :
        impact = { "+1" : prev_impact["+1"] + impact["+1"], "-1" : prev_impact["-1"] + impact["-1"] }
      else :
        impact = [ { "+1" : pi["+1"] + ci["+1"], "-1" : pi["-1"] + ci["-1"] } for pi, ci in zip(prev_impact, impact) ]
    sample['impacts'][modifier_name] = impact


  for pyhf_channel in pyhf_model['channels'] :
    channel = {}
    try :
      channel['name'] = pyhf_channel['name']
      nbins = len(pyhf_channel['samples'][0]['data'])
      if nbins == 1 :
        channel['type'] = 'bin'
      else :
        channel['type'] = 'multi_bin'
        channel['bins'] = []
      channel['samples'] = []
      for pyhf_sample in pyhf_channel['samples'] :
        if options.verbosity > 2 : print('** Channel %s, sample %s' % (pyhf_channel['name'], pyhf_sample['name']))
        sample = {}
        nominal_yields = pyhf_sample['data']
        if channel['type'] == 'multi_bin' and len(channel['bins']) == 0 : channel['bins'] = [ 'bin%d' % b for b in range(0, len(nominal_yields)) ]
        sample['name'] = pyhf_sample['name']
        sample['nominal_yields'] = nominal_yields
        normfactors = [ modifier['name'] for modifier in pyhf_sample['modifiers'] if modifier['type'] == 'normfactor' ]
        if len(normfactors) > 1 : raise("Sample '%s' has %d normfactors, can only handle one -- exiting." % (sample['name'], len(normfactors)))
        if len(normfactors) == 1 :
          #sample['norm_type'] = 'parameter'
          sample['nominal_norm'] = 1 # FIXME: put the nominal POI or NP value here instead
          sample['norm'] = normfactors[0]
        sample['impacts'] = {} 
        for modifier in pyhf_sample['modifiers'] :
          if modifier['name'] in normfactors : continue
          if modifier['type'] == 'lumi' :
            lumi_impact = pyhf_meas[modifier['name']]['sigmas'][0]
            impact = [ { "+1" : lumi_impact, "-1" : -lumi_impact } ]
            if options.verbosity > 2 : print('  Parameter %s of type %s : lumi_impact = %g' % (modifier['name'], modifier['type'], lumi_impact))
            add_impact(sample, modifier['name'], impact)
          elif modifier['type'] == 'normsys' :
            impact = [ { "+1" : modifier['data']['hi'] - 1, "-1" : modifier['data']['lo'] - 1 } ]
            if options.verbosity > 2 : print('  Parameter %s of type %s : impact_up = %s' % (modifier['name'], modifier['type'],
                                                                                                     str(modifier['data']['hi'])))
            add_impact(sample, modifier['name'], impact)
          elif modifier['type'] == 'histosys' :
            impact = [ { "+1" : hi/y - 1, "-1" : lo/y - 1 } for y, hi, lo in zip(nominal_yields, modifier['data']['hi_data'], modifier['data']['lo_data']) ]
            if options.verbosity > 2 : print('  Parameter %s of type %s : impact_up = %s / %s = %s' % (modifier['name'], modifier['type'],
                                                                                                     str(modifier['data']['hi_data']),
                                                                                                     str(nominal_yields), str(impact)))
            add_impact(sample, modifier['name'], impact)
          elif modifier['type'] == 'shapesys' or modifier['type'] == 'staterror' :
            for b, (y, sigma) in enumerate(zip(nominal_yields, modifier['data'])) :
              impact = [ { "+1" : sigma/y if b == b2 else 0, "-1" : -sigma/y if b == b2 else 0 } for b2 in range(0, len(nominal_yields)) ]
              if options.verbosity > 2 : print('  Parameter %s of type %s : impact = %s / %s = %s' % (('%s_bin%d' % (modifier['name'], b)), modifier['type'],
                                                                                                      str(modifier['data']), str(nominal_yields), str(impact)))
              add_impact(sample, '%s_bin%d' % (modifier['name'], b), impact)
          else :
            raise ValueError("Unknown modifier type '%s' in channel '%s'." % (modifier['type'], pyhf_channel['name']))
          if options.verbosity > 3 : print(sample)
        channel['samples'].append(sample)
    except Exception as inst :
      print(inst)
      raise KeyError("ERROR: could not read model information for channel '%s.'" % pyhf_channel['name'])
    fastprof_model['channels'].append(channel)
 
  if 'observations' in pyhf_model :
    fastprof_data['channels'] = []
    for pyhf_channel in pyhf_model['observations'] :
      try :
        if len(pyhf_channel['data']) == 1 :
          fastprof_data['channels'].append({'name' : pyhf_channel['name'],
                                            'counts' : pyhf_channel['data'][0] })
        else :
          fastprof_data['channels'].append({'name' : pyhf_channel['name'],
                                            'bins' : [ { 'name' : 'bin%d' % b, 'counts' : c } for b, c in enumerate(pyhf_channel['data']) ]})
      except Exception as inst :
        print(inst)
        raise KeyError("ERROR: could not read data information for channel '%s.'" % pyhf_channel['name'])
    fastprof_data['aux_obs'] = []
    for par in fastprof_model['NPs'].values() :
      if 'aux_obs' in par : fastprof_data['aux_obs'].append({ 'name' : par['aux_obs'], 'value' : par['nominal_value'] })

  with open(options.output_file, 'w') as fd :
    sdict = {}
    fastprof_model['POIs'] = list(fastprof_model['POIs'].values())
    fastprof_model['NPs'] = list(fastprof_model['NPs'].values())
    fastprof_model['channels'] = fastprof_model['channels']
    sdict['model'] = fastprof_model
    if len(fastprof_data) > 0 :  sdict['data'] = fastprof_data
    if options.markup == 'json' : return json.dump(sdict, fd, ensure_ascii=True, indent=3)
    if options.markup == 'yaml' : return yaml.dump(sdict, fd, sort_keys=False, default_flow_style=None, width=10000)
    raise KeyError("Unknown markup flavor '%s',  so far only 'json' or 'yaml' are supported" % options.markup)

if __name__ == '__main__' : run()
