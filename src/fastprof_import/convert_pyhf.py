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
      nominal_value = 0
      if 'aux_data' in par_spec :
        nominal_value = par_spec['aux_data'][0] 
      elif 'inits' in par_spec :
        nominal_value = par_spec['inits'][0] 
      np_spec = { 'name' : par_spec['name'], 'nominal_value' : nominal_value }
      if constraint is not None :
        np_spec['constraint'] = constraint
        np_spec['aux_obs'] = 'aux_' + par_spec['name']
      fastprof_model['NPs'][par_spec['name']] = np_spec
  except Exception as inst :
    print(inst)
    raise KeyError('ERROR: could not read nuisance parameters information from measurements section.')
  
  if not 'channels' in pyhf_model :
    raise KeyError('ERROR: could not read channel information from model.')

  for pyhf_channel in pyhf_model['channels'] :
    channel = {}
    try :
      channel['name'] = pyhf_channel['name']
      nbins = len(pyhf_channel['samples'][0]['data'])
      if nbins == 1 :
        channel['type'] = 'bin'
        channel['samples'] = []
        for pyhf_sample in pyhf_channel['samples'] :
          sample = {}
          nominal_yields = pyhf_sample['data'][0]
          sample['name'] = pyhf_sample['name']
          sample['nominal_yields'] = [ nominal_yields ]
          normfactors = [ modifier['name'] for modifier in pyhf_sample['modifiers'] if modifier['type'] == 'normfactor' ]
          if len(normfactors) > 1 : raise("Sample '%s' has %d normfactors, can only handle one -- exiting." % (sample['name'], len(normfactors)))
          if len(normfactors) == 1 :
            #sample['norm_type'] = 'parameter'
            sample['nominal_norm'] = 1 # FIXME: put the nominal POI or NP value here inste
            sample['norm'] = normfactors[0]
          #else :
          #  sample['norm_type'] = 'number'
          #  sample['norm'] = 1
          sample['impacts'] = {} 
          for modifier in pyhf_sample['modifiers'] :
            if not modifier['name'] in fastprof_model['NPs'] and not modifier['name'] in fastprof_model['POIs'] :
              fastprof_model['NPs'][modifier['name']] = { 'name' : modifier['name'], 'nominal_value' : 0, 'constraint' : 1, 'aux_obs' : 'aux_' + modifier['name'] }
            if modifier['name'] in normfactors : continue
            if modifier['type'] == 'lumi' :
              #impact = { "+1" : +1, "-1" : -1 }
              impact = { "+1" : 1.0, "-1" : -1.0 }
            elif modifier['type'] == 'normsys' :
              impact = { "+1" : modifier['data']['hi'] - 1, "-1" : modifier['data']['lo'] - 1 }
            elif modifier['type'] == 'histosys' :
              impact = { "+1" : modifier['data']['hi_data'][0]/nominal_yields - 1, "-1" : modifier['data']['lo_data'][0]/nominal_yields - 1 }
            elif modifier['type'] == 'shapesys' :
              sigma = modifier['data'][0]/nominal_yields
              if sigma > 0 :
                fastprof_model['NPs'][modifier['name']]['nominal_value'] = 1
                fastprof_model['NPs'][modifier['name']]['constraint'] = sigma
                impact = { "+1" : 1.0, "-1" : -1.0 }
              else :
                del fastprof_model['NPs'][modifier['name']]
            else :
              raise ValueError("Unknown modifier type '%s' in channel '%s'." % (modifier['type'], pyhf_channel['name']))
            if impact["+1"]*impact["-1"] > 0 : 
              print("WARNING: +1 sigma and -1 sigma impacts for '%s' in sample '%s' of channel '%s' (%g, %g) have the same sign, setting to default." % (modifier['name'], sample['name'], channel['name'], impact["+1"], impact["-1"]))
              #impact = { "+1" : 0.0, "-1" : 0.0 }
            sample['impacts'][modifier['name']] = impact
          channel['samples'].append(sample)
      else :
        raise TypeError("Conversion of multi-bin channels is not yet implemented, cannot handle channel '%s' with %d bins." % (pyhf_channel['name'], nbins))
        # TODO: implement fully multi-bin channels, and treat single-bin case as a special case (do everything the multi-bin way, then simplify to single-bin if nbins=1) 
        #for par in fastprof_model['NPs'] :
        #  if 'aux_obs' in par : 
    except Exception as inst :
      print(inst)
      raise KeyError("ERROR: could not read model information for channel '%s.'" % pyhf_channel['name'])
    fastprof_model['channels'].append(channel)
 
  if 'observations' in pyhf_model :
    fastprof_data['channels'] = []
    for pyhf_channel in pyhf_model['observations'] :
      try :
        fastprof_data['channels'].append({ 'name' : pyhf_channel['name'], 'counts' : pyhf_channel['data'][0] })
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
