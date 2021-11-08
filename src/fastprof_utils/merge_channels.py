#! /usr/bin/env python

__doc__ = """
*Merge model channels*

"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import json

from fastprof import Model, Data, ChannelMerger


####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("merge_channels.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"    , type=str  , required=True    , help="Name of markup file defining model")
  parser.add_argument("-d", "--data-file"     , type=str  , default=None     , help="Use the dataset stored in the specified markup file")
  parser.add_argument("-o", "--output-file"   , type=str  , required=True    , help="Name of output file")
  parser.add_argument("-c", "--channels"      , type=str  , required=True    , help="Merging specification, as new_channel1=channel1(range),channel2(range):...")
  parser.add_argument("-v", "--verbosity"     , type=int  , default=0        , help="Verbosity level")
  parser.add_argument("-n", "--obs-name"      , type=str  , default=None     , help="Name of observable in meged channel")
  parser.add_argument("-u", "--obs-unit"      , type=str  , default=None     , help="Unit of observable in meged channel")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args(argv)
  if not options :
    parser.print_help()
    sys.exit(0)
  if options.verbosity >= 1 : print('Loading model from file %s' % options.model_file)
  model = Model.create(options.model_file, verbosity=options.verbosity)
  if model is None : raise ValueError('No valid model definition found in file %s.' % options.model_file)

  if options.data_file :
    data = Data(model).load(options.data_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    if options.verbosity >= 1 : print('Using dataset stored in file %s.' % options.data_file)

  merge_specs = []
  try:
    merges = options.channels.replace(' ', '').replace('\n', '').split('~')
    for merge in merges :
      split1 = merge.split('=')
      name = split1[0]
      split2 = split1[1].split(',')
      merged_channels = []
      ranges = []
      for channel in split2 :
        specs = channel.split(':')
        if len(specs) != 1 and len(specs) != 3 :
          raise ValueError("ERROR: Could not parse channel specification '%s'." % channel)
        merged_channels.append(specs[0])
        if len(specs) == 3 : ranges.append((float(specs[1]), float(specs[2]))) 
      if len(ranges) == 0 :
        ranges = None
      elif len(ranges) != len(merged_channels) :
        raise ValueError('ERROR: Should specify ranges for all channels or none at all, here got ranges for %d of %d.' % (len(ranges), len(merged_channels)))
      merge_specs.append({ 'name' : name, 'merged_channels' : merged_channels, 'ranges' : ranges })
  except Exception as inst :
    print(inst)
    raise ValueError('Could not parse merged channel specification')

  current_model = model
  current_data = data
  for spec in merge_specs :
     merger = ChannelMerger(current_model, spec['merged_channels'], spec['name'], spec['ranges'], options.obs_name, options.obs_unit)
     new_model = merger.merge()
     if current_data is not None : current_data = merger.merge_data(current_data, new_model)
     current_model = new_model

  current_model.save('%s.json' % options.output_file)
  if current_data is not None : current_data.save('%s_data.json' % options.output_file)


if __name__ == '__main__' : run()
