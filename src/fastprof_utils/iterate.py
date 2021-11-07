#! /usr/bin/env python

__doc__ = "Produce scripts iterating over parameter values"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import copy
import json
import math
import numpy as np
import matplotlib.pyplot as plt

####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("iterate.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-p", "--positions"  , type=str  , default='', help="Parameter values to scan over")
  parser.add_argument("-c", "--cmd-pattern", type=str  , default='', help="command to enter, with the value indicated by a * or a %% sign")
  parser.add_argument("-o", "--output-file", type=str  , default='', help="Output file name (leave blank for stdout)")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args()
  if not options :
    parser.print_help()
    sys.exit(0)
  
  try:
    pos_spec = options.positions.split(':')
    if len(pos_spec) == 3 :
      positions = np.linspace(float(pos_spec[0]), float(pos_spec[1]), int(pos_spec[2]))
    elif len(pos_spec) == 4 and pos_spec[3] == 'int' :
      positions = np.linspace(float(pos_spec[0]), float(pos_spec[1]), int(pos_spec[2]))
      positions = [ math.floor(pos) for pos in positions ]
    elif len(pos_spec) == 4 and pos_spec[3] == 'log' :
      positions = np.logspace(1, math.log(float(pos_spec[1]))/math.log(float(pos_spec[0])), int(pos_spec[2]) + 1, True, float(pos_spec[0]))
    elif len(pos_spec) == 4 and pos_spec[3] == 'logint' :
      positions = np.logspace(1, math.log(float(pos_spec[1]))/math.log(float(pos_spec[0])), int(pos_spec[2]) + 1, True, float(pos_spec[0]))
      positions = [ math.floor((positions[i] + positions[i+1] + 1)/2) for i in range(0, len(positions) - 1) ]
    else :
      positions = options.positions.split(',')
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid value specification %s : the format should be either vmin:vmax:nvals or v1,v2,...' % options.positions)
  
  #print('positions:', positions)
  result = ''
  for pos in positions :
    str_pos = str(pos)
    command = options.cmd_pattern.replace('*', str_pos).replace('%', str_pos).replace('\\n', '\n')
    result += command + '\n'
  
  if options.output_file != '' :
    with open(options.output_file, 'w') as fd:
      fd.write(result)
  else :
    print(result)

if __name__ == '__main__' : run()
