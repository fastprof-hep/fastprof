#! /usr/bin/env python

__doc__ = "Ccompute limits from sampling distributions"
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

parser = ArgumentParser("compute_limits.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument("-p", "--positions"    , type=str  , default='', help="Parameter values to scan over")
parser.add_argument("-i", "--input-pattern", type=str  , default='', help="Pattern of result files to load, with the value indicated by a * or a %")
parser.add_argument("-k", "--key"          , type=str  , default='', help="Key indexing the output result")
parser.add_argument("-o", "--output-file"  , type=str  , default='', help="Output file name")
parser.add_argument("-r", "--root-output"  , type=str  , default='', help="Output a ROOT file with the specified name")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

try:
  pos_spec = options.positions.split(':')
  if len(pos_spec) == 3 : 
    positions = np.linspace(float(pos_spec[0]), float(pos_spec[1]), int(pos_spec[2]))
  elif len(pos_spec) == 3 and pos_spec[3] == 'int' : 
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

print('positions:', positions)
results = []
for pos in positions :
  filename = options.input_pattern.replace('*', str(pos)).replace('%', str(pos))
  try :
    with open(filename, 'r') as fd :
      jdict = json.load(fd)
  except Exception as inst :
    print(inst)
    print('Skipping file %s with missing data' % filename)
  try :
    res = jdict[options.key]
    results.append(float(res) if res != None else None)
  except Exception as inst :
    print(inst)
    raise ValueError('Floating-point result not found at key %s in file %s' % (options.key, filename))

jdict = {}
jdict['positions'] = positions
jdict['results'] = results

with open(options.output_file, 'w') as fd:
  json.dump(jdict, fd, ensure_ascii=True, indent=3)

if options.root_output :
  import ROOT
  g = ROOT.TGraph(len(positions))
  g.SetName(options.key)
  for i in range(0, len(positions)) : g.SetPoint(i, positions[i], results[i] if results[i] != None else 0)
  f = ROOT.TFile.Open(options.root_output, 'RECREATE')
  g.Write()
  f.Close()

plt.ion()
plt.plot(positions, results, 'b')
plt.show()
