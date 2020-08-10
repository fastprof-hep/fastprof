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
parser.add_argument("-p", "--positions"    , type=str  , default=''  , help="Parameter values to scan over")
parser.add_argument("-i", "--input-pattern", type=str  , default=''  , help="Pattern of result files to load, with the value indicated by a * or a %")
parser.add_argument("-k", "--key"          , type=str  , default=''  , help="Key indexing the output result")
parser.add_argument("-b", "--bands"        , type=int  , default=None, help="Name of JSON file containing full-model fit results")
parser.add_argument("-o", "--output-file"  , type=str  , default=''  , help="Output file name")
parser.add_argument("-r", "--root-output"  , type=str  , default=''  , help="Output a ROOT file with the specified name")
parser.add_argument("-l", "--log-scale"    , action='store_true'     , help="Use log scale for plotting")

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

if options.bands :
  results = {}
  for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
    results[band] = []
else :
  results = []
good_pos = []
i = 0

for pos in positions :
  filename = options.input_pattern.replace('*', str(pos)).replace('%', str(pos))
  try :
    with open(filename, 'r') as fd :
      jdict = json.load(fd)
  except Exception as inst :
    print(inst)
    print('Skipping file %s with missing data' % filename)
    continue
  if options.bands :
    try :
      for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
        res = jdict[options.key + '_%+d' % band]
        if res == None : raise ValueError('Result is None')
      for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
        res = jdict[options.key + '_%+d' % band]
        results[band].append(float(res))
    except Exception as inst :
      print(inst)
      print('Floating-point result not found for band %+d at key %s in file %s. Available keys:\n%s' % (band, options.key, filename, '\n'.join(jdict.keys())))
      continue
  else :
    try :
      res = jdict[options.key]
      if res == None : raise ValueError('Result is None')
      results.append(float(res))
    except Exception as inst :
      print(inst)
      print('Floating-point result not found at key %s in file %s. Available keys:\n%s' % (options.key, filename, '\n'.join(jdict.keys())))
      continue
  good_pos.append(pos)
  
jdict = {}
jdict['positions'] = good_pos
jdict['results'] = results

with open(options.output_file, 'w') as fd:
  json.dump(jdict, fd, ensure_ascii=True, indent=3)

if options.root_output :
  import ROOT
  if options.bands :
    colors = [ 1, 8, 5 ]
    taes = {}
    for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
      for band in range(1, options.bands + 1) :
        tae = ROOT.TGraphAsymmErrors(len(good_pos))
        tae.SetName(options.key + '_%d' % band)
        tae.SetTitle('')
        tae.SetLineWidth(2)
        tae.SetLineColor(1)
        tae.SetLineStyle(2)
        tae.SetFillColor(colors[band])
        taes[band] = tae
  else :
    g = ROOT.TGraph(len(good_pos))
    g.SetName(options.key)
  for i in range(0, len(good_pos)) : 
    if options.bands :
      for band in range(1, options.bands + 1) :
        taes[band].SetPoint(i, good_pos[i], results[0][i])
        taes[band].SetPointEYhigh(i, results[band][i] - results[0][i])
        taes[band].SetPointEYlow (i, results[0][i] - results[-band][i])
    else :
      g.SetPoint(i, good_pos[i], results[i])
  f = ROOT.TFile.Open(options.root_output, 'RECREATE')
  if options.bands :
    for band in range(1, options.bands + 1) :
      taes[band].Write()
  else :
    g.Write()
  f.Close()
# In ROOT, draw as follows
# limit_sampling_CLs_expected_band_2->Draw("AL4")
# limit_sampling_CLs_expected_band_1->Draw("L4SAME")
# limit_sampling_CLs_expected_band_1->Draw("LX")

plt.ion()
if options.log_scale : plt.yscale('log')

if options.bands :
  colors = [ 'k', 'g', 'y', 'c', 'b' ]
  for i in reversed(range(1, options.bands + 1)) :
    plt.fill_between(good_pos, results[+i], results[-i], color=colors[i])
    plt.plot(good_pos, results[0], 'k--')
else :
  plt.plot(good_pos, results, 'b')
plt.show()
