#! /usr/bin/env python

__doc__ = """
*Collect results from different model points and plot them together*

Results computed at specified model points, stored in output JSON files,
are collected together to make plots as a function of model parameters.

For now only one model parameter is considered. Parameter values are specified
using the `--positions` argument, in the format `min:max:bins[:[log]int]`. If the
last argument is omitted, a linear grid with equal spacing between `min` and
`max` is used. If the last argument contains `log`, the grid is prepared in log
scale, and if it containts `int`, values are rounded to the nearest integer.
The format of the results files to look for is given by `--input_pattern`. The
symbols `*` or `%` in the pattern are replaced by position values.

In each file, the result labeled by `--key` is extracted, along with its
variation bands if `--bands` is specified. If `--errors` is specified, 
statistical uncertainties in the results are included if they are available
in the input results files.

The results are plotted and written to the JSON file specfied by `--output-file`.
If `--root-output` is also provided, results are formatted as `TGraph` or
`TGraphAsymError` ROOT objects and written to the specified ROOT file.
"""

# In ROOT, draw the output as follows
#
# - Simple
# limit_sampling_CLs->Draw("CA")
#
# - With errors
# limit_sampling_CLs->Draw("A4")
# limit_sampling_CLs->Draw("LX")
#
# - Bands
# limit_sampling_CLs_expected_band_2->Draw("A4")
# limit_sampling_CLs_expected_band_1->Draw("L4SAME")
# limit_sampling_CLs_expected_band_1->Draw("LX")


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
  parser = ArgumentParser("compute_limits.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-p", "--positions"    , type=str  , default=''  , help="Parameter values to scan over")
  parser.add_argument("-i", "--input-pattern", type=str  , default=''  , help="Pattern of result files to load, with the value indicated by a * or a %")
  parser.add_argument("-k", "--key"          , type=str  , default=''  , help="Key indexing the output result")
  parser.add_argument("-b", "--bands"        , type=int  , default=None, help="Number of expected limit bands to include")
  parser.add_argument("-e", "--errors"       , action='store_true'     , help="Include sampling uncertainties on the limit values")
  parser.add_argument("-o", "--output-file"  , type=str  , default=''  , help="Output file name")
  parser.add_argument("-r", "--root-output"  , type=str  , default=''  , help="Output a ROOT file with the specified name")
  parser.add_argument("-l", "--log-scale"    , action='store_true'     , help="Use log scale for plotting")
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
          res = jdict[options.key + '_expected_band_%+d' % band]
          if res == None : raise ValueError('Result is None')
        for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
          res = jdict[options.key + '_expected_band_%+d' % band]
          results[band].append(float(res))
      except Exception as inst :
        print(inst)
        print('Floating-point result not found for band %+d at key %s in file %s. Available keys:\n%s' % (band, options.key, filename, '\n'.join(jdict.keys())))
        continue
    else :
      try :
        res = jdict[options.key]
        if res == None :
          print('Result at position %g is None, skipping' % pos)
          continue
      except Exception as inst :
        print(inst)
        print('Floating-point result not found at key %s in file %s. Available keys:\n%s' % (options.key, filename, '\n'.join(jdict.keys())))
        continue
      if options.errors :
        try :
          resup = jdict[options.key + '_up']
          resdn = jdict[options.key + '_dn']
          if resup == None :
            print('Positive error at position %g is None, skipping' % pos)
            continue
          if resdn == None :
            print('Negative error at position %g is None, skipping' % pos)
            continue
        except Exception as inst :
          print(inst)
          print('Positive and negative errors on result not found at key %s in file %s. Available keys:\n%s' % (options.key, filename, '\n'.join(jdict.keys())))
          continue
        results.append([ float(res), float(resup), float(resdn) ])
      else :
        results.append(float(res))
    good_pos.append(pos)
    
  jdict = {}
  jdict['positions'] = good_pos
  jdict['results'] = results
  
  with open(options.output_file, 'w') as fd:
    json.dump(jdict, fd, ensure_ascii=True, indent=3)
  
  if options.root_output :
    import ROOT
    colors = [ 1, 8, 5 ]
    if options.bands :
      gs = {}
      for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
        for band in range(1, options.bands + 1) :
          g = ROOT.TGraphAsymmErrors(len(good_pos))
          g.SetName(options.key + '_expected_band_%d' % band)
          g.SetTitle('')
          g.SetLineWidth(2)
          g.SetLineColor(1)
          g.SetLineStyle(2)
          g.SetFillColor(colors[band])
          gs[band] = g
    elif options.errors :
      g = ROOT.TGraphAsymmErrors(len(good_pos))
      g.SetName(options.key)
      g.SetLineWidth(2)
      g.SetLineColor(1)
      g.SetFillColor(colors[1])
    else :
      g = ROOT.TGraph(len(good_pos))
      g.SetName(options.key)
    for i in range(0, len(good_pos)) : 
      if options.bands :
        for band in range(1, options.bands + 1) :
          gs[band].SetPoint(i, good_pos[i], results[0][i])
          gs[band].SetPointEYhigh(i, results[band][i] - results[0][i])
          gs[band].SetPointEYlow (i, results[0][i] - results[-band][i])
      elif options.errors :
        g.SetPoint(i, good_pos[i], results[i][0])
        g.SetPointEYhigh(i, results[i][1] - results[i][0])
        g.SetPointEYlow (i, results[i][0] - results[i][2])
      else :
        g.SetPoint(i, good_pos[i], results[i])
    f = ROOT.TFile.Open(options.root_output, 'RECREATE')
    if options.bands :
      for band in range(1, options.bands + 1) :
        gs[band].Write()
    else :
      g.Write()
    f.Close()
  
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


if __name__ == '__main__' : run()
