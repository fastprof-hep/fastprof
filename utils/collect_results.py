#! /usr/bin/env python

__doc__ = """
*Collect results from different model points and plot them together*

Results computed at specified model points, stored in output markup files,
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

The results are plotted and written to the markup file specfied by `--output-file`.
If `--root-output` is also provided, results are formatted as `TGraph` or
`TGraphAsymError` ROOT objects and written to the specified ROOT file.

In ROOT, the output can then be drawn as follows:

- *Simple*
  limit_sampling_CLs->Draw("CA")

- *With errors*
  limit_sampling_CLs->Draw("A4")
  limit_sampling_CLs->Draw("LX")

- *With bands*
  limit_sampling_CLs_expected_band_2->Draw("A4")
  limit_sampling_CLs_expected_band_1->Draw("L4SAME")
  limit_sampling_CLs_expected_band_1->Draw("LX")
"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

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
  parser = ArgumentParser("collect_results.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-p", "--positions"    , type=str  , default=''  , help="Parameter values to scan over")
  parser.add_argument("-i", "--input-pattern", type=str  , default=''  , help="Pattern of result files to load, with the value indicated by a * or a %%")
  parser.add_argument("-k", "--key"          , type=str  , default=''  , help="Key indexing the output result")
  parser.add_argument("-v", "--scan-var"     , type=str  , default=None, help="Name of the scanned variable")
  parser.add_argument("-u", "--scan-unit"    , type=str  , default=None, help="Unit of the scanned variable")
  parser.add_argument("-b", "--bands"        , type=int  , default=None, help="Number of expected limit bands to include")
  parser.add_argument("-e", "--errors"       , action='store_true'     , help="Include sampling uncertainties on the limit values")
  parser.add_argument("-o", "--output-file"  , type=str  , default=None, help="Output file name")
  parser.add_argument("-r", "--root-output"  , type=str  , default=None, help="Output a ROOT file with the specified name")
  parser.add_argument("-l", "--log-scale"    , action='store_true'     , help="Use log scale for plotting")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args()
  if not options :
    parser.print_help()
    return

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

  keys = options.key.split(',')

  results = {}
  bands = {}
  good_pos = {}
  good_pos_bands = {}
  
  poi_name = None
  poi_unit = None
  
  for k, key in enumerate(keys) :
    if options.bands and k == 0 :
      bands[key] = {}
      good_pos_bands[key] = []
      for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
        bands[key][band] = []
    results[key] = []
    good_pos[key] = []
    for pos in positions :
      filename = options.input_pattern.replace('*', str(pos)).replace('%', str(pos))
      try :
        with open(filename, 'r') as fd :
          jdict = json.load(fd)
      except Exception as inst :
        print(inst)
        print('Skipping file %s with missing data' % filename)
        continue
      if poi_name == None :
        try :
          poi_name = jdict['poi_name']
          poi_unit = jdict['poi_unit']
        except Exception as inst :
          poi_name = ''
          poi_unit = ''
      if options.bands and k == 0 :
        try :
          for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
            res = jdict[key + '_expected_band_%+d' % band]
            if res == None : raise ValueError('Result is None')
          for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
            res = jdict[key + '_expected_band_%+d' % band]
            bands[key][band].append(float(res))
        except Exception as inst :
          print(inst)
          print('Floating-point result not found for band %+d at key %s in file %s. Available keys:\n%s' % (band, key, filename, '\n'.join(jdict.keys())))
          continue
        good_pos_bands[key].append(pos)
      try :
        res = jdict[key]
        if res == None :
          print('Result at position %g is None, skipping' % pos)
          continue
      except Exception as inst :
        print(inst)
        print('Floating-point result not found at key %s in file %s. Available keys:\n%s' % (key, filename, '\n'.join(jdict.keys())))
        continue
      if options.errors :
        try :
          resup = jdict[key + '_up']
          resdn = jdict[key + '_dn']
          if resup == None :
            print('Positive error at position %g is None, skipping' % pos)
            continue
          if resdn == None :
            print('Negative error at position %g is None, skipping' % pos)
            continue
        except Exception as inst :
          print(inst)
          print('Positive and negative errors on result not found at key %s in file %s. Available keys:\n%s' % (key, filename, '\n'.join(jdict.keys())))
          continue
        results[key].append([ float(res), float(resup), float(resdn) ])
      else :
        results[key].append(float(res))
      good_pos[key].append(pos)

  jdict = {}
  for k, key in enumerate(keys) :
    key_jdict = {}
    key_jdict['positions'] = good_pos[key]
    key_jdict['results'] = results[key]
    if options.bands and k == 0 : key_jdict['bands'] = bands[key]
    jdict[key] = key_jdict

  if options.output_file is not None :
    with open(options.output_file, 'w') as fd:
      json.dump(jdict, fd, ensure_ascii=True, indent=3)
  else :
    print(jdict)

  if options.root_output is not None :
    import ROOT
    colors = [ 1, 8, 5 ]
    gs = {}
    g = {}
    xlabel = options.scan_var + (' ['  + options.scan_unit + ']') if options.scan_unit != '' else ''
    ylabel = poi_name + (' ['  + poi_unit + ']') if poi_unit != '' else ''
    for k, key in enumerate(keys) :
      if options.bands and k == 0 :
        gs[key] = {}
        for band in np.linspace(-options.bands, options.bands, 2*options.bands + 1) :
          for band in range(1, options.bands + 1) :
            g[key] = ROOT.TGraphAsymmErrors(len(good_pos[key]))
            g[key].SetName(key + '_expected_band_%d' % band)
            g[key].SetTitle('')
            g[key].SetLineWidth(2)
            g[key].SetLineColor(1)
            g[key].SetLineStyle(2)
            g[key].SetFillColor(colors[band])
            g[key].GetXaxis().SetTitle(xlabel)
            g[key].GetYaxis().SetTitle(ylabel)
            gs[key][band] = g[key]
      if options.errors :
        g[key] = ROOT.TGraphAsymmErrors(len(good_pos[key]))
        g[key].SetName(key)
        g[key].SetLineWidth(2)
        g[key].SetLineColor(1)
        g[key].SetFillColor(colors[1])
        g[key].GetXaxis().SetTitle(xlabel)
        g[key].GetYaxis().SetTitle(ylabel)
      else :
        g[key] = ROOT.TGraph(len(good_pos[key]))
        g[key].SetName(key)
        g[key].GetXaxis().SetTitle(xlabel)
        g[key].GetYaxis().SetTitle(ylabel)
      for i in range(0, len(good_pos[key])) :
        if options.bands and k == 0 :
          for band in range(1, options.bands + 1) :
            gs[key][band].SetPoint(i, good_pos[key][i], bands[key][0][i])
            gs[key][band].SetPointEYhigh(i, bands[key][band][i] - bands[key][0][i])
            gs[key][band].SetPointEYlow (i, bands[key][0][i] - bands[key][-band][i])
        if options.errors :
          g[key].SetPoint(i, good_pos[key][i], results[key][i][0])
          g[key].SetPointEYhigh(i, results[key][i][1] - results[key][i][0])
          g[key].SetPointEYlow (i, results[key][i][0] - results[key][i][2])
        else :
          g[key].SetPoint(i, good_pos[key][i], results[key][i])
    f = ROOT.TFile.Open(options.root_output, 'RECREATE')
    for k, key in enumerate(keys) :
      if options.bands and k == 0 :
        for band in range(1, options.bands + 1) :
          gs[key][band].Write()
      g[key].Write()
    f.Close()

  plt.ion()
  if options.log_scale : plt.yscale('log')

  items = []
  for k, key in enumerate(keys) :
    if options.bands is not None and k == 0 :
      band_colors = [ 'k', 'g', 'y', 'c', 'b' ]
      for i in reversed(range(1, options.bands + 1)) :
        plt.fill_between(good_pos_bands[key], bands[key][+i], bands[key][-i], color=band_colors[i], label=key + ' +- %dsigma band' % i)
        items.append(key + ' +- %dsigma band' % i)
      plt.plot(good_pos_bands[key], bands[key][0], 'k--', label=key + ' expected')
      items.append(key + ' expected')
    line_colors = [ 'b', 'r', 'g', 'k', 'y', 'c' ]
    plt.plot(good_pos[key], results[key], color=line_colors[k], label=key)
    items.append(key)
  handles, labels = plt.gca().get_legend_handles_labels()
  label_handles = { l : h for l,h in zip(labels, handles) }
  plt.legend([ label_handles[l] for l in items ], items)
  plt.gca().set_xlabel('$' + options.scan_var + '$' + ((' ['  + options.scan_unit + ']') if options.scan_unit != '' else ''))
  plt.gca().set_ylabel('$' + poi_name + '$' + ((' ['  + poi_unit + ']') if poi_unit != '' else ''))
  plt.show()
  if options.output_file is not None :
    split_name = os.path.splitext(options.output_file)
    plt.savefig(split_name[0] + '.png')

if __name__ == '__main__' : run()
