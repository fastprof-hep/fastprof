#! /usr/bin/env python

__doc__ = """
*Plot model validation data*

Produces validation plots for each NP, in which
validation data extracted from the full model, is compared
with predictions from the fast model.

The validation data is produced by the `convert_ws.py` script, using
the `--validation-data` option. It consists in a event yield
variations for a range of values of each NP.

If the linear model is a good approximation to the full model, it
should reproduce these variations to good precision. If this is not
the case, the plots can be used to set a range of NP values for which
the model is valid, which can be used for instance to define the
`--bounds` argument to `compute_limits.py`.

By default two linear model preductions are shown: one for log-normal
NP impact, which is the default and usually provides the best
prediction; and one fo linear NP impact.

Additional plots can also be shown to gauge the precision
of approximating 1/(N0(1+e)) ~ (1-e)/N0, although this is
usually verified when the linear approximation in the denominator
is valid.
"""
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fastprof import Model, Data, OptiMinimizer, Serializable
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import math

####################################################################################################################################
###

class ValidationData (Serializable) :
  def __init__(self, model, filename = '') :
    super().__init__()
    self.model = model
    if filename != '' : self.load(filename)

  def load_dict(self, jdict) :
    self.poi_value  = jdict[self.model.poi(0).name]
    self.points     = jdict['points']
    self.variations = {}
    for chan in self.model.channels :
      channel_variations = {}
      for par in self.model.nps.keys() :
        if not par in jdict[chan] :
          print('No validation data found for NP %s' % par)
        else :
          channel_variations[par] = np.array(jdict[chan][par])
      self.variations[chan] = channel_variations
    return self

  def dump_jdict(self) :
    return {}


def make_parser() :
  parser = ArgumentParser("plot_valid.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"     , type=str  , required=True, help="Name of markup file defining model")
  parser.add_argument("-l", "--validation-file", type=str  , default=None , help="Name of markup file containing validation data (default: <model_file>_validation.json)")
  parser.add_argument("-c", "--channel"        , type=str  , default=None , help="Name of selected channel (default: first one in the model)")
  parser.add_argument("-s", "--sample"         , type=str  , default=None , help="Name of selected sample (default: first one in the channel)")
  parser.add_argument("-b", "--bins"           , type=str  , default=None , help="List of bins for which to plot validation data")
  parser.add_argument("-p", "--nps"            , type=str  , default=None , help="List of nuisance parameters to plot (default: all)")
  parser.add_argument("-y", "--yrange"         , type=str  , default=''   , help="Vertical range for variations, in the form min,max")
  parser.add_argument(      "--cutoff"         , type=float, default=None , help="Cutoff to regularize the impact of NPs")
  parser.add_argument(      "--inversion-plots", action='store_true'      , help="Only plot variations, not inversion impact")
  parser.add_argument("-i", "--inv-range"      , type=float, default=None , help="Vertical range for inversion impact")
  parser.add_argument(      "--no-nli"         , action='store_true'      , help="Only plot linear reference")
  parser.add_argument(      "--no-title"       , action='store_true'      , help="Omit the plot title")
  parser.add_argument(      "--symmetric"      , action='store_true'      , help="Show symmetric impacts")
  parser.add_argument("-t", "--style"          , type=str  , default=None , help="List of styles to apply (comma-separated)")
  parser.add_argument("-o", "--output-file"    , type=str  , default=None , help="Output file name")
  return parser

def run(argv = None) :
  parser = make_parser()

  options = parser.parse_args()
  if not options :
    parser.print_help()
    sys.exit(0)

  if options.style is not None : plt.style.use(options.style.split(','))
  try :
    bins = [ int(b) for b in options.bins.split(',') ] if options.bins is not None else [ 0 ]
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid bin specification %s : the format should be bin1,bin2,...' % options.bins)

  if options.yrange != '' :
    try:
      y_min, y_max = [ float(p) for p in options.yrange.split(',') ]
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid range specification %s, expected y_min,y_max' % options.yrange)

  model = Model().load(options.model_file)
  if model is None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
  if not options.cutoff is None : model.cutoff = options.cutoff
  if options.symmetric : model.use_asym_impacts = False

  if options.channel != None :
    channel = model.channel(options.channel)
    if not channel : raise KeyError('Channel %s not found in model.' % options.channel)
  else :
    channel = list(model.channels.values())[0]

  if options.sample != None :
    sample = channel.sample(options.sample)
    if not sample : raise KeyError('Sample %s not found in channel %s.' % (options.sample, channel.name))
  else :
    sample = list(channel.samples.values())[0]

  if options.validation_file is not None :
    validation_file = options.validation_file
  else :
    split_name = os.path.splitext(options.model_file)
    validation_file = split_name[0] + '_validation' + split_name[1]
  print("Loading from '%s'" % validation_file)
  data = ValidationData(model, validation_file)
  poi_name = model.poi(0).name
  print('Validating for POI value %s = %g' % (poi_name, data.poi_value))
  plt.ion()
  
  if options.nps is None :
    nps = list(model.nps.keys())
  else :
    nps = options.nps.split(',')

  nplots = len(nps)
  nc = math.ceil(math.sqrt(nplots))
  nr = math.ceil(nplots/nc)

  cont_x = np.linspace(data.points[0], data.points[-1], 100)

  nexp0 = model.channel_n_exp(pars=model.expected_pars(data.poi_value), channel=channel.name, sample=sample.name)
  ax_vars = []
  ax_invs = []

  sample_index = list(channel.samples.keys()).index(sample.name)

  def nexp_var(pars, par, x) :
    return model.channel_n_exp(pars=pars.set(par, x), channel=channel.name, sample=sample.name)

  for b in bins :
    fig = plt.figure()
    if not options.no_title :
      if options.bins is not None :
        fig.suptitle('Linearity checks for sample %s, bin [%g, %g]'  % (sample.name, channel.bins[b]['lo_edge'], channel.bins[b]['hi_edge']))
      else :
        fig.suptitle('Linearity checks for sample %s'  % sample.name)
    gs = gridspec.GridSpec(nrows=nr, ncols=nc, wspace=0.3, hspace=0.3, top=0.9, bottom=0.05, left=0.1, right=0.95)
    for i, par in enumerate(nps) :
      if options.inversion_plots :
        sgs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i//nc, i % nc], wspace=0.1, hspace=0.1)
      else :
        sgs = [ gs[i//nc, i % nc] ]
      pars = model.expected_pars(data.poi_value)
      model.use_linear_nps = True
      vars_lin = [ nexp_var(pars, par, x)[b]/nexp0[b] - 1 for x in cont_x ]
      rvar_lin = [ -((nexp_var(pars, par, x)[b] - nexp0[b])/nexp0[b])**2 for x in cont_x ]
      model.use_linear_nps = False
      vars_nli = [ nexp_var(pars, par, x)[b]/nexp0[b] - 1 for x in cont_x ]
      rvar_nli = [ -((nexp_var(pars, par, x)[b] - nexp0[b])/nexp0[b])**2 for x in cont_x ]

      ax_var = fig.add_subplot(sgs[0])
      ax_var.plot(data.points, data.variations[channel.name][par][sample_index, b, :] - 1, 'ko', label='Full model impact')
      ax_var.plot([0], [0], marker='o', color='purple')
      ax_var.plot(sample.pos_vars[par], sample.pos_imps[par][:,b], marker='o', color='red')
      ax_var.plot(sample.neg_vars[par], sample.neg_imps[par][:,b], marker='o', color='red')
      ax_var.plot(cont_x, vars_lin, 'r--', label='Linear impact')
      if not options.no_nli : ax_var.plot(cont_x, vars_nli, 'b', label='Non-linear impact')
      ax_var.legend(title=par)
      ax_var.set_xlabel('Normalized parameter value')
      ax_var.set_ylabel('Relative impact on bin yield')
      ax_vars.append(ax_var)
      if options.yrange : ax_var.set_ylim(y_min, y_max)
      if options.inversion_plots :
        ax_inv = fig.add_subplot(sgs[1], sharex=ax_var)
        ax_inv.plot(cont_x, rvar_lin, 'r--')
        ax_inv.plot(cont_x, rvar_nli, 'b')
        if options.inv_range : ax_inv.set_ylim(-options.inv_range, 0)
        ax_invs.append(ax_inv)
    #fig.canvas.set_window_title('Linearity checks for sample %s, bin  %g' % (sample.name, b))

    if options.output_file != '' :
      if options.output_file is not None :
        output_file = options.output_file
      else :
        split_name = os.path.splitext(options.model_file)
        output_file = split_name[0] + '-%s-bin_%d.png' % (options.sample, b)
      plt.savefig(output_file)


if __name__ == '__main__' : run()
