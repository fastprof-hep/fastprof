#! /usr/bin/env python

__doc__ = """
*Plot linear model predictions and datasets*

The expected event yields for the model specified by `--model-file` in
channel `--channel` are shown in a histogram plot.

If a dataset is specified, its bin contents are also shown as a bar graph.
The dataset can be either observed data, using the `--data-file` option, or
an Asimov dataset using the `--asimov` option.

The POI values for which to report the model are specified by the `--poi` option.
If POIs are provided along with the `--profile` option, then NPs will be profiled
to the POI value.
If no POIs are provided but a dataset is, then the best-fit POIs for this dataset
are used.
"""
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fastprof import Model, Data, Parameters, OptiMinimizer
from fastprof_utils import process_setvals
import matplotlib.pyplot as plt
import math
import os

####################################################################################################################################
###
def make_parser() :
  parser = ArgumentParser("plot.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-m", "--model-file"  , type=str  , required=True , help="Name of markup file defining model")
  parser.add_argument("-c", "--channel"     , type=str  , default=None  , help="Name of selected channel (default: first one in the model)")
  parser.add_argument("-i", "--plot-alone"  , type=str  , default=None  , help="Name of samples to plot by itself in a second (dashed) model line")
  parser.add_argument("-e", "--plot-without", type=str  , default=None  , help="Name of samples to exclude in a second (dashed) model line")
  parser.add_argument("-p", "--setval"      , type=str  , default=None  , help="Parameter values, in the form par1=val1,par2=val2,...")
  parser.add_argument("-d", "--data-file"   , type=str  , default=None  , help="Name of markup file defining the dataset (optional, otherwise taken from model file)")
  parser.add_argument("-a", "--asimov"      , type=str  , default=None  , help="Use an Asimov dataset for the specified POI values (format: 'poi1=xx,poi2=yy'")
  parser.add_argument("-x", "--x-range"     , type=str  , default=None  , help="X-axis range, in the form min,max")
  parser.add_argument("-y", "--y-range"     , type=str  , default=None  , help="Y-axis range, in the form min,max")
  parser.add_argument(      "--profile"     , action='store_true'       , help="Perform a conditional fit for the provided POI value before plotting")
  parser.add_argument("-l", "--log-scale"   , action='store_true'       , help="Use log scale for plotting")
  parser.add_argument("-s", "--stack"       , action='store_true'       , help="Use log scale for plotting")
  parser.add_argument(      "--variations"  , type=str  , default=None  , help="Plot variations for parameters par1=val1[:color],par2=val2[:color]... or a single value for all parameters")
  parser.add_argument("-r", "--residuals"   , action='store_true'       , help="Show model - data residuals in an inset plot")
  parser.add_argument("-o", "--output-file" , type=str  , default=None  , help="Output file name")
  parser.add_argument("-w", "--window"      , type=str  , default="8x8" , help="Window size (format: (width)x(height) )")
  parser.add_argument("-v", "--verbosity"   , type=int  , default=0     , help="Verbosity level")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args()
  if not options :
    parser.print_help()
    return

  model = Model.create(options.model_file)
  if model is None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
  if options.channel is not None and not options.channel in model.channels() : raise KeyError('Channel %s not found in model.' % options.channel)

  if options.data_file is not None :
    data = Data(model).load(options.data_file)
    if data is None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    print('Using dataset stored in file %s.' % options.data_file)
  elif options.asimov is not None :
    try:
      sets = process_setvals(options.asimov, model)
      data = model.generate_expected(sets)
    except Exception as inst :
      print(inst)
      raise ValueError("Cannot define an Asimov dataset from options '%s'." % options.asimov)
    print('Using Asimov dataset with POIs %s.' % str(sets))
  else :
    data = Data(model).load(options.model_file)

  if options.setval is not None :
    try :
      poi_dict = process_setvals(options.setval, model, match_nps = False)
    except Exception as inst :
      print(inst)
      raise ValueError("ERROR : invalid POI specification string '%s'." % options.setval)
    pars = model.expected_pars(poi_dict)
    if options.profile :
      mini = OptiMinimizer()
      mini.profile_nps(pars, data)
      print('Minimum: nll = %g @ parameter values : %s' % (mini.min_nll, mini.min_pars))
      pars = mini.min_pars
  elif data is not None and options.profile :
    mini = OptiMinimizer().set_pois_from_model(model)
    mini.minimize(data)
    pars = mini.min_pars
  else :
    pars = model.expected_pars([0]*model.npois)

  try:
    width, height = tuple( [ float(dim) for dim in options.window.split('x') ] )
  except Exception as inst :
    raise ValueError('ERROR: expected window dimentions in (width)x(height) format.')

  xmin = None
  xmax = None
  ymin = None
  ymax = None
  if options.x_range is not None :
    try:
      xmin, xmax = [ float(p) for p in options.x_range.split(',') ]
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid X-axis range specification %s, expected x_min,x_max' % options.x_range)

  if options.y_range is not None :
    try:
      ymin, ymax = [ float(p) for p in options.y_range.split(',') ]
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid Y-axis range specification %s, expected y_min,y_max' % options.y_range)

  plt.ion()
  if not options.residuals :
    model.plot(pars, figsize=(width, height), data=data, labels=options.variations is None, stack=options.stack, logy=options.log_scale)
    if options.plot_without is not None or options.plot_alone is not None :
      model.plot(pars, canvas=fig1, only=options.plot_alone, exclude=options.plot_without, labels=options.variations is None, stack=options.stack, logy=options.log_scale)
    if xmin is not None : plt.xlim(xmin, xmax)
    if ymin is not None : plt.ylim(ymin, ymax)
  else :
    fig1, ax1 = plt.subplots(nrows=2, ncols=1, figsize=(width, height), dpi=96)
    model.plot(pars, data=data, canvas=ax1[0], stack=options.stack, logy=options.log_scale)
    if xmin is not None : ax1[0].set_xlim(xmin, xmax)
    if ymin is not None : ax1[0].set_ylim(ymin, ymax)
    model.plot(pars, data=data, canvas=ax1[1], residuals=options.residuals, labels=options.variations is None, stack=options.stack, logy=options.log_scale)
  if options.output_file is not None : plt.savefig(options.output_file)

  variations = None
  colors_pos = [ 'purple', 'green', 'darkblue', 'lime' ]
  colors_neg = [ 'darkred', 'red', 'orange', 'magenta' ]
  if options.variations is not None :
    # First try the comma-separated format
    try:
      float(options.variations)
      variations = 'all'
    except :
      pass
    if variations is None :
      variations = []
      try :
        for spec in options.variations.split(',') :
          specfields = spec.split(':')
          varval = specfields[0]
          color = specfields[1] if len(specfields) == 2 else None 
          var,val = varval.split('=')
          try :
            val = float(val)
          except:
            raise ValueError('Invalid numerical value %s.' % val)
          if not var in model.nps :
            raise KeyError('Parameter %s is not defined in the model.' % var)
          colors = colors_pos if val > 0 else colors_neg
          if color is None : color = colors[len(variations) % len(colors)]
          variations.append( (var, val, color,) )
      except Exception as inst :
        print(inst)
        raise ValueError('Invalid variations specification %s : should be a comma-separated list of var=val[:color] items, or a single number' % options.variations)

  if variations == 'all' :
    n1 = math.ceil(math.sqrt(model.nnps))
    n2 = math.ceil(model.nnps/n1)
    fig_nps, ax_nps = plt.subplots(nrows=n1, ncols=n2, figsize=(width, height), dpi=96)
    for i in range(len(model.nps), n1*n2) : fig_nps.delaxes(ax_nps.flatten()[i])
    for par, ax in zip(model.nps, ax_nps.flatten()) :
      model.plot(pars, data=data, variations = [ (par, var_val, 'r'), (par, -var_val, 'g') ], canvas=ax)
      if options.plot_without is not None or options.plot_alone is not None :
        model.plot(pars, variations = [ (par, var_val, 'r'), (par, -var_val, 'g') ], canvas=ax, only=options.plot_alone, exclude=options.plot_without)
      if options.log_scale : ax.set_yscale('log')
      if xmin is not None : ax.set_xlim(xmin, xmax)
      if ymin is not None : ax.set_ylim(ymin, ymax)
  elif variations is not None :
    model.plot(pars, variations=variations, figsize=(width, height))
    if options.plot_without is not None or options.plot_alone is not None :
      model.plot(pars, variations=variations, only=options.plot_alone, exclude=options.plot_without)
    if options.log_scale : plt.yscale('log')
    if options.output_file is not None :
      split_name = os.path.splitext(options.output_file)
      plt.savefig(split_name[0] + '_variations' + split_name[1])


if __name__ == '__main__' : run()
