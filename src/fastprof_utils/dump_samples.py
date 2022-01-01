__doc__ = """
*Plot a sampling distribution*

Draws in histogram form the sampling distribution
contained in the specified input file.

By default the asymptotic p-value is shown, or the
test statistic value can be shown instead (specified
using the `--t-value` option).

For test statistics, the model and hypothesis considered
must be specified usign the `--model` and `--hypo` options.
This is not needed for p-values, since the sampling
distribution is stored in p-value form in the file.

The expected asymptotic distribution is shown on the plot
if the `--reference` option is passed.

An output file for the plot can be specified using the
`--output-file` option.
"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"


import numpy as np
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.stats import chi2
from scipy.integrate import quad
from fastprof import Model, Raster, QMuCalculator, QMuTildaCalculator

####################################################################################################################################
###
def make_parser() :
  parser = ArgumentParser("dump_samples.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument('filename'           , type=str, nargs=1     , help='Name of the npy file in which samples are stored')
  parser.add_argument("-b", "--nbins"      , type=int, default=100 , help="Number of bins to use")
  parser.add_argument("-l", "--log-scale"  , action='store_true'   , help="Use log scale for plotting")
  parser.add_argument("-x", "--x-range"    , type=str, default=''  , help="X-axis range, in the form min,max")
  parser.add_argument("-t", "--t-value"    , type=str, default=''  , help="Show t-value instead of p-value")
  parser.add_argument("-y", "--hypo"       , type=str, default=''  , help="Generation hypothesis, format: <file>:<index>")
  parser.add_argument("-m", "--model-file" , type=str, default=''  , help="Name of markup file defining model")
  parser.add_argument("-r", "--reference"  , action='store_true'   , help="Use log scale for plotting")
  parser.add_argument("-o", "--output-file", type=str  , default='', help="Output file name")
  return parser

def run(argv = None) :
  parser = make_parser()

  options = parser.parse_args()
  if not options :
    parser.print_help()
    sys.exit(0)

  samples = np.load(options.filename[0])

  if options.x_range :
    try:
      x_min, x_max = [ float(p) for p in options.x_range.split(',') ]
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid X-axis range specification %s, expected x_min,x_max' % options.x_range)
  else :
    if options.t_value == '' :
      x_min, x_max = 0, 1
    else :
      x_min, x_max = -10, 10

  plr_data = None
  if options.hypo != '' :
    model = Model.create(options.model_file)
    if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
    try:
      filename, index = options.hypo.split(':')
      index = int(index)
      raster = Raster('data', model=model, filename=filename)
      plr_data = list(raster.plr_data.values())[index]
      hypo = list(raster.plr_data.keys())[index]
      print('Using hypothesis %s' % str(hypo.dict(pois_only=True)))
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid hypothesis spec, should be in the format <filename>:<index>')

  plt.ion()
  if options.log_scale : plt.yscale('log')
  plt.suptitle(options.filename[0])

  if options.t_value == 'q_mu' :
    if plr_data is None : raise ValueError('A signal hypothesis must be provided (--hypo option) to convert to q_mu values')
    q = QMuCalculator.make_q(plr_data)
    data = np.array([ q.asymptotic_ts(pv) for pv in samples ])
    plt.hist(data[:], bins=options.nbins, range=[x_min, x_max])
  elif options.t_value == 'q~mu' :
    if plr_data is None : raise ValueError('A signal hypothesis must be provided (--hypo option) to convert to q~mu values')
    q = QMuTildaCalculator.make_q(plr_data)
    data = np.array([ q.asymptotic_ts(pv) for pv in samples ])
    plt.hist(data[:], bins=options.nbins, range=[x_min, x_max])
  else :
    plt.hist(samples[:], bins=options.nbins, range=[x_min, x_max])
  plt.show()

  if options.reference :
    xx = np.linspace(x_min, x_max, options.nbins+1)
    dx = xx[1] - xx[0]
    bin_norm = len(samples)
    if options.t_value == 'q_mu' or options.t_value == 'q~mu':
      yy = [ bin_norm*quad(lambda t : q.asymptotic_pdf(t), x, x+dx)[0] for x in xx[:-1] ]
    else :
      yy = [ bin_norm*dx for x in xx[:-1] ]
    plt.plot(xx[:-1] + dx/2, yy)
    plt.ylim(1E-1)

  if options.output_file != '' : plt.savefig(options.output_file)


if __name__ == '__main__' : run()
