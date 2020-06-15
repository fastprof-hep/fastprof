# Usage : python3 -i fastprof/examples/dump_samples.py <sample_file.npy>

import numpy as np
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.stats import chi2
from fastprof import Model, FitResults, QMu, QMuTilda

####################################################################################################################################
###

parser = ArgumentParser("dump_samples.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument('filename'           , type=str, nargs=1     , help='Name of the npy file in which samples are stored')
parser.add_argument("-b", "--nbins"      , type=int, default=100 , help="Number of bins to use")
parser.add_argument("-l", "--log-scale"  , action='store_true'   , help="Use log scale for plotting")
parser.add_argument("-x", "--x-range"    , type=str, default=''  , help="X-axis range, in the form min,max")
parser.add_argument("-t", "--t-value"    , type=str, default=''  , help="Show t-value instead of p-value")
parser.add_argument("-y", "--hypo"       , type=str, default=''  , help="Generation hypothesis, format: <file>:<index>")
parser.add_argument("-m", "--model-file", type=str, default='' , help="Name of JSON file defining model")
parser.add_argument("-r", "--reference"  , action='store_true'   , help="Use log scale for plotting")
parser.add_argument("-o", "--output-file", type=str  , default='', help="Output file name")

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

fit_result = None
if options.hypo != '' :
  model = Model.create(options.model_file)
  if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
  try:
    filename, index = options.hypo.split(':')
    index = int(index)
    fit_result = FitResults(model, filename).fit_results[index]
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid hypothesis spec, should be in the format <filename>:<index>')

plt.ion()
if options.log_scale : plt.yscale('log')
plt.suptitle(options.filename[0])

if options.t_value == 'q_mu' :
  if fit_result == None : raise ValueError('A signal hypothesis must be provided (--hypo option) to convert to q_mu values')
  q = QMu(test_poi = fit_result['hypo_pars'].poi, tmu=0, best_poi=fit_result['free_pars'].poi, tmu_A = fit_result['tmu_0'])
  data = np.array([ q.asymptotic_ts(pv) for pv in samples ])
  plt.hist(data[:], bins=options.nbins, range=[x_min, x_max])
elif options.t_value == 'q~mu' :
  if fit_result == None : raise ValueError('A signal hypothesis must be provided (--hypo option) to convert to q~mu values')
  q = QMuTilda(test_poi = fit_result['hypo_pars'].poi, tmu=0, best_poi=fit_result['free_pars'].poi, tmu_A = fit_result['tmu_0'], tmu_0 = fit_result['tmu_0'])
  data = np.array([ q.asymptotic_ts(pv) for pv in samples ])
  plt.hist(data[:], bins=options.nbins, range=[x_min, x_max])
else :
  plt.hist(samples[:], bins=options.nbins, range=[x_min, x_max])
plt.show()

if options.reference :
  xx = np.linspace(x_min, x_max, options.nbins)
  bin_norm = len(samples)*(xx[1] - xx[0])
  if options.t_value == 'q_mu':
    yy = [ bin_norm*q.asymptotic_pdf(x) for x in xx ]
  elif options.t_value == 'q~mu':
    yy = [ bin_norm*q.asymptotic_pdf(x) for x in xx ]
  else :
    yy = [ bin_norm for x in xx ]
  plt.plot(xx,yy)
  plt.ylim(1E-1)

if options.output_file != '' : plt.savefig(options.output_file)
