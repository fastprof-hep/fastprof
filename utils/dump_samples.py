# Usage : python3 -i fastprof/examples/dump_samples.py <sample_file.npy>

import numpy as np
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fastprof import QMu
from scipy.stats import chi2

####################################################################################################################################
###

parser = ArgumentParser("dump_samples.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument('filename'           , type=str, nargs=1     , help='Name of the npy file in which samples are stored')
parser.add_argument("-b", "--nbins"      , type=int, default=100 , help="Number of bins to use")
parser.add_argument("-l", "--log-scale"  , action='store_true'   , help="Use log scale for plotting")
parser.add_argument("-t", "--t-value"    , type=str, default=''  , help="Show t-value instead of p-value")
parser.add_argument("-x", "--x-range"    , type=str, default=''  , help="X-axis range, in the form min,max")
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

plt.ion()
if options.log_scale : plt.yscale('log')
plt.suptitle(options.filename[0])
if options.t_value == 'q_mu' :
  data = np.array([ QMu(0,0,0).asymptotic_tmu(pv) for pv in samples ])
  plt.hist(data[:], bins=options.nbins, range=[x_min, x_max])
else :
  plt.hist(samples[:], bins=options.nbins, range=[x_min, x_max])
plt.show()

if options.reference :
  xx = np.linspace(x_min, x_max, options.nbins)
  if options.t_value == 'q_mu' :
    yy = [ len(samples)/2*(xx[1] - xx[0])*chi2.pdf(abs(x), 1) for x in xx ]
  else :
    yy = [len(samples)*(xx[1] - xx[0]) for x in xx ]
  plt.plot(xx,yy)

if options.output_file != '' : plt.savefig(options.output_file)
