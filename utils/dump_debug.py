# Usage : python3 -i fastprof/examples/dump_samples.py <sample_file.npy>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

####################################################################################################################################
###

parser = ArgumentParser("dump_samples.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument('filename'         , type=str, nargs=1  , help='Name of the npy file in which samples are stored')
parser.add_argument("-b", "--nbins"    , type=int, default=100, help="Number of bins to use")
parser.add_argument("-l", "--log-scale", action='store_true'  , help="Use log scale for plotting")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

debug = pd.read_csv(options.filename[0])

plt.ion()
debug.hist('mu_hat',bins=options.nbins)
if options.log_scale : plt.yscale('log')
plt.show()
