# Usage : python3 -i fastprof/examples/dump_samples.py <sample_file.npy>

import numpy as np
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

samples = np.load(options.filename[0])

# Samples data:
# 0, 1 : data bin contents (0=SR, 1=CR)
# 2, 3 : aux measurement values of signal (2) and background (3) NPs
# 4, 5, 6, 7 : fast-fit values of mu (4) signal (5) and background (6) pars + CLs+b (7)
# 8, 9, 10, 11 : best-fit values of mu (8) signal (9) and background (10) pars  + CLsb @ mu=mu_min (11)
# 12, 13, 14 :  mu_min (12) and best-fit values of signal (13) and background (14) pars

plt.ion()
if options.log_scale : plt.yscale('log')
plt.suptitle(options.filename[0])
plt.hist(samples[:], bins=options.nbins, range=[0, 1])
plt.show()
