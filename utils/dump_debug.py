import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

####################################################################################################################################
###

parser = ArgumentParser("dump_debug.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument('filename'         , type=str, nargs=1    , help='Name of the CSV file in which samples are stored')
parser.add_argument("-b", "--nbins"    , type=int, default=100, help="Number of bins to use")
parser.add_argument("-l", "--log-scale", action='store_true'  , help="Use log scale for plotting")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

debug = pd.read_csv(options.filename[0])

plt.ion()
fig,ax = plt.subplots(2,2)

debug.hist('mu_hat', ax=ax[0,0], bins=options.nbins)
debug.hist('tmu'   , ax=ax[0,1], bins=np.linspace(0,20,options.nbins))
debug.hist('cl'    , ax=ax[1,0], bins=options.nbins)
debug.hist('nfev'  , ax=ax[1,1])

if options.log_scale : ax[0,1].set_yscale('log')
plt.show()
