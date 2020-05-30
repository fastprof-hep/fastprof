import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.stats import norm, chi2
from fastprof import Model, FitResults

####################################################################################################################################
###

parser = ArgumentParser("dump_debug.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument('filename'          , type=str, nargs=1    , help='Name of the CSV file in which samples are stored')
parser.add_argument("-b", "--nbins"     , type=int, default=100, help="Number of bins to use")
parser.add_argument("-l", "--log-scale" , action='store_true'  , help="Use log scale for plotting")
parser.add_argument("-r", "--reference" , action='store_true'  , help="Use log scale for plotting")
parser.add_argument(      "--hypo"      , type=str, default='' , help="Generation hypothesis, format: <file>:<index>")
parser.add_argument("-m", "--model-file", type=str, default='' , help="Name of JSON file defining model")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

debug = pd.read_csv(options.filename[0])

plt.ion()
fig1,ax1 = plt.subplots(2,2)

debug.hist('mu_hat', ax=ax1[0,0], bins=options.nbins)
debug.hist('tmu'   , ax=ax1[0,1], bins=np.linspace(0, 10, options.nbins))
debug.hist('cl'    , ax=ax1[1,0], bins=options.nbins)
debug.hist('nfev'  , ax=ax1[1,1])

if options.log_scale :
  ax1[0,0].set_yscale('log')
  ax1[0,1].set_yscale('log')
  ax1[0,1].set_ylim(bottom=1)

if options.reference :
  mu_hat = debug['mu_hat']
  xx = np.linspace(np.min(mu_hat), np.max(mu_hat), options.nbins)
  yy = [ mu_hat.shape[0]*(xx[1] - xx[0])*norm.pdf(x, np.mean(mu_hat), np.std(mu_hat)) for x in xx ]
  ax1[0,0].plot(xx,yy)
  tmu = debug['tmu']
  xx = np.linspace(0, 10, options.nbins)
  yy = [ tmu.shape[0]*(xx[1] - xx[0])*chi2.pdf(x, 1) for x in xx ]
  ax1[0,1].plot(xx,yy)
  xx = np.linspace(0,1, options.nbins)
  yy = [ mu_hat.shape[0]*(xx[1] - xx[0]) for x in xx ]
  ax1[1,0].plot(xx,yy)

if options.hypo != '' :
  model = Model.create(options.model_file)
  if model == None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
  try:
    filename, index = options.hypo.split(':')
    index = int(index)
    fit_result = FitResults(model, filename).fit_results[index]
    hypo_pars = fit_result['hypo_pars']
    print('Got hypo pars for hypothesis %g :' % fit_result[model.poi])
    print(hypo_pars)
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid hypothesis spec, should be in the format <filename>:<index>')

z = 3
pars = [ col[len('free_'):] for col in debug.columns if col.startswith('free_') and not col.endswith('nll') ]
fig2,ax2 = plt.subplots(2, len(pars),figsize=(15,5), sharey=True)
fig2.subplots_adjust(left=0.05,right=0.98)
if options.reference :
  xx = np.linspace(-z, z, options.nbins)
  yy = [ mu_hat.shape[0]*(xx[1] - xx[0])*norm.pdf(x, 0, 1) for x in xx ]
for i, par in enumerate(pars) :
  ax2[0,i].set_title(par)
  free_delta = debug['free_' + par] -  - debug['aux_' + par] if 'aux_' + par in debug.columns else debug['free_' + par]
  hypo_delta = debug['hypo_' + par] -  - debug['aux_' + par] if 'aux_' + par in debug.columns else debug['hypo_' + par]
  if options.hypo != '' :
    print('Shifting distributions of %s by %g' % (par, hypo_pars[par]))
    free_delta -= hypo_pars[par]
    hypo_delta -= hypo_pars[par]
  ax2[0,i].hist(free_delta, bins=np.linspace(-z, z, options.nbins))
  ax2[1,i].hist(hypo_delta, bins=np.linspace(-z, z, options.nbins))
  if options.reference :
    ax2[0,i].plot(xx,yy)
    ax2[1,i].plot(xx,yy)
  if options.log_scale : ax2[0,i].set_yscale('log')
  if options.log_scale : ax2[1,i].set_yscale('log')
ax2[0,0].set_ylabel('free fit')
ax2[1,0].set_ylabel('hypothesis fit')


  #free_g = sns.jointplot(free_delta, debug['mu_hat'], kind="kde", xlim=(-z,z), ax=ax2[1,i]) 
  #hypo_g = sns.jointplot(hypo_delta, debug['mu_hat'], kind="kde", xlim=(-z,z), ax=ax2[2,i]) 
  #free_g.ax_joint.axhline(0,c='r',ls='--')
  #free_g.ax_joint.axvline(0,c='r',ls='--')
  #hypo_g.ax_joint.axhline(0,c='r',ls='--')
  #hypo_g.ax_joint.axvline(0,c='r',ls='--')

plt.show()
