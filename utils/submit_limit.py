#! /usr/bin/env python

__doc__ = "Submit a limit job to the CCIN2P3 batch farm"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import collections

####################################################################################################################################
###

parser = ArgumentParser("submit_limit.py", formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = __doc__
parser.add_argument(      "--name"          , type=str  , required=True      , help="Name of job")
parser.add_argument("-q", "--queue"         , type=str  , default='long'     , help="Name of batch queue to submit the job to")
parser.add_argument("-c", "--cores"         , type=int  , default=1          , help="Number of cores to use")
parser.add_argument("-m", "--model-file"    , type=str  , required=True      , help="Name of JSON file defining model")
parser.add_argument("-f", "--fits-file"     , type=str  , required=True      , help="Name of JSON file containing full-model fit results")
parser.add_argument("-n", "--ntoys"         , type=int  , default=10000      , help="Number of toy iterations to generate")
parser.add_argument("-s", "--seed"          , type=int  , default='0'        , help="Random generation seed")
parser.add_argument("-%", "--print-freq"    , type=int  , default=1000       , help="Verbosity level")
parser.add_argument("-d", "--data-file"     , type=str  , default=''         , help="Perform checks using the dataset stored in the specified JSON file")
parser.add_argument("-a", "--asimov"        , type=float, default=None       , help="Perform checks using an Asimov dataset for the specified POI value")
parser.add_argument("-i", "--iterations"    , type=int  , default=1          , help="Numer of iterations to perform for NP computation")
parser.add_argument("-r", "--regularize"    , type=float, default=None       , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
parser.add_argument("-b", "--break-locks"   , action='store_true'            , help="Allow breaking locks from other sample production jobs")
parser.add_argument(      "--debug"         , action='store_true'            , help="Produce debugging output")
parser.add_argument("-v", "--verbosity"     , type=int  , default=0          , help="Verbosity level")

options = parser.parse_args()
if not options :
  parser.print_help()
  sys.exit(0)

print('Running in directory %s' % os.getcwd())
os.makedirs('Batch', exist_ok=True)
os.makedirs('Batch/%s' % options.name)
os.chdir('Batch/%s' % options.name)
print('Now in directory %s' % os.getcwd())
os.symlink('../../run', 'run')
os.symlink('../../fastprof', 'fastprof')
os.makedirs('samples')

opts = ''
if options.asimov      : opts += ' --asimov %d' % options.asimov
if options.data_file   : opts += ' --data-file %s' % options.data_file
if options.regularize  : opts += ' --regularize %d' % options.regularize
if options.iterations  : opts += ' --iterations %d' % options.iterations
if options.print_freq  : opts += ' --print-freq %d' % options.print_freq
if options.verbosity   : opts += ' --verbosity %d' % options.verbosity
if options.debug       : opts += ' --debug'
if options.break_locks : opts += ' --break-locks'

command = './fastprof/utils/compute_limits.py -m %s -f %s -n %d -s %d %s -o samples/%s' % (options.model_file, options.fits_file, options.ntoys, options.seed, opts, options.name)
with open('job.sh', 'w') as f :
  f.write(command)
os.chmod('job.sh', 0o555)

submit_opts = ''
if options.cores > 1 : submit_opts += ' -pe multicores %d' % option.cores

submit_command = 'qsub -N %s -q %s -cwd -V -l sps=1 %s -o stdout -e stderr job.sh' % (options.name, options.queue, submit_opts)
print(command)
print(submit_command)
os.system(submit_command)
