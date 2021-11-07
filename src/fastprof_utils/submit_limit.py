#! /usr/bin/env python

__doc__ = "Submit a limit job to the CCIN2P3 batch farm"
__author__ = "Nicolas Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import collections

####################################################################################################################################
###
def make_parser() :
  parser = ArgumentParser("submit_limit.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument(      "--name"          , type=str  , required=True , help="Name of job")
  parser.add_argument("-q", "--queue"         , type=str  , default='long', help="Name of batch queue to submit the job to")
  parser.add_argument("-c", "--cores"         , type=int  , default=1     , help="Number of cores to use")
  parser.add_argument("-m", "--model-file"    , type=str  , required=True , help="Name of markup file defining model")
  parser.add_argument("-f", "--fits-file"     , type=str  , required=True , help="Name of markup file containing full-model fit results")
  parser.add_argument("-n", "--ntoys"         , type=int  , default=10000 , help="Number of toy iterations to generate")
  parser.add_argument("-s", "--seed"          , type=int  , default='0'   , help="Random generation seed")
  parser.add_argument("-%", "--print-freq"    , type=int  , default=1000  , help="Verbosity level")
  parser.add_argument("-d", "--data-file"     , type=str  , default=''    , help="Perform checks using the dataset stored in the specified markup file")
  parser.add_argument("-a", "--asimov"        , type=str  , default=None  , help="Perform checks using an Asimov dataset for the specified POI value")
  parser.add_argument("-i", "--iterations"    , type=int  , default=1     , help="Numer of iterations to perform for NP computation")
  parser.add_argument("-r", "--regularize"    , type=float, default=None  , help="Set loose constraints at specified N_sigmas on free NPs to avoid flat directions")
  parser.add_argument(      "--cutoff"        , type=float, default=None  , help="Cutoff to regularize the impact of NPs")
  parser.add_argument(      "--truncate-dist" , type=float, default=None  , help="Truncate high p-values (just below 1) to get reasonable bands")
  parser.add_argument(      "--bounds"        , type=str  , default=None  , help="Parameter bounds in the form name1:[min]:[max],name2:[min]:[max],...")
  parser.add_argument(      "--sethypo"       , type=str  , default=''    , help="Change hypo parameter values, in the form par1=val1,par2=val2,...")
  parser.add_argument("-t", "--test-statistic", type=str  , default='q~mu', help="Test statistic to use")
  parser.add_argument(      "--break-locks"   , action='store_true'       , help="Allow breaking locks from other sample production jobs")
  parser.add_argument(      "--resume"        , type=int  , default=0     , help="Resume an interrupted job ")
  parser.add_argument(      "--debug"         , action='store_true'       , help="Produce debugging output")
  parser.add_argument(      "--dry-run"       , action='store_true'       , help="Run the command locally for debugging")
  parser.add_argument("-v", "--verbosity"     , type=int  , default=0     , help="Verbosity level")

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args()
  if not options :
    parser.print_help()
    sys.exit(0)
  
  print('Running in directory %s' % os.getcwd())
  os.makedirs('batch', exist_ok=True)
  
  if options.resume == 0 :
    try :
      os.makedirs('batch/%s' % options.name)
    except Exception as inst :
      print(inst)
      print('Directory %s exists already, will not submit again an existing job' % options.name)
      sys.exit(1)
  
  os.chdir('batch/%s' % options.name)
  print('Now in directory %s' % os.getcwd())
  
  if options.resume == 0 :
    os.symlink('../..', 'run')
    os.symlink('../../..', 'fastprof')
    os.makedirs('samples')
  
  opts = ''
  if options.asimov         : opts += ' --asimov %s' % options.asimov
  if options.data_file      : opts += ' --data-file %s' % options.data_file
  if options.iterations     : opts += ' --iterations %d' % options.iterations
  if options.regularize     : opts += ' --regularize %g' % options.regularize
  if options.cutoff         : opts += ' --cutoff %g' % options.cutoff
  if options.truncate_dist  : opts += ' --truncate-dist %g' % options.truncate_dist
  if options.bounds         : opts += ' --bounds %s' % options.bounds
  if options.sethypo        : opts += ' --sethypo %s' % options.sethypo
  if options.test_statistic : opts += ' --test-statistic %s' % options.test_statistic
  if options.print_freq     : opts += ' --print-freq %d' % options.print_freq
  if options.verbosity      : opts += ' --verbosity %d' % options.verbosity
  if options.debug          : opts += ' --debug'
  if options.break_locks or options.resume : opts += ' --break-locks'
  
  resume = options.resume
  while os.path.exists('job_%d.sh' % resume) : resume += 1
  job = 'job_%d.sh' % resume
  out = 'stdout_%d' % resume
  err = 'stderr_%d' % resume
  
  if resume > 0 :
    try :
      prev_out_size = os.path.getsize('stdout_%d' % (resume-1))
    except FileNotFoundError :
      print('Trying to submit iteration %d but no log file found for the previous one, stopping' % resume)
      sys.exit(0)
    if prev_out_size == 0 :
      print('Trying to submit iteration %d but previous iteration still seems active (file %s has size 0), aborting job submission' % (resume, 'stdout_%d' % (resume-1)))
      sys.exit(0)
  
  command = './fastprof/fastprof_utils/compute_limits.py -m %s -f %s -n %d -s %d %s -o samples/%s' % (options.model_file, options.fits_file, options.ntoys, options.seed, opts, options.name)
  with open(job, 'w') as f :
    f.write(command)
  os.chmod(job, 0o555)
  
  if options.dry_run :
    os.system(job)
    sys.exit(0)
  
  submit_opts = ''
  if options.cores > 1 : submit_opts += ' -pe multicores %d' % option.cores
  
  submit_command = 'qsub -N %s -q %s -cwd -V -l sps=1 %s -o %s -e %s %s' % (options.name, options.queue, submit_opts, out, err, job)
  print(command)
  print(submit_command)
  os.system(submit_command)


if __name__ == '__main__' : run()
