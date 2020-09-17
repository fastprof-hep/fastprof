#! /usr/bin/env python
# Script to produce fastprof limits
# The options below should be customized for the intended application, and the script can then be
# run once for each scan point (e.g. ./make_limits 2020)

# =====================================
# User-modifiable configuration options
# =====================================

# Base arguments
basedir     = 'run/root'                  # Base directory for file paths
name        = 'highMass_NW-prod1000'      # Base name of output files
scan_var    = 'mX'                        # Scan variable
ntoys       = 10000                       # Number of toys
bands       = 2
computation = 'limit-reg'                 # Name of computation

# Regularization arguments
regularize     = ''
sethypo        = 'dSig=0'
bounds         = 'dSig:-2:2'
truncate_dists = ''

opts = [ '--break-locks' ]

# ==============================================
# No user modifications intended below this line
# ==============================================

import os, sys

# Command-line inputs
scan_val = sys.argv[1] # scan points to process

dname  = os.path.join(basedir, name, 'datasets', 'data-%s.json' % name)
mname  = os.path.join(basedir, name, 'models'  , 'model-%s-%s.json' % (name, scan_val))
fname  = os.path.join(basedir, name, 'wsfits'  , 'wsfits-%s-%s.json' % (name, scan_val))
outdir = os.path.join(basedir, name, 'limits'  , '%s-%s-%s' % (computation, name, scan_val))
outlog = os.path.join(outdir, 'log_%d.txt')
oname  = os.path.join(outdir, 'sampling')

os.makedirs(outdir, exist_ok=True)

if not sys.flags.interactive :
  nlog = 1
  while os.path.exists(outlog % nlog) : nlog += 1
  logfile = open(outlog % nlog, 'w')
  sys.stdout = logfile

args = []
args += [ '-m', mname ]
args += [ '-d', dname ]
args += [ '-f', fname ]
args += [ '-o', oname ]
args += [ '-n', str(ntoys) ]
args += [ '--bands', str(bands) ]
if regularize     != '' : args += [ '--regularize', regularize ]
if sethypo        != '' : args += [ '--sethypo', sethypo ]
if bounds         != '' : args += [ '--bounds', bounds ]
if truncate_dists != '' : args += [ '--truncate-dists', truncate_dists ]
args += opts

import compute_limits
compute_limits.run(args)
