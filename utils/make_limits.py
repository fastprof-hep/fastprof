#! /usr/bin/env python
# Script to produce fastprof limits
# The options below should be customized for the intended application, and the script can then be
# run once for each scan point (e.g. ./make_limits 2020)

# =====================================
# User-modifiable configuration options
# =====================================

basedir     = 'run/fastprof'              # Base directory for file paths
name        = 'highMass_NW-prod1000'      # Base name of output files
scan_var    = 'mX'                        # Scan variable
ntoys       = 10000                       # Number of toys
                                         
computation = 'limit'                     # Name of computation

opts = [ '--break-locks' ]

# ==============================================
# No user modifications intended below this line
# ==============================================

import os, sys

# Command-line inputs
scan_val = sys.argv[1] # scan points to process


dname = os.path.join(basedir, name, 'data-%s.json' % name)
mname = os.path.join(basedir, name, 'models', 'model-%s-%s.json' % (name, scan_val))
fname = os.path.join(basedir, name, 'wsfits', 'wsfits-%s-%s.json' % (name, scan_val))
outdir = os.path.join(basedir, name, 'limits', '%s-%s-%s' % (computation, name, scan_val))
oname = os.path.join(outdir, 'sampling')

os.makedirs(outdir, exist_ok=True)

args = []
args += [ '-m', mname ]
args += [ '-d', dname ]
args += [ '-f', fname ]
args += [ '-o', oname ]
args += [ '-n', str(ntoys) ]
args += opts

import compute_limits
compute_limits.run(args)
