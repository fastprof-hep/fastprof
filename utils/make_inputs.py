#! /usr/bin/env python
# Script to produce fastprof inputs over a range of a model variable (e.g. a resonance mass)
# The options below should be customized for the intended application, and the script can then be
# run once for each scan point (e.g. ./make_inputs.py 2020)
# run commands using subprocess as otherwise roofit output cannot be redirected

# =====================================
# User-modifiable configuration options
# =====================================

name    = 'highMass_NW-prod1000' # Common naming of output files
scan_var= 'mX'                   # Scan variable

# Input workspace options
ws_file   = 'highMass_NW.root'   # Input workspace file
data_name = 'obsData'            # Input dataset (RooAbsData object name)

# Common workspace tweaks
setval    = 'a0=-1.68'        # Workspace variable value changes
setconst  = 'a0'              # Workspace variable constness changes
setrange  = 'xs:0:500'        # Workspace variable range changes

# Model-building options
refit   = 'xs=0'              # Use a model that is refit to the data under the specified POI values
binning = '150:4000:1000:log' # Output binning specification
default_sample = 'Background' # Name of sample collecting contributions that don't scale with the POIs (e.g. spurious signal)

# PLR computation options
hypos = '17'                  # Specification of the fit hypotheses (list of values, or just a number of points to use)

# ==============================================
# No user modifications intended below this line
# ==============================================

import os, sys
from contextlib import redirect_stdout

# Command-line inputs
scan_val = sys.argv[1]

# Output files
os.makedirs(os.path.join(name, 'models'), exist_ok=True)
os.makedirs(os.path.join(name, 'wsfits'), exist_ok=True)

dname = os.path.join(name, 'data-%s.json' % name)
mname = os.path.join(name, 'models', 'model-%s-%s.json' % (name, scan_val))
vname = os.path.join(name, 'models', 'valid-%s-%s.json' % (name, scan_val))
fname = os.path.join(name, 'wsfits', 'wsfits-%s-%s.json' % (name, scan_val))

dlog = os.path.join(name, 'data-%s.json' % name)
mlog = os.path.join(name, 'models', 'log-model-%s-%s.txt' % (name, scan_val))
flog = os.path.join(name, 'wsfits', 'log-wsfits-%s-%s.txt' % (name, scan_val))

if not os.path.exists(dname) :
  sys.stderr.write('=== Making %s' % dname)
  args = []
  args += [ '-x' ]
  args += [ '-f'        , ws_file ]
  args += [ '-d'        , data_name ]
  args += [ '--setconst', setconst ]
  args += [ '-b'        , binning ]
  args += [ '-o'        , dname ]
  import convert_ws
  import subprocess
  with open(dlog, "w") as fd :
    subprocess.call(['python', 'convert_ws.py', *args], stdout=fd)
else :
  sys.stderr.write('--- File %s already exists\n' % dname)

if not os.path.exists(mname) :
  sys.stderr.write('=== Making %s\n' % mname)
  args = []
  args += [ '-f'        , ws_file ]
  args += [ '-d'        , data_name ]
  args += [ '--setval'  , '%s=%s,' % (scan_var, scan_val) + setval ]
  args += [ '--setconst', setconst ]
  args += [ '--setrange', setrange ]
  args += [ '--default-sample', default_sample ]
  args += [ '-b'        , binning ]
  args += [ '--refit'   , refit ]
  args += [ '--binned'  ]
  args += [ '-o'        , mname ]
  args += [ '-l'        , vname ]
  import convert_ws
  import subprocess
  with open(mlog, "w") as fd :
    subprocess.call(['python', 'convert_ws.py', *args], stdout=fd)
else :
  sys.stderr.write('--- File %s already exists\n' % mname)

if not os.path.exists(fname) :
  sys.stderr.write('=== Making %s\n' % fname)
  args = []
  args += [ '-f'        , ws_file ]
  args += [ '-d'        , data_name ]
  args += [ '--setval'  , '%s=%s,' % (scan_var, scan_val) + setval ]
  args += [ '--setconst', setconst ]
  args += [ '--setrange', setrange ]
  args += [ '--binned'  ]
  args += [ '-y'        , hypos ]
  args += [ '-o'        , fname ]
  import fit_ws
  import subprocess
  with open(flog, "w") as fd :
    subprocess.call(['python', 'fit_ws.py', *args], stdout=fd)
else :
  sys.stderr.write('--- File %s already exists\n' % fname)
