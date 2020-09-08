#! /bin/zsh
# Script to produce fastprof inputs over a range of a model variable (e.g. a resonance mass)
# The options below should be customized for the intended application, and the script can then be
# run once for each scan point (e.g. ./make_inputs 2020)

# =====================================
# User-modifiable configuration options
# =====================================

name=highMass_NW-prod1000 # Common naming of output files
scan_var=mX               # Scan variable

# Input workspace options
ws_file=highMass_NW.root  # Input workspace file
data_name=obsData         # Input dataset (RooAbsData object name)

# Common workspace tweaks
setval=a0=-1.68           # Workspace variable value changes
setconst=a0               # Workspace variable constness changes
setrange=xs:0:500         # Workspace variable range changes

# Model-building options
refit=xs=0                # Use a model that is refit to the data under the specified POI values
binning=150:4000:1000:log # Output binning specification
default_sample=Background # Name of sample collecting contributions that don't scale with the POIs (e.g. spurious signal)

# PLR computation options
hypos=17                  # Specification of the fit hypotheses (list of values, or just a number of points to use)

# ==============================================
# No user modifications intended below this line
# ==============================================

# Command-line inputs
scan_val=$1

# Output files
mkdir -p $name/models
mkdir -p $name/wsfits
dname=$name/data-$name.json
mname=$name/models/model-$name-$scan_val.json
vname=$name/models/valid-$name-$scan_val.json
fname=$name/wsfits/wsfits-$name-$scan_val.json
dlog=$name/log-data_$name.txt
mlog=$name/models/log-model-$name-$scan_val.txt
flog=$name/wsfits/log-wsfits-$name-$scan_val.txt

if [[ ! -e $dname ]]; then
    echo "Making $dname"
    python -u ./convert_ws.py -f $ws_file -x -d $data_name --setconst $setconst -b $binning  -o $dname >! $dlog
else
  echo "File $dname already exists"
fi

if [[ ! -e $mname ]]; then
    echo "Making $mname"
    python -u ./convert_ws.py -f $ws_file --setval $scan_var=$scan_val,$setval --setconst $setconst \
                              --default-sample $default_sample -d $data_name --refit $refit --binned --setrange $setrange \
                              -b $binning  -o $mname -l $vname >! $mlog
else
  echo "File $mname already exists"
fi

if [[ ! -e $fname ]]; then
    echo "Making $fname"
    python -u ./fit_ws.py -f $ws_file --setval $scan_var=$scan_val,$setval --setconst $setconst -y $hypos --setrange $setrange \
                          -d $data_name --binned -o $fname >! $flog
else
  echo "File $fname already exists"
fi
