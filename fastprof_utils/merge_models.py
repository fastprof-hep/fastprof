#! /usr/bin/env python

__doc__ = """
*Perform a PLR scan over one or more parameters*

"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys, re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fastprof import Model, ModelMerger, SamplePruner
import glob



####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("merge_models.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("input_files"           , type=str  , nargs='+'     , help="List of input files, either comma-separated or specified using wildcards.")
  parser.add_argument("-o", "--output-file"   , type=str  , required=True , help="Name of output file")
  parser.add_argument("-n", "--numeric-sort"  , action='store_true'       , help="Sort input files numerically")
  parser.add_argument("-z", "--min-signif"    , type=float, default=None  , help="Prune away samples with significance below the specified threshold")  
  parser.add_argument("-v", "--verbosity"     , type=int  , default=0     , help="Verbosity level")
  return parser

def run(argv = None) :
  parser = make_parser()
  options = parser.parse_args(argv)
  if not options :
    parser.print_help()
    sys.exit(0)

  files = []
  try :
    for spec in options.input_files :
      if spec.find('*') :
        files.extend(glob.glob(spec))
      else :
        files.append(spec)
  except Exception as inst :
    print(inst)
    print("Could not parse input file specification '%s', expected list of filenames." % options.input_files)
    return 1
  
  if options.numeric_sort :
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('(\W)', key)]
    files = sorted(files, key=alphanum)
    if options.verbosity > 0 : 
      print('Will merge in this order :')
      print('\n'.join(files))
  models = []
  for i, f in enumerate(files) :
    if not os.access(f, os.R_OK) : raise FileNotFoundError("Could not access input file '%s'." % f) 
    print("Loading model file '%s' [%g/%g]." % (f, i+1, len(files)))
    model = Model.create(f, verbosity=options.verbosity)
    if options.min_signif is not None : SamplePruner(model, options.verbosity).prune(options.min_signif)
    models.append(model)

  if options.verbosity > 0 : print('Merging models')
  big_model = ModelMerger(models).merge()
  print("Exporting merged model to '%s'." % options.output_file)
  big_model.save(options.output_file)

if __name__ == '__main__' : run()


