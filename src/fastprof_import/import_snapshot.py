#! /usr/bin/env python

__doc__ = """
*Convert a ROOT workspace into fastprof markup format*

The script takes as input a ROOT workspace, and converts the contents
into the definition file of a linear model, as follows:

* The model POIs and NPs are taken from the ModelConfig file.

* The model PDF is also taken from the ModelConfig. Two cases are currently
  implemented:
  
  * *The PDF is a `RooAddPdf`*: the components of the sum are then 
    used to define the samples of a single channel. The channel
    observable is taken from the PDF, and the binning from the `-b` option.
      
  * *The PDF is a `RooSimultaneous`* : the states of the PDF are taken
    to correspond each to a separate channel. Each channel must have a 
    PDF of *RooAddPdf* type, which is then treated as above.

* Nominal yields in each bin are computed from integrating the PDF of each
  sample in each channel.

* Linear impacts are computed by changing the values of the NPs as 
  specified by the `--variations` option. By default :math:`\pm 1 \sigma`
  variations are used. The impact on each sample are separated by setting
  the normalizations of all but one to zero for each in turn. Variations
  which are present when all normalizations are set to 0 are assigned to
  the default sample, specified by the `--default-sample` option.

* NP central values and uncertainties taken from directly from the workspace,
  or from a fit to data (`--data-file` option) or to an Asimov dataset
  (`--asimov` option). The same applied to the POI value used to define
  the nominal yields. If the POI value leads to a normalization of 0, the POI
  is instead set to twice its uncertainty.
  
* Prior to computations, the model can be adjusted using the `--setval`,
  `--setconst` and `--setrange` options, which set respectively the value,
  constness and range of model parameters.

The output is a single markup file which defines the model as well as the 
dataset if one was specified. A validation file is also produced if the 
`--validation-output` option was set: this contains information that can be
used to assess if the linearity assumption is valid for the model.
"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import json,yaml
import ROOT

class WSChannel() : pass
class WSSample() : pass

####################################################################################################################################
###

def make_parser() :
  parser = ArgumentParser("import_snapshot.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-f", "--ws-file"          , type=str  , required=True    , help="Name of file containing the workspace")
  parser.add_argument("-w", "--ws-name"          , type=str  , default='modelWS', help="Name workspace object inside the specified file")
  parser.add_argument("-s", "--snapshot"         , type=str  , required=True    , help="Name of the relevant snapshot")
  parser.add_argument("-o", "--output-file"      , type=str  , required=True    , help="Name of output file")
  parser.add_argument(      "--markup"           , type=str  , default='json'   , help="Output markup flavor (supported : 'json', 'yaml')")
  parser.add_argument("-v", "--verbosity"        , type=int  , default=0        , help="Verbosity level")
  return parser

def run(argv = None) :
  """
  Main method in the script
  """
  parser = make_parser()
  options = parser.parse_args(argv)
  if not options :
    parser.print_help()
    sys.exit(0)

  f = ROOT.TFile(options.ws_file)
  if not f or not f.IsOpen() :
    raise FileNotFoundError('Cannot open file %s' % options.ws_file)

  ws = f.Get(options.ws_name)
  if not ws :
    raise KeyError('Workspace %s not found in file %s.' % (options.ws_name, options.ws_file))

  snapshot = ws.getSnapshot(options.snapshot)
  if not snapshot :
    raise KeyError("Snapshot '%s' not found in workspace '%s' of file '%s'." % (options.snapshot, options.ws_name, options.ws_file))
  
  pars_dict = {}
  pars_list = ROOT.RooArgList(snapshot)
  for i in range(0, pars_list.getSize()) :
    try :
      par = pars_list.at(i)
      pars_dict[par.GetName()] = par.getVal()
    except :
      print("Skipping non-real parameter '%s'." % par.GetName())

  with open(options.output_file, 'w') as fd:
    if options.markup == 'json' : 
      json.dump(pars_dict, fd, ensure_ascii=True, indent=3)
    else :
      yaml.dump(pars_dict, fd, sort_keys=False, default_flow_style=None, width=10000)


if __name__ == '__main__' : run()
