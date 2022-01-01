#! /usr/bin/env python

__doc__ = """
*Inspect a ROOT workspace*

The script takes as input a ROOT workspace, and plots the PDF information.

* The model POIs and NPs are taken from the ModelConfig file.

* The model PDF is also taken from the ModelConfig. Only the case of a binned or unbinned 
  distribution is considered.

* If the PDF is a `RooSimultaneous`*, each component is inspected separately 

The output is a PDF file containing the mode information.
"""
__author__ = "N. Berger <Nicolas.Berger@cern.ch"

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import ROOT
from fastprof_import.tools import make_asimov

####################################################################################################################################
###

class WSChannel() : pass

def make_parser() :
  parser = ArgumentParser("inspect_ws.py", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.description = __doc__
  parser.add_argument("-f", "--ws-file"          , type=str  , required=True    , help="Name of file containing the workspace")
  parser.add_argument("-w", "--ws-name"          , type=str  , default='modelWS', help="Name workspace object inside the specified file")
  parser.add_argument("-m", "--model-config-name", type=str  , default='mconfig', help="Name of model config within the specified workspace")
  parser.add_argument("-c", "--channels"         , type=str  , default=None     , help="Names of the model channels, in the form c1,c2,..., in the same order as the RooSimPdf components")
  parser.add_argument("-d", "--data-name"        , type=str  , default=''       , help="Name of dataset object within the input workspace")
  parser.add_argument("-a", "--asimov"           , type=str  , default=None     , help="Perform an Asimov fit before conversion")
  parser.add_argument("-o", "--output-file"      , type=str  , required=True    , help="Name of output file")
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

  # 1 - Retrieve workspace contents
  # ---------------------------------------------------------

  f = ROOT.TFile(options.ws_file)
  if not f or not f.IsOpen() :
    raise FileNotFoundError('Cannot open file %s' % options.ws_file)

  ws = f.Get(options.ws_name)
  if not ws :
    raise KeyError('Workspace %s not found in file %s.' % (options.ws_name, options.ws_file))

  mconfig = ws.obj(options.model_config_name)
  if not mconfig : raise KeyError('Model config %s not found in workspace.' % options.model_config_name)

  main_pdf = mconfig.GetPdf()

  data = None
  if options.data_name != '' :
    data = ws.data(options.data_name)
    if data == None :
      ds = [ d.GetName() for d in ws.allData() ]
      raise KeyError('Dataset %s not found in workspace. Available datasets are: %s' % (options.data_name, ', '.join(ds)))
  elif options.asimov != None :
    data = make_asimov(mconfig, options.asimov)
  else:
    raise ValueError('ERROR: no dataset was specified either using --data-name or --asimov')
  

  # 2. Identify the model channels
  # ------------------------------
  
  user_channels = {}
  if options.channels is not None :
    try:
      channel_specs = options.channels.split(',')
      for channel_spec in channel_specs :
        fields = channel_spec.split(':')
        user_channels[fields[0]] = fields[1] if len(fields) == 2 else ''  
    except Exception as inst :
      print(inst)
      raise ValueError('Invalid channel name specification %s : should be of the form index1:name1,index2:name2,...' % options.channels)

  channels = []

  for component in main_pdf.getComponents() :
    if isinstance(component, ROOT.RooSimultaneous) :
      cat = component.indexCat()
      for i in range(0, cat.size()) :
        channel = WSChannel()
        channel_name = cat.lookupType(i).GetName()
        if len(user_channels) > 0 and not str(i) in user_channels and not channel_name in user_channels : continue
        channel.pdf = component.getPdf(channel_name)
        channel.name = user_channels[str(i)] if str(i) in user_channels and user_channels[str(i)] != '' else channel_name
        channel.index = i
        channel.cat = cat
        channels.append(channel)
      break
    elif isinstance(component, ROOT.RooAddPdf) :
      channel = WSChannel()
      channel.pdf = component
      channel.name = user_channels[0] if 0 in user_channels else component.GetName()
      channel.cat = None
      channels.append(channel)
      break

  for channel in channels : 
    channel_obs = mconfig.GetObservables().selectCommon(channel.pdf.getVariables())
    if channel_obs.getSize() == 0 :
      raise ValueError('Cannot identify observables for channel %s.')
    if channel_obs.getSize() > 1 :
      raise ValueError('Channel %s has %d observables -- multiple observables not supported (yet).')
    channel.obs = ROOT.RooArgList(channel_obs).at(0)
    channel.nevt = channel.pdf.expectedEvents(0)

  canvas = ROOT.TCanvas("canvas", "Per-channel plots", 800, 600)

  # 3 - Dump the model information
  # ------------------------------

  for i, channel in enumerate(channels) :
    frame = channel.obs.frame()
    if channel.cat is None :
      channel_data = data
    else :
      channel_data = data.reduce('%s==%d' % (channel.cat.GetName(), channel.index))
    channel_data.plotOn(frame)
    channel.pdf.plotOn(frame, ROOT.RooFit.Normalization(channel.nevt, ROOT.RooAbsReal.NumEvent))
    frame.SetTitle(channel.name)
    frame.Draw()
    if len(channels) > 1 and i == 0 :
      suffix = '('
    elif len(channels) > 1 and i == len(channels) - 1 :
      suffix = ')'
    else :
      suffix = ''
    canvas.Print(options.output_file + suffix)

if __name__ == '__main__' : run()
