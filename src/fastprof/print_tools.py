"""
Utility classes for model operations

"""
import re
import math
import numpy as np

from .core  import Model, Data, Parameters

# -------------------------------------------------------------------------
def print_model(model : Model, verbosity : int = 1) :
  idt = '   '
  print()
  print('Parameters of interest')
  print('======================\n')
  for par in model.pois.values() : 
    print(' - ' + par.string_repr(verbosity, indent='   '))

  print()
  print('Nuisance parameters')
  print('===================\n')
  for par in model.nps.values() :
    print(' - ' + par.string_repr(verbosity, indent='   '))

  print()
  print('Auxiliary observables')
  print('=====================\n')
  for par in model.aux_obs.values() :
    print(' - ' + par.string_repr(verbosity, indent='   '))

  print()
  print('Channels')
  print('=====================\n')
  for channel in model.channels.values() :
    print(' - ' + channel.string_repr(verbosity, indent='   '))

