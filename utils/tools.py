"""
Common functions for utils/ scripts

"""

import re
from fastprof import Model
import numpy as np


def process_setvals(setvals : str, model : Model, match_pois : bool = True, match_nps : bool = True, check_val : bool = True) -> dict :
  """Parse a set of model POI value assignments

  The input string is expected in the form

  par1=val1,par2=val2,...

  The return value is then the dict

  { "par1" : val1, "par2" : val2, ...}

  The assignments are parsed, and not applied to the model
  parameters (the values of which are not stored in the model)

  Exception are raised if the `parX` are not POIs of the
  model, or if the `valX` are not float values.

  Args:
    setvals : a string specifying parameter assignements
    model   : model containing the parameters

  Returns:
    the parsed POI assignments, as a dict in the form { par_name : par_value }
  """
  par_dict = {}
  try:
    sets = [ a.replace(' ', '').split('=') for a in setvals.split(',') ]
  except Exception as inst :
    print(inst)
    raise ValueError("ERROR : invalid variable assignment specification '%s'." % setvals)
  for (var, val) in sets :
    if match_pois :
      matching_pois = [ p for p in model.pois.keys() if re.match(var, p) ]
      if len(matching_pois) == 0 and not match_nps : raise ValueError("No POIs matching '%s' defined in model." % var)
    else : matching_pois = []
    if match_nps :
      matching_nps = [ p for p in model.nps.keys() if re.match(var, p) ]
      if len(matching_nps) == 0 and not match_pois : raise ValueError("No NPs matching '%s' defined in model." % var)
    else : matching_nps = []
    matching_all = matching_pois + matching_nps
    if len(matching_all) == 0 : raise ValueError("No parameters matching '%s' defined in model." % var)
    try :
      float_val = float(val)
    except ValueError as inst :
      if check_val : raise ValueError("Invalid numerical value '%s' in assignment to variable(s) '%s'." % (val, var))
      float_val = None
    for var in matching_all : par_dict[var] = float_val if float_val is not None else val
  return par_dict

def process_values_spec(spec : str) :
  values = []
  try:
    range_specs = spec.split('|')
    for range_spec in range_specs :
      range_fields = range_spec.split(':')
      if len(range_fields) == 1 :
        values.append(float(range_fields[0]))
        continue
      add_last = False
      if range_fields[2][-1] == '+' :
        add_last = True
        range_fields[2] = range_fields[2][:-1]
      nbins = int(range_fields[2])
      if len(range_fields) == 4 and range_fields[3] == 'log' :
        values.extend(np.logspace(1, math.log(float(range_fields[1]))/math.log(float(range_fields[0])), nbins, add_last, float(range_fields[0])))
      else :
        values.extend(np.linspace(float(range_fields[0]), float(range_fields[1]), nbins, add_last))
  except Exception as inst :
    print(inst)
    raise ValueError('Invalid value range specification %s : the format should be xmin[:xmax:nbins[:log]]|...|...' % spec)
  return values


def process_setval_list(setvals : str, model : Model, match_pois : bool = True, match_nps : bool = True) -> dict :
  return process_setval_dicts([ process_setvals(spec, model, check_val=False) for spec in setvals.split('#') ])
  
  
def process_setval_dicts(setval_dicts : dict) -> dict :
  new_svds = []
  any_expanded = False
  for svd in setval_dicts :
    has_expanded = False
    this_svds = []
    this_new = {}
    for var, val in svd.items() :
      if not has_expanded :
        if isinstance(val, float) :
          this_new[var] = val
        else :
          newvals = process_values_spec(val)
          this_svds = [ {**this_new, var : newval } for newval in newvals ]
          has_expanded = True
          any_expanded = True
      else :
        for d in this_svds : d[var] = val
    if not has_expanded :
      new_svds.append(this_new)
    else :
      new_svds.extend(this_svds)
  if not any_expanded : return new_svds
  return process_setval_dicts(new_svds)

def process_setranges(setranges : str, model : Model) :
  """Parse a set of POI range assignments

  The input string is expected in the form

  par1:[min1]:[max1],par2:[min2]:[max2],...

  where either the `minX` or the `maxX` values can be omitted (but not the ':' separator!)
  to indicate open ranges. The

  The parameter ranges are applied directly on the model
  parameters

  Exception are raised if the `parX` parameters are not defined
  in the model, or if the `minX` or `maxX` are not float values.

  Args:
    setvals : a string specifying the parameter ranges
    model   : model containing the parameters
  """
  try:
    sets = [ v.replace(' ', '').split(':') for v in setranges.split(',') ]
    for (var, minval, maxval) in sets :
      if not var in model.pois : raise ValueError("Parameter of interest '%s' not defined in model." % var)
      if minval != '' :
        try :
          float_minval = float(minval)
        except ValueError as inst :
          raise ValueError("Invalid numerical value '%s' for the lower bound of variable '%s'." % (minval, var))
        model.pois[var].min_value = float_minval
        print("INFO : setting lower bound of %s to %g" % (var, float_minval))
      if maxval != '' :
        try :
          float_maxval = float(maxval)
        except ValueError as inst :
          raise ValueError("Invalid numerical value '%s' for the upper bound of variable '%s'." % (maxval, var))
        model.pois[var].max_value = float_maxval
        print("INFO : setting upper bound of %s to %g" % (var, float_maxval))
  except Exception as inst :
    print(inst)
    raise ValueError("ERROR : invalid parameter range specification '%s'." % setranges)
