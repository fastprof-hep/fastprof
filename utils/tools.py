"""utility functions


"""

import re
from .core import Model


def process_setvals(setvals : str, model : Model, match_pois : bool = True, match_nps : bool = True) -> dict :
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
      raise ValueError("Invalid numerical value '%s' in assignment to variable(s) '%s'." % (val, var))
    for var in matching_all : par_dict[var] = float_val
  return par_dict


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
