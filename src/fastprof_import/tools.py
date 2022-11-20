import ROOT
import numpy as np

def process_setvals(setvals, ws = None, model = None, init_pars = None) :
  try:
    sets = [ v.replace(' ', '').split('=') for v in setvals.split(',') ]
    output = []
    for (var, val) in sets :
      if ws is not None :
        if not ws.var(var) :
          raise ValueError("ERROR: Cannot find variable '%s' in workspace" % var)
        save_val = ws.var(var).getVal()
        ws.var(var).setVal(float(val))
        print('Setting %s = %g' % (var, float(val)))
      elif model is not None :
        if not var in model.config.parameters :
          raise ValueError("ERROR: Cannot find variable '%s' in model" % var)
        save_val = init_pars[model.config.par_slice(k)]
        if len(save_val) != 1 :
          raise ValueError("ERROR: invalid variable '%s' in model" % var)
        save_val = save_val[0]
        init_pars[model.config.par_slice(var)] = float(val)
        print('Setting %s = %g' % (var, float(val)))
      else :
        save_val = None
      output.append((ws.var(var), float(val), save_val))
  except Exception as inst :
    print(inst)
    raise ValueError("ERROR : invalid variable assignment string '%s'." % setvals)
  return output


def process_setconsts(setconsts, ws = None, model = None, init_pars = None, fixed_pars = None) :
  if ws is None and (model is None or init_pars is None or fixed_pars is None) :
    raise ValueError('ERROR: should specify either a model or a workspace to work with.')
  varlist = setconsts.split(',')
  for var in varlist :
    if ws is not None :
      if var.find('*') >= 0 :
        matching_vars = ROOT.RooArgList(ws.allVars().selectByName(var))
        if matching_vars.getSize() == 0 :
          raise ValueError("ERROR : no variables matching '%s' in model" % var)
      else :
        var_obj = ws.var(var)
        if var_obj is None : raise ValueError("ERROR : variable '%s' not found in model" % var)
        matching_vars = ROOT.RooArgList(var_obj)
      for i in range(0, matching_vars.getSize()) :
        thisvar =  matching_vars.at(i)
        thisvar.setConstant(const)
        print("INFO : setting variable '%s' %s (current value: %g)" % (thisvar.GetName(), 'constant' if const else 'free', thisvar.getVal()))
    elif model is not None :
      matching_vars = [ par for par in model.config.parameters if re.match(var, par) ]
      if len(matching_vars) == 0 :
        raise ValueError("ERROR : no variables matching '%s' in model" % var)
      for var in matching_vars :
        current_val = init_pars[model.config.par_slice(var)]
        if len(current_val) != 1 :
          raise ValueError("ERROR: invalid variable '%s' in model" % var)
        fixed_pars[model.config.par_slice(var)] = True
        print("INFO : setting variable '%s' %s (current value: %g)" % (var, 'constant' if const else 'free', current_val))


def process_setranges(setranges, ws = None, model = None, par_ranges = None) :
  if ws is None and (model is None or par_ranges is None) :
    raise ValueError('ERROR: should specify either a model or a workspace to work with.')
  try:
    sets = [ v.replace(' ', '').split('=') for v in setranges.split(',') ]
    for (var, var_range) in sets :
      if ws is not None :
        if var.find('*') >= 0 :
          matching_vars = ROOT.RooArgList(ws.allVars().selectByName(var))
          if matching_vars.getSize() == 0 :
            raise ValueError("ERROR : no variables matching '%s' in model" % var)
        else :
          var_obj = ws.var(var)
          if not var_obj : raise ValueError("Cannot find variable '%s' in workspace" % var)
          matching_vars = ROOT.RooArgList(var_obj)
        minval, maxval = var_range.split(':')
        for i in range(0, matching_vars.getSize()) :
          thisvar =  matching_vars.at(i)
          if minval == '' : 
            thisvar.setMax(float(maxval))
            print("INFO : setting upper bound of %s to %g" % (thisvar.GetName(), float(maxval)))
          elif maxval == '' :
            thisvar.setMin(float(minval))
            print("INFO : setting lower bound of %s to %g" % (thisvar.GetName(), float(minval)))
          else :
            thisvar.setRange(float(minval), float(maxval))
            print("INFO : setting range of %s to [%g, %g]" % (thisvar.GetName(), float(minval), float(maxval)))
      elif model is not None :
        matching_vars = [ par for par in model.config.parameters if re.match(var, par) ]
        if len(matching_vars) == 0 :
          raise ValueError("ERROR : no variables matching '%s' in model" % var)
        minval, maxval = var_range.split(':')
        if minval == '' : 
          minval = None
          maxval = float(maxval)
          print("INFO : setting upper bound of %s to %g" % (var, maxval))
        elif maxval == '' :
          minval = float(minval)
          maxval = None
          print("INFO : setting lower bound of %s to %g" % (var, minval))
        else :
          minval = float(minval)
          maxval = float(maxval)
          print("INFO : setting range of %s to [%g, %g]" % (var, minval, maxval))
        for var in matching_vars :
          min_max = par_ranges[model.config.par_slice(var)]
        if len(min_max) != 1 :
          raise ValueError("ERROR: invalid variable '%s' in model" % var)
        min_max = [ (minval, maxval) ]
  except Exception as inst :
    print(inst)
    raise ValueError("ERROR : invalid variable range specification '%s'." % setranges)


def make_binned(dataset, rebinnings) :
  for obs, bins in rebinnings.items() : obs.setBins(bins)
  return dataset.binnedClone()


def fit(pdf, dataset, robust = False, n_max = 3, ref_nll = 0, silent=True) :
  result = pdf.fitTo(dataset, ROOT.RooFit.Offset(), ROOT.RooFit.SumW2Error(False), ROOT.RooFit.Minimizer('Minuit2', 'migrad'), ROOT.RooFit.Hesse(True),
                     ROOT.RooFit.Save(), ROOT.RooFit.Verbose(not silent), ROOT.RooFit.PrintLevel(-1 if silent else 1))
  if robust and (result.status() != 0 or abs(result.minNll() - ref_nll) > 1) and n_max > 0 :
    return fit(pdf, dataset, robust, n_max - 1, result.minNll())
  else :
    return result

def make_asimov(setvals, mconfig, pdf = None, data = None, robust = True) :
  saves = process_setval(setvals)
  if data : fit(pdf, data, robust)
  asimov = ROOT.RooStats.AsymptoticCalculator.MakeAsimovData(mconfig, ROOT.RooArgSet(), ROOT.RooArgSet())
  print('INFO: generating Asimov for parameter values :')
  for (var, val, save_val) in saves : print("INFO :   %s=%g" % (var.GetName(), val)) 
  print('INFO: and NP values :')
  mconfig.GetNuisanceParameters().selectByAttrib('Constant', False).Print('V')
  for (var, val, save_val) in saves : var.setVal(save_val)
  return asimov

def format_float(x, num_digits = 7) :
  if x == 0 or abs(np.log10(abs(x))) < 4 : return np.format_float_positional(x, num_digits, True, False, trim='-')
  return np.format_float_scientific(x, num_digits, trim='-')

def trim_float(x, num_digits = 7) :
  return float(np.format_float_scientific(x, num_digits, trim='-'))
