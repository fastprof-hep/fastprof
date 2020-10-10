import ROOT

def process_setvals(setvals, ws, parse_only = False) :
  try:
    sets = [ v.replace(' ', '').split('=') for v in setvals.split(',') ]
    output = []
    for (var, val) in sets :
      if not ws.var(var) :
        raise ValueError("ERROR: Cannot find variable '%s' in workspace" % var)
      save_val = ws.var(var).getVal()
      if not parse_only :
        ws.var(var).setVal(float(val))
        print('Setting %s = %g' % (var, float(val)))
      output.append((ws.var(var), float(val), save_val))
  except Exception as inst :
    print(inst)
    raise ValueError("ERROR : invalid variable assignment string '%s'." % setvals)
  return output


def process_setconsts(setconsts, ws) :
  varlist = setconsts.split(',')
  for var in varlist :
    matching_vars = ROOT.RooArgList(ws.allVars().selectByName(var))
    if matching_vars.getSize() == 0 :
      print("ERROR : no variables matching '%s' in model" % var)
      raise ValueError
    for i in range(0, matching_vars.getSize()) :
      thisvar =  matching_vars.at(i)
      thisvar.setConstant()
      print("INFO : setting variable '%s' constant (current value: %g)" % (thisvar.GetName(), thisvar.getVal()))


def process_setranges(setranges, ws) :
  try:
    sets = [ v.replace(' ', '').split(':') for v in setranges.split(',') ]
    for (var, minval, maxval) in sets :
      if not ws.var(var) :
        raise ValueError("Cannot find variable '%s' in workspace" % var)
      if minval == '' : 
        ws.var(var).setMax(float(maxval))
        print("INFO : setting upper bound of %s to %g" % (var, float(maxval)))
      elif maxval == '' :
        ws.var(var).setMin(float(minval))
        print("INFO : setting lower bound of %s to %g" % (var, float(minval)))
      else :
        ws.var(var).setRange(float(minval), float(maxval))
        print("INFO : setting range of %s to [%g, %g]" % (var, float(minval), float(maxval)))
  except Exception as inst :
    print(inst)
    raise ValueError("ERROR : invalid variable range specification '%s'." % setranges)


def make_binned(dataset, rebinnings) :
  for obs, bins in rebinnings.items() : obs.setBins(bins)
  return dataset.binnedClone()


def fit(pdf, dataset, robust = False, n_max = 3, ref_nll = 0, silent=True) :
  print("INFO : fit initial state = ")
  pdf.getVariables().Print('V')
  result = pdf.fitTo(dataset, ROOT.RooFit.Offset(), ROOT.RooFit.SumW2Error(False), ROOT.RooFit.Minimizer('Minuit2', 'migrad'), ROOT.RooFit.Hesse(True),
                     ROOT.RooFit.Save(), ROOT.RooFit.Verbose(not silent), ROOT.RooFit.PrintLevel(-1 if silent else 1))
  if robust and (result.status() != 0 or abs(result.minNll() - ref_nll) > 1) :
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

