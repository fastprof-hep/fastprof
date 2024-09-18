from fastprof import POIHypo, Parameters, Model, Data, OptiMinimizer, NPMinimizer, Raster, ParBound

from fastprof_utils import process_setval_list, process_setvals

def make_model(options) :
  model = Model.create(options.model_file, verbosity=options.verbosity)
  if model is None : raise ValueError('No valid model definition found in file %s.' % options.model_file)
  return model

def make_data(model, options) :
  if options.data_file :
    data = Data(model).load(options.data_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    print('Using dataset stored in file %s.' % options.data_file)
  elif options.asimov is not None :
    try:
      sets = [ v.replace(' ', '').split('=') for v in options.asimov.split(',') ]
      sets_dict = { name : value for name, value in sets }
      data = model.generate_expected(sets_dict)
    except Exception as inst :
      print(inst)
      raise ValueError("Cannot define an Asimov dataset from options '%s'." % options.asimov)
    print('Using Asimov dataset with POIs %s.' % str(sets))
  else :
    data = Data(model).load(options.model_file)
    if data == None : raise ValueError('No valid dataset definition found in file %s.' % options.data_file)
    print('Using dataset stored in file %s.' % options.model_file)
  return data

def make_hypos(model, options) :
  try :
    hypos = [ POIHypo(setval_dict) for setval_dict in process_setval_list(options.hypos, model) ]
  except Exception as inst :
    print(inst)
    raise ValueError("Could not parse list of hypothesis values '%s' : expected |-separated list of variable assignments" % options.hypos)
  return hypos

def set_pois(model, data, options) :
  if options.setval is not None :
    try :
      poi_dict = process_setvals(options.setval, model, match_nps = False)
    except Exception as inst :
      print(inst)
      raise ValueError("ERROR : invalid POI specification string '%s'." % options.setval)
    pars = model.expected_pars(poi_dict)
    if options.profile :
      mini = OptiMinimizer()
      mini.profile_nps(pars, data)
      print('Minimum: nll = %g @ parameter values : %s' % (mini.min_nll, mini.min_pars))
      pars = mini.min_pars
  elif data is not None and options.profile :
    mini = OptiMinimizer().set_pois(model)
    mini.minimize(data)
    pars = mini.min_pars
  else :
    pars = model.ref_pars
  return pars

def init_calc(calc, model, options) :
  par_bounds = []
  if options.bounds :
    bound_specs = options.bounds.split(',')
    try :
      for spec in bound_specs :
        var_range = spec.split('=')
        range_spec = var_range[1].split(':')
        if len(range_spec) == 2 :
          par_bounds.append(ParBound(var_range[0], float(range_spec[0]) if range_spec[0] != '' else None, float(range_spec[1]) if range_spec[1] != '' else None))
        elif len(range_spec) == 1 :
          par_bounds.append(ParBound(var_range[0], float(range_spec[0]), float(range_spec[0]))) # case of fixed parameter
    except Exception as inst:
      print(inst)
      raise ValueError('Could not parse parameter bound specification "%s", expected in the form name1=[min]#[max],name2=[min]#[max],...' % options.bounds)

  for bound in par_bounds :
    print('Overriding bound %s by user-provided %s' % (str(calc.minimizer.bounds[bound.par]), str(bound)))
  calc.minimizer.set_pois(model, bounds=par_bounds)
  return par_bounds


def try_loading_results(model, raster_file, options) :
  try :
    faster = Raster('fast', model=model)
    faster.load(raster_file)
  except FileNotFoundError :
    faster = None
  if faster is not None and options.overwrite :
    print("INFO: will recompute results and overwrite output file '%s' as requested." % raster_file)
    faster = None
  if faster is not None :
    print("INFO: using cached results from file '%s'." % raster_file)
  return faster
