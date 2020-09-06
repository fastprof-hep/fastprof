import json
import math
import scipy
import numpy as np
from abc import abstractmethod

from .core import Model, Data, Parameters, JSONSerializable, ModelPOI
from .minimizers import OptiMinimizer
from .test_statistics import QMu, QMuTilda


class FitResult(JSONSerializable) :
  def __init__(self, name = '', fitpars = None, nll = None, model=None) :
    super().__init__()
    self.name = name
    self.fitpars = { name : ModelPOI(name, value=fitpars[name]) for name in model.all_pars() } if not fitpars is None and not model is None else {}
    self.nll = nll
    self.model = model
  def pars(self) :
    par_dict = { par.name : par.value for par in self.fitpars }
    return Parameters(par_dict, model=self.model).set_from_dict(par_dict)
  def load_jdict(self, jdict) :
    for par_name, par_dict in jdict['fit_pars'].items() :
      fitpar = ModelPOI(par_name).load_jdict(par_dict)
      self.fitpars[fitpar.name] = fitpar
    self.nll = jdict['nll']
    return self
  def save_jdict(self, jdict) :
    jdict['fit_pars'] = { par.name : par.dump_jdict() for par in self.fitpars }
    jdict['nll'] = self.nll
  def __str__(self) :
    return  "  Fit '%s' : nll = %g, pars : %s" % (self.name, self.nll, str(self.pars))


class PLRData(JSONSerializable) :
  def __init__(self, name = '', hypo = None, free_fit = None, hypo_fit = None, test_statistics = None, pvs = None, asimov = None, model = None) :
    super().__init__()
    self.name = name
    self.hypo = hypo
    self.free_fit = free_fit
    self.hypo_fit = hypo_fit
    self.test_statistics = test_statistics if test_statistics != None else {}
    self.pvs = pvs if pvs != None else {}
    self.asimov = asimov
    self.model = model
    if self.free_fit != None and self.hypo_fit != None and not 'tmu' in self.test_statistics : self.compute_tmu()
  def pois(self) :
    return self.hypo.model.pois
  def hypo_pars(self) :
    return Parameters(model=self.model).set_from_dict(self.hypo)
  def compute_tmu(self) :
    self.test_statistics['tmu'] = -2*(self.hypo_fit.nll - self.free_fit.nll)
  def set_asimov(self, asimov_plr_data, local_key = 'tmu_0') :
    self.test_statistics[local_key] = asimov_plr_data.test_statistics['tmu']
    self.asimov = asimov_plr_data
  def load_jdict(self, jdict) :
    self.hypo = Parameters(jdict['hypo'], model=self.model)
    self.free_fit = FitResult('free_fit', model=self.model).load_jdict(jdict['free_fit'])
    self.hypo_fit = FitResult('hypo_fit', model=self.model).load_jdict(jdict['hypo_fit'])
    self.test_statistics = jdict['test_statistics'] if 'test_statistics' in jdict else {}
    self.pvs = jdict['pvs'] if 'pvs' in jdict else {}
    if not 'tmu' in self.test_statistics : self.compute_tmu()
    return self
  def save_jdict(self, jdict) :
    jdict['free_fit'] = self.free_fit.dump_jdict()
    jdict['hypo_fit'] = self.hypo_fit.dump_jdict()
    jdict['test_statistics'] = self.test_statistics
    jdict['pvs'] = self.pvs
  def __str__(self) :
    s = 'Profile-likelihood ratio data for hypothesis:' + str(self.hypo)
    s += '\n  test statistics : %s' % str(self.test_statistics)
    s += '\n  p-values : %s' % str(self.pvs)
    s += '\n  Unconditional fit:' + str(self.free_fit)
    s += '\n  Conditional fit:' + str(self.hypo_fit)


class Raster(JSONSerializable) :
  def __init__(self, name = '', plr_data = None, use_global_best_fit = True, fill_missing = True, filename = '', load_asimov = 'asimov', model = None) :
    super().__init__()
    self.name = name
    self.model = model
    self.plr_data = plr_data if plr_data != None else {}
    self.use_global_best_fit = use_global_best_fit
    self.fill_missing = fill_missing
    if filename != '' : self.load_with_asimov(filename, 'asimov') if load_asimov else self.load(filename)
    if self.use_global_best_fit : self.set_global_best_fit()
    if self.fill_missing : self.compute_tmu()

  def pois(self) :
    return list(self.plr_data.values())[0].pois() if len(self.plr_data) > 0 else None

  def set_global_best_fit(self) :
    if len(self.plr_data) == 0 : return
    nlls = { hypo : plr_data.free_fit.nll for hypo, plr_data in self.plr_data.items() }
    best_hypo = min(nlls, key=nlls.get)
    for plr_data in self.plr_data : plr_data.free_fit = self.plr_data[best_hypo].free_fit

  def set_asimov(self, asimov_raster, local_key = 'tmu_0') :
    if len(self.plr_data) != len(asimov_raster.plr_data) :
      raise KeyError('Cannot set Asimov scan data with a different number of hypotheses (%d instead of %d).' % (len(self.plr_data), len(asimov_raster.plr_data)))
    for plr_data, asimov_data in zip(self.plr_data.values(), asimov_raster.plr_data.values()) : 
      plr_data.set_asimov(asimov_data, local_key)

  def compute_tmu(self) :
    for plr_data in self.plr_data.values() : plr_data.compute_tmu()

  def compute_limit(self, pv_key = 'cls', cl = 0.05, order = 3, log_scale = True) :
    if len(self.pois()) > 1 : raise ValueError('Cannot interpolate limit in more than 1 dimension.')
    poi_name = list(self.pois())[0]
    hypos = [ hypo[poi_name] for hypo in self.plr_data.keys() ]
    values = []
    for hypo, plr_data in self.plr_data.items() :
      if not pv_key in plr_data.pvs : raise KeyError("P-value '%s' not found at hypo %s." % (pv_key, str(hypo.dict(pois_only=True))))
      if log_scale :
        value = math.log(plr_data.pvs[pv_key]/cl) if plr_data.pvs[pv_key] > 0 else -999
      else :
        value = plr_data.pvs[pv_key] - cl
      values.append(value)
    finder = scipy.interpolate.InterpolatedUnivariateSpline(hypos, values, k=order)
    roots = finder.roots() 
    if len(roots) == 0 :
      print("No solution found for %s[%s] = %g. Interpolation set:" % (pv_key, poi_name, cl))
      print([a for a in zip(hypos, values)])
      return None
    if len(roots) > 1 :
      print('Multiple solutions found for %s[%s] = %g, returning the first one' % (pv_key, poi_name, cl))
    return roots[0]

  def load_jdict(self, jdict) :
    for i, plr_dict in enumerate(jdict[self.name]) :
      plr_data = PLRData('%s_%d' % (self.name, i), model=self.model).load_jdict(plr_dict)
      self.plr_data[plr_data.hypo] = plr_data
    if self.use_global_best_fit : self.set_global_best_fit()
    if self.fill_missing : self.compute_tmu()
    return self

  def fill_jdict(self, jdict) :
    jdict[self.name] = []
    for plr_data in self.plr_data.values() :
      jdict[self.name].append(plr_data.dump_jdict())

  def load_with_asimov(self, filename, asimov_key = 'asimov') :
    self.load(filename)
    asimov_raster = Raster(asimov_key, model=self.model).load(filename)
    self.set_asimov(asimov_raster)
    return self

  def __str__(self) :
    s = ''
    s += 'POIs : ' + str(self.pois().keys()) + '\n'
    s += 'PLR data : '
    for hypo, plr_data in self.plr_data.items() :
      s += '\nHypo :' + str(hypo.dict(pois_only=True))
      s += '\n' + str(pv_result)
    return s

  def key_value(self, key, hypo) :
    if not hypo in self.plr_data : raise KeyError('While trying to access key %s, hypo %s was not found in raster %s.' % (key, str(hypo.dict(pois_only=True)), self.name))
    for poi in self.pois().keys() :
      if key == poi : return hypo[poi]
      if key == 'best_' + poi : return self.plr_data[hypo].free_fitpars[poi].value
    if key in self.plr_data[hypo].pvs : return self.plr_data[hypo].pvs[key]
    if key in self.plr_data[hypo].test_statistics : return self.plr_data[hypo].test_statistics[key]
    raise KeyError('No data found for key %s in hypo %s in raster %s.' % (key, str(hypo.dict(pois_only=True)), self.name))
  
  def print(self, print_keys = None, verbosity = 0, print_limits=True) :
    if len(self.plr_data) == 0 : return ''
    plr_template = list(self.plr_data.values())[0]
    if print_keys == None :
      print_keys = list(self.pois().keys())
      if 'pv' in plr_template.pvs : print_keys += [ 'pv' ]
      if verbosity > 0 :
        if 'cls' in plr_template.pvs : print_keys += [ 'cls' ]
        if 'clb' in plr_template.pvs : print_keys += [ 'clb' ]
      if verbosity > 1 :
        print_keys.extend([ 'tmu' ] + [ 'best_' + k for k in self.pois().keys() ])
    s = ''
    for k in print_keys : s += '| %-15s ' % k
    for hypo, plr_data in self.plr_data.items() :
      s += '\n'
      for key in print_keys :
        if key in self.pois().keys() : 
          s += '| %-15g ' % self.key_value(key, hypo)
    if print_limits and len(self.pois()) == 1 and 'cls' in plr_template.pvs :
      limit = self.compute_limit('cls', 0.05)
      limit_str = '%g' % limit if limit != None else 'not computable'
      s += '\n' + "Asymptotic 95%% CLs limit for raster '%s' = %s" % (self.name, limit_str)
    print(s)
    return s


class TestStatisticCalculator :
  def __init__(self, minimizer) :
    self.minimizer = minimizer

  def poi(self, plr_data) :
    if len(plr_data.pois()) != 1 : raise ValueError('Can currently only compute test statistics for a single POI.')
    return list(plr_data.pois().values())[0]

  @abstractmethod
  def fill_pv(self, plr_data) :
    pass

  def compute_fast_plr(self, hypo, data, name = 'fast') :
    plr_data =  PLRData(name, hypo, model=data.model)
    tmu = self.minimizer.tmu(hypo, data, hypo)
    plr_data.test_statistics['tmu'] = tmu
    plr_data.free_fit = FitResult('free_fit', self.minimizer.free_pars, self.minimizer.free_nll, model=data.model)
    plr_data.hypo_fit = FitResult('hypo_fit', self.minimizer.hypo_pars, self.minimizer.hypo_nll, model=data.model)
    return plr_data

  def compute_fast_q(self, hypo, data) :
    fast_plr_data = self.compute_fast_plr(hypo, data, 'fast')
    asimov = data.model.generate_expected(self.poi(fast_plr_data).value, self.minimizer, data)
    asimov_plr_data = self.compute_fast_plr(hypo, data, 'fast_asimov')
    fast_plr_data.set_asimov(asimov_plr_data)
    self.fill_pv(fast_plr_data)
    return fast_plr_data

  def fill_all_pv(self, raster) :
    for plr_data in raster.plr_data.values() : self.fill_pv(plr_data)

  def compute_fast_results(self, raster, data, name = 'fast') :
    fast_plr_data = {}
    for hypo in raster.plr_data :
      fast_plr_data[hypo] = self.compute_fast_q(hypo, data)
    fast = Raster(name, fast_plr_data)
    self.fill_all_pv(fast)
    return fast


class QMuCalculator(TestStatisticCalculator) :
  def __init__(self, minimizer) :
    super().__init__(minimizer)

  def fill_pv(self, plr_data) :
    try :
      # since we use tmu_A to compute CLb, we need tmu_A = tmu_0 (computed from an Asimov with mu'=0)
      q = QMu(test_poi = plr_data.hypo[self.poi(plr_data).name], tmu = plr_data.test_statistics['tmu'],
              best_poi = plr_data.free_fit.fitpars[self.poi(plr_data).name].value, tmu_A = plr_data.test_statistics['tmu_0'])
      plr_data.test_statistics['q_mu'] = q.value()
      plr_data.pvs['pv' ] = q.asymptotic_pv()
      plr_data.pvs['cls'] = q.asymptotic_cls()
      plr_data.pvs['clb'] = q.asymptotic_clb()
    except Exception as inst:
      print("q_mu computation failed for PLR '%s', hypothesis %s, with exception below:" % (plr_data.name, plr_data.hypo.dict(pois_only=True)))
      raise(inst)
    return self


class QMuTildaCalculator(TestStatisticCalculator) :
  def __init__(self, minimizer) :
    super().__init__(minimizer)
    self.qs = []

  def fill_pv(self, plr_data) :
    try :
      # since we use tmu_A to compute CLb, we need tmu_A = tmu_0 (computed from an Asimov with mu'=0)
      q = QMuTilda(test_poi = plr_data.hypo[self.poi(plr_data).name], tmu = plr_data.test_statistics['tmu'],
                   best_poi = plr_data.free_fit.fitpars[self.poi(plr_data).name].value, tmu_A = plr_data.test_statistics['tmu_0'],
                   tmu_0 = plr_data.test_statistics['tmu_0'])
      plr_data.test_statistics['qm~u'] = q.value()
      plr_data.pvs['pv' ] = q.asymptotic_pv()
      plr_data.pvs['cls'] = q.asymptotic_cls()
      plr_data.pvs['clb'] = q.asymptotic_clb()
    except Exception as inst:
      print("q~mu computation failed for computation '%s', hypothesis %s, with exception below:" % (plr_data.name, plr_data.hypo.dict(pois_only=True)))
      raise(inst)
    return self


class ParBound :
  def __init__(self, par, minval = None, maxval = None) :
    self.par = par
    self.minval = minval
    self.maxval = maxval
  def test(self, pars) :
    try :
      return (pars[self.par] >= self.minval if self.minval != None else True) and (pars[self.par] <= self.maxval if self.maxval != None else True)
    except KeyError :
      return True
  def __str__(self) :
    smin = '%s >= %g' % (self.par, self.minval) if self.minval != None else ''
    smax = '%s <= %g' % (self.par, self.maxval) if self.maxval != None else ''
    if smin == '' : return smax
    if smax == '' : return smin
    return smin + ' and ' + smax
