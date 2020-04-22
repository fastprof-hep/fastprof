import pyhf
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import fastprof
from scipy.interpolate import InterpolatedUnivariateSpline

def fit(model, data) :
  return pyhf.infer.mle.fit(data, model, return_fitted_val=True) # return [mhat, ahat, bhat], nll_min

def clsb(model, mu, data) :
  return pyhf.infer.hypotest(mu, data, model, return_tail_probs = True)[1][0]

def profile(model, mu, data) :
  return pyhf.infer.mle.fixed_poi_fit(mu, data, model, return_fitted_val=True)

# debug : each toy stores data_bin1, .., data_binN, aux_NP1, ... aux_NPN, fitval_mu, fitval_NP1 ... fitval_NPN, profA, profB, cl
def fast_sampling_dist(model, gen_mu, scan_mus, ntoys = 100, debug = False, full_model = None) :
  cls = np.zeros(ntoys)
  if debug : 
    n_dat = model.n_bins() + model.n_syst()
    n_np = model.n_syst()
    debug_info = np.zeros((ntoys, n_dat + 3*n_np + 6))
  gen_hypo = fastprof.Parameters(gen_mu, 0, 0)
  for k in range(0, ntoys) :
    if k % 1000 == 0 : print(k)
    data = model.generate_data(gen_hypo)
    minimizer = fastprof.ScanMinimizer(data, scan_mus)
    nll_min, min_pos = minimizer.minimize(debug)
    nll_hypo = fastprof.NPMinimizer(gen_mu, data).profile_nll()
    q = fastprof.QMu(2*(nll_hypo - nll_min), gen_mu, min_pos)
    cls[k] = q.asymptotic_cl()
    if debug:
      pyhf_data = data.export_pyhf_data(full_model)
      debug_info[k, :n_dat] = pyhf_data # 0,1,2,3 : data
      debug_info[k, n_dat:n_dat + n_np + 2] = [ min_pos, minimizer.min_pars.alpha, minimizer.min_pars.beta, cls[k] ] # 4,5,6,7 : fast best-fit pars & cls
      pars, val = fit(full_model, pyhf_data)
      debug_info[k, n_dat + n_np + 2:n_dat + 2*n_np + 3] = pars # 8,9,10 : best-fit pars 
      debug_info[k, n_dat + 2*n_np + 3:n_dat + 2*n_np + 4] = clsb(full_model, gen_mu, pyhf_data) # 11: CLs+b @ mu=min
      pars, val = profile(full_model, scan_mus[minimizer.min_idx], pyhf_data)
      debug_info[k, n_dat + 2*n_np + 4:n_dat + 3*n_np + 5] = pars # 11,12,13 : full best-fit pars @ mu=min sample
  if debug : return cls, debug_info
  return cls

def fast_save_samples(model, mus, scan_mus, ntoys, samples_file = 'fast_samples', break_lock = False, debug_file = None, full_model = None) :
  for mu in mus :
    filename = samples_file + '_%g' % mu
    if os.path.exists(filename + '.lock') and not break_lock : 
      print('Samples for mu = %g already being produced, skipping' % mu)
      continue
    if os.path.exists(filename + '.npy') and not break_lock :
      print('Samples for mu = %g already done, skipping' % mu)
      continue
    print('Creating sampling distribution for %g' % mu)
    with open(filename + '.lock', 'w') as f :
      f.write(str(os.getpid()))
    if debug_file:
      dist, debug_info = fast_sampling_dist(model, mu, scan_mus, ntoys, True, full_model)
    else :
      dist = fast_sampling_dist(model, mu, scan_mus, ntoys)
    np.sort(dist)
    np.save(filename, dist)
    if debug_file :
      debug_filename = debug_file + '_%g' % mu
      np.save(debug_filename, debug_info)
    print('Done')
    os.remove(filename + '.lock')

if __name__ == "__main__":
  spec = json.load(open('examples/test1.json', 'r'))
  ws = pyhf.Workspace(spec)
  full_model = ws.model()
  fast_model = fastprof.Model(sig = np.array([1.0, 0]),
                              bkg = np.array([1.0, 10.0]),
                              a = np.array([[0.2], [0.2]]),
                              b = np.array([[0.2], [0.2]]))
  gen_mus = np.linspace(0.1, 4.1, 11)
  print('Will generate the following hypotheses: ', gen_mus)
  scan_mus = np.linspace(0, 10, 11)
  print('Will scan over the following hypotheses: ', scan_mus)
  np.random.seed(131071)
  #fast_save_samples(fast_model, gen_mus, scan_mus, 200, 'samples/fast_test1', False, 'samples/fast_debug_test1', full_model)
  fast_save_samples(fast_model, gen_mus, scan_mus, 20000, 'samples/fast_test1')
