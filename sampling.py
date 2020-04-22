import pyhf
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import os
import fastprof

spec = json.load(open('fastprof/models/test1.json', 'r'))
parameters = [ 1.0, 0.0, 0.0 ]

ws = pyhf.Workspace(spec)
model = ws.model()

data = ws.data(model)
main_data = ws.data(model, False)

n_poi = 1
n_np = len(data) - len(main_data)

calc = pyhf.infer.calculators.AsymptoticCalculator(data, model)

mus = np.linspace(0.1, 4.1, 11)
print('Will test the following hypotheses: ', mus)

def update_bins(data, mu, n_np) :
  params = [mu] + n_np*[0]
  expected = model.expected_data(params)
  for i in range(0, len(data) - n_np) :
    data[i] = np.random.poisson(expected[i])
  
def update_aux(data, n_np) :
  for i in range(len(data) - n_np, len(data)) :
    data[i] = np.random.normal(0, 1)

def clsb(mu, data) :
  return pyhf.infer.hypotest(mu, data, model, return_tail_probs = True)[1][0]

def profile(mu, data) :
  return pyhf.infer.mle.fixed_poi_fit(mu, data, model, return_fitted_val=True)

# debug : each toy stores data_bin1, .., data_binN, aux_NP1, ... aux_NPN, fitval_NP1 ... fitval_NPN, profA, profB, cl
def sampling_dist(mu, n_np, ntoys = 100, debug = False) :
  cls = np.zeros(ntoys)
  if debug : debug_info = np.zeros((ntoys, len(data) + 2*n_np + 1))
  sig = np.array([1.0, 0])*mu  # specific to this case!
  bkg = np.array([1.0, 10.0])  # specific to this case!
  a = np.array([[0.2], [0.2]]) # specific to this case!
  b = np.array([[0.2], [0.2]]) # specific to this case!
  for k in range(0, ntoys) :
    if k % 100 == 0 : print(k)
    update_bins(data, mu, n_np)
    update_aux(data, n_np)
    cls[k] = clsb(mu, data)
    if debug:
      pars, val = profile(mu, data)
      debug_info[k, :len(data)] = data
      debug_info[k, len(data):len(data) + n_np] = pars[1:]
      debug_info[k, -1] = cls[k]
      n   = np.array(data[:-n_np])
      alpha = np.array([ data[3] ]) # specific to this case!
      beta  = np.array([ data[2] ]) # specific to this case!
      model = fastprof.Model(n, sig, bkg, alpha, beta, a, b)
      prof_alpha, prof_beta = fastprof.profile(model)
      debug_info[k, len(data) + n_np:-1] = prof_alpha[0], prof_beta[0]
  if debug : return cls, debug_info
  return cls


def save_samples(mus, n_np, ntoys, samples_file = 'samples', break_lock = False, debug_file = None) :
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
      dist, debug_info = sampling_dist(mu, n_np, ntoys, True)
    else :
      dist = sampling_dist(mu, n_np, ntoys)
    np.sort(dist)
    np.save(filename, dist)
    if debug_file :
      debug_filename = debug_file + '_%g' % mu
      np.save(debug_filename, debug_info)
    print('Done')
    os.remove(filename + '.lock')

def load_samples(mus, samples_file = 'samples') :
  samples = {}
  for mu in mus :
    filename = samples_file + '_%g.npy' % mu
    try:
      dist = np.load(filename)
    except:
      print('File %s not found, for samples at mu = %g' % (filename, mu))
      return {}
    
    samples[mu] = np.sort(dist)
  return samples

def sampling_cl(mu, cl, samples) :
  return np.searchsorted(samples[mu], cl)/len(samples[mu])

def sampling_cls(mu, data) :
  return np.array([ sampling_cl(mu, clsb(mu, data), samples) for mu in mus ])

def fast_sampling_cls(mu, data) :
  return np.array([ sampling_cl(mu, clsb(mu, data), fast_samples) for mu in mus ])

def asymptotic_cls(mus, data) :
 return np.array([ clsb(mu, data) for mu in mus ])

def find_hypo(mus, cls, cl = 0.05) :
  logcls = [ math.log(c) for c in cls ]
  finder = scipy.interpolate.interp1d(logcls, mus, 'quadratic')
  #print(logcls, math.log(cl))
  return finder(math.log(cl))


np.random.seed(131071)
save_samples(mus, 2, 20000, 'samples/test1', False, 'samples/test1_debug')
samples = load_samples(mus, 'samples/test1')
fast_samples = load_samples(mus, 'samples/fast_test1')

fig = plt.figure()
plt.ion()
#cls = sampling_dist(1, 2, 2000)
#plt.hist(cls, bins = 10, range = (0, 1))
#plt.show()
#f = lambda x: sampling_cl(mus[2], x, samples)
#x = np.linspace(0, 0.5, 100)
#plt.plot(x, f(x))
acls = asymptotic_cls(mus, data)
scls = sampling_cls(mus, data)
fcls = fast_sampling_cls(mus, data)
plt.plot(mus, acls, 'r')
plt.plot(mus, scls, 'b')
plt.plot(mus, fcls, 'g')
print(find_hypo(mus, acls))
print(find_hypo(mus, scls))
print(find_hypo(mus, fcls))
plt.show()
