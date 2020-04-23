import fastprof
from sampling import SamplingManager, FastSampler, PyhfSampler, DebuggingFastSampler
import pyhf
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

# Define the models

spec = json.load(open('fastprof/models/test1.json', 'r'))
ws = pyhf.Workspace(spec)
pyhf_model = ws.model()
fast_model = fastprof.Model(sig = np.array([1.0, 0]),
                            bkg = np.array([1.0, 10.0]),
                            a = np.array([[0.2], [0.2]]),
                            b = np.array([[0.2], [0.2]]))

gen_mus = np.linspace(0.1, 4.1, 11)
print('Will generate the following hypotheses: ', gen_mus)

scan_mus = np.linspace(0, 10, 11)
print('Will scan over the following hypotheses: ', scan_mus)

# Generate the samples, if needed

np.random.seed(131071)
SamplingManager(DebuggingFastSampler(fast_model, scan_mus, pyhf_model), 'samples/fast_debug_test1').generate_and_save(gen_mus, 200)

fast_samples = SamplingManager(FastSampler(fast_model, scan_mus), 'samples/fast_test1').generate_and_save(gen_mus, 20000)
pyhf_samples = SamplingManager(PyhfSampler(pyhf_model, 2, 2), 'samples/pyhf_test1').generate_and_save(gen_mus, 5000)

# Compute 

def clsb(mu, data, model) :
  return pyhf.infer.hypotest(mu, data, model, return_tail_probs = True)[1][0][0]

pyhf_data = fastprof.Data(fast_model).set_expected(fastprof.Parameters(0,0,0)).export_pyhf_data(pyhf_model)
acls = [ clsb(mu, pyhf_data, pyhf_model) for mu in gen_mus ]
fcls = [ fast_samples.cl(acl, mu) for mu, acl in zip(gen_mus, acls) ]
scls = [ pyhf_samples.cl(acl, mu) for mu, acl in zip(gen_mus, acls) ]

# Plot

def find_hypo(mus, cls, cl = 0.05) :
  logcls = [ math.log(c) for c in cls ]
  #print(logcls, math.log(cl))
  finder = scipy.interpolate.interp1d(logcls, mus, 'quadratic')
  return finder(math.log(cl))

fig = plt.figure()
plt.ion()
plt.plot(gen_mus, acls, 'r')
plt.plot(gen_mus, scls, 'b')
plt.plot(gen_mus, fcls, 'g')
print(find_hypo(gen_mus, acls))
print(find_hypo(gen_mus, scls))
print(find_hypo(gen_mus, fcls))
plt.show()
