import fastprof
from sampling import Samples, CLsSamples, FastSampler, OptiSampler, PyhfSampler, DebuggingFastSampler
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
Samples(DebuggingFastSampler(fast_model, scan_mus, pyhf_model), 'samples/fast_debug_test1').generate_and_save(gen_mus, 200)

fast_samples = CLsSamples(
  Samples(FastSampler(fast_model, scan_mus)               , 'samples/fast_test1'),
  Samples(FastSampler(fast_model, scan_mus, do_CLb = True), 'samples/fast_test1_clb')).generate_and_save(gen_mus, 20000)

opti_samples = CLsSamples(
  Samples(OptiSampler(fast_model, x0 = 1, bounds=(0.1, 20))               , 'samples/opti_test1'),
  Samples(OptiSampler(fast_model, x0 = 1, bounds=(0.1, 20), do_CLb = True), 'samples/opti_test1_clb')).generate_and_save(gen_mus, 20000)

pyhf_samples = CLsSamples(
  Samples(PyhfSampler(pyhf_model, 2, 2)               , 'samples/pyhf_test1'),
  Samples(PyhfSampler(pyhf_model, 2, 2, do_CLb = True), 'samples/pyhf_test1_clb')).generate_and_save(gen_mus, 10000)

# Compute 

def clsb(mu, data, model) :
  return pyhf.infer.hypotest(mu, data, model, return_tail_probs = True)[1][0][0]

def cls(mu, data, model) :
  return pyhf.infer.hypotest(mu, data, model, return_tail_probs = True)[0]

#pyhf_data = fastprof.Data(fast_model).set_expected(fastprof.Parameters(0,0,0)).export_pyhf_data(pyhf_model) # Asimov case
pyhf_data = np.array( [0, 10, 0, 0 ] ) # nobs = 0 case

asym_clsb = [ clsb(mu, pyhf_data, pyhf_model) for mu in gen_mus ]
fast_clsb = [ fast_samples.clsb.cl(acl, mu) for mu, acl in zip(gen_mus, asym_clsb) ]
opti_clsb = [ opti_samples.clsb.cl(acl, mu) for mu, acl in zip(gen_mus, asym_clsb) ]
samp_clsb = [ pyhf_samples.clsb.cl(acl, mu) for mu, acl in zip(gen_mus, asym_clsb) ]
asym_cl_s = [ cls(mu, pyhf_data, pyhf_model) for mu in gen_mus ]
fast_cl_s = [ fast_samples.cl(acl, mu) for mu, acl in zip(gen_mus, asym_clsb) ]
opti_cl_s = [ opti_samples.cl(acl, mu) for mu, acl in zip(gen_mus, asym_clsb) ]
samp_cl_s = [ pyhf_samples.cl(acl, mu) for mu, acl in zip(gen_mus, asym_clsb) ]

# Plot

def find_hypo(mus, cls, cl = 0.05, n = 0) :
  logcls = [ math.log(c/cl) for c in cls[n:] ]
  #print(logcls, math.log(cl))
  finder = scipy.interpolate.InterpolatedUnivariateSpline(mus[n:], logcls, k=3)
  return finder.roots()[0]

plt.ion()

plt.figure(1)
plt.suptitle('$CL_{s+b}$')
plt.xlabel('$\mu$')
plt.ylabel('$CL_{s+b}$')
plt.plot(gen_mus, asym_clsb, 'r:' , label = 'Asymptotics')
plt.plot(gen_mus, samp_clsb, '--', label = 'pyhf sampling')
plt.plot(gen_mus, fast_clsb, 'b'  , label = 'Fast sampling')
plt.plot(gen_mus, opti_clsb, 'g'  , label = 'Opti sampling')
plt.legend()

plt.figure(2)
plt.suptitle('$CL_s$')
plt.xlabel('$\mu$')
plt.ylabel('$CL_s$')
plt.plot(gen_mus, asym_cl_s, 'r:' , label = 'Asymptotics')
plt.plot(gen_mus, samp_cl_s, '--', label = 'pyhf sampling')
plt.plot(gen_mus, fast_cl_s, 'b'  , label = 'Fast sampling')
plt.plot(gen_mus, opti_cl_s, 'g'  , label = 'Opti sampling')
plt.legend()

print('asymptotics  , CLsb : UL(95) =', find_hypo(gen_mus, asym_clsb))
print('fast sampling, CLsb : UL(95) =', find_hypo(gen_mus, fast_clsb))
#print('opti sampling, CLsb : UL(95) =', find_hypo(gen_mus, opti_clsb, n=1))
print('pyhf sampling, CLsb : UL(95) =', find_hypo(gen_mus, samp_clsb))

print('asymptotics  , CLs  : UL(95) =', find_hypo(gen_mus, asym_cl_s))
print('fast sampling, CLs  : UL(95) =', find_hypo(gen_mus, fast_cl_s))
print('opti sampling, CLs  : UL(95) =', find_hypo(gen_mus, opti_cl_s, n=1))
print('pyhf sampling, CLs  : UL(95) =', find_hypo(gen_mus, samp_cl_s))
plt.show()
