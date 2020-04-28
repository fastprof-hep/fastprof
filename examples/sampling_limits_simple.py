import fastprof
from sampling import Samples, CLsSamples, FastSampler, OptiSampler, DebuggingFastSampler
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

# Define the models
model_filename = 'run/high_mass_gg_1300.json'
hypos_filename = 'run/hypos_high_mass_gg_1300.json'
poi = 'mu'
poi_init = 1
poi_min = 0
poi_max = 20
output_filename = 'samples/high_mass_gg_1300'
fast_model = fastprof.Model.create(model_filename)
ntoys = 10000

with open(hypos_filename, 'r') as fd :
  hypo_dicts = json.load(fd)

hypo_mus = [ hd[poi] for hd in hypo_dicts ]

print(fast_model)

np.random.seed(131071)
opti_samples = CLsSamples(
  Samples(OptiSampler(fast_model, x0 = poi_init, bounds=(poi_min, poi_max))               , output_filename),
  Samples(OptiSampler(fast_model, x0 = poi_init, bounds=(poi_min, poi_max), do_CLb = True), output_filename + '_clb')).generate_and_save(hypo_mus, ntoys)

for hd in hypo_dicts :
  if not 'cl' in hd :
    hd['cl'] = fastprof.QMu(hd['qmu'], hd[poi], hd['best_fit_val']).asymptotic_cl()
  if not 'cls' in hd :
    hd['cls'] = fastprof.QMu(hd['qmu'], hd[poi], hd['best_fit_val']).asymptotic_cls(hd['best_fit_err'])
  print(hd['cl'])
  hd['sampling_cl'] = opti_samples.clsb.cl(hd['cl'], hd[poi])
  hd['sampling_cls'] = opti_samples.cl(hd['cl'], hd[poi])

# Plot

def find_hypo(mus, cls, cl = 0.05, n = 0) :
  logcls = [ math.log(c/cl) if c > 0 else -999 for c in cls[n:] ]
  finder = scipy.interpolate.InterpolatedUnivariateSpline(mus[n:], logcls, k=3)
  return finder.roots()[0]

plt.ion()

plt.figure(1)
plt.suptitle('$CL_{s+b}$')
plt.xlabel('$\mu$')
plt.ylabel('$CL_{s+b}$')
plt.plot(hypo_mus, [ hd['cl']          for hd in hypo_dicts ], 'r:' , label = 'Asymptotics')
plt.plot(hypo_mus, [ hd['sampling_cl'] for hd in hypo_dicts ], 'b'  , label = 'Sampling')
plt.legend()

plt.figure(2)
plt.suptitle('$CL_s$')
plt.xlabel('$\mu$')
plt.ylabel('$CL_s$')
plt.plot(hypo_mus, [ hd['cls']          for hd in hypo_dicts ], 'r:' , label = 'Asymptotics')
plt.plot(hypo_mus, [ hd['sampling_cls'] for hd in hypo_dicts ], 'b'  , label = 'Sampling')
plt.legend()

plt.show()

print('Asymptotics, CLsb : UL(95) =', find_hypo(hypo_mus, [ hd['cl']           for hd in hypo_dicts ]))
print('Sampling,    CLsb : UL(95) =', find_hypo(hypo_mus, [ hd['sampling_cl']  for hd in hypo_dicts ]))
print('Asymptotics, CLs  : UL(95) =', find_hypo(hypo_mus, [ hd['cls']          for hd in hypo_dicts ]))
print('Sampling,    CLs  : UL(95) =', find_hypo(hypo_mus, [ hd['sampling_cls'] for hd in hypo_dicts ]))
