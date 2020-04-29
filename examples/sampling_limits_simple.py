# For now this is a script -- should be upgraded to a generally usable class when the process is mature

from fastprof import Model, Data, QMu, Samples, CLsSamples, OptiSampler, OptiMinimizer
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

# Define the models
model_filename = 'run/high_mass_gg_1300-100bins.json'
data_filename = model_filename

hypos_filename = 'run/hypos_high_mass_gg_1300.json'
poi = 'mu'
poi_init = 1
poi_min = -3
poi_max = 20
output_filename = 'samples/high_mass_gg_1300-100bins'
ntoys = 10000

with open(hypos_filename, 'r') as fd :
  hypo_dicts = json.load(fd)

hypo_mus = [ hd[poi] for hd in hypo_dicts ]

print(fast_model)

np.random.seed(131071)
opti_samples = CLsSamples(
  Samples(OptiSampler(fast_model, mu0 = poi_init, bounds=(poi_min, poi_max))               , output_filename),
  Samples(OptiSampler(fast_model, mu0 = poi_init, bounds=(poi_min, poi_max), do_CLb = True), output_filename + '_clb')).generate_and_save(hypo_mus, ntoys)

for hd in hypo_dicts :
  if not 'cl' in hd :
    hd['cl'] = QMu(hd['qmu'], hd[poi], hd['best_fit_val']).asymptotic_cl()
  if not 'cls' in hd :
    hd['cls'] = QMu(hd['qmu'], hd[poi], hd['best_fit_val']).asymptotic_cls(hd['best_fit_err'])
  hd['sampling_cl'] = opti_samples.clsb.cl(hd['cl'], hd[poi])
  hd['sampling_cls'] = opti_samples.cl(hd['cl'], hd[poi])

# Check the fastprof CLs against the ones in the reference: in principle this should match well,
# otherwise it means what we generate isn't exactly comparable to the observation, which would be a problem...
fast_model = Model.create(model_filename)
fast_data = Data(fast_model).load(data_filename)

for hd in hypo_dicts :
  tmu, min_mu = OptiMinimizer(fast_data, hd[poi], (-5, 20)).tmu(hd[poi])
  q = QMu(tmu, hd[poi], min_mu)
  print('mu = %g : observed asymptotic reference CL = %6.4f , fast CL = %6.4f (a large difference would require to correc the sampling distributions)' % (hd[poi], hd['cl'], q.asymptotic_cl()))

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
