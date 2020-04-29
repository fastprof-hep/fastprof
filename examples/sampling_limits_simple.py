# For now this is a script -- should be upgraded to a generally usable class when the process is mature

from fastprof import Model, Data, QMu, Samples, CLsSamples, OptiSampler, OptiMinimizer, FitResults
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

# Define the models
#model_filename = 'run/high_mass_gg_1300-100bins.json'
#data_filename = 'run/high_mass_gg_1300-100bins-data.json'
#hypos_filename = 'run/hypos_high_mass_gg_1300.json'
#poi = 'mu'
#poi_init = 1
#poi_min = -3
#poi_max = 20
#output_filename = 'samples/high_mass_gg_1300-100bins'
#ntoys = 10000

model_filename = 'run/high_mass_gg_1700-100bins.json'
data_filename = 'run/high_mass_gg_1700-100bins-data.json'
fits_filename = 'run/fits_high_mass_gg_1700.json'
output_filename = 'samples/high_mass_gg_1700-100bins'
ntoys = 10000

full_results = FitResults(fits_filename)
fit_results = full_results.fit_results

fast_model = Model.create(model_filename)

np.random.seed(131071)
fr = full_results
opti_samples = CLsSamples(
  Samples(OptiSampler(fast_model, mu0=fr.poi_initial_value, bounds=(fr.poi_min, fr.poi_max))               , output_filename),
  Samples(OptiSampler(fast_model, mu0=fr.poi_initial_value, bounds=(fr.poi_min, fr.poi_max), do_CLb = True), output_filename + '_clb')).generate_and_save(fr.hypos, ntoys)

full_results.fill() # fill asymptotic results
for fit_result in fit_results :
  fit_result['sampling_cl'] = opti_samples.clsb.cl(fit_result['cl'], fit_result[fr.poi_name])
  fit_result['sampling_cls'] = opti_samples.cl(fit_result['cl'], fit_result[fr.poi_name])

# Check the fastprof CLs against the ones in the reference: in principle this should match well,
# otherwise it means what we generate isn't exactly comparable to the observation, which would be a problem...
print('Check CL computed from fast model against those of the full model (a large difference would require to correct the sampling distributions) :')
fast_data = Data(fast_model).load(data_filename)
full_results.check(fast_data)

# Plot
def find_hypo(mus, cls, cl = 0.05, n = 0) :
  logcls = [ math.log(c/cl) if c > 0 else -999 for c in cls[n:] ]
  finder = scipy.interpolate.InterpolatedUnivariateSpline(mus[n:], logcls, k=3)
  if len(finder.roots()) == 0 :
    print('No solution found')
    return None
  if len(finder.roots()) > 1 :
    print('Multiple solutions found, returning the first one')
  return finder.roots()[0]

plt.ion()

plt.figure(1)
plt.suptitle('$CL_{s+b}$')
plt.xlabel('$\mu$')
plt.ylabel('$CL_{s+b}$')
plt.plot(fr.hypos, [ fit_result['cl']          for fit_result in fit_results ], 'r:' , label = 'Asymptotics')
plt.plot(fr.hypos, [ fit_result['sampling_cl'] for fit_result in fit_results ], 'b'  , label = 'Sampling')
plt.legend()

plt.figure(2)
plt.suptitle('$CL_s$')
plt.xlabel('$\mu$')
plt.ylabel('$CL_s$')
plt.plot(fr.hypos, [ fit_result['cls']          for fit_result in fit_results ], 'r:' , label = 'Asymptotics')
plt.plot(fr.hypos, [ fit_result['sampling_cls'] for fit_result in fit_results ], 'b'  , label = 'Sampling')
plt.legend()

plt.show()

print('Asymptotics, CLsb : UL(95) =', find_hypo(fr.hypos, [ fit_result['cl']           for fit_result in fit_results ]))
print('Sampling,    CLsb : UL(95) =', find_hypo(fr.hypos, [ fit_result['sampling_cl']  for fit_result in fit_results ]))
print('Asymptotics, CLs  : UL(95) =', find_hypo(fr.hypos, [ fit_result['cls']          for fit_result in fit_results ]))
print('Sampling,    CLs  : UL(95) =', find_hypo(fr.hypos, [ fit_result['sampling_cls'] for fit_result in fit_results ]))
