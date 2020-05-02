from fastprof import Model, Data, QMu, Samples, CLsSamples, OptiSampler, OptiMinimizer
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

# Define the models
model_filename = 'run/high_mass_gg_1300_100.json'
hypos_filename = 'run/hypos_high_mass_gg_1300.json'
poi = 'mu'
poi_init = 1
poi_min = -3
poi_max = 20
output_filename = 'samples/high_mass_gg_1300'
ntoys = 10000

fast_model = Model.create(model_filename)
#fast_data = Data(fast_model).load(model_filename)
fast_data = Data(fast_model).set_expected(fast_model.expected_pars(0))

with open(hypos_filename, 'r') as fd :
  hypo_dicts = json.load(fd)

hypo_mus = [ hd[poi] for hd in hypo_dicts ]

print(fast_model)

np.random.seed(131071)

for hd in hypo_dicts :
  if not 'cl' in hd :
    hd['cl'] = QMu(hd[poi], hd['tmu'], hd['best_fit_val']).asymptotic_cl()
  if not 'cls' in hd :
    hd['cls'] = QMu(hd[poi], hd['tmu'], hd['best_fit_val']).asymptotic_cls(hd['best_fit_err'])
  # DEBUG below
  tmu, min_mu = OptiMinimizer(fast_data, hd[poi], (-5, 20)).tmu(hd[poi])
  q = QMu(tmu, hd[poi], min_mu)
  print(hd[poi], hd['tmu'], hd['cl'], hd['best_fit_val'], '  <-->  ', q.value(), q.asymptotic_cl(), min_mu)
