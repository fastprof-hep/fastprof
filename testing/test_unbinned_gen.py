import numpy as np
import matplotlib.pyplot as plt
from fastprof import Model, ScanSampler, OptiSampler

model = Model.create('models/highMass-1164.json')

gen_mu = 3
print('Will generate the following hypothesis: ', gen_mu)

np.random.seed(131071)
dist = OptiSampler(model, test_hypo=model.expected_pars(0.1), mu0=0.1, poi_bounds=(0,2), method='scalar').generate(1000)
dist.save('test')
