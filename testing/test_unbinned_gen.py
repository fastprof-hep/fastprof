import numpy as np
import matplotlib.pyplot as plt
from fastprof import Model, ScanSampler, OptiSampler

model = Model.create('run/high_mass_gg_1700-100bins.json')

gen_mu = 3
print('Will generate the following hypothesis: ', gen_mu)

np.random.seed(131071)
dist = OptiSampler(model, mu0=1, bounds=(0,20), method='scalar').generate(gen_mu, 2000)
dist.save('test')
