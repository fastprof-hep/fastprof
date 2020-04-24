import pyhf
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import os


def clsb(mu, data, model) :
  return pyhf.infer.hypotest(mu, data, model, return_tail_probs = True)[1][0]

spec = json.load(open('fastprof/models/test1.json', 'r'))

ws = pyhf.Workspace(spec)
pyhf_model = ws.model()

mu = 3.7
#pyhf_data = ws.data(pyhf_model)
#pyhf_data = np.array([7, 12, 1, -1.6])
#pyhf_data = np.array([7, 12, 0, 1])
#pyhf_data = np.array([4, 8, -0.15946151719059748, -0.11580946974871631])
pyhf_data = np.array([ 2,  16,  0.24, 0.15 ])

n_poi = 1
n_np = 2

calc = pyhf.infer.calculators.AsymptoticCalculator(pyhf_data, pyhf_model)
free_pars = pyhf.infer.mle.fit(pyhf_data, pyhf_model, return_fitted_val=True)
print(free_pars)
pars = pyhf.infer.mle.fixed_poi_fit(mu, pyhf_data, pyhf_model, return_fitted_val=True)
print(pars)
print(clsb(mu, pyhf_data, pyhf_model))
parameters = [ mu, 0.0, 0.0 ]
