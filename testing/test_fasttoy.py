import pyhf
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from fastprof import Model, NPMinimizer

mus = np.linspace(0.1, 10.1, 11)
n   = np.array([5, 12])
bkg = np.array([1.0, 10.0])

alpha = np.array([1])
beta  = np.array([-1.6])

a = np.array([[0.2], [0.2]])
b = np.array([[0.2], [0.2]])
data = np.array([7, 12, 0, 1])

models = {}
n_np = 2

for mu in mus :
  sig = np.array([1.0, 0])*mu  # specific to this case!
  models[mu] = Model(np.array(data[:-n_np]), sig, bkg, np.array([ data[3] ]), np.array([ data[2] ]), a, b)

nlls = np.zeros(mus.shape[0])

for i, mu in enumerate(mus) :
  model = models[mu]
  model.set_all_data(np.array(data[:-n_np]), np.array([ data[3] ]), np.array([ data[2] ]))
  nlls[i] = NPMinimizer().profile_nll(model)

plt.ion()
plt.plot(mus, nlls, 'b')

smooth_nll = InterpolatedUnivariateSpline(mus, nlls, k=4)
cr_pts = smooth_nll.derivative().roots()
x = np.linspace(0, mus[-1], 100)
plt.plot(x, smooth_nll(x), 'g')

print(cr_pts)

