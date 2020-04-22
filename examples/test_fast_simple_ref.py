import numpy as np
import fastprof_ref
import pyhf, json

test_mus = np.linspace(0,5,6)
n_np = 2
bkg = np.array([1.0, 10.0])  # specific to this case!
sig = np.array([1.0, 0])  # specific to this case!
a = np.array([[0.2], [0.2]]) # specific to this case!
b = np.array([[0.2], [0.2]]) # specific to this case!
#pyhf_data = np.array([ 6.00000000e+00,  9.00000000e+00,  1.29864346e-01, -5.74496450e-01 ])
pyhf_data = np.array([ 2,  10,  0, 0 ])

spec = json.load(open('examples/test1.json', 'r'))
ws = pyhf.Workspace(spec)
pyhf_model = ws.model()

for mu in test_mus :
  print('-------------------------------------')
  print('Testing the following hypothesis: ', mu)
  model = fastprof_ref.Model(pyhf_data[0:2], sig*mu, bkg, pyhf_data[2:3], pyhf_data[3:4], a, b)
  prof_a, prof_b = fastprof_ref.profile(model) 
  pars, val  = pyhf.infer.mle.fixed_poi_fit(mu, pyhf_data, pyhf_model, return_fitted_val=True)
  print('fast pars |', np.array([ mu, prof_a, prof_b]))
  print('pyhf data |', pyhf_data)
  print('pyhf pars |', np.array([pars[0], pars[1], pars[2]]))
