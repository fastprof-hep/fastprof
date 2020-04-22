import numpy as np
import fastprof
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
  model = fastprof.Model(sig, bkg, a, b)
  data = fastprof.Data(model).set_from_pyhf_data(pyhf_data, pyhf_model)
  #print(model)
  npm = fastprof.NPMinimizer(mu, data)
  nll = npm.profile_nll()
  pars, val  = pyhf.infer.mle.fixed_poi_fit(mu, pyhf_data, pyhf_model, return_fitted_val=True)
  print('fast pars |', npm.min_pars.array())
  print('fast nexp |', model.s_exp(npm.min_pars), model.b_exp(npm.min_pars), model.n_exp(npm.min_pars))
  print(nll)
  pyhf_pars = fastprof.Parameters(pars[0], pars[1], pars[2])
  print('pyhf data |', pyhf_data)
  print('pyhf pars |', pyhf_pars.array())
  print('pyhf nexp |', model.s_exp(pyhf_pars), model.b_exp(pyhf_pars), model.n_exp(pyhf_pars))
  print(model.nll(pyhf_pars, data))
