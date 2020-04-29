import numpy as np
import matplotlib.pyplot as plt
from fastprof import Model, ScanSampler, OptiSampler

model = Model(sig = np.array([1.0, 0]),
              bkg = np.array([1.0, 10.0]),
              alphas = ['acc_sys' ], betas = [ 'bkg_sys' ],
              a = np.array([[0.2], [0.2]]),
              b = np.array([[0.2], [0.2]]))

gen_mu = 3.7
print('Will generate the following hypothesis: ', gen_mu)

scan_mus = np.linspace(0, 10, 21)
print('Will scan over the following hypotheses: ', scan_mus)

np.random.seed(131071)
#dist = ScanSampler(model, scan_mus).generate(gen_mu, 10000)
dist = OptiSampler(model, mu0=1, bounds=(0,20), method='scalar').generate(gen_mu, 10000)
dist.sort()

plt.ion()
plt.yscale('log')
plt.hist(dist.samples[:], bins=50)

