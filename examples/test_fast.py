import numpy as np
import sampling
import fastprof

model = fastprof.Model(sig = np.array([1.0, 0]),
                       bkg = np.array([1.0, 10.0]),
                       a = np.array([[0.2], [0.2]]),
                       b = np.array([[0.2], [0.2]]))

gen_mu = 3.7
print('Will generate the following hypothesis: ', gen_mu)

scan_mus = np.linspace(0, 10, 21)
print('Will scan over the following hypotheses: ', scan_mus)

np.random.seed(131071)
dist = sampling.FastSampler(model, scan_mus).generate(gen_mu, 10000)
dist.sort()

