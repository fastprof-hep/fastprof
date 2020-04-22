import numpy as np
from fast_sampling import fast_sampling_dist

gen_mu = 3.7
print('Will generate the following hypothesis: ', gen_mu)

scan_mus = np.linspace(0, 10, 21)
print('Will scan over the following hypotheses: ', scan_mus)

np.random.seed(131071)
dist = fast_sampling_dist(gen_mu, scan_mus, 10000)
np.sort(dist)

