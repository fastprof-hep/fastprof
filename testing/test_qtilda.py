import scipy.stats
import numpy as np
from fastprof import QMu, QMuTilda
import matplotlib.pyplot as plt

x0 = 0.5
sigma = 1
ntoys = 10000
tmu_0 = (x0/sigma)**2

ts_type = 'q~mu'

ts_min = -9
ts_max = +9
ts_bins = 100
tsx = np.linspace(ts_min, ts_max, ts_bins)
bin_norm = ntoys*(tsx[1] - tsx[0])

xs = np.array([ np.random.normal(x0, sigma) for i in range(0, ntoys) ])

if ts_type == 'q_mu' :
  tmus = np.array([ ((x - x0)/sigma)**2 for x in xs ])
  pvs  = np.array([ QMu(test_poi = x0, best_poi = x, tmu = t, tmu_A = tmu_0).asymptotic_pv() for x,t in zip(xs, tmus) ])
  tss  = np.array([ QMu(test_poi = x0, best_poi = x, tmu = t, tmu_A = tmu_0).value() for x,t in zip(xs, tmus) ])
  q = QMu(test_poi = x0, best_poi = 0, tmu = 0, tmu_A = tmu_0)
  tsy = [ bin_norm*q.asymptotic_pdf(t) for t in tsx ]
elif ts_type == 'q~mu' :
  tmus = np.array([ ((x - x0)/sigma)**2 if x > 0 else (x0/sigma)**2 - 2*x0*x/sigma**2 for x in xs ])
  pvs = np.array([ QMuTilda(test_poi = x0, best_poi = x, tmu = t, tmu_A = tmu_0, tmu_0 = tmu_0).asymptotic_pv() for x,t in zip(xs, tmus) ])
  tss = np.array([ QMuTilda(test_poi = x0, best_poi = x, tmu = t, tmu_A = tmu_0, tmu_0 = tmu_0).value() for x,t in zip(xs, tmus) ])
  q = QMuTilda(test_poi = x0, best_poi = 0, tmu = 0, tmu_A = tmu_0, tmu_0 = tmu_0)
  tsy = [ bin_norm*q.asymptotic_pdf(t) for t in tsx ]

plt.ion()
plt.figure(1)
plt.hist(pvs, bins=np.linspace(0,1,100))
plt.figure(2)
plt.yscale('log')
plt.hist(tss, bins=np.linspace(ts_min, ts_max,100))
plt.plot(tsx, tsy)
plt.ylim(1E-1, 2*ntoys)

plt.show()
