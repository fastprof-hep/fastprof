import matplotlib.pyplot as plt
import numpy as np
import math
import os
import numpy as np
from scipy import optimize
from scipy.stats import poisson, norm

pv = 0.05
z_cls = norm.isf(pv/2)

# ==============================================
#bkgs = np.linspace(0,100,101)
#toys = np.load('limit_guess_toys_1k.npy')
#ntoys = 1000
# ==============================================
bkgs = np.linspace(0,20,101)
toys = np.load('limit_guess_toys.npy')
ntoys = 25000
# ==============================================
#toys = np.zeros((len(bkgs), ntoys))

bkg = 1 
nobs = 0

def poisson_pval(n,s,b) :
  return poisson.cdf(n, s+b)

def delta_clsb(s) :
   clsb = poisson.cdf(math.floor(bkg), s + bkg)
   return clsb - pv

def delta_cls(s) :
   cls = poisson.cdf(nobs, s + bkg)/poisson.cdf(nobs, bkg)
   return cls - pv

def limit(n, b) :
  global bkg
  global nobs
  bkg = b
  nobs = n
  return optimize.root_scalar(delta_cls, bracket=[0, 200], method='brentq').root

def sampling_bands(toys, max_sigma = 2) :
  bands = {}
  for i in range (-max_sigma, max_sigma+1) :
    index = int(ntoys*norm.cdf(i))
    bands[i] = np.array([ toys[b,index] for b in range(0, len(bkgs)) ])
  return bands

def asymptotic_band(bkgs, pv, n = 0) :
  z = norm.isf(pv*norm.cdf(n))
  return np.array([ (n + z)**2/2*(1 + math.sqrt(1 + 4*b/(n+z)**2)) for b in bkgs ])

def w(bkgs) :
  return np.exp(-bkgs/3)

def guess(unc, i) :
  return (3 + 0.5*i)*np.exp(-unc**2/3) + (1 - np.exp(-unc**2/3))*(i + norm.isf(pv*norm.cdf(i)))*np.sqrt(9 + unc**2)

if toys[0,0] == 0 :
  for i, b in enumerate(bkgs) :
    print('Testing b = %g' % b)
    for k in range(0, ntoys) :
      if k % 5000 == 0 : print('  toy %d' % k)
      n = np.random.poisson(b)
      toys[i,k] = limit(n,b)
for b in range(0, len(bkgs)) : toys[b,:].sort()
toylim = np.array([ np.median(toys[b,:]) for b in range(0, len(bkgs)) ])
asilim = np.array([ limit(math.floor(b), b) for b in bkgs ])

asymp = { i:asymptotic_band(bkgs, pv, i) for i in np.linspace(-2,2,5) }
sampl = sampling_bands(toys, 2)

approx = { i:((3 + 0.5*i)*w(bkgs) + (1 - w(bkgs))*(i + norm.isf(pv*norm.cdf(i)))*np.sqrt(9 + bkgs)) for i in np.linspace(-5,5,11) }
approx = { i:np.array([ guess(math.sqrt(b), i) for b in bkgs]) for i in np.linspace(-5,5,11) }


plt.ion()

fig1 = plt.figure(1)
colors = [ 'k', 'g', 'y' ]
styles = { -2: 'r', -1: 'm', 0: 'k', 1:'c', 2:'b' }

for i in reversed(range(1, 3)) :
  plt.fill_between(bkgs, sampl[+i], sampl[-i], color=colors[i])
plt.plot(bkgs, sampl[0], 'k')

#plt.plot(bkgs, asymp[+2], styles[+2] + '--')
#plt.plot(bkgs, asymp[+1], styles[+1] + '--')
#plt.plot(bkgs, asymp[ 0], styles[ 0] + '--')
#plt.plot(bkgs, asymp[-1], styles[-1] + '--')
#plt.plot(bkgs, asymp[-2], styles[-2] + '--')

plt.plot(bkgs, approx[+5], styles[+2] + ':')
plt.plot(bkgs, approx[+4], styles[+2] + ':')
plt.plot(bkgs, approx[+3], styles[+2] + ':')
plt.plot(bkgs, approx[+2], styles[+2] + ':')
plt.plot(bkgs, approx[+1], styles[+1] + ':')
plt.plot(bkgs, approx[ 0], styles[ 0] + ':')
plt.plot(bkgs, approx[-1], styles[-1] + ':')
plt.plot(bkgs, approx[-2], styles[-2] + ':')
plt.plot(bkgs, approx[-2], styles[-2] + ':')
plt.plot(bkgs, approx[-3], styles[-2] + ':')
plt.plot(bkgs, approx[-4], styles[-2] + ':')
plt.plot(bkgs, approx[-5], styles[-2] + ':')
