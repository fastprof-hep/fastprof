import pyhf
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math

spec = json.load(open('examples/test1.json', 'r'))
parameters = [ 1.0, 0.0, 0.0 ]
#spec = json.load(open('examples/test2.json', 'r'))
#parameters = [ 1.0 ]


print(spec)


ws = pyhf.Workspace(spec)
model = ws.model()

data = ws.data(model)
calc = pyhf.infer.calculators.AsymptoticCalculator(data, model)

print(model.logpdf(parameters, data))

def my_qmu(mu) : 
  parameters[0] = mu
  return -2*model.logpdf(parameters, data)[0]

def qmu(mu) :
   return calc.teststatistic(mu)
 
def cl(q) :
  return 0.5*scipy.stats.chi2.sf(q, 1)

def my_arr_logcl(my_qmus) :
  my_qmu0 = np.amin(my_qmus)
  print("min NLL = %g" % my_qmu0)
  return  np.array([math.log(cl(my_qmu - my_qmu0)) for my_qmu in my_qmus])

def interval(mus, cls, cl) :
  imax = np.argmax(cls)
  print('min NLL at %d' % imax)
  find_lo = scipy.interpolate.interp1d(logcls[:imax], mus[:imax], 'quadratic')
  print('find_lo', find_lo(math.log(cl)))
  find_hi = scipy.interpolate.interp1d(logcls[imax:], mus[imax:], 'quadratic')
  print('find_hi', find_hi(math.log(cl)))
  return find_lo(math.log(cl)), find_hi(math.log(cl))

def arr_logcl(mus) :
 return np.array([ math.log(pyhf.infer.hypotest(mu, data, model, return_tail_probs = True)[1][0]) for mu in mus ])

mus = np.linspace(0.1, 4.1, 41)
print(mus)

qmus = np.array(list(map(qmu, mus)))
print(qmus)

my_qmus = np.array(list(map(my_qmu, mus)))
print(my_qmus)

#logcls = arr_logcl(my_qmus)
logcls = arr_logcl(mus)
cls = np.array([math.exp(logcl) for logcl in logcls])
print(logcls)

#print(interval(mus, logcls, 2*scipy.stats.norm.sf(1)))

fig = plt.figure()
plt.plot(mus, cls)
plt.show()

