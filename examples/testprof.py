import fastprof as fastprof
import numpy as np

mu = 3.7
print('mu = ', mu)
n   = np.array([7, 12])
sig = np.array([1.0, 0])*mu
bkg = np.array([1.0, 10.0])

alpha = np.array([1])
beta  = np.array([-1.6])

a = np.array([[0.2], [0.2]])
b = np.array([[0.2], [0.2]])

model = fastprof.Model(n, sig, bkg, alpha, beta, a, b)

print(model)
print(model.na(), model.nb())

p,q = fastprof.PQ(model)
print('p =', p)
print('q =', q)

best_a, best_b = fastprof.profile(model)
print('hat(a) =', best_a)
print('hat(b) =', best_b)

#der_a, der_b = fastprof.derivatives(model, best_a, best_b)
der_a, der_b = model.derivatives(best_a, best_b)
print('dl/da =', der_a)
print('dl/db =', der_b)
