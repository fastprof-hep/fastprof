from fastprof import Model, Data, NPMinimizer
import numpy as np

mu = 3.7
print('mu = ', mu)
n   = np.array([7, 12])
sig = np.array([1.0, 0])*mu
bkg = np.array([1.0, 10.0])

aux_a = np.array([1])
aux_b = np.array([-1.6])

a = np.array([[0.2], [0.2]])
b = np.array([[0.2], [0.2]])

model = Model(sig, bkg, alphas=['acc_sys'], betas=['bkg_sys'], a=a, b=b)
data = Data(model, n, aux_a, aux_b)

print(model)

best_a, best_b, best_c = NPMinimizer(mu, data).profile()
print('hat(a) =', best_a)
print('hat(b) =', best_b)
print('hat(c) =', best_c)

#der_a, der_b = fastprof.derivatives(model, best_a, best_b)
#der_a, der_b = model.derivatives(best_a, best_b)
#print('dl/da =', der_a)
#print('dl/db =', der_b)
