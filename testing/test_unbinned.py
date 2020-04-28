import numpy as np
import matplotlib.pyplot as plt
from fastprof import Model, Data, Parameters, NPMinimizer


model = Model.create('run/high_mass_gg_1500.json')

n = np.array([6, 4, 2, 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0 ])
aux_a = np.array([-1])
aux_b = np.array([])
data = Data(model, n, aux_a, aux_b)

data.set_expected(Parameters(mu))

print(model)

mu = 5
print('mu = ', mu)

plt.ion()
plt.figure(1)
plt.suptitle('Prefit model')
model.plot(model.expected_pars(mu), data=data, variations=[ ('dEff', 5, 'r'), ('xi', +0.5, 'g') ])
#plt.yscale('log')

mini = NPMinimizer(mu, data)
mini.profile()
plt.suptitle('Prefit model')

print('hat(a) =', mini.min_pars.alphas)
print('hat(b) =', mini.min_pars.betas)
print('hat(c) =', mini.min_pars.gammas)

plt.figure(2)
plt.suptitle('Postfit model')
model.plot(mini.min_pars, data=data)

#der_a, der_b = fastprof.derivatives(model, best_a, best_b)
#der_a, der_b = model.derivatives(best_a, best_b)
#print('dl/da =', der_a)
#print('dl/db =', der_b)

