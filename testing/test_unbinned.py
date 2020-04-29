import numpy as np
import matplotlib.pyplot as plt
from fastprof import Model, Data, Parameters, NPMinimizer, OptiMinimizer

filename = 'run/high_mass_gg_1300.json'

model = Model.create(filename)
ws_data = Data(model).load(filename)

#n = np.array([6, 4, 2, 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0 ])
#aux_a = np.array([-1])
#aux_b = np.array([])
#data = Data(model, n, aux_a, aux_b)
# data.set_expected(Parameters(mu))

print(model)

opti = OptiMinimizer(ws_data)
opti.minimize()
min_pars = opti.min_pars
print(min_pars)

mu = 1
print('mu = ', mu)

mini = NPMinimizer(mu, ws_data)
mini.profile()
min1 = mini.min_pars
print(min1,model.nll(min1, ws_data))
print('mu = ', min_pars.mu)

mini = NPMinimizer(min_pars.mu, ws_data)
mini.profile()
min2 = mini.min_pars
print(min2,model.nll(min2, ws_data))

plt.ion()

fig1, (ax11, ax12) = plt.subplots(2)
fig1.suptitle('Prefit model')
plt.axes(ax11)
model.plot(model.expected_pars(1), data=ws_data)
plt.yscale('log')
plt.axes(ax12)
model.plot(model.expected_pars(1), data=ws_data, residuals=True)


print('hat(a) =', mini.min_pars.alphas)
print('hat(b) =', mini.min_pars.betas)
print('hat(c) =', mini.min_pars.gammas)

fig2, (ax21, ax22) = plt.subplots(2)
fig2.suptitle('Postfit model')
plt.axes(ax21)
model.plot(mini.min_pars, ws_data)
plt.yscale('log')
plt.axes(ax22)
model.plot(mini.min_pars, data=ws_data, residuals=True)

#der_a, der_b = fastprof.derivatives(model, best_a, best_b)
#der_a, der_b = model.derivatives(best_a, best_b)
#print('dl/da =', der_a)
#print('dl/db =', der_b)

