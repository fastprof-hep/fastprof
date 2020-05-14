import numpy as np
import matplotlib.pyplot as plt
from fastprof import Model, Data, Parameters, NPMinimizer, OptiMinimizer

np.random.seed(7)

model = Model().load('fastprof/models/HighMass_NW-700-log500-gammas-only.json')
pars0 = model.expected_pars(5)

pars = model.expected_pars(5)
pars.gammas = np.random.normal(pars.gammas, 1)
print('Randomized gammas = ', pars.gammas)

model.linear_nps = False
data_exp = Data(model).set_expected(pars)
model.linear_nps = True
data_lin = Data(model).set_expected(pars)

npm = NPMinimizer(5, data_lin)
print(npm.profile())
min_pars = npm.min_pars

plt.ion()
plt.show()

fig1 = plt.figure(1)
fig1.canvas.set_window_title('True pars, data from linear model')
model.plot(pars, data_lin, residuals=True)
plt.xlim(150,300)

fig2 = plt.figure(2)
fig2.canvas.set_window_title('True pars, data from exp model')
model.plot(pars, data_exp, residuals=True)
plt.xlim(150,300)

fig3 = plt.figure(3)
fig3.canvas.set_window_title('Nominal pars, data from linear model')
model.plot(pars0, data_lin, residuals=True)
plt.xlim(150,300)

fig4 = plt.figure(4)
fig4.canvas.set_window_title('Profiled pars, data from linear model')
model.plot(min_pars, data_lin, residuals=True)
plt.xlim(150,300)

t1 = np.einsum('i,ij,i->j',model.bkg,model.c, 1 - data_lin.n/(model.n_exp(pars0))*(1 - model.bkg/model.n_exp(pars0)*model.c.dot(pars.gammas)) )
t2 = np.einsum('i,ij,i,i,i->j',model.bkg,model.c, -data_lin.n/model.n_exp(pars0), model.bkg/model.n_exp(pars0)*model.c.dot(pars.gammas), model.bkg/model.n_exp(pars0)*model.c.dot(pars.gammas) )
t3 = np.einsum('i,ij,i,i,i,i->j',model.bkg,model.c, data_lin.n/model.n_exp(pars0), model.bkg/model.n_exp(pars0)*model.c.dot(pars.gammas), model.bkg/model.n_exp(pars0)*model.c.dot(pars.gammas), model.bkg/model.n_exp(pars0)*model.c.dot(pars.gammas))
t4 = np.einsum('i,ij,i,i,i,i,i->j',model.bkg,model.c, -data_lin.n/model.n_exp(pars0), model.bkg/model.n_exp(pars0)*model.c.dot(pars.gammas), model.bkg/model.n_exp(pars0)*model.c.dot(pars.gammas), model.bkg/model.n_exp(pars0)*model.c.dot(pars.gammas), model.bkg/model.n_exp(pars0)*model.c.dot(pars.gammas))

model2 = Model().load('fastprof/models/HighMass_NW-700-log500-gammas-only.json')
model2.sig = model.s_exp(min_pars)
model2.bkg = model.b_exp(min_pars)
model.init_vars()
model2.linear_nps = True
pars02 = model.expected_pars(5)
pars02.gammas -= min_pars.gammas
pars2 = model.expected_pars(5)
pars2.gammas = pars.gammas - min_pars.gammas
data2_lin = Data(model2).set_expected(pars2)

fig5 = plt.figure(5)
fig5.canvas.set_window_title('Nominal pars @ model2, data from linear model')
model2.plot(pars02, data2_lin, residuals=True)
plt.xlim(150,300)

fig6 = plt.figure(6)
fig6.canvas.set_window_title('Iter 1 pars @ model2, data from linear model')
model2.plot(model.expected_pars(5), data2_lin, residuals=True)
plt.xlim(150,300)

npm2 = NPMinimizer(5, data2_lin)
npm2.profile()
min_pars2 = npm2.min_pars

print(npm2.profile())
print(pars.gammas)
print(min_pars.gammas)
print(min_pars.gammas + min_pars2.gammas)

npm_off = NPMinimizer(5, data_lin)
npm_off.profile(min_pars)
min_pars_off = npm_off.min_pars

print(min_pars_off)

opti1 = OptiMinimizer(data_lin, 0, (0,10), niter=1)
opti1.profile_nps(5)
print(opti1.min_pars)

opti2 = OptiMinimizer(data_lin, 0, (0,10), niter=2)
opti2.profile_nps(5)
print(opti2.min_pars)

toy_lin = model.generate_data(pars0)
opti = OptiMinimizer(toy_lin, 0, (0,10), niter=2)
print(opti.tmu(5))
