import numpy as np
import matplotlib.pyplot as plt
from fastprof import Model, Data, Parameters, NPMinimizer, OptiMinimizer
import copy

np.set_printoptions(precision=4, suppress=False, linewidth=180)

np.random.seed(4)

# ==========================================
model = Model().load('run/fastprof/HighMass_NW-1700-log200-noDSig.json')
pars0 = model.expected_pars(0.1)
data = model.generate_data(pars0)

opti = OptiMinimizer(data, 0.1, (0,20))
t = opti.tmu(0.1)

print(opti.hypo_pars)
print(model.closure_approx(opti.hypo_pars, data))
print(model.closure_exact(opti.hypo_pars, data))

# ==========================================
model2 = model.regularize(2)
data2 = copy.copy(data)
data2.model = model2
data2.aux_betas = np.zeros(model2.nb)

opti2 = OptiMinimizer(data2, 0.1, (0,20))
t2 = opti2.tmu(0.1)

print(opti2.hypo_pars)
print(model2.closure_approx(opti2.hypo_pars, data2))
print(model2.closure_exact(opti2.hypo_pars, data2))

# ==========================================
plt.ion()

plt.figure(1)
model.plot(pars0, data, residuals=True)
plt.xlim(150,800)
plt.ylim(-200,200)

plt.figure(2)
model.plot(opti.hypo_pars, data, residuals=True)
plt.xlim(150,800)
plt.ylim(-200,200)

plt.figure(3)
model2.plot(opti2.hypo_pars, data2, residuals=True)
plt.xlim(150,800)
plt.ylim(-200,200)

print(t)
print(t2)
