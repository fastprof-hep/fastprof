import fastprof
import matplotlib.pyplot as plt

model = fastprof.Model.create('run/high_mass_gg_1500.json')
pars = model.expected_pars(10)
data = model.generate_data(pars)
plt.ion()
plt.figure(1)
model.plot(pars, data=data, variations=[ ('dEff', 5, 'r'), ('xi', -0.2, 'g') ])
plt.yscale('log')

