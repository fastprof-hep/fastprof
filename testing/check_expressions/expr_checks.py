from fastprof import ProductRatio, SingleParameter, LinearCombination
import numpy as np

np.set_printoptions(linewidth=np.inf)

a = SingleParameter('a')
b = SingleParameter('b')
c = SingleParameter('c')

p = ProductRatio('p', numerator=['a','b','c'])

pab = ProductRatio('pab', numerator=['a','b'])
pbc = ProductRatio('pbc', numerator=['b','c'])
pac = ProductRatio('pac', numerator=['a','c'])

l = LinearCombination('l', 9, { 'pab' : 4, 'pbc' : 5, 'pac' : 6 })

pars = { p.name : p for p in [a,b,c] }
reals = { r.name : r for r in [a, b , c, pab, pbc, pac ,  l] }
real_vals = { 'a' : 1, 'b' : 2, 'c' : 3 }
for r in reals.values() : real_vals[r.name] = r.value(real_vals)

print('@', p.value(real_vals))
print('@', p.gradient(pars, reals, real_vals))
print('@', str(p.hessian(pars, reals, real_vals)).replace('\n', ','))

print('@', l.value(real_vals))
print('@', l.gradient(pars, reals, real_vals))
print('@', str(l.hessian(pars, reals, real_vals)).replace('\n', ','))

