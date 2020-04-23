import numpy as np
import matplotlib.pyplot as plt
plt.ion()

samples = np.load('samples/fast_test1_clb_3.7.npy')

# Samples data:
# 0, 1 : data bin contents (0=SR, 1=CR)
# 2, 3 : aux measurement values of signal (2) and background (3) NPs
# 4, 5, 6, 7 : fast-fit values of mu (4) signal (5) and background (6) pars + CLs+b (7)
# 8, 9, 10, 11 : best-fit values of mu (8) signal (9) and background (10) pars  + CLsb @ mu=mu_min (11)
# 12, 13, 14 :  mu_min (12) and best-fit values of signal (13) and background (14) pars

plt.yscale('log')
plt.hist(samples[:], bins=100, range=[0, 0.5])
plt.show()
