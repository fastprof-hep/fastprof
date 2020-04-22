import numpy as np
import matplotlib.pyplot as plt
plt.ion()

samples = np.load('samples/fast_debug_test1_3.7.npy')

# Samples data:
# 0, 1 : data bin contents (0=SR, 1=CR)
# 2, 3 : aux measurement values of signal (2) and background (3) NPs
# 4, 5, 6, 7 : fast-fit values of mu (4) signal (5) and background (6) pars + CLs+b (7)
# 8, 9, 10, 11 : best-fit values of mu (8) signal (9) and background (10) pars  + CLsb @ mu=mu_min (11)
# 12, 13, 14 :  mu_min (12) and best-fit values of signal (13) and background (14) pars

SR_pred      = 1.0*samples[:,8]*(1 + 0.2*samples[:,9]) + 1.0*(1 + 0.2*samples[:,10])
SR_pred_fast = 1.0*samples[:,4]*(1 + 0.2*samples[:,5]) + 1.0*(1 + 0.2*samples[:,6])
CR_pred      = 10.0*(1 + 0.2*samples[:,10])
CR_pred_fast = 10.0*(1 + 0.2*samples[:,6])

fig1, axs1 = plt.subplots(2, 2)
axs1[0,0].hist2d(samples[:,0], SR_pred     , bins=(11,20), range=[[0, 10],[0, 10]], cmap=plt.cm.viridis)
axs1[0,1].hist2d(samples[:,0], SR_pred_fast, bins=(11,20), range=[[0, 10],[0, 10]], cmap=plt.cm.viridis)
axs1[1,0].hist2d(samples[:,1], CR_pred     , bins=(21,20), range=[[0, 20],[0, 20]])
axs1[1,1].hist2d(samples[:,1], CR_pred_fast, bins=(21,20), range=[[0, 20],[0, 20]])
fig1.show()

fig2, axs2 = plt.subplots(2, 2)
axs2[0,0].hist2d(samples[:, 8], samples[:, 4], bins=(20,20), range=[[0, 15],[0, 15]], cmap=plt.cm.viridis)
axs2[0,1].hist2d(samples[:, 9], samples[:, 5], bins=(20,20), range=[[-3, 3],[-3,3]])
axs2[1,0].hist2d(samples[:,10], samples[:, 6], bins=(20,20), range=[[-3, 3],[-3,3]])
axs2[1,1].hist2d(samples[:, 7], samples[:,11], bins=(20,20), range=[[0, 0.5],[0, 0.5]])
fig2.show()

fig3, axs3 = plt.subplots(2, 2)
axs3[0,0].hist2d(samples[:, 8], samples[:,12], bins=(15,15), range=[[0, 15],[0, 15]], cmap=plt.cm.viridis)
axs3[0,1].hist2d(samples[:, 9], samples[:,13], bins=(20,20), range=[[-3, 3],[-3,3]])
axs3[1,0].hist2d(samples[:,10], samples[:,14], bins=(20,20), range=[[-3, 3],[-3,3]])
fig3.show()
