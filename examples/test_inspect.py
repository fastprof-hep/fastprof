import numpy as np
import matplotlib.pyplot as plt
plt.ion()

samples = np.load('samples/test1_debug_3.7.npy')

# Samples data:
# 0, 1 : data bin contents (0=SR, 1=CR)
# 2, 3 : aux measurement values of signal (2) and background (3) NPs
# 4, 5 : best-fit values of signal (4) and background (5) NPs
# 6, 7 : fastprof values of signal (6) and background (7) NPs

fig1, axs1 = plt.subplots(2, 2)
axs1[0,0].hist2d(samples[:,0], samples[:,4], bins=(11,20), range=[[0, 10],[-3,3]], cmap=plt.cm.viridis)
axs1[0,1].hist2d(samples[:,1], samples[:,5], bins=(21,20), range=[[0, 20],[-3,3]])
axs1[1,0].hist2d(samples[:,2], samples[:,4], bins=(20,20), range=[[-3, 3],[-3,3]])
axs1[1,1].hist2d(samples[:,3], samples[:,5], bins=(20,20), range=[[-3, 3],[-3,3]])

fig1.show()

fig2, axs2 = plt.subplots(2, 2)
axs2[0,0].hist2d(samples[:,0], samples[:,6], bins=(11,20), range=[[0, 10],[-3,3]], cmap=plt.cm.viridis)
axs2[0,1].hist2d(samples[:,1], samples[:,7], bins=(21,20), range=[[0, 20],[-3,3]])
axs2[1,0].hist2d(samples[:,2], samples[:,6], bins=(20,20), range=[[-3, 3],[-3,3]])
axs2[1,1].hist2d(samples[:,3], samples[:,7], bins=(20,20), range=[[-3, 3],[-3,3]])

fig2.show()

fig3, axs3 = plt.subplots(2, 2)
axs3[0,0].hist2d(samples[:,4], samples[:,6], bins=(20,20), range=[[-3, 3],[-3,3]])
axs3[0,1].hist2d(samples[:,5], samples[:,7], bins=(20,20), range=[[-3, 3],[-3,3]])

fig3.show()

