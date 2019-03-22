import numpy as np
import matplotlib.pyplot as plt

def loadNpy(test):
    poss = [np.load('pos' + test + str(i) + '.npy')  for i in range(3)]
    n = min([pos.shape[1] for pos in poss])
    X = np.vstack([pos[:,0:n] for pos in poss])
    X = X - np.mean(X,axis=1).reshape((6,1))
    return X

# ideal case, vertical motion only and stable camera
X1 = loadNpy('1')
u1,s1,vs1 = np.linalg.svd(X1,full_matrices=False)
# vertical motion with a shakey camera
X2 = loadNpy('2')
u2,s2,vs2 = np.linalg.svd(X2,full_matrices=False)
# stable camera, now with vertical and horizontal motion
X3 = loadNpy('3')
u3,s3,vs3 = np.linalg.svd(X3,full_matrices=False)
# stable camera, with vertical horizontal, and rotation
X4 = loadNpy('4')
u4,s4,vs4 = np.linalg.svd(X4,full_matrices=False)

s = [s1,s2,s3,s4]
titles = ['Ideal Case','Noisy Case',
          'Horizontal Oscillation',
          'Horizontal Oscillation \n and Rotation']
fig,axes = plt.subplots(1,4,figsize=(16,4),sharex=True,sharey=True)
for i in range(4):
    axes[i].plot(s[i],'o')
    axes[i].set_title(titles[i])
    axes[i].set_xticks([])
axes[0].set_ylabel('$\mathbf{\sigma_j}$',fontsize=12)
fig.text(0.5,0.95,'The singular values in each of the four cases',
         ha='center',fontsize=14)

# project in principle component directions
Y = []
X = [X1,X2,X3,X4]
V = [vs1,vs2,vs3,vs4]
fig,axes = plt.subplots(1,4,figsize=(16,4),sharey=True)
for i in range(4):
    line1 = axes[i].plot(s[i][0]*V[i][0,:])
    line2 = axes[i].plot(s[i][1]*V[i][1,:])
    axes[i].set_title(titles[i])
axes[0].set_ylabel('position')
line1[0].set(label='PC1')
line2[0].set(label='PC2')
fig.text(0.5,0.95,'Position along the first two principle components',
         ha='center',fontsize=14)
fig.text(0.5,0.01,'frame number',ha='center')
fig.legend()
    
    
    
