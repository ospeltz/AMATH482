import numpy as np
import matplotlib.pyplot as plt
import imageio as imio
import os
plt.rcParams['image.cmap'] = 'gray'

imsize = (192,168)
nums = ['B' + (n < 10)*'0' + str(n) for n in range(1,40)]
nums.remove('B14') # for some reason the dataset is missing 'B14'
path = 'yalefaces_cropped/CroppedYale/yale{}/'

# load face matrix
X = []
for n in nums:
    for d in os.listdir(path.format(n)):
        im = imio.imread(path.format(n)+d)
        X.append(im.flatten().astype('float64'))

# each face flattened into column        
X = np.array(X).T

def imshow(ax,im):
    ax.imshow(im)
    ax.set_xticks([]), ax.set_yticks([])

X_ave = X.mean(axis=1).reshape((X.shape[0],1))
X0 = X - X_ave # remove mean from each row

u,s,v = np.linalg.svd(X,full_matrices=False)
u0,s0,v0 = np.linalg.svd(X0,full_matrices=False)

fig,ax = plt.subplots()
imshow(ax,X_ave.reshape(imsize))
ax.set_title('The "average" face')

# plot singular values
def plot_singvals(s,title):  
    cumu = []
    tot = np.sum(s)
    for i in range(len(s)):
        cumu.append(np.sum(s[:i])/tot)
    
    fig,axes = plt.subplots(2,1,sharex=True)
    axes[0].plot(s/tot,'ro',markersize=2)
    axes[0].set_ylabel('$\sigma_j$ normalized')
    axes[1].plot(cumu,'r',linewidth=3)
    axes[1].axhline(0.95,label='95%')
    axes[1].legend()
    axes[1].set_ylabel('cumulative variance')
    axes[0].set_title(title)

plot_singvals(s0,'Singular values of cropped faces')

# plot principle components of the face matrix
def plot_PCs(u,s,title,rows=3,cols=5):
    fig,axes = plt.subplots(rows,cols)
    
    for r in range(rows):
        for c in range(cols):  
            n = r*cols+c
            im = u[:,n].reshape(imsize)
            imshow(axes[r,c],im)
            axes[r,c].set_xlabel('$\sigma_{' + str(n+1) +
                '}$ = ' + str(int(s[n])))
    
    fig.text(0.5,0.95,title,ha='center',fontsize=12)

plot_PCs(u0,s0,'The first 15 principle components of the cropped faces')

# projection of face matrix
def plot_proj(X,u,n,seed,title):
    fig,axes = plt.subplots(3,4)
    np.random.seed(seed)
    for i in range(3):
        face = np.random.randint(X.shape[1])
        print(face)
        for j in range(axes.shape[1]-1):
            im = X[:,face+j].reshape(imsize)
            proj = u[:,:n].T @ X[:,face+j]
            imshow(axes[i,j],im)
            axes[i,j].set_xlabel('shot ' + str(j+1))
            axes[i,axes.shape[1]-1].plot(proj)
            axes[i,axes.shape[1]-1].set_xticks([])
            axes[i,axes.shape[1]-1].set_yticks([])
    fig.text(0.5,0.95,title,ha='center',fontsize=12)
    
plot_proj(X0,u0,15,10,
          'Projection of 3 faces onto the first 15 principle componenets') 

# low rank approximations of faces   
fig,axes = plt.subplots(2,5)
for row in range(len(axes)):
    face = np.random.randint(X.shape[1])
    face_full = X[:,face].reshape(imsize)
    axes[row,0].imshow(face_full)
    axes[row,0].set_xlabel('full rank')
    face_r = np.zeros(face_full.size)
    rank = [0,10,100,250,500]
    for i in range(1,len(rank)):
        for r in range(rank[i-1],rank[i]):
            face_r += s[r] * u[:,r] * v[r,face]
        axes[row,i].imshow(face_r.reshape(imsize))
        axes[row,i].set_xlabel('rank {}'.format(rank[i]))
    for ax in axes[row]:
        ax.set_xticks([]), ax.set_yticks([])
        
fig.text(0.5,0.95,'Low rank reconstructions of faces',
         ha='center',fontsize=12)
        
# uncropped images
path = 'yalefaces/'
imsize = (243,320)
subjs = ['subject' + (n < 10)*'0' + str(n) for n in range(1,16)]
X = []
for pic in os.listdir(path):
    im = imio.imread(os.path.join(path,pic))
    X.append(im.flatten())
X = np.array(X).T

X = X - np.mean(X,axis=1).reshape((X.shape[0],1))
u,s,v = np.linalg.svd(X,full_matrices=False)

# plot singular values
plot_singvals(s,'Singular Values of uncropped faces')

# plot PCs of uncropped faces
plot_PCs(u,s,'The first 15 principle components of the cropped faces')

plot_proj(X,u,25,70,
          'Projection of 3 faces onto the first 15 priciple components')   