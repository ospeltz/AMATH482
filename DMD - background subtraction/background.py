import numpy as np
import matplotlib.pyplot as plt
import imageio as imio
from PIL import Image
la = np.linalg
plt.rcParams['image.cmap'] = 'gray'
# (height, width)
scale = 2
imsize = (scale*108,scale*192)
fps = 30

def load_vid(test,rotate=False):
    '''loads the video and saves it as a 2d np array, each frame 
    flattened as a column'''
    path = 'vids/' + test
    vid = imio.get_reader(path + '.mp4','ffmpeg')
    X = []
    for frame in vid:
        im = Image.fromarray(frame)
        if rotate:
            im = im.rotate(90,expand=True)
        im = im.convert('L') # grayscale
        im = im.resize((imsize[1],imsize[0]))
        X.append(np.array(im).flatten())
    vid.close()
    # frames in the columns
    X = np.array(X).T
    np.save(path,X)
    return X

def imshow(im,ax):
    if len(im.shape) < 2:
        im = im.reshape(imsize)
    ax.imshow(im)
    ax.set_xticks([]),ax.set_yticks([])

test = 'test6'
X = load_vid(test)  

u,s,v = la.svd(X,full_matrices=False)
# show principal components of the video
fig = plt.figure()
ax = fig.add_subplot(311)
ax.plot((s/s.sum())[:50],'ro',fillstyle='none')
ax.set_xticks([]); ax.set_yticks([0,0.5])
ax.set_title('Singular values and principle components')
axes = [fig.add_subplot(3,2,i) for i in range(3,7)]
for i in range(4):
    imshow(-1*u[:,i],axes[i]) # PCs are inverted
    sig = np.round(s[i]/s.sum(),3)
    axes[i].set_xlabel('$\sigma_{} = {}$'.format(i+1,sig))

# subtract first principle component from 
# each frame
ubg = s[0] * u[:,0:1] @ v[0:1,:]
Xfg1 = X - ubg

# dmd 
X1 = X[:,:-1]; X2 = X[:,1:]
U,S,V = la.svd(X1,full_matrices=False)
r = 100
Ur = U[:,:r]; Sr = S[:r]; Vr = V[:r,:]
A_til = Ur.T @ X2 @ Vr.T @ np.diag(1/Sr)
lam,W = la.eig(A_til)
Phi = X2 @ Vr.T @ np.diag(1/Sr) @ W
omega = np.log(lam)

# threshold for ln of DMD frequency to be
# considered part of the background
thresh = 0.01
# this command returns a tuple, the first value
# are the indices we need
bg_i = np.nonzero(np.abs(omega) < thresh)[0]

x0 = X[:,0]
Phibg = Phi[:,bg_i]
# initial DMD projection of background
bg0 = la.pinv(Phibg) @ x0

# it needs to hold complex values, so that must 
# be declared off the bat
bg_modes = np.zeros((len(bg_i),X.shape[1]),dtype=complex)
# DMD projection of background for each frame
for i in range(X.shape[1]):
    bg_modes[:,i] = bg0 * np.exp(omega[bg_i]*i)
Xbg = Phibg @ bg_modes
Xfg2 = X - np.abs(Xbg)

fig,axes = plt.subplots(3,5)
titles = ['Frame','SVD background','SVD foreground',
          'DMD background','DMD foreground']
for i in range(5):
    axes[0,i].set_title(titles[i])
for i,frame in enumerate([40,80,160]):
    imshow(X[:,frame],axes[i,0])
    imshow(ubg[:,frame],axes[i,1])
    imshow(Xfg1[:,frame],axes[i,2])
    imshow(np.abs(Xbg)[:,frame],axes[i,3])
    imshow(Xfg2[:,frame],axes[i,4])
    t = np.round(frame/fps,2)
    axes[i,0].set_ylabel('$t = {}$'.format(t),
        fontsize=12)

# predict "future" of video
t = [100,100000,1000000000]
y0 = la.pinv(Phi) @ x0
modes = np.zeros((r,len(t)),dtype=complex)
for i in range(len(t)):
    modes[:,i] = y0 * np.exp(omega*t[i])
X_future = np.abs( Phi @ modes )
fig,axes = plt.subplots(1,len(t))
for i in range(len(t)):
    imshow(X_future[:,i],axes[i])
    time = np.round(t[i]/fps,1)
    axes[i].set_title('$t = {} s$'.format(time))
    