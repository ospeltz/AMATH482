import numpy as np
import scipy.io as sio
import scipy.fftpack as ft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting

L = 15 # spatial domain
n = 64 # fourier nodes

# set up spatial domain, throw out last point because fourier assumes a
# periodic domain, that is f(-L) = f(L)
x = np.linspace(-L,L,n,endpoint=False); y = x.copy(); z = x.copy();
# set up frequency domain, fft assumes a interval of 2pi so we must rescale 
# our 2L interval accordingly
k = np.pi/L * np.hstack((np.arange(0,n/2),np.arange(-n/2,0)))
# the fft algorithm puts things out of order, "butterfly shift"

# set up grids for X,Y and Z so any point is accesses as u[xi,yi,zi]
Y,X,Z = np.meshgrid(x,y,z)
# use butterfly shifted domain so we dont have to shift our transforms later
Ky,Kx,Kz = np.meshgrid(k,k,k)

# load data from .mat file and reshape it so it has the form
# [time slice,x,y,z]
dat = sio.loadmat('Testdata.mat')
un = dat['Undata'].reshape((20,n,n,n))

# fourier transform on the 1st, 2nd and 3rd axes so that the 0th axis,
# our time slices, is maintained
unt = ft.fftn(un,axes=(1,2,3))
# sum up all the time slices to average out the noise
ave = np.sum(unt,axis=0)

# plotda a histogram of the normalized signal strength
#a = (np.abs(ave)/np.max(np.abs(ave))).flatten()
#plt.hist(a)

# get the index of the maximum value of the average transformed signal
# and then get the corresponding (kx,ky,kz)
ind = np.argmax(ave)
xi,yi,zi = np.unravel_index(ind,(n,n,n))
x_cen = Kx.flatten()[ind]
y_cen = Ky.flatten()[ind]
z_cen = Kz.flatten()[ind]
print('The central frequency of the marble is:')
print('kx =',x_cen,'ky =',y_cen,'kz = ',z_cen)

# make a Gaussian filter around this frequency center
filt = np.exp(-0.2*((Kx-x_cen)**2 + (Ky-y_cen)**2 + (Kz-z_cen)**2))

# apply the filter to each slice of time in the frequency domain
untf = filt*np.ones_like(unt)*unt
# reverse the transform so we have our filtered signal in the spatial domain
unf = ft.ifftn(untf,axes=(1,2,3))

coords = np.zeros((20,3))
for j in range(20):
    # take the maximum value of the filtered signal to 
    # to be the location of the marble
    ind = np.argmax(np.abs(unf[j]))
    coords[j,:] = [X.flatten()[ind],Y.flatten()[ind],Z.flatten()[ind]]
  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(coords[:,0],coords[:,1],coords[:,2],label='trajectory',
        linewidth=3)
ax.scatter(coords[0,0],coords[0,1],coords[0,2],label='start')
ax.scatter(coords[19,0],coords[19,1],coords[19,2],label='end')
ax.legend()
ax.set(xlabel='X',ylabel='Y',zlabel='Z',
       title='Trajectory of the marble in poor, poor Fluffy',
       yticks=np.arange(-10,11,5))
plt.show()

print('To break up the marble with an acoustic wave, we should place it at:')
print('x =', coords[19,0], 'y =', coords[19,1], 'z = ', coords[19,2])
    
    
    