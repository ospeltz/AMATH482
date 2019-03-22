import numpy as np 
import scipy.io as spio
import matplotlib.pyplot as plt
import time 
import winsound as ws

def loadCams(test):
    ''' test is a string for the test number (1-4), loads the frames for all
    the different perspectives for this test in a list, in grayscale'''
    
    camNames = [str(k) + '_' + test for k in range(1,4)]
    
    gray = [0.3, 0.6, 0.1]
    t = time.time()
    cams = [spio.loadmat('cams/cam' + c)['vidFrames' + c] for c in camNames]
    print('load mat time',time.time()-t)
    
    views = []
    for i,cam in enumerate(cams):
        xx,yy,rgb,fs = cam.shape
        t = time.time()
        out = np.zeros((xx,yy,fs))
        
        for f in range(fs):
            out[:,:,f] = np.dot(cam[:,:,:,f],gray)
        print('grayscale convert',i,time.time()-t)
        t = time.time()
        views.append(out)
    return views

def extractPos(view,thresh,box,crop=None):
    '''takes a single viewpoint of the bouncing mass and extracts the position
    of it as a numpy array (2,n_frames) with pos[0,:] = dim1, pos[0,:] = dim2,
    thresh is the threshold of variance for a pixel value to be kept, box is 
    box size to search for with the brightest pixels, crop is a length 4 list
    [dim0 start, dim0 stop, dim1 start dim1 stop]'''
    if crop:
        view = view[crop[0]:crop[1],crop[2]:crop[3],:]
    x,y,fs = view.shape
    pos = -1 * np.ones((2,fs))
    t = time.time()
    pixel_var = np.std(view,axis=2)
    boo = (pixel_var > thresh).reshape((x,y,1))
    boo = np.dot(boo,np.ones((1,fs)))
    view = view * boo
    print('filter time:',time.time()-t)
    t = time.time()
    for f in range(fs):
        frame = view[:,:,f]
        s = np.zeros_like(frame)
        for i in range(box):
            for j in range(box):
                s[0:x-box+1,0:y-box+1] += \
                frame[i:x+i-box+1,j:y+j-box+1]
        
        pos[:,f] = np.unravel_index(np.argmax(s),(x,y))
    print('add time:',time.time()-t)
    return pos, view, pixel_var

# crop coordinates for the noisy case
crops = [[0,480,300,450],
         [0,480,200,400],
         [210,350,0,640]]

for test in ['1','2','3','4']:
    views = loadCams(test)
    for i in range(1):
        view = views[i]
        if test == '2':
            crop = crops[i]
        else:
            crop = None
        pos,filtered,pv = extractPos(view,20,15)
        ws.Beep(880,500)
        np.save('pos'+test+str(i),pos)