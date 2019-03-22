import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wv
import sounddevice as sd
import os
import scipy.fftpack as ft

def read_audio(path,start=None,stop=None):
    '''returns the song as a sample rate and a np array, combines left and
    right channels into one, normalized volume to one, returns the portion of
    the song between start and stop, in seconds'''
    f,y = wv.read(path)
    # cast as float to avoid overflow in in16 dtype
    y = y / (y.max(axis=0).astype('float64')*2) 
    y = y[:,0] + y[:,1]
    if start != None and stop != None:
        y = y[int(f*start):int(f*stop)]
    return f,y

def spec(y,step,dt,width,ax=None):
    '''returns a 2d spectrogram of y, resampling every step points,
    using dt in between window centers, and the provided window width
    for a shannon filter, if ax is provided, it plots it'''
    inds = np.arange(0,len(y),step)
    y = y[inds]
    tslide = np.arange(0,len(y),dt)
    G = np.arange(len(y)) * np.ones((len(tslide),len(y)))
    win_start = (tslide - width).reshape((len(tslide),1)) * np.ones_like(G)
    win_stop = (tslide + width).reshape((len(tslide),1)) * np.ones_like(G)
    G = np.logical_and((G > win_start),(G < win_stop)) # create Gabor windows
    Yf = y * G # filter y with each step function
    Yft = ft.fft(Yf) # ft the rows 
    Yft = np.abs(Yft[:,:int(len(y)/2)]) # only save positive wave numbers
    if ax:
        ax.imshow(Yft.T,aspect='auto',cmap='hot')
        ax.set_xticks([]), ax.set_yticks([])
    return Yft

def test_train_split(genres,bands,n,specargs):
    '''returns (trainX,trainY,testX,testY), spectrogramed random song data
    n is the number of samples to take per song'''
    np.random.seed(100)
    if bands == None:
        splits = split_by_genres([],[],[],[],genres,n,specargs)
    else:
        splits = split_by_bands([],[],[],[],genres,bands,n,specargs)
    return (np.array(x).T for x in splits)
        
def split_by_bands(train,train_y,test,test_y,genres,bands,n,specargs):
    '''splits up data into training and testing sets for specific bands'''
    for genre in genres:
        artists = os.listdir(genre)
        for art in artists:
            if art in bands:
                path = os.path.join(genre,art)
                songs = os.listdir(path)
                t = np.random.randint(len(songs))
                for s in range(len(songs)):
                    song_path = os.path.join(path,songs[s])
                    for k in range(n):
                        start = np.random.randint(90)
                        _,y = read_audio(song_path,start,start+5)
                        yft = spec(y,*specargs)
                        if s == t:
                            test.append(yft.flatten())
                            test_y.append(art)
                        else:
                            train.append(yft.flatten())
                            train_y.append(art)
    return (train,train_y,test,test_y)
    
def split_by_genres(train,train_y,test,test_y,genres,n,specargs):
    '''splits up the data with no specific bands'''
    for genre in genres:
        for band in os.listdir(genre):
            path = os.path.join(genre,band)
            for song in os.listdir(path):
                song_path = os.path.join(path,song)
                if np.random.rand() > 0.2:
                    X = train
                    lab = train_y
                else:
                    X = test
                    lab = test_y
                for k in range(n):
                    start = np.random.randint(90)
                    _,y = read_audio(song_path,start,start+5)
                    yft = spec(y,*specargs)
                    X.append(yft.flatten())
                    lab.append(genre)
    return (train,train_y,test,test_y)
    
# (n_features,n_samples)
from sklearn.svm import SVC
from KNN import KNN

def trial(model,rank,genres,bands,n):
    '''runs a trial of the model identifying the genre or bands, returns
    the accuracy for the test set and training set averaged over five trials'''
    trials = []
    for i in range(5):
        tr,tr_y,te,te_y = test_train_split(genres,
                        bands,n,(4,44100/40,600))
    #    tr0 = tr.mean(axis=1).reshape((len(tr),1))
        u,s,v = np.linalg.svd(tr,full_matrices=False)
        PC_tr = u[:,:rank].T @ (tr)
        PC_te = u[:,:rank].T @ (te)
        model.fit(PC_tr.T,tr_y)
        pred = model.predict(PC_tr.T)
        acc1 = sum(pred==tr_y)/len(pred)
        print('train acc, rank =',rank,acc1)
        pred = model.predict(PC_te.T)
        acc2 = sum(pred==te_y)/len(pred)
        print('test acc',acc2)
        trials.append([acc1,acc2])
    trials = np.array(trials)
    return np.mean(trials,axis=0)

# assess the effect of the rank of the PC's included on model accuracy
#accs = []
#for rank in range(1,30):
#    svm = SVC(kernel='linear')
#    accs.append(trial(svm,rank,['hip hop','alt','house'],['EARTHGANG',
#          'RHCP','Phlegmatic Dogs'],5))
#    
## assess the effect of the k parameter on a KNN model
from sklearn.neighbors import KNeighborsClassifier
accs =  []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    accs.append(trial(knn,18,['hip hop','alt','house'],['EARTHGANG',
          'RHCP','Phlegmatic Dogs'],5))
#
## assess the model applied to different situations, identifying artists
## among similar artists
accs1 = []
for i in range(50):
    svm = SVC(kernel='linear')
    accs1.append(trial(svm,18,['house'],
                      ['Tchami','Phlegmatic Dogs','FISHER'],5))

# assess the model applied to identifying genres
#accs = []
#for k in range(1,21):
#    svm = SVC(kernel='linear')
#    accs.append(trial(svm,18,['house','alt','hip hop'],None,3))