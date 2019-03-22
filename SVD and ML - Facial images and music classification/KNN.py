import numpy as np
class KNN:
    def __init__(self,k):
        self.k = k
        self.X = []
        self.y = []
    
    def fit(self,X,y):
        ''' X = (n_samples,n_features), y = n_samples'''
        self.X = X
        self.y = y
    
    def predict(self,test):
        ''' test = (n_samples, n_features)'''
        if test.shape[-1] != self.X.shape[1]:
            raise ValueError('incosistent feature dims')
        labels = []
        for p in range(len(test)):
            diff = self.X - test[p,:]
            dist = (diff**2).sum(axis=1)
            inds = np.argsort(dist) < self.k
            labs = self.y[inds]
            uniq,freq = np.unique(labs,return_counts=True)
            labels.append(uniq[np.argmax(freq)])
        return np.array(labels)