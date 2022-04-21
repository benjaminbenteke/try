import numpy as np
from utils import bag_of_words

class BernoulliNB(object):
    def __init__(self, voc,alpha=1.0):
        self.alpha = alpha
        self.feature_prob_= None
        self.voc= voc

    def fit(self, X, y):
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        smoothing = 2 * self.alpha
        n_doc = np.array([len(i) + smoothing for i in separated])
        # print(n_doc)
        self.feature_prob_ = count / n_doc[np.newaxis].T
        return self

    def predict_log_proba(self, X):
        return [(np.log(self.feature_prob_) * x + \
                 np.log(1 - self.feature_prob_) * np.abs(x - 1)
                ).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)
    
    def predic_new(self, x_new):

      x_new= x_new.split()
      x_new= [i.lower() for i in x_new]
      x_new= bag_of_words(x_new, self.voc)


      return np.argmax(self.predict(x_new))