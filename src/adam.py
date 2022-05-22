from sklearn.naive_bayes import GaussianNB
from tensorflow import keras as K
from tsai.all import *
import numpy as np

#machine learning classificator
from martin import TSAI


class MLClassificator(GaussianNB):
    def __init__(self):
        super().__init__(priors=None, var_smoothing=1e-09)

    @staticmethod
    def transform(X):
        return X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    def fit(self, X, y, **fit_params):
        X = self.transform(X)
        super().fit(X, y.argmax(axis=-1))
        return self

    def predict(self, X):
        return super().predict(self.transform(X))

    @property
    def name(self):
        return 'GaussianNB'

#deep learning calssificator



class LSTMClassificator(TSAI):
    def __init__(self):
        super().__init__(LSTM)
    