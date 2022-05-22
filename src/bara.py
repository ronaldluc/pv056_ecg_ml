from sklearn.svm import SVC
from tsai.all import *

from martin import TSAI


class MLClassificator(SVC):
    def __init__(self):
        super().__init__()

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
        return 'SVM'


class TSTClassificator(TSAI):
    def __init__(self):
        super().__init__(TST)
