from sklearn.naive_bayes import GaussianNB
from tensorflow import keras as K
from tsai.all import *
import numpy as np

#machine learning classificator
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

class TSAI:
    def __init__(self, arch):
        self.arch = arch

    def fit(self, X_train, y_train, **args):
        X_val, y_val = args['validation_data']
        X, y, splits = combine_split_data([X_train, X_val], [y_train, y_val])
        y = OneHot(n_classes=y.shape[1]).decode(y)
        self.model = TSClassifier(
            X,
            y,
            splits=splits,
            bs=[64, 128],
            batch_tfms=[TSStandardize()],
            arch=self.arch,
            metrics=accuracy
        )
        self.model.fit_one_cycle(25, lr_max=1e-3)

    def predict(self, X):
        y = np.array(self.model.get_X_preds(X)[2], dtype='int')
        return y

    @property
    def name(self):
        return str(self.arch).split('.')[-1][:-2]


class LSTMClassificator(TSAI):
    def __init__(self):
        super().__init__(LSTM)
    