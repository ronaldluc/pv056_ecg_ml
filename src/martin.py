from tsai.all import *
import numpy as np


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


class MiniRocketClassificator(TSAI):
    def __init__(self):
        super().__init__(MiniRocket)


class InceptionTimeClassificator(TSAI):
    def __init__(self):
        super().__init__(InceptionTime)


class TSTClassificator(TSAI):
    def __init__(self):
        super().__init__(TST)


class XCMClassificator(TSAI):
    def __init__(self):
        super().__init__(XCM)
