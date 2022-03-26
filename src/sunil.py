from sklearn.neighbors import KNeighborsClassifier


class MLClassificator(KNeighborsClassifier):
    def __init__(self):
        super().__init__(n_neighbors=7, algorithm='auto', n_jobs=-1)

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
        return 'KNN'
