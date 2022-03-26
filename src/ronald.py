import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras as K


class MLClassificator(RandomForestClassifier):
    def __init__(self):
        super().__init__(criterion='entropy', max_depth=70, n_estimators=20, max_features=500)

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
        return 'Random Forest'


class DLClassificator(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self):
        self.model = None
        self.kwargs = dict(kernel_size=7,  # 11
                           strides=2,  # 2
                           activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.callbacks = [K.callbacks.EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True, ), ]

        # print(f'Test acc: {model.evaluate(x[test], y[test], verbose=0)[1] * 100:7.4f} %')

    def fit(self, X, y, **fit_params):
        self.model = K.models.Sequential([K.layers.Input(shape=X.shape[1:]),
                                          #  K.layers.AveragePooling1D(),
                                          K.layers.Conv1D(128, **self.kwargs),
                                          K.layers.BatchNormalization(),
                                          K.layers.MaxPool1D(),
                                          K.layers.Conv1D(64, **self.kwargs),
                                          K.layers.BatchNormalization(),
                                          K.layers.MaxPool1D(),
                                          K.layers.Conv1D(64, **self.kwargs),
                                          # K.layers.BatchNormalization(),
                                          # K.layers.MaxPool1D(),
                                          # K.layers.Conv1D(64, **self.kwargs),
                                          K.layers.BatchNormalization(),
                                          K.layers.MaxPool1D(),
                                          K.layers.Flatten(),
                                          K.layers.Dense(32, 'relu'),
                                          K.layers.Dense(32, 'relu'),
                                          K.layers.Dense(y.shape[1], 'softmax')])
        optimizer = K.optimizers.Adam(learning_rate=0.005)
        self.model.summary()
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=8, batch_size=64, verbose=2, callbacks=self.callbacks, **fit_params)

    def predict(self, X):
        return self.model.predict(X).argmax(axis=-1)

    @property
    def name(self):
        return '1D CNN'
