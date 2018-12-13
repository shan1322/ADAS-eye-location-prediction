import numpy as np
from keras.layers import Conv2D
from keras.layers import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

X_train = np.load('values/train_images.npy')
Y_train = np.load('values/train_label.npy')
X_train = X_train[0:250]
Y_train = Y_train[0:250]
X_train = X_train.reshape(X_train.shape[0], 96, 96, 1)


def cnn(x_train):
    """
    returns Cnn
    """
    model = Sequential()

    model.add(BatchNormalization(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(Conv2D(8, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(16, (2, 2), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.50))

    model.add(Dense(4))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    return model


Cnn = cnn(X_train)
epochs = 1000

Cnn.fit(X_train, Y_train, epochs=epochs, batch_size=1, verbose=1)
keras_file = "models/cnn.h5"
Cnn.save(keras_file)
