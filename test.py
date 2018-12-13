import numpy as np
from keras.models import load_model

X_test = np.load('values/test_images.npy')
Y_test = np.load('values/test_label.npy')
X_test = X_test.reshape(X_test.shape[0], 96, 96, 1)


def func_test():
    """
            :returns test accuracy and loss
            """
    model = load_model("Models/CNN.h5")
    pred = model.evaluate(x=X_test, y=Y_test, verbose=1)
    return pred
