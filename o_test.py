import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

model = load_model("Models/CNN.h5")
images = np.load("values/o_images.npy")
images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
test = images[12]
print(test.shape)
out = model.predict(test.reshape(1,96,96,1), batch_size=1)
print(out[0])
lx, ly, rx, ry = out[0]
plt.scatter(lx, ly)
plt.scatter(rx, ry)
plt.imshow(test.reshape(96,96))
plt.show()
