import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('training.csv')
df1 = pd.read_csv('test.csv')
images1 = df['Image']
images = df['Image']
label = []
lx = df['left_eye_center_x']
ly = df['left_eye_center_y']
rx = df['right_eye_center_x']
ry = df['right_eye_center_y']

for i in range(len(df)):
    temp = []
    temp.append(lx[i])
    temp.append(ly[i])
    temp.append(rx[i])
    temp.append(ry[i])
    label.append(temp)


def preprocess_image(img):
    """
            returns image in after reshaping from csv files
            """
    formated_image = []
    for i in range(len(img)):
        results = (img[i].split(' '))
        results = list(map(int, results))
        results = np.asarray(results)
        results = results.reshape(96, 96)
        formated_image.append(results)
    return np.asarray(formated_image)


train_images, test_images, train_labels, test_labels = train_test_split(preprocess_image(images), label, test_size=0.1,
                                                                        random_state=42)
o_images = preprocess_image(images1)
np.save('values/train_images.npy', train_images)
np.save('values/train_label.npy', train_labels)
np.save('values/test_images.npy', test_images)
np.save('values/test_label.npy', test_labels)
np.save('values/o_images.npy', o_images)
