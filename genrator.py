import os
import cv2
import random
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np



n_label = 3
# # ['林草地', '耕地', '其他用地']
classes = [2., 3., 4.]

labelencoder = LabelEncoder()
labelencoder.fit(classes)


def generateData(batch_size, path, size=256):
    while True:
        images = []
        labels = []
        batch = 0
        data = os.listdir(path + 'images/')
        random.shuffle(data)
        for i in (range(len(data))):
            name = data[i]
            batch += 1
            img = cv2.imread(path + 'images/' + name)
            img = img / 255.0
            images.append(img)
            label = cv2.imread(path + 'labels/' + name, cv2.IMREAD_GRAYSCALE)
            label = label.reshape((size * size,))
            labels.append(label)
            if batch % batch_size == 0:
                images = np.array(images)
                labels = np.array(labels).flatten()
                labels = labelencoder.transform(labels)
                labels = to_categorical(labels, num_classes=n_label)
                labels = labels.reshape((batch_size, size * size, n_label))
                yield (images, labels)
                images = []
                labels = []
                batch = 0


def generateAllData(path, size=256):
    while True:
        images = []
        labels = []
        batch = 0
        data = os.listdir(path + 'images/')
        random.shuffle(data)
        for i in (range(len(data))):
            name = data[i]
            batch += 1
            img = cv2.imread(path + 'images/' + name)
            img = img / 255.0
            images.append(img)
            label = cv2.imread(path + 'labels/' + name, cv2.IMREAD_GRAYSCALE)
            label = label.reshape((size * size,))
            labels.append(label)
        images = np.array(images)
        labels = np.array(labels).flatten()
        labels = labelencoder.transform(labels)
        labels = to_categorical(labels, num_classes=n_label)
        labels = labels.reshape((-1, size * size, n_label))

        return images, labels
