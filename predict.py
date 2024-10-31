from tensorflow.keras.models import load_model

from tqdm import tqdm
import time
from SERESUNET import SE
from losses import dice_loss
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import os

n_label = 3
# # ['耕地','林草地',  '其他用地']
classes = [2., 3., 4.]

labelencoder = LabelEncoder()
labelencoder.fit(classes)


def predict(model_dir, model_name, test_path):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    custom_objects = {
              '_dice_loss' : dice_loss(),
              'SE': SE,
            }
    model = load_model(model_dir + model_name + '.h5', custom_objects={'_dice_loss': dice_loss(), 'SE': SE})
    test_set = os.listdir(test_path + 'images/')
    save_dir = test_path + model_name
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    for n in tqdm(range(len(test_set))):
        path = test_set[n]
        name = os.path.splitext(path)[0]
        img = cv2.imread(test_path + 'images/' + path)
        img = img / 255
        h, w, _ = img.shape
        mask_whole = np.zeros((h, w), dtype=np.uint8)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img, verbose=0)
        pred = np.argmax(pred, axis = 2)
        pred = labelencoder.inverse_transform(pred[0])
        pred = pred.reshape((h,w)).astype(np.uint8)
        #color_image = color_annotation(pred)
        save_name = name + '.png'
        cv2.imwrite(save_dir + '/' + path, pred)


rootPath = 'H:e/d//save_data/'
trainPath = rootPath + 'train/'
valPath = rootPath + 'val/'
modelSaveDir = rootPath + 'save_models/'
modelName = 'seresUNet***'

testPath = rootPath + 'test/'

start_time = time.perf_counter()
predict(modelSaveDir, modelName, testPath)
end_time = time.perf_counter()
log_time = "预测总时间: " + str(end_time - start_time) + "s"
print(log_time)

