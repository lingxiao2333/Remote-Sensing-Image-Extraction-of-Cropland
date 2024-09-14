import cv2
import numpy as np
from PIL import Image
import random
import os
from tqdm import tqdm
import shutil
from matplotlib import pyplot as plt

# 定义数据增强函数

# 添加点噪声
def add_noise(img,label):
    size = img.shape[0]
    for i in range(size):

        temp_x = np.random.randint(0, img.shape[0])

        temp_y = np.random.randint(0, img.shape[1])

        img[temp_x][temp_y] = 255

    return img,label

# 旋转
def rotate(img,label, angle):
    size = img.shape[0]
    M_rotate = cv2.getRotationMatrix2D((size /2, size / 2), angle, 1)
    img = cv2.warpAffine(img, M_rotate, (size, size))
    label = cv2.warpAffine(label, M_rotate, (size, size))

    return img, label


def random_rotate(img, label, angel):
    size = img.shape[0]
    scale_size = int(size*1.1)
    crop_size = int(size*0.1/2)
    M_rotate = cv2.getRotationMatrix2D((size /2, size / 2), angel, 1)
    img = cv2.warpAffine(img, M_rotate, (size, size))
    if label.shape[0] > 0 and label.shape[1] > 0:  # 添加检查
        label = cv2.warpAffine(label, M_rotate, (size, size))
    label = cv2.warpAffine(label, M_rotate, (size, size))
    img = cv2.resize(img, (scale_size, scale_size), interpolation=cv2.INTER_NEAREST)
    label = cv2.resize(label, (scale_size, scale_size), interpolation=cv2.INTER_NEAREST)
    img = img[crop_size:crop_size+size, crop_size:crop_size+size, :]
    label = label[crop_size:crop_size+size, crop_size:crop_size+size]

    return img, label


def random_rotate(img, label, angel):
    size = img.shape[0]
    print(img.shape[0])
    scale_size = int(size * 1.1)
    crop_size = int(size * 0.1 / 2)
    M_rotate = cv2.getRotationMatrix2D((size / 2, size / 2), angel, 1)
    img = cv2.warpAffine(img, M_rotate, (size, size))

    if label is not None and label.shape[0] > 0 and label.shape[1] > 0:
        label = cv2.warpAffine(label, M_rotate, (size, size))
    else:
        label = None

    img = cv2.resize(img, (scale_size, scale_size), interpolation=cv2.INTER_NEAREST)

    if label is not None and label.shape[0] > 0 and label.shape[1] > 0:
        label = cv2.resize(label, (scale_size, scale_size), interpolation=cv2.INTER_NEAREST)
    else:
        label = None

    img = img[crop_size:crop_size + size, crop_size:crop_size + size, :]

    if label is not None:
        label = label[crop_size:crop_size + size, crop_size:crop_size + size]

    return img, label


# 伽马变换
def gamma_transform(img,label):

    gamma_vari = np.random.random()
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    img = cv2.LUT(img, gamma_table)

    return img,label

#随机翻转
def flip(img,label, code):
    # code为0：沿x轴翻转,1：沿y轴翻转
    img = cv2.flip(img,code)
    label = cv2.flip(label,code)

    return img,label


def data_augment(img, label, count, augmentation_threshold=20):

    label_4_percentage = (np.sum(label == 4) / label.size) * 100

    if label_4_percentage <= augmentation_threshold:
        return img, label

    if count == 0:
        img, label = random_rotate(img, label, 90)
    elif count == 1:
        img, label = random_rotate(img, label, 180)
    elif count == 2:
        img, label = random_rotate(img, label, 270)
    elif count == 3:
        img, label = flip(img, label, 0)
    elif count == 4:
        img, label = flip(img, label, 1)
    elif count == 5:
        img, label = gamma_transform(img, label)
    else:
        img, label = add_noise(img, label)

    return img, label

random.seed(42)

image_folder = 'G:/newimage/DUIBI/3N/data_process/images/'
label_folder = 'G:/newimage/DUIBI/3N/data_process/labels/'

# Get a list of image and label files
image_files = os.listdir(image_folder)
label_files = os.listdir(label_folder)

image_files.sort()
label_files.sort()
# Combine image and label files into pairs
file_pairs = list(zip(image_files, label_files))

# Shuffle the pairs
random.shuffle(file_pairs)

# Split the shuffled pairs into training, validation, and test sets
total_files = len(file_pairs)
train_size = int(0.8 * total_files)
val_size = int(0.1 * total_files)
test_size = total_files - train_size - val_size

train_pairs = file_pairs[:train_size]
val_pairs = file_pairs[train_size:train_size+val_size]
test_pairs = file_pairs[train_size+val_size:]

# Unzip the pairs back into separate lists of images and labels
train_images, train_labels = zip(*train_pairs)
val_images, val_labels = zip(*val_pairs)
test_images, test_labels = zip(*test_pairs)

# Modify the copy_files function to accept lists of files
def copy_files(source_folder, dest_folder, files):
    for file in files:
        source_path = os.path.join(source_folder, file)
        dest_path = os.path.join(dest_folder, file)
        shutil.copy(source_path, dest_path)



copy_files(image_folder, 'G:/newimage/DUIBI/3N/data_process/save_data/train/images/', train_images)
copy_files(label_folder, 'G:/newimage/DUIBI/3N/data_process/save_data/train/labels/', train_labels)
copy_files(image_folder, 'G:/newimage/DUIBI/3N/data_process/save_data/val/images/', val_images)
copy_files(label_folder, 'G:/newimage/DUIBI/3N/data_process/save_data/val/labels/', val_labels)
copy_files(image_folder, 'G:/newimage/DUIBI/3N/data_process/save_data/test/images/', test_images)
copy_files(label_folder, 'G:/newimage/DUIBI/3N/data_process/save_data/test/labels/', test_labels)


train_image_files = os.listdir('G:/newimage/DUIBI/3N/data_process/save_data/train/' + 'images/')
train_label_files = os.listdir('G:/newimage/DUIBI/3N/data_process/save_data/train/' + 'labels/')

for i in tqdm(range(len(train_image_files))):
    image_path = os.path.join('G:/newimage/DUIBI/3N/data_process/save_data/train/' + 'images/', train_image_files[i])
    label_path = os.path.join('G:/newimage/DUIBI/3N/data_process/save_data/train/' + 'labels/', train_label_files[i])

    image = cv2.imread(image_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    label_4_percentage = (np.sum(label == 4) / label.size) * 100

    if label_4_percentage <= 20:
        continue

    for count in range(7):
        augmented_image, augmented_label = data_augment(image, label, count)

        if augmented_label is not None and augmented_label.size != 0:
            augmented_image_path = os.path.join('G:/newimage/DUIBI/3N/data_process/save_data/train/' + 'images/',f'{count}_augmented_{i}.tif')
            augmented_label_path = os.path.join('G:/newimage/DUIBI/3N/data_process/save_data/train/' + 'labels/',f'{count}_augmented_{i}.tif')

            cv2.imwrite(augmented_image_path, augmented_image)
            cv2.imwrite(augmented_label_path, augmented_label)
            print(f"Saved augmented images for pair {i} with count {count} | Label 4 Percentage: {label_4_percentage:.2f}%")

print(f"Total: {total_files} | Train: {train_size} | Val: {val_size} | Test: {test_size}")




