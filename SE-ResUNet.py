from tensorflow.keras.layers import Multiply
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, Reshape
from tensorflow.keras.layers import BatchNormalization, Dense, Add
from tensorflow.keras import Model
from tensorflow.keras import layers
from keras.models import *
from keras.layers import *
from keras import layers
import tensorflow as tf

class SE(Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super(SE, self).__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.fc1 = Dense(self.units, activation=self.activation, kernel_initializer='he_normal')
        self.fc2 = Dense(input_shape[-1], activation='sigmoid', kernel_initializer='glorot_normal')

    def call(self, inputs):
        x = K.mean(inputs, axis=1, keepdims=True)
        x = self.fc1(x)
        x = self.fc2(x)
        x = Multiply()([inputs, x])
        return Add()([inputs, x])

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(SE, self).get_config()
        config.update({'units': self.units, 'activation': self.activation})
        return config


IMAGE_ORDERING = 'channels_last'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    shortcut = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING, strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def get_resnet50_encoder(input_height=256, input_width=256, pretrained='imagenet',
                         include_top=True, weights='imagenet',
                         input_tensor=None, input_shape=None,
                         pooling=None,
                         classes=1000):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input) #64个3×3的卷积核
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    f1 = x
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING, strides=(2, 2))(x)
    x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1))

    x = SE(128)(x)


    x = identity_block(x, 3, [32, 32, 128], stage=2, block='b')
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='c')
    f2 = x
    x = conv_block(x, 3, [64, 64, 256], stage=3, block='a')

    x = SE(64)(x)

    x = identity_block(x, 3, [64, 64, 256], stage=3, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='c')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='d')
    f3 = x
    x = conv_block(x, 3, [128, 128, 512], stage=4, block='a')

    x = SE(32)(x)


    x = identity_block(x, 3, [128, 128, 512], stage=4, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='d')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='e')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='f')
    f4 = x
    x = conv_block(x, 3, [256, 256, 1024], stage=5, block='a')

    x = SE(16)(x)


    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='c')
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]

def _unet(num_classes, encoder, input_height=256, input_width=256):
    img_input, levels = encoder(input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, f5] = levels

    o = f5
    o = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(512, (2, 2), padding='same', data_format=IMAGE_ORDERING)(o)
    o = Activation('relu')(o)
    o = BatchNormalization()(o)
    o = concatenate([o, f4], axis=3)
    o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING)(o)
    o = Activation('relu')(o)
    o = BatchNormalization()(o)
    o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING)(o)
    o = Activation('relu')(o)
    o = BatchNormalization()(o)




    o = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(256, (2, 2), padding='same', data_format=IMAGE_ORDERING)(o)
    o = Activation('relu')(o)
    o = BatchNormalization()(o)
    o = concatenate([o, f3], axis=3)
    o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)



    o = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(128, (2, 2), padding='same', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    o = concatenate([o, f2], axis=3)
    o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING)(o)
    o = Activation('relu')(o)
    o = BatchNormalization()(o)
    o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING)(o)
    o = Activation('relu')(o)
    o = BatchNormalization()(o)



    o = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(64, (2, 2), padding='same', data_format=IMAGE_ORDERING)(o)
    o = Activation('relu')(o)
    o = BatchNormalization()(o)
    o = concatenate([o, f1], axis=3)
    o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING)(o)
    o = Activation('relu')(o)
    o = BatchNormalization()(o)
    o = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(o)
    o = Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING)(o)
    o = Activation('relu')(o)
    o = BatchNormalization()(o)



    o = Conv2D(num_classes, (3, 3), activation='softmax', padding='same', data_format=IMAGE_ORDERING)(o)
    o = Reshape((int(input_height) * int(input_width), num_classes))(o)
    o = Softmax()(o)
    model = Model(img_input, o)

    return model


def SE_ResUnet(num_classes, input_height=256, input_width=256):
    model = _unet(num_classes, get_resnet50_encoder, input_height=input_height, input_width=input_width)
    return model

model = SE_ResUnet(3)
model.summary()
