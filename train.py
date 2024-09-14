from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam,SGD
from data_process.generator import get_train_val,generateData,generateValidData
from loss.focal_loss import multi_category_focal_loss1
import tensorflow.keras.backend as K
import tensorflow as tf
from dice_loss import dice_loss
from loss.bce_loss import bce_dice_loss
from lovasz_losses import lovasz_softmax
from tensorflow.keras.losses import categorical_crossentropy
from loss1 import acfloss,acfloss2
from WCCE import weighted_categorical_crossentropy
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_set,val_set = get_train_val(val_rate=0.25,num_rate=1)
train_numb = len(train_set)  
valid_numb = len(val_set)

size = 256
n_label = 3
EPOCHS = 100
BS = 16


#model = unet(height=size, width=size, channel=3, num_classes=n_label)
#model = Res_UNet(height=size, width=size, channel=3, num_classes=n_label)
#model = unet_se(height=size, width=size, channel=3, num_classes=n_label)
#model = seg_hrnet(height=size, width=size, channel=3, classes=n_label)
#model = SegNet(size=size, channel=3, num_classes=n_label)
model = SE_ResUnet(num_classes=n_label)
model.compile(  optimizer=SGD(learning_rate=0.001,momentum=0.9, decay=0.0001),
#                 loss={
#                   'fine_segmentation': 'categorical_crossentropy',
#                   'coarse_segmentation': 'categorical_crossentropy',
#                   'auxiliary': 'categorical_crossentropy'},
#                 loss_weights={
#                   'fine_segmentation': 0.7,
#                   'coarse_segmentation': 0.6,
#                   'auxiliary': 0.4},
#                 metrics=['accuracy'])
# loss = multi_category_focal_loss1(alpha=[2,1,2,1,2,3], gamma=2)

model.compile(  optimizer=Adam(learning_rate=0.0005),
                loss=['categorical_crossentropy'],
                metrics=['accuracy']
             )
#poly策略
def poly_decay(epoch):
    maxEpochs = EPOCHS
    step_each_epoch = train_numb / BS
    baseLR = 0.0005
    power = 0.9
    ite = K.get_value(model.optimizer.iterations)
    # compute the new learning rate based on polynomial decay
    alpha = baseLR*((1 - (ite / float(maxEpochs*step_each_epoch)))**power)
    # return the new learning rate
    return alpha


def train(dir): 
    modelcheck = ModelCheckpoint(dir ,monitor='val_accuracy',save_best_only=True,mode='max',verbose=1) 
    # callable = [modelcheck]
    es = EarlyStopping(
                        monitor='val_accuracy', 
                        min_delta=0, 
                        patience=10, 
                        verbose=1,
                        mode='max'
                    )
    
    reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss', 
                        factor=0.5, 
                        patience=3, 
                        verbose=1
                    )
    
    lrate = LearningRateScheduler(poly_decay)
    model_name = 'unet-{}'.format(int(time.time()))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(model_name))
    callable = [modelcheck,lrate,es,tensorboard] 

    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    #  获取当前时间
    start_time = datetime.datetime.now()

    H = model.fit_generator(generator=generateData(BS,train_set,size),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                            validation_data=generateValidData(BS,val_set,size),validation_steps=valid_numb//BS,callbacks=callable,max_queue_size=1)  

    #  训练总时间
    end_time = datetime.datetime.now()
    log_time = "训练总时间: " + str((end_time - start_time).seconds / 60) + "m"
    print(log_time)


if __name__=='__main__':  
    train(dir = './save_model/xxx.h5')
