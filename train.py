from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam,SGD
from generator_github import generateData,generateAllData
import tensorflow.keras.backend as K
from loss_gifhub import dice_loss
from cosine_decay import WarmUpCosineDecayScheduler
import os
import datetime
import time

n_label = 3
# # ['林草地', '耕地', '其他用地']
classes = [2., 3., 4.]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

size = 256
EPOCHS = 100
BS = 16


# 训练
def train(dir, model_name, size, n_label, EPOCHS, BS, lr, decay, train_path, val_path):
    train_numb = len(os.listdir(train_path + 'images/'))
    valid_numb = len(os.listdir(val_path + 'images/'))
    #     loss = binary_focal_loss()
    loss = dice_loss()
    #model = seg_hrnet(height=size, width=size, channel=3, classes=n_label)
    #model = SegNet(size=size, channel=3, num_classes=n_label)
    #model = ResUnet(num_classes=n_label)
    model = SE_ResUnet(num_classes=n_label)
    #model = unet_se(height=size, width=size, channel=3, num_classes=n_label)
    #model = unet(height=size, width=size, channel=3, num_classes=n_label)
    #model = DenseNet(input_shape=(size,size,3), dense_blocks=4, dense_layers=[6, 12, 24, 16], growth_rate=12, nb_classes=n_label,
    #        dropout_rate=0.5,bottleneck=False, compression=1.0, weight_decay=1e-4, depth=121)
    model.compile(
        #             optimizer=SGD(learning_rate=lr, momentum=0.9, decay=decay),
        optimizer=Adam(learning_rate=lr, decay=decay),
        #             optimizer=Nadam(learning_rate=lr),
        # loss=losses.categorical_crossentropy,
        loss=loss,
        metrics=['accuracy']
    )
    modelcheck = ModelCheckpoint(dir + model_name + '.h5', monitor='val_loss', save_best_only=True, verbose=1)
    # callable = [modelcheck]
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    def poly_decay(epoch):
        maxEpochs = EPOCHS
        step_each_epoch = train_numb / BS
        baseLR = lr
        power = 0.9
        ite = K.get_value(model.optimizer.iterations)
        # compute the new learning rate based on polynomial decay
        alpha = baseLR * ((1 - (ite / float(maxEpochs * step_each_epoch))) ** power)
        # return the new learning rate
        return alpha

    lrate = LearningRateScheduler(poly_decay, verbose=1)
    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=lr,
                                            total_steps=int(EPOCHS * train_numb / BS),
                                            warmup_learning_rate=5e-05,
                                            warmup_steps=10,
                                            hold_base_rate_steps=3,
                                            verbose=0
                                            )
    # model_name = 'unet-{}'.format(int(time.time()))
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(model_name))
    # callable = [modelcheck,lrate,es,tensorboard]
    callable = [modelcheck, reduce_lr, es]
    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)
    #  获取当前时间
    start_time = datetime.datetime.now()

    H = model.fit_generator(generator=generateData(BS, train_path, size), steps_per_epoch=train_numb // BS,
                            epochs=EPOCHS, verbose=1,
                            validation_data=generateData(BS, val_path, size), validation_steps=valid_numb // BS,
                            callbacks=callable,
                            max_queue_size=1)

    #     train_x, train_y = generateAllData(train_path,size)
    #     val_x, val_y = generateAllData(val_path,size)
    #     H = model.fit(x=train_x, y=train_y, batch_size=BS, epochs=EPOCHS, verbose=1, callbacks=callable,
    #                   validation_data=(val_x,val_y), shuffle=True)
    #  训练总时间
    end_time = datetime.datetime.now()
    log_time = "训练总时间: " + str((end_time - start_time).seconds / 60) + "m"
    print(log_time)

    return H

rootPath = 'H:e/d//save_data/'

trainPath = rootPath + 'train/'
valPath = rootPath + 'val/'
modelSaveDir = rootPath + 'save_models/'
modelName = 'seresUNet***'
if not os.path.exists(modelSaveDir): os.mkdir(modelSaveDir)
size = 256
classNum = 3
epochs = 150
batchsize = 8
learningRate = 0.001
decay = 0.00
# 训练
H = train(modelSaveDir,modelName,size,classNum,epochs,batchsize,learningRate,decay,trainPath,valPath)
