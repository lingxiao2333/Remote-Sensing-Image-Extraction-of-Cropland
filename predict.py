from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tqdm import tqdm
import time
from losses import dice_loss
import time
from SEUNet import SE
# from denseCRF import CRFs

labelencoder = LabelEncoder()
labelencoder.fit(classes)

n_label = 3
# # ['林草地', '耕地', '其他用地']
classes = [85., 170., 255.]


custom_objects = {  'relu6': K.ReLU(6.),
                    'tf':tf,
                    'DepthwiseConv2D': K.DepthwiseConv2D,
                    'loss':weighted_categorical_crossentropy,
                    '_dice_loss':dice_loss
                    'SE': SE,
                }


def color_annotation(img):
    color = np.ones([img.shape[0], img.shape[1], 3])
    color[img == 85.] = [0, 100, 0]
    color[img == 170.] = [34, 187, 255]
    color[img == 255.] = [240, 240, 240]
    return color

    return color

def predict(dir,output_path):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(dir,custom_objects=custom_objects)
    for n in tqdm(range(len(TEST_SET))):
        path = TEST_SET[n]
        name = os.path.splitext(path)[0]
        image = load_img(img_path + path)
        image = img_to_array(image)
        image = image / 255
        # image = image[0:image_size,0:image_size,:]
        h,w,_ = image.shape 
        mask_whole = np.zeros((h,w),dtype=np.uint8)
        crop = np.expand_dims(image, axis=0)
        pred = model.predict(crop,verbose=1)
        pred = np.argmax(pred,axis = 2)
        # print(pred) 
        pred = labelencoder.inverse_transform(pred[0])  
        # print (np.unique(pred))  
        pred = pred.reshape((image_size,image_size)).astype(np.uint8)
        mask_whole[0:h,0:w] = pred[:,:]
        color_image = color_annotation(mask_whole[0:h,0:w])
        # print(np.unique(color_image))
        filename = name + '_pre.png'
        # print('图片保存为{}'.format(filename))
        cv2.imwrite(output_path+filename,color_image)

if __name__ == '__main__':
    # dir = './model/xxx.h5'
    # output_path='./xxx/xxx/'
    dir = './model/xxx.h5'
    output_path='./xxx/xxx/'
    if not os.path.exists(output_path): os.mkdir(output_path)
    start_time = time.clock()
    predict(dir = dir,output_path=output_path)
    end_time = time.clock()
    log_time = "训练总时间: " + str(end_time - start_time) + "s"
    print(log_time)
   
