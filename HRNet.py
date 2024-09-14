import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Multiply
from tensorflow.keras.layers import UpSampling2D, Add, concatenate, Concatenate, DepthwiseConv2D, Lambda, Reshape
from attention import PAM, CAM, CFAM
from tensorflow import linalg as lg

def seg_hrnet(height, width, channel, classes):
    inputs = Input((height, width, channel))

    x = stem_net(inputs)

    x = transition_layer1(x)
    x0 = make_branch1_0(x[0])
    x1 = make_branch1_1(x[1])
    x = fuse_layer1([x0, x1])

    x = transition_layer2(x)
    x0 = make_branch2_0(x[0])
    x1 = make_branch2_1(x[1])
    x2 = make_branch2_2(x[2])
    x = fuse_layer2([x0, x1, x2])

    x = transition_layer3(x)
    x0 = make_branch3_0(x[0])
    x1 = make_branch3_1(x[1])
    x2 = make_branch3_2(x[2])
    x3 = make_branch3_3(x[3])
    x = fuse_layer3([x0, x1, x2, x3])

    out = final_layer(x, classes=classes)
    out = Reshape((height * width, classes))(out)

    model = Model(inputs=inputs, outputs=out)

    return model
  
if __name__ == '__main__':
    a = seg_hrnet(256, 256, 3, 3)
    a.summary()
