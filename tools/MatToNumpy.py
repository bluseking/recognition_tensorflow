import sys
import cv2
import numpy as np
print("-----haha-------")
print(sys.path)
from recognition_tensorflow.tools import predict_shadownet
#import recognition_tensorflow.tools.predict_shadownet
# print("-----haha-------")
# print(sys.path)
# import recognition_tensorflow.tools.predict_shadownet

w=227
h=227
sess = None

def arrayreset(array):
    # for i inrange(array.shape[1]/3):
    #     pass
    # print(array.shape)
    # np.save("~/image/array.npy",array)
    a = array[:,0:len( array[0] -2 ):3]
    b = array[:, 1:len( array[0] - 2 ):3]
    c = array[:, 2:len( array[0] - 2 ):3]
    a = a[:, :, None]
    b = b[:, :, None]
    c = c[:, :, None]
    m = np.concatenate((a,b,c),axis=2)
    return m

# def load_model():
#     global sess
#     sess = tf.Session()
#     saver = tf.train.import_meta_graph( './model/model.ckpt.meta')
#     saver.restore( sess, tf.train.latest_checkpoint('./model/') )


def load_image(image,model_text):
    #image_array = arrayreset(image)
    print("1")
    returns = predict_shadownet.predict_shadownet(image, str(model_text), save_dir='/home/zht/recognition_tensorflow/tfrecords_dir', weights_path='model/shadownet/shadownet_2018-08-11-20-29-49.ckpt-10390', is_recursive=True)
    if returns == True:
        r1=1
    else:
        r1=0
    isinstance(r1,int)
    print("输出结果")
    print(r1)
    return r1


