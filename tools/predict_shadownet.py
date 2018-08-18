#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use shadow net to recognize the scene text
"""
import tensorflow as tf
import numpy as np
import os
import operator as op
import  skimage
from skimage import transform
try:
    from cv2 import cv2
except ImportError:
    pass

from recognition_tensorflow.crnn_model import crnn_model
from recognition_tensorflow.global_configuration import config
from recognition_tensorflow.local_utils import log_utils, data_utils

logger = log_utils.init_logger()


def predict_shadownet(image_path,template_string, weights_path='model/shadownet/shadownet_2018-08-11-20-29-49.ckpt-10390'):

    """

    :param image_path:
    :param weights_path:
    :param is_vis:
    :return:
    """
    print("enter in the predict function")
    print("row = ",image_array.shape[0])
    #rows = image_array.shape[0]
    #cols = image_array.shape[1] // 3
    #print("col = ",cols)
    image = cv2.imread(image_path,cv2.IMREAD_COLOR)

    #调整像素的大小
    #image_array = transform.resize(image_array,(32,100))
    #image_array = skimage.img_as_ubyte(image_array)
    #image = np.expand_dims(image_array, axis=0).astype(np.float32)
    #print("after skimage")
    image = cv2.resize(image, (100, 32))
    inputdata = tf. placeholder(dtype=tf.float32, shape=[1, 32, 100,3], name='input')

    net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=38)

    print("after crnn_model")

    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=inputdata)
    print("predict_shadownet")
    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=25*np.ones(1), merge_repeated=False)

    decoder = data_utils.TextFeatureIO()

    # config tf session
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    print("sflsadflsdfjkdsfj")

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        preds = sess.run(decodes, feed_dict={inputdata: image})

        preds = decoder.writer.sparse_tensor_to_str(preds[0])

        logger.info('Predict image label {:s}'.format(preds[0]))

        #if is_vis:
            #plt.figure('CRNN Model Demo')
            #plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
            #plt.show()

        sess.close()

    return op.eq(preds[0],template_string)
