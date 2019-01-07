#coding:utf-8
import os
import tensorflow as tf
import numpy as np
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]
# 封装Vgg19模型及操作
class Vgg19:
    # 初始化 加载预训练模型 vgg19.npy
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")
    # 编码器（特征提取）
    def encoder(self, inputs, target_layer):
        # 构建指定层与数字的对应关系
        layer_num =dict(zip(['relu1', 'relu2', 'relu3', 'relu4', 'relu5'],range(1, 6)))[target_layer]
        encode = inputs
        # 定义 编码器参数表
        encoder_arg={
                '1': [('conv1_1', 64),
                      ('conv1_2', 64),
                      ('pool1', 64)],
                '2': [('conv2_1', 128),
                      ('conv2_2', 128),
                      ('pool2', 128)],
                '3': [('conv3_1', 256),
                      ('conv3_2', 256),
                      ('conv3_3', ),
                      ('conv3_4', 256),
                      ('pool3', 256)],
                '4': [('conv4_1', 512),
                      ('conv4_2', 512),
                      ('conv4_3', 512),
                      ('conv4_4', 512),
                      ('pool4', 512)],
                '5': [('conv5_1', 512),
                      ('conv5_2', 512),
                      ('conv5_3', 512),
                      ('conv5_4', 512)]}
        # 根据需要提取的指定层特征，将输入 依次进行 该层之前的所有变换，获得该层特征
        for d in range(1, layer_num+1):
            for layer in encoder_arg[str(d)]:
                # 如果是卷积层，进行卷积操作
                if 'conv' in layer[0] :
                    encode =self.conv_layer(encode, layer[0])
                # 如果是池化层，进行池化操作
                if 'pool' in layer[0] and d < layer_num:
                    encode = self.max_pool(encode, layer[0])
        return encode

    # 对输入进行池化操作
    def max_pool(self, bottom, name):
        # 对输入进行池化操作，尺寸为2 步长为 2  全0填充
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # 对输入进行卷积操作
    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            # 从模型中取卷积核参数
            filt = self.get_conv_filter(name)
            # 定义卷积核尺寸
            filt_size = 3
            # 对输入图片进行边缘填充，消除边界效应
            bottom = tf.pad(bottom,[[0,0],[int(filt_size/2),int(filt_size/2)],[int(filt_size/2),int(filt_size/2)],[0,0]],mode= 'REFLECT')
            # 对bottom 以 filt为卷积核进行卷积
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            # 预训练模型中读取偏置项参数
            conv_biases = self.get_bias(name)
            # 引入偏置
            bias = tf.nn.bias_add(conv, conv_biases)
            # 进行 relu 非线性变换
            relu = tf.nn.relu(bias)
            return relu

    # 从模型中获取 卷积核的值
    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    # 从模型中获取 偏置项的值
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    # 从模型中获取 权重的值
    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
