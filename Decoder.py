#coding:utf-8
import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]

# 封装Decoder
class Decoder:
    """
    A trainable version VGG19.
    """
    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None
        # 权重值
        self.var_dict = {}
        # 区分是训练过程还是测试过程
        self.trainable = trainable
        # 定义dropout大小 防止过拟合
        self.dropout = dropout  # dropout

    def decoder(self, encode, target_layer):
        # zip建立映射关系 relu1:1 .. dict根据zip 生成的tuple 建立字典  获取到targe_layer的层数
        layer_num = dict(zip(['relu1','relu2','relu3','relu4','relu5'],range(1,6)))[target_layer]  # 层数
        var_list=[]                     # 存放权重值
        height = int(encode.shape[1])   # 输入图像的高
        width = int(encode.shape[2])    # 输入图像的宽
        scal = width / height           # 长宽比例
        # 定义解码器的参数表
        decode_arg={
                '5':[('upsample', 28,  56),
                     ('dconv5_1', 512, 512),
                     ('dconv5_2', 512, 512),
                     ('dconv5_3', 512, 512),
                     ('dconv5_4', 512, 512)],

                '4':[('upsample', 56, 112),
                     ('dconv4_1', 512, 256),
                     ('dconv4_2', 256, 256),
                     ('dconv4_3', 256, 256),
                     ('dconv4_4', 256, 256)],

                '3':[('upsample', 112, 224),
                     ('dconv3_1', 256, 128),
                     ('dconv3_2', 128, 128),
                     ('dconv3_3', 128, 128),
                     ('dconv3_4', 128, 128)],

                '2':[('upsample', 224, 448),
                     ('dconv2_1', 128, 64),
                     ('dconv2_2', 64, 64)],
                '1':[('dconv1_1', 64, 64),
                     ('output', 64, 3)]}
        # 输入特征
        decode = encode
        # 反向读取解码器参数表 反卷积过程需要
        for d in reversed(range(1,layer_num+1)):
            for layer in decode_arg[str(d)]:
                if 'up' in layer[0]:
                    # 锁定比例调整图像大小
                    decode = self.upsample(decode, height, scal)
                    height = height*2
                if 'dconv' in layer[0] :
                    # 从解码器的参数表中 解析 输入输出 通道数
                    decode, var_list= self.conv_layer(decode, layer[1], layer[2], layer[0]+'_'+target_layer, var_list) # 卷积
                if 'out' in layer[0] :
                    decode, var_list = self.output_layer(decode, layer[1], layer[2], layer[0]+'_'+target_layer, var_list) # 输出
                    
        return decode, var_list

    # 调整图像大小 为了得到和原图等大的分割图，我们需要上采样 对应 卷积的池化的逆操作
    def upsample(self, bottom, height, scal):
        height = height
        # 等比例放大
        width = int(height * scal)
        # 图像大小翻倍
        new_height = height*2
        new_width = width*2
        # 将输入调整为新的大小，调整方式为 最近邻 还有0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法
        return tf.image.resize_images(bottom, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR) #调整图像大小

    # 输出层参数提取
    def output_layer(self, bottom, in_channels, out_channels, name, var_list):
        with tf.variable_scope(name):
            # 定义卷积核尺寸
            filt_size = 9
            # 获取卷积层相关变量 filt conv_biases
            filt, conv_biases = self.get_conv_var(filt_size, in_channels, out_channels, name)
            # 扩充边界，防止边界效应
            bottom = tf.pad(bottom,[[0,0],[int(filt_size/2), int(filt_size/2)], [int(filt_size/2),int(filt_size/2)],[0,0]],mode= 'REFLECT')
            # filt为卷积核 步长为1 不填充 对bottom进行卷积
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            # 加入偏置项
            bias = tf.nn.bias_add(conv, conv_biases)
            # 保存参数
            var_list.append(filt)
            var_list.append(conv_biases)
            return bias, var_list

    # 卷积层参数提取
    def conv_layer(self, bottom, in_channels, out_channels, name, var_list, trainable=True):
        # 卷积核尺寸
        filt_size = 3
        with tf.variable_scope(name):
            # 获取卷积核  获取 卷积偏置项
            filt, conv_biases = self.get_conv_var(filt_size, in_channels, out_channels, name)
            # 对边缘进行扩充，防止边界效应
            bottom = tf.pad(bottom,[[0,0],[int(filt_size/2),int(filt_size/2)],[int(filt_size/2),int(filt_size/2)],[0,0]],mode= 'REFLECT')
            # 定义卷积函数，对bottom 以filt为卷积核，步长为1 非0填充  进行卷积
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            # 添加偏置项
            bias = tf.nn.bias_add(conv, conv_biases)
            # 非线性激活
            relu = tf.nn.relu(bias)
            # 保存权重值filt
            var_list.append(filt)
            # 保存权重值bias
            var_list.append(conv_biases)
            return relu, var_list

    # 获取卷积层的变量
    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        # 从截断的正态分布中输出随机值。 均值为0.0 标准差为0.001
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_var(self, initial_value, name, idx, var_name):
        # 有预训练结果 且 命名空间再数据字典中
        if self.data_dict is not None and name in self.data_dict:
            # 直接从预训练中提取变量 idx指示 idx = 0为filter ， idx = 1 为 bias
            value = self.data_dict[name][idx]
            print('resore %s weight' % (name))
        else:
            # 否则 初始化value
            value = initial_value

        if self.trainable:
            # 是训练过程，则初始化了变量
            var = tf.Variable(value, name=var_name)
        else:
            # 非训练过程， 锁定提取的变量为常量
            var = tf.constant(value, dtype=tf.float32, name=var_name)
        # 将var 再写入var_dict中， 之后进行保存
        self.var_dict[(name, idx)] = var
        # print var_name, var.get_shape().as_list()
        # 设置断言 防止初始化变量尺寸 与 模型尺寸不一
        assert var.get_shape() == initial_value.get_shape()
        return var

