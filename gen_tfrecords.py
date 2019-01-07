#coding:utf-8
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tqdm import tqdm
from os import listdir
import os

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

# 定义训练集路径
images_path='/Users/sunqibin/Documents/data/数据集/train2017'
# 生成训练集图片列表 glob获取所有匹配的路径
image_dir = listdir(images_path)
images_list=[os.path.join(images_path, j) for j in image_dir]
# 获取训练集 长度
num=len(images_list)
# 定义tfrecoder 文件路径
tfrecords_filename =  'tfrecords/train.tfrecords'
# 定义TFRecord 写入
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# 处理图片数据（进度条形式）
for image_path in tqdm(images_list[:10000]):
    # 加载图片
    img = image.load_img(image_path,target_size=[224,224,3])
    # 图片的 数据形式为 array 且 格式为  unit8 像素0-255
    img = image.img_to_array(img).astype(np.uint8)
    # 转换，方便TFRecord 的 指定属性的转换
    img_raw = img.tostring()
    # 定义数据读取结构，以及将img_raw 转为指定属性
    feature = {'image_raw':_bytes_feature(img_raw)}
    # 创建一个 example protocol buffer
    example=tf.train.Example(features=tf.train.Features(feature=feature))
    # 序列化 转为字符串 并写入磁盘
    writer.write(example.SerializeToString())    
# 关闭写入
writer.close()
print ('record done')

