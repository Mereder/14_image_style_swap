#coding:utf-8
from __future__ import division, print_function
import tensorflow as tf
import numpy as np
from skimage.io import imsave,imshow,imread
from keras.preprocessing import image
from vgg19 import Vgg19
from Decoder import  Decoder
from wct import wct_tf, wct_style_swap
import time
from PIL import Image
import os

# 封装训练模型
class WTC_Model:
    # 模型初始化 
    def __init__(self, target_layer=None, pretrained_path=None, max_iterator=None, checkpoint_path=None, tfrecord_path=None, batch_size=None):
        self.pretrained_path = pretrained_path  #预处理模型存放路径
        self.target_layer = target_layer    #目标层名称
        # 加载vgg19 模型，调用Vgg19 
        self.encoder = Vgg19(self.pretrained_path)  
        self.max_iterator = max_iterator    #训练最大次数
        self.checkpoint_path = checkpoint_path  #decoder存放路径
        self.tfrecord_path = tfrecord_path  #tfrecord存放路径
        self.batch_size = batch_size #训练batch大小
        # 预设训练的字典， {名称：ckpt文件名}
        self.save_model_dir = {
            'relu1':'decoder_1.ckpt',
            'relu2': 'decoder_2.ckpt',
            'relu3': 'decoder_3.ckpt',
            'relu4': 'decoder_4.ckpt',
            'relu5': 'decoder_5.ckpt',
        }
    # 定义  编码（特征提取）与 解码（反卷积生成图片）
    # 也是前向传播过程
    def encoder_decoder(self,inputs):
        encoded = self.encoder.encoder(inputs,self.target_layer)   #特征提取
        model=Decoder()   #生成图像model
        decoded,_ = model.decoder(encoded,self.target_layer)   #生成图像
        # 再对生成图像，进行特征提取
        decoded_encoded= self.encoder.encoder(decoded,self.target_layer)   
        # 返回值为 a.提取的特征 b.由特征生成的图片 c.由生成图片提取的特征
        return encoded,decoded,decoded_encoded
    
    # 定义反向传播
    def train(self):
        # 定义大小为【None,224,224,3】大小的，类型为float32 的 占位符常量 inputs（后面需要喂进去数据）
        inputs = tf.placeholder('float',[None,224,224,3])
        # 定义大小为【None,224,224,3】大小的，类型为float32 的 占位符常量 outputs（后面需要喂进去数据）
        outputs = tf.placeholder('float',[None,224,224,3])
        # 定义训练次数，且初始化变量
        global_step = tf.get_variable(name="global_step", initializer=0)
        # 通过前向传播的结果 获得 三个返回值
        encoded,decoded,decoded_encoded = self.encoder_decoder(inputs)
        # 定义像素损失 原图和生成图像之间的loss  均方误差
        pixel_loss = tf.losses.mean_squared_error(decoded,outputs) 
        # 定义特征损失 原图的feature和生成图像的feature的loss  均方误差
        feature_loss = tf.losses.mean_squared_error(decoded_encoded,encoded)
        # 定义总的 loss
        loss = pixel_loss+ feature_loss
        # 定义优化方法  Adam 自适应矩估计  学习率=0.0001  计算总的loss最小值
        opt= tf.train.AdamOptimizer(0.0001).minimize(loss) 

        # tfrecord 数据文件输入路径
        tfrecords_filename =  self.tfrecord_path
        # string_input_producer函数接收一个文件名列表，并自动返回一个对应的文件名队列filename_queue
        # 方便后续的多线程读数据
        filename_queue = tf.train.string_input_producer([tfrecords_filename],num_epochs=100)
        # 实例化tf.TFRecordReader()类生成reader对象，接收filename_queue参数，
        reader = tf.TFRecordReader()  
        # 读取队列中 文件名对应的文件，得到serialized_example(读到的就是.tfrecords序列化文件) 
        _, serialized_example = reader.read(filename_queue)
        # 定义读取的数据结构，然后解析读取的样例
        feature2 = {  
                    'image_raw': tf.FixedLenFeature([], tf.string)} 
        # tf.parse_single_example函数，按照feature2的结构从serialized_example中解析出一条数据
        features = tf.parse_single_example(serialized_example, features=feature2)
        # 将读取数据还原为像素数组
        image = tf.decode_raw(features['image_raw'], tf.uint8) 
        # 重新生成图片格式 224*224*3
        image = tf.reshape(image,[224,224,3])   
        # 批数据读取
        images = tf.train.shuffle_batch([image], batch_size=self.batch_size, capacity=30, min_after_dequeue=10)  #训练数据
        # 指定GPU使用的是按需分配的：
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # tf.Session(config = config)
        # 初始化会话，并开始训练过程。
        with tf.Session()as sess:
            # 变量初始化
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            # tf.train.Coordinator()来创建一个线程管理器
            coord = tf.train.Coordinator()
            # 启动多线程训练模型
            threads = tf.train.start_queue_runners(coord=coord)  
            # 定义保存模型对象
            saver = tf.train.Saver()
            # 模型保存路径
            save_model = self.save_model_dir[self.target_layer]
            # 断点续训  检测是否存在模型，若存在则恢复 否则 未找到
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Not Found, train From 0 step")
            # 训练的迭代过程
            for i in range (self.max_iterator):
                # 获取 输入文件
                batch_x=sess.run(images)
                # 喂数据  输入输出为同一幅图片
                feed_dict = {inputs:batch_x, outputs : batch_x}
                # 计算各个损失loss 以及生成图片
                _,p_loss,f_loss,reconstruct_imgs=sess.run([opt,pixel_loss,feature_loss,decoded],feed_dict=feed_dict)
                # 输出
                print('step %d |  pixel_loss is %f   | feature_loss is %f  |'%(i,p_loss,f_loss))
                # 生成图片 设置像素数组 并进行修剪 限定 0-255 
                if i % 5 ==0:
                    result_img = np.clip(reconstruct_imgs[0],0,255).astype(np.uint8)
                    # 保存图片
                    imsave('result.jpg',result_img)
                    # 保存模型
                    saver.save(sess,os.path.join(self.checkpoint_path, save_model), global_step=global_step)
                # 训练步数+1
                global_step = tf.add(global_step, 1)
            # 保存训练模型
            saver.save(sess, os.path.join(self.checkpoint_path, save_model), global_step=global_step)
            # 线程管理器请求停止，并关闭线程
            coord.request_stop()
            coord.join(threads)

class WCT_test_single_layer:
    def __init__(self,target_layer,content_path,style_path,alpha,pretrained_vgg,output_path,decoder_weights) :
        self.target_layer = target_layer
        self.content_path = content_path
        self.style_path = style_path
        self.output_path = output_path
        self.alpha = alpha
        self.encoder = Vgg19(pretrained_vgg)
        self.decoder = Decoder()  
        self.decoder_weights = decoder_weights
    def test(self):
        content = tf.placeholder('float',[1,304,304,3])
        style = tf.placeholder('float',[1,304,304,3])
        
        content_encode = self.encoder.encoder(content,self.target_layer)
        style_encode = self.encoder.encoder(style,self.target_layer)

        #blended = wct_tf(content_encode,style_encode,self.alpha)
        #blended = Adain(content_encode,style_encode)
        #blended = style_swap(content_encode, style_encode, 3, 2)
        blended = wct_style_swap(content_encode, style_encode, self.alpha)
        
        stylized = self.decoder.decoder(blended,self.target_layer)
        saver = tf.train.Saver()
        
        with tf.Session()as sess:
             tf.global_variables_initializer().run()
             tf.local_variables_initializer().run()
             saver.restore(sess,self.decoder_weights)
             img_c = image.load_img(self.content_path,target_size=(304,304,3))
             img_c = image.img_to_array(img_c)
             img_c = np.expand_dims(img_c,axis=0)

             img_s = image.load_img(self.style_path,target_size = (304,304,3))
             img_s = image.img_to_array(img_s)
             img_s = np.expand_dims(img_s,axis=0)
             
             feed_dict = {content : img_c , style : img_s}
             
             result,e = sess.run([stylized,content_encode],feed_dict= feed_dict)
             result = result[0][0]
             result = np.clip(result,0,255)/255.
             print(result.shape)
             imsave(self.output_path,result)


# 封装测试模型
class WCT_test_all_layer:
    # 测试模型初始化，导入命令行参数
    def __init__(self,content_path,style_path,alpha_wct, alpha_swap,pretrained_vgg,output_path) :
        self.content_path = content_path       # 内容图片路径
        self.style_path = style_path           # 风格图片路径
        self.output_path = output_path         # 融合后图片输出路径
        self.alpha_wct = alpha_wct                     # wct方法内容与风格图片比重
        self.alpha_swap = alpha_swap            #wct_swap方法内容与风格图片比重
        self.encoder = Vgg19(pretrained_vgg)   # 导入VGG19 模型
        self.decoder = Decoder()               # 导入Decoder 反卷积
        # 导入Decoder模型参数
        self.decoder_weights = ['models/decoder_1.ckpt','models/decoder_2.ckpt','models/decoder_3.ckpt','models/decoder_4.ckpt']
    def test(self):
        # 加载content图像,并转成数组,同时进行扩维度（-，h,w,c）
        img_c = Image.open(self.content_path)
        img_c = np.array(img_c)
        img_c = np.expand_dims(img_c, axis=0)

        # 加载style图像,并转成数组,同时进行扩维度（-，h,w,c）
        img_s = Image.open(self.style_path)
        img_s = np.array(img_s)
        img_s = np.expand_dims(img_s, axis=0)
        # 定义尺寸为  内容图片大小的  占位符变量
        content = tf.placeholder('float',shape=img_c.shape)
        # 定义尺寸为  风格图片大小的  占位符变量
        style = tf.placeholder('float',shape=img_s.shape)

        wct_tf_alpha = self.alpha_wct  #wct方法比重
        wct_swap_alpha = self.alpha_swap    #swap方法比重

        # 加入relu4_1的style特征，使用wct方法加入
        content_encode_4 = self.encoder.encoder(content,'relu4')            # relu4_1的content特征
        style_encode_4 = self.encoder.encoder(style,'relu4')                # relu4_1的style特征
        blended_4 = wct_tf(content_encode_4,style_encode_4,wct_tf_alpha)    # wct方法生成融合的特征图
        stylized_4 ,var_list4= self.decoder.decoder(blended_4,'relu4')      # 还原成图片，同时记录变量

        #加入relu3_1的style特征，使用wct_swap方法加入
        content_encode_3 = self.encoder.encoder(stylized_4,'relu3' )                # relu3_1的content特征
        style_encode_3 = self.encoder.encoder(style,'relu3')                        # relu3_1的style特征
        blended_3 = wct_style_swap(content_encode_3, style_encode_3, wct_swap_alpha)# wct_style_swap方法生成融合的特征图
        stylized_3 ,var_list3= self.decoder.decoder(blended_3,'relu3')              # 还原成图片，同时记录变量

        # 加入relu2_1的style特征，使用wct方法加入
        content_encode_2 = self.encoder.encoder(stylized_3,'relu2')         # relu2_1的content特征
        style_encode_2 = self.encoder.encoder(style,'relu2')                # relu2_1的style特征
        blended_2 = wct_tf(content_encode_2,style_encode_2,wct_tf_alpha)    # wct方法生成融合的特征图
        stylized_2 ,var_list2= self.decoder.decoder(blended_2,'relu2')      # 还原成图片，同时记录变量

        # 加入relu1_1的style特征，使用wct方法加入
        content_encode_1 = self.encoder.encoder(stylized_2,'relu1')         # relu2_1的content特征
        style_encode_1 = self.encoder.encoder(style,'relu1')                # relu2_1的style特征
        blended_1 = wct_tf(content_encode_1,style_encode_1,wct_tf_alpha)    # wct方法生成融合的特征图
        stylized_1,var_list1 = self.decoder.decoder(blended_1,'relu1')      # 还原成图片，同时记录变量
        # 保存模型
        saver1 = tf.train.Saver(var_list1)
        saver2 = tf.train.Saver(var_list2)
        saver3 = tf.train.Saver(var_list3)
        saver4 = tf.train.Saver(var_list4)
        # 初始化会话，并开始测试过程
        with tf.Session()as sess:
            # 变量初始化
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            # 从训练好的模型中加载 参数
            saver1.restore(sess, self.decoder_weights[0])   # 加载decoder_1的权重
            saver2.restore(sess, self.decoder_weights[1])   # 加载decoder_2的权重
            saver3.restore(sess, self.decoder_weights[2])   # 加载decoder_3的权重
            saver4.restore(sess, self.decoder_weights[3])   # 加载decoder_4的权重
            # 分别喂入内容图片和风格图片
            feed_dict = {content: img_c, style: img_s}
            # 产生结果
            result = sess.run(stylized_1, feed_dict= feed_dict)
            # 图片输出前的处理
            result = result[0]
            # 数据处理使用的  float 变到0-1之间，正确表达图片信息
            result = np.clip(result, 0, 255) / 255.
            # 存储图片
            imsave(self.output_path, result)

#时间处理
def hms(seconds):
    seconds = int(seconds)
    hours = (seconds // (60 * 60))
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    if hours > 0:
        return '%d hr %d min' % (hours, minutes)
    elif minutes > 0:
        return '%d min %d sec' % (minutes, seconds)
    else:
        return '%d sec' % seconds

if __name__ == '__main__':
    # model = WCT_test_single_layer(
    #     target_layer='relu3',
    #     content_path= './content/im5.jpg',
    #     style_path='./style/im6.jpg',
    #     alpha=1,
    #     pretrained_vgg = '/Users/sunqibin/Documents/data/权重模型/vgg19.npy',
    #     output_path = './output.jpg',
    #     decoder_weights = './models/decoder_3.ckpt'
    # )

    # model = WCT_test_all_layer(
    #         content_path= './content/im5.jpg',
    #         style_path='./style/im6.jpg',
    #         alpha=1,
    #         pretrained_vgg = '/Users/sunqibin/Documents/data/权重模型/vgg19.npy',
    #         output_path = './output.jpg',
    # )

    #model.test()
    start_time = time.time()
    model = WCT_test_all_layer(pretrained_vgg='/Users/sunqibin/Documents/data/权重模型/vgg19.npy',
                               content_path='./content/im3.jpg',
                               style_path='./style/s3.jpg',
                               output_path='./result/show/im3_s3.jpg',
                               alpha_wct=0.6,
                               alpha_swap=0.4)
    model.test()
    use_sende = time.time() - start_time
    print("用时%s" %(hms(use_sende)))

