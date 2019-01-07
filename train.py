# coding:utf-8
from argparse import ArgumentParser
from model import WTC_Model

# 定义一系列命令行参数
parser = ArgumentParser()
# 定义 --target_layer   待训练目标层
parser.add_argument('--target_layer', type=str,
                        dest='target_layer', help='target_layer(such as relu5)',
                        metavar='target_layer', required=True)
# 定义 --pretrained_path VGG19预训练模型路径
parser.add_argument('--pretrained_path',type=str,
                        dest='pretrained_path',help='the pretrained vgg19 path',
                        metavar='Pretrained',required = True)
# 定义 --max_iterator  最大迭代次数，训练次数
parser.add_argument('--max_iterator',type=int,
                        dest='max_iterator',help='the max iterator',
                        metavar='MAX',required = True)
# 定义 --checkpoint_path 断点续训文件路径
parser.add_argument('--checkpoint_path',type=str,
                        dest='checkpoint_path',help='checkpoint path',
                        metavar='CheckPoint',required = True)
# 定义 --tfrecord_path  定义图像数据处理文件路径
parser.add_argument('--tfrecord_path',type=str,
                        dest='tfrecord_path',help='tfrecord path',
                        metavar='Tfrecord',required = True)
# 定义 --batch_size   批处理数据大小
parser.add_argument('--batch_size',type=int,
                        dest='batch_size',help='batch_size',
                        metavar='Batch_size',required = True)


def main():
    # 解析函数 根据上面定义结果，进行使用
    #opts = parser.parse_args()
    # 将一系列命令行参数 输入给模型 进行训练使用
    # model = WTC_Model(target_layer = opts.target_layer,
    #                   pretrained_path = opts.pretrained_path,
    #                   max_iterator = opts.max_iterator,
    #                   checkpoint_path = opts.checkpoint_path,
    #                   tfrecord_path = opts.tfrecord_path,
    #                   batch_size = opts.batch_size)
    # 调用训练函数
    model = WTC_Model(target_layer = "relu3",
                      pretrained_path = '/Users/sunqibin/Documents/data/权重模型/vgg19.npy',
                      max_iterator = 10000,
                      checkpoint_path = 'test_model',
                      tfrecord_path = './tfrecords/train.tfrecords',
                      batch_size = 8
                      )
    model.train()
    
if __name__=='__main__' :
    main()