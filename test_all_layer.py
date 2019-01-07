#coding:utf-8
from argparse import ArgumentParser
from model import WCT_test_all_layer


parser = ArgumentParser()


    
parser.add_argument('--pretrained_vgg',type=str,
                        dest='pretrained_vgg',help='the pretrained vgg19 path',
                        metavar='Pretrained',required = True)
parser.add_argument('--content_path',type=str,
                        dest='content_path',help='the content path',
                        metavar='Content',required = True)

parser.add_argument('--style_path',type=str,
                        dest='style_path',help='style path',
                        metavar='Style',required = True)
    
parser.add_argument('--output_path',type=str,
                        dest='output_path',help='output_path',
                        metavar='Output',required = True)
    
parser.add_argument('--alpha_wct',type=float,
                        dest='alpha_wct',help='the blended wct weight',
                        metavar='ALphwct',required = True)

parser.add_argument('--alpha_swap',type=float,
                        dest='alpha_swap',help='the blended swap weight',
                        metavar='ALphaswap',required = True)


def main():
    opts = parser.parse_args()

    model = WCT_test_all_layer(
                     pretrained_vgg = opts.pretrained_vgg,
                     content_path = opts.content_path,
                     style_path = opts.style_path,
                     output_path = opts.output_path,
                     alpha_wct = opts.alpha_wct,
                     alpha_swap= opts.alpha_swap
                     )

    # model = WCT_test_all_layer(pretrained_vgg='/Users/sunqibin/Documents/data/权重模型/vgg19.npy',
    #                            content_path='./content/im3.jpg',
    #                            style_path='./style/s3.jpg',
    #                            output_path='./result/myoutput9.jpg',
    #                            alpha=1)
    model.test()
    
if __name__ == '__main__' :
    main()