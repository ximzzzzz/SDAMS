from resnet import Resnet
import argparse
from utils import *

def parse_args():
    parser = argparse.ArgumentParser('hi im resnet')
    parser.add_argument('--name', type=str, default='Resnet', help='naming your model')
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--res_n', type=int, default=34,choices=[18,34,50,101,152])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--checkpoint_dir', type=str, default='c:/users/sf-1300f14mg/dki/resnet/ckpt_dir')
    parser.add_argument('--log_dir', type=str, default='c:/users/sf-1300f14mg/dki/resnet/log_dir')
    parser.add_argument('--image_dir', type=str, default='c:/users/sf-1300f14mg/dki/sdams/dataset_20000.npy', help='you should load "*.npy" image dataset')
    parser.add_argument('--label_dir', type=str, default='c:/users/sf-1300f14mg/dki/sdams/dataset_20000_test.npy',
                        help='you should load "*.npy" image label with one-hot-encoded ')

    return check_args(parser.parse_args())


def check_args(args):
    check_folder(args.checkpoint_dir)
    check_folder(args.log_dir)

    #epoch
    try:
        assert args.epoch >=1
    except:
        print('epoch must be larger than 0')

    #batch_size
    try:
        assert args.batch_size >=1
    except:
        print('batch_size must be larger than 0')

    return args

def main():
    args = parse_args()
    if args is None:
        exit()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        cnn = Resnet(sess,args)

        cnn.build_model()
        show_all_variables()

        if args.phase=='train':
            cnn.train()
            print('Train finished! \n')

            cnn.test()
            print('Test finished!')

        if args.phase=='test':
            cnn.test()
            print('Test finished!!')

if __name__=='__main__':
    main()