import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        #basic parameters
        parser.add_argument('--dataroot', required=True, help='path to video images')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        #parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: 0; 1; 0,1')
        parser.add_argument('--classes', type=int, default=101, help='# classes of the dataset')

        #model parameters
        #TODO

        self.initialized=True
        return parser

    def gather_options(self):
        """initialize the parser with basic options"""
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------Options---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}:{:<30}{}\n'.format(str(k), str(v), comment)

        message += '----------------End---------------------\n'

        #save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        self.print_options(opt)

        #str_ids = opt.gpu_ids.split(',')
        #opt.gpu_ids = []
        #for str_id in str_ids:
        #    id = int(str_id)
        #    if id >= 0:
        #        opt.gpu_ids.append(id)
        #if len(opt.gpu_ids) > 0:
        #    torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt

        return self.opt


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='freqeuncy of saving checkpoints of models')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves the model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training, load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test')

        #training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--train_id', type=int, default=1, help='The txt index for loading train videos')
        parser.add_argument('--batch_size', type=int, default=40, help='Batch size')
        parser.add_argumanet('--log_interval', type=int, default=10, help='frequency of displaying the log information')

        self.isTrain = True
        return parser


class TestOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--test_id', type=int, default=1, help='The txt index for loading test videos')
        parser.add_argument('--results_dir', type=str, default='./results', help='Save results here')
        parser.add_argument('--phase', type=str, default='test', help='train, val ,test')

        self.isTrain = False
        return parser
