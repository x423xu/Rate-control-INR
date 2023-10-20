import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=str, default='Rate-control-INR', help='project name')
# Dataset arguments
parser.add_argument('--data_path', type=str, default='../../data/UVG', help='data path for vid')
parser.add_argument('--crop_list', type=str, default='960_1920', help='video crop size',)
parser.add_argument('--resize_list', type=str, default='256_256', help='video resize size',)
parser.add_argument('--num_frame', type=int, default=1, help='number of frames',)
parser.add_argument('--coords_transform', action='store_true', default=False, help='Whetehr to transform coords')
parser.add_argument('--add_ij', action='store_true', default=False, help='Whetehr to add_ij')

# model arguments
parser.add_argument('--num_grid', type=int, default=4, help='number of girds at each row',)
parser.add_argument('--num_dim', type=int, default=2, help='number of input dim',)
parser.add_argument('--num_hidden', type=int, default=3, help='number of hidden layers',)
parser.add_argument('--hidden_width', type=int, default=64, help='width of hidden layers',)
parser.add_argument('--out_channels', type=int, default=3, help='number of out channels',)

# training arguments
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size',)
parser.add_argument('--parallel', action='store_true', default=False, help='Whetehr to parallel')
args = parser.parse_args()

if args.num_dim == 2:
    args.num_frame = 1
    args.batch_size = 1