import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=str, default='Rate-control-INR', help='project name')
# Dataset parameters
parser.add_argument('--data_path', type=str, default='../../data/UVG', help='data path for vid')
parser.add_argument('--crop_list', type=str, default='640_1280', help='video crop size',)
parser.add_argument('--resize_list', type=str, default='-1', help='video resize size',)
parser.add_argument('--num_grid', type=int, default=4, help='number of girds at each row',)
parser.add_argument('--num_dim', type=int, default=2, help='number of input dim',)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()