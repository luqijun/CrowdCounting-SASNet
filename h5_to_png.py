import cv2
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw


def get_args_parser():
    # define the argparse for the script
    parser = argparse.ArgumentParser('Convert to heatmap setting', add_help=False)
    parser.add_argument('--image_path', type=str, default='datas/part_test/train_data/ground_truth/IMG_1_sigma4.h5',
                        help='root path of the image')
    parser.add_argument('--target_path', type=str, default='', help='target path of the image')
    return parser


def main(args):
    f = h5py.File(args.image_path, 'r')
    dset = f['density']
    data = np.array(dset[:, :])

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, aspect='auto', cmap=plt.cm.viridis)
    ax.text(0, 0, '1000', horizontalalignment='right', verticalalignment='bottom')
    # plt.imshow(data)
    file = 'heatmap.png'
    fig.savefig(args.target_path + file, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image conversion h5 to png heatmap', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)