import os
import numpy as np
import torch
import argparse
from model import SASNet
import warnings
import random
from PIL import Image
import cv2
import torchvision.transforms as standard_transforms

warnings.filterwarnings('ignore')

# define the GPU id to be used
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_args_parser():
    # define the argparse for the script
    parser = argparse.ArgumentParser('Inference setting', add_help=False)
    parser.add_argument('--model_path', type=str, help='path of pre-trained model')
    parser.add_argument('--image_path', type=str, help='image path of the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--log_para', type=int, default=1000, help='magnify the target density map')
    parser.add_argument('--block_size', type=int, default=32, help='patch size for feature level selection')

    return parser

def predict(args):
    img_path = args.image_path
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])

    # open the image
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = torch.Tensor(img)
    img = img.unsqueeze(0)
    img = img.cuda()

    print(img.shape)

    model = SASNet(args=args).cuda()
    # load the trained model
    model.load_state_dict(torch.load(args.model_path))
    print('successfully load model from', args.model_path)

    with torch.no_grad():
        model.eval()

        # get the predicted density map
        pred_map = model(img)
        pred_map = pred_map.data.cpu().numpy()

        for i_img in range(pred_map.shape[0]):

            pred_cnt = np.sum(pred_map[i_img]) / args.log_para
            print(f'Predice Count:{pred_cnt}')

            img_to_draw = pred_map[i_img].transpose((1, 2, 0)) * 255
            # img_to_draw = cv2.cvtColor(np.array(pred_map[i_img]).transpose((1, 2, 0)) * 255, cv2.COLOR_RGB2BGR)
            img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_GRAY2BGR)

            cv2.putText(img_to_draw, f'pred_cnt: {pred_cnt}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)  # 在图片上写文字
            cv2.imwrite('demo1_predict.jpg', img_to_draw)
            pass

        print("Predict Over")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SASNet inference', parents=[get_args_parser()])
    args = parser.parse_args()
    predict(args)