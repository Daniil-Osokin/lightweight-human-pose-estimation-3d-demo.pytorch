import argparse

import torch
from torch2trt import torch2trt

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def convert_to_trt(net, output_name, height, width):
    net.eval()
    input = torch.randn(1, 3, height, width).cuda()
    net_trt = torch2trt(net, [input])
    torch.save(net_trt.state_dict(), output_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height', type=int, default=256, help='network input height')
    parser.add_argument('--width', type=int, default=448, help='network input width')
    parser.add_argument('--output-name', type=str, default='human-pose-estimation-3d-trt.pth',
                        help='name of output model in TensorRT format')
    args = parser.parse_args()
    print('TensorRT does not support dynamic network input size reshape.\n'
          'Make sure you have set proper network input height, width. If not, there will be no detections.\n'
          'Default values work for a usual video with 16:9 aspect ratio (1280x720, 1920x1080).\n'
          'You can check the network input size with \'print(scaled_img.shape)\' in demo.py')

    net = PoseEstimationWithMobileNet().cuda()
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)

    convert_to_trt(net, args.output_name, args.height, args.width)
