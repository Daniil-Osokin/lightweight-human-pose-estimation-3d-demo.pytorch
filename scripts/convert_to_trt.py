import argparse

import torch
from torch2trt import torch2trt

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def convert_to_trt(net, output_name):
    net.eval()
    input = torch.randn(1, 3, 256, 448).cuda()
    net_trt = torch2trt(net, [input])
    torch.save(net_trt.state_dict(), output_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--output-name', type=str, default='human-pose-estimation-3d-trt.pth',
                        help='name of output model in TensorRT format')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet().cuda()
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)

    convert_to_trt(net, args.output_name)
