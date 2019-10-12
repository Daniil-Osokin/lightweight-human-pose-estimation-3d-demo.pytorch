# Real-time 3d Multi-person Pose Estimation Demo

This repository contains 3d multi-person pose estimation demo in PyTorch. Intel OpenVINO&trade; backend can be used for fast inference on CPU. This demo is based on [Lightweight OpenPose](https://arxiv.org/pdf/1811.12004.pdf) and [Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB](https://arxiv.org/pdf/1712.03453.pdf) papers. It detects 2d coordinates of up to 18 types of keypoints: ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees, and ankles, as well as their 3d coordinates. It was trained on [MS COCO](http://cocodataset.org/#home) and [CMU Panoptic](http://domedb.perception.cs.cmu.edu/) datasets and achieves 100 mm MPJPE (mean per joint position error) on CMU Panoptic subset. *This repository significantly overlaps with https://github.com/opencv/open_model_zoo/tree/master/demos/python_demos/human_pose_estimation_3d_demo, however contains just the necessary code for 3d human pose estimation demo.*

<p align="center">
  <img src="data/human_pose_estimation_3d_demo.jpg" />
</p>

## Table of Contents

* [Requirements](#requirements)
* [Prerequisites](#prerequisites)
* [Pre-trained model](#pre-trained-model)
* [Running](#running)
* [Inference with OpenVINO](#inference-openvino)

## Requirements
* Python 3.5 (or above)
* CMake 3.10 (or above)
* C++ Compiler (g++ or MSVC)
* OpenCV 4.0 (or above)

> [Optional] [Intel OpenVINO](https://software.intel.com/en-us/openvino-toolkit) for fast inference on CPU.

## Prerequisites
1. Install requirements:
`pip install -r requirements.txt`

2. Build `pose_extractor` module. To build `pose_extractor` module, please run in command line:
`python setup.py build_ext`

3. Then add build folder to `PYTHONPATH`:
`export PYTHONPATH=pose_extractor/build/:$PYTHONPATH`

## Pre-trained model <a name="pre-trained-model"/>

Pre-trained model is available at [Google Drive](https://drive.google.com/file/d/1niBUbUecPhKt3GyeDNukobL4OQ3jqssH/view?usp=sharing).

## Running

To run the demo, pass path to the pre-trained checkpoint and camera id (or path to video file):
`python demo.py --model human-pose-estimation-3d.pth --video 0`

> Camera can capture scene under different view angles, so for correct scene visualization, please pass camera extrinsics and focal length with `--extrinsics` and `--fx` options correspondingly (extrinsics sample format can be found in data folder). In case no camera parameters provided, demo will use the default ones.

## Inference with OpenVINO <a name="inference-openvino"/>

To run with OpenVINO, it is necessary to convert checkpoint to OpenVINO format:
1. Set OpenVINO environment variables:
    * `source <OpenVINO_INSTALL_DIR>/bin/setupvars.sh`
2. Convert checkpoint to ONNX:
    * `python scripts/convert_to_onnx.py --checkpoint-path human-pose-estimation-3d.pth`
3. Convert to OpenVINO format:
    * `python <OpenVINO_INSTALL_DIR>/deployment_tools/model_optimizer/mo.py --input_model human-pose-estimation-3d.onnx --input=data --mean_values=data[128.0,128.0,128.0] --scale_values=data[255.0,255.0,255.0] --output=features,heatmaps,pafs`

To run the demo with OpenVINO inference, pass `--use-openvino` option and specify device to infer on:

* `python demo.py --model human-pose-estimation-3d.xml --device CPU --use-openvino --video 0`
