# Mask-rcnn-v2.1-win-tf2
TensorFlow2 Mask R-CNN for windows by using python3.  
This is the branch to compile Mask RCNN on Windows, it is heavily inspired by the great work done [here](https://github.com/matterport/Mask_RCNN/releases). I have not implemented anything new but fixed the implementation for tensorflow2.

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow 2.1.0. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.  

## Requirements
* Python 3.4+
* TensorFlow 1.3+
* Keras 2.0.8+
* Jupyter Notebook
* Numpy, skimage, scipy, Pillow, cython, h5py

## Installation
1. Clone this repository
2. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases). And place it the the samples\balloon dir.
3. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

## How to start
After doing the above preparations, you also need to find the file \Lib\site-packages\keras\callbacks\tensorboard_v2.py in the python, and add  
`import tensorflow.compat.v1 as tf`  
`tf.disable_v2_behavior()`  
in the front of the file. This modification is to run the code by the compative tf1. And dont forget to comment out them when you run other programs.

And having done all these stuff, you can run the .\samples\balloon\balloon.py with the parameters 'train --dataset=../../balloon --weights=coco'

