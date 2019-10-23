# Mask_RCNN in Civil Engineering applications for object detection and segmentation
This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow based on the [matterpot](https://github.com/matterport/Mask_RCNN) repository for object detection and instance segmentation. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone. This is a project of [Smart Infrastructure & Technology Lab](http://sitl.cau.ac.kr/) for concrete crack detection and segmentation under the supervision of Prof. Jongwoong Park.

The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101
* Instruction and training code for crack dataset
* Pre-trained weights for MS COCO and ImageNet
* Jupyter notebooks to visualize the detection result
* Demo file for running prediction on your own dataset
* Example of training on your own dataset, with emphasize on how to build and adapt codes to dataset with multiple classes.

# Getting Started
* [demo.ipynb](): This is the easiest way to start and it shows an example of using a model pre-trained on crack dataset to segment your own images. It includes code to run object detection and instance segmentation on arbitrary images.
* [(model.py, utils.py, config.py)](): These files contain the main Mask RCNN implementation files.

* ~~[inspect_data.ipynb](): This notebook visualizes the different pre-processing steps to prepare the training data~~

* ~~[inspect_model.ipynb](): This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline~~

# Contributing
Contributions to this repository are welcome. Examples of things you can contribute:

* Increasing dataset
* Accuracy Improvements
* Visualization and examples

# Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in > requirements.txt.

# Installation
1. Clone this repository

2. Install dependencies

`pip3 install -r requirements.txt`

3. Run setup from the repository root directory

`python3 setup.py install`

4. Use pre-trained Crack weights (mask_rcnn_crack_0081.h5).
