# Mask R-CNN for crack detection in civil infrastructure
This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow based on the [matterpot](https://github.com/matterport/Mask_RCNN) repository for crack damage detection and segmentation. The model generates bounding boxes and segmentation masks for each instance of a crack in the image.

The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101
* Instruction and training code for crack dataset
* Pre-trained weights for MS COCO and ImageNet
* Jupyter notebooks to visualize the detection result
* Demo file for running prediction on your own dataset
* Example of training on your own dataset, with emphasize on how to build and adapt codes to dataset with multiple classes.

# Getting Started
The pre-trained weights from MS COCO and ImageNet are provided to fine-tune the new dataset and for starters, begin with this blog [post](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) about the balloon color splash sample. It covers the process starting from annotating images to training to using the results in a sample application.

* [demo.ipynb](https://github.com/rakehsaleem/Custom_Mask_RCNN/blob/master/crack/demo.ipynb): This is the easiest way to start and it shows an example of using a model pre-trained on crack dataset to segment your own images. It includes code to run crack detection and instance segmentation on arbitrary images.
* [(model.py, utils.py, config.py)](https://github.com/rakehsaleem/Custom_Mask_RCNN/tree/master/mrcnn): These files contain the main Mask RCNN implementation.

* ~~[inspect_data.ipynb](): This notebook visualizes the different pre-processing steps to prepare the training data~~

* ~~[inspect_model.ipynb](): This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline~~

# Contributing
Contributions to this repository are welcome. Examples of things you can contribute:

* Increasing dataset
* Accuracy Improvements
* Visualization and examples

# Requirements
Python 3.6, TensorFlow >=1.14.x, Keras 2.2.4 and other common packages listed in > requirements.txt.

# Installation
1. Clone this repository

2. Install dependencies

   ```bash
   pip3 install -r requirements.txt
   ```

3. Run setup from the repository root directory

   ```bash
   python3 setup.py install
   ```

4. Use pre-trained Crack weights (mask_rcnn_crack_0081.h5).
