# Mask R-CNN for crack detection in civil infrastructure
This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras 2.0.x, and TensorFlow 1.14.x based on the [matterpot](https://github.com/matterport/Mask_RCNN) repository for crack damage detection and segmentation. The model generates bounding boxes and segmentation masks for each instance of a crack in the image.

The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101
* Instruction for training on your own dataset
* Pre-trained weights for MS COCO and ImageNet
* Jupyter notebooks to visualize the detection result
* Demo file for running prediction on your own dataset (crack here in example)

# Getting Started
The pre-trained weights from MS COCO and ImageNet are provided to fine-tune the new dataset and for starters, begin with this blog [post](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) about the balloon color splash sample. It covers the process starting from annotating images to training to using the results in a sample application.

* [demo.ipynb](https://github.com/rakehsaleem/Custom_Mask_RCNN/blob/master/crack/demo.ipynb): This is the easiest way to start and it shows an example of using a model pre-trained on crack dataset to segment your own images. It includes code to run crack detection and instance segmentation on arbitrary images.
* [(model.py, utils.py, config.py)](https://github.com/rakehsaleem/Custom_Mask_RCNN/tree/master/mrcnn): These files contain the overall Mask RCNN implementation.

* [crack.py](https://github.com/rakehsaleem/Custom_Mask_RCNN/blob/master/crack/crack.py): This file contains the main configuration and attributes for training crack instances.

# Training on your own dataset
In summary, to train the model on your own dataset you'll need to extend two classes:

```CrackConfig``` This class contains the default configuration. Subclass it and modify the attributes you need to change.

```CrackDataset``` This class provides a consistent way to work with any dataset. It allows you to use new datasets for training without having to change the code of the model. It also supports loading multiple datasets at the same time, which is useful if the objects you want to detect are not all available in one dataset.
To start training on your dataset, you can run following commands directly from the command line as such:

1. Train a new model starting from pre-trained COCO weights
   ```bash
   python3 crack.py train --dataset=/home/.../mask_rcnn/data/crack/ --weights=coco  
   ```
2. Train a new model starting from pre-trained ImageNet weights
   ```bash
   python3 crack.py train --dataset=/home/.../mask_rcnn/data/crack/ --weights=imagenet
   ```
3. Continue training the last model you trained. This will find the last trained weights in the model directory.
   ```bash
   python3 crack.py train --dataset=/home/.../mask_rcnn/data/crack/ --weights=last
   ```
The code in the ```crack.py``` is set to train for 240K steps (300 epochs of 800) with a batch size of 4. Update the schedule to fit your system needs. 

# Contributing
Contributions to this repository are welcome. Examples of things you can contribute:

* Increase crack dataset
* Accuracy Improvements
* Visualization and examples

# System configurations
Anaconda==4.6.8, Python==3.6.7, TensorFlow==1.14.0, Keras==2.0.8 and other common packages listed in ```requirements.txt```.
* Note keras and tensorflow versions are based of **_CPU_** requirments!

With Tensorflow2.0 releases, earlier versions can not be configured via pip, and therefore, you can simply download the required version from [here](https://pypi.org/project/tensorflow/#history) depending on your OS.

# Installation
1. Clone this repository
2. Create virtual environment for your specicfc project using
   ```bash
   conda create -n [env. name] python=3.x anaconda
   ```

3. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
4. Install tensorflow by running the folowing command (ex. ```tensorflow-1.x.x-cpxx-cpxxm-OS.whl```).
   ```bash
   pip install [whl file path]
   ```
5. Install keras by downloading the keras library from releases page into the project directory and build the file from that directory.
   ```bash
   python setup.py install
   ```
6. Run setup file from the mask r-cnn root directory
   ```bash
   python3 setup.py install
   ```
   File hierarchy as follow to avoid any confusion among two setup files.
   ```bash
   -->Project->
             |-> Mask R-CNN->
             |             |-> setup.py file. 
             |-> Keras ->
                       |-> setup.py file
   ```   
7. You can choose to download ```mask_rcnn_crack.h5``` from the [releases page](https://github.com/rakehsaleem/Custom_Mask_RCNN/releases) and save it in the root directory.
8. The code will automatically download pre-trained COCO weights, but in case it doesn't work, download from [releases page](https://github.com/rakehsaleem/Custom_Mask_RCNN/releases).
