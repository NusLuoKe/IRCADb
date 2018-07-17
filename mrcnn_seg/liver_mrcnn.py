#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/12 15:03
# @File    : liver_mrcnn.py
# @Author  : NUS_LuoKe

"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet
    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
from mrcnn_seg.config import Config
from mrcnn_seg import model as modellib, utils
from mrcnn_seg import load_dicom
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import json
import datetime
import numpy as np
import skimage.draw
import pydicom

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class LiverConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "liver"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + liver

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Learning rate
    LEARNING_RATE = 1e-2

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Input image resizing
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Image mean(RGB), for gray CT image, copy one channel to three channels
    MEAN_PIXEL = np.array([-534.3, -534.3, -534.3])


############################################################
#  Dataset
############################################################

class LiverDataset(utils.Dataset):
    '''
    构造的数据集需要继承utils.Dataset类，使用load_shapes()方法向外提供加载数据的方法，并需要重写下面的方法：
    load_image()
    load_mask()
    image_reference()
    '''

    def load_livers(self, base_dir, patient_id_list, shuffle=True, filter_liver=True, reserve_some=False,
                    reserve_num=None):
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("livers", 1, "livers")

        # Get image ids from directory names
        # # base_dir = "/home/guest/notebooks/datasets/3Dircadb"
        # # patient_id_list = list(range(1, 19))
        slice_path_list, mask_path_list = load_dicom.get_slice_mask_path(base_dir,
                                                                         patient_id_list=patient_id_list,
                                                                         shuffle=shuffle)
        if filter_liver:
            slice_with_liver, mask_with_liver, _ = load_dicom.filter_useless_data(slice_path_list, mask_path_list,
                                                                                  reserve_some=reserve_some,
                                                                                  reserve_num=reserve_num)
        else:
            slice_with_liver = slice_path_list

        # Add images
        for slice_path in slice_with_liver:
            # patient_id = os.path.split(slice_path)[0].split("/")[3].split(".")[1] # server
            patient_id = os.path.split(slice_path)[0].split("/")[2].split(".")[1]  # workstation
            patient_image_id = os.path.split(slice_path)[1]
            image_id = "p" + patient_id + "_" + patient_image_id  # eg: p1_image_71,
            self.add_image(
                source="livers",
                image_id=image_id,
                path=slice_path)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        image_path = self.image_info[image_id]['path']
        image_file = pydicom.read_file(image_path)
        image = image_file.pixel_array

        # Truncated pixel value
        image[image < -1024] = -1024
        image[image > 1024] = 1024

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_name = os.path.split(info['path'])[1]
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "MASKS_DICOM/liver")

        # Read mask files from .png image
        mask = []
        mask_path = os.path.join(mask_dir, mask_name)
        mask_file = pydicom.read_file(mask_path)
        m = mask_file.pixel_array.astype(np.bool)

        mask.append(m)
        mask = np.stack(mask, axis=-1)
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    config = LiverConfig()

    # create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    print("load model successfully")

    # data set information
    base_dir = "../3Dircadb1"
    train_patient_id_list = list(range(1, 19))
    validation_patient_id_list = [20]

    # training dataset
    dataset_train = LiverDataset()
    dataset_train.load_livers(base_dir=base_dir, patient_id_list=train_patient_id_list, reserve_some=True,
                              reserve_num=10)
    dataset_train.prepare()

    print("Train Image Count: {}".format(len(dataset_train.image_ids)))
    print("Train Class Count: {}".format(dataset_train.num_classes))
    for i, info in enumerate(dataset_train.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    # Validation dataset
    dataset_val = LiverDataset()
    dataset_val.load_livers(base_dir=base_dir, patient_id_list=validation_patient_id_list)
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all',
                augmentation=augmentation)
