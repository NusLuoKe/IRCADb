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

import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import nibabel as nib
import numpy as np
import pydicom
import skimage.draw
import model as modellib, utils
from config import Config

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
    LEARNING_RATE = 1e-3

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Input image resizing
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Image mean(RGB), for gray CT image, copy one channel to three channels
    MEAN_PIXEL = np.array([-157.4, -157.4, -157.4])  # mean for LITS data set


############################################################
#  Dataset
############################################################

class LiverDataset(utils.Dataset):
    '''
    The constructed dataset needs to inherit the utils.Dataset class and override the following methods:
    load_image()
    load_mask()
    image_reference()
    '''

    def load_livers(self, slice_nii_dir):
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("livers", 1, "livers")

        for slice_nii in os.listdir(slice_nii_dir):
            patient_id = os.path.split(slice_nii)[0].split("-")[-1]
            nii_path = os.path.join(slice_nii_dir, slice_nii)
            nii = nib.load(nii_path)
            nii_arr = nii.get_fdata()  # eg: nii_arr.shape = (512, 512, 78), 78 is the number of slices
            patient_slice_num = nii_arr.shape[-1]
            for i in range(patient_slice_num):
                image_id = "p" + patient_id + "_" + i  # eg: p1_71,
                # slice_path: eg: "./volume-1/i"
                # this is not a real path, i means the index in the nii_arr.
                # slice_arr = nii_arr[:, :, i]
                slice_path = os.path.join(nii_path, i)
                self.add_image(
                    source="livers",
                    image_id=image_id,
                    path=slice_path)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        image_path = self.image_info[image_id]['path']
        # TODO
        corres_nii_path = os.path.split(image_path)[0]
        slice_id = os.path.split(image_path)[-1]
        nii = nib.load(corres_nii_path)
        nii_arr = nii.get_fdata()
        image = nii_arr[:, :, slice_id]

        # Truncated pixel value
        image[image < -200] = -200
        image[image > 250] = 250
        image = image - 48

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        mask_path = self.image_info[image_id]['path']  # eg: mask_path = "./volume-1/i"
        patient_id = os.path.split(mask_path)[0].split("/")[-1].split("-")[-1]
        corres_nii_path = os.path.join(os.path.split(os.path.split(mask_path)[0])[0],
                                       "segmentation-{}".format(patient_id))
        mask_id = os.path.split(mask_path)[-1]

        nii = nib.load(corres_nii_path)
        nii_arr = nii.get_fdata()

        mask = []
        m = nii_arr[:, :, mask_id].astype(np.bool)
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
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all',
                augmentation=augmentation)
