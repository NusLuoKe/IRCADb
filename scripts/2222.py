# import dicom as dicom
# import numpy as np
# import os
#
# img = np.zeros([512, 512])
#
#
# def binaryzation(ds, label):
#     arr = ds.pixel_array
#     arr = np.uint(arr > 0) * label
#     return arr
#
#
# patient_list = [1, 4, 5, 6, 7]
# [1, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 19, ]
#
# for i in range(300):
#     try:
#         img = np.zeros([512, 512])
#         portalvein = binaryzation(dicom.read_file("D:/3Dircadb1/3Dircadb1.1/MASKS_DICOM/portalvein/image_" + str(i)),
#                                   1)
#         venoussystem = binaryzation(
#             dicom.read_file("D:/3Dircadb1/3Dircadb1.1/MASKS_DICOM/venoussystem/image_" + str(i)), 2)
#
#         image = dicom.read_file("D:/3Dircadb1/3Dircadb1.1/MASKS_DICOM/portalvein/image_" + str(i))
#         img[portalvein == 1] = 1
#         img[venoussystem == 2] = 2
#         image.pixel_array.flat = np.int16(img)
#         image.PixelData = image.pixel_array.tostring()
#         save_dir = "D:/3Dircadb1/3Dircadb1.1/MASKS_DICOM/vessel_sum"
#         if not os.path.isdir(save_dir):
#             os.mkdir(save_dir)
#         image.save_as("D:/3Dircadb1/3Dircadb1.1/MASKS_DICOM/vessel_sum/image_" + str(i))
#     except:
#         continue
#
# print("done")

#
# import pydicom
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator
#
# # load dicom文件
# x_path = "F:/IRCAD/3Dircadb1/3Dircadb1.10/PATIENT_DICOM/image_49"
# # image_result = pydicom.read_file(x_path)
# image_result = pydicom.read_file(x_path)
# print(np.max(image_result.pixel_array))
# print(np.min(image_result.pixel_array))
#
# a = ["123456"]
# print(a[-1][::-1])
# print("".join(a[0].reverse()))

# from skimage import io
# from skimage.transform import resize
# import matplotlib.pyplot as plt
#
# image_path = "D:/hua.jpg"
# img_arr = io.imread(image_path)
# re_img_arr = resize(img_arr, (100, 200))
# plt.imshow(re_img_arr)
# plt.show()
# import pydicom
# from skimage.measure import regionprops
#
# test_slice_path = "F:/IRCAD/3Dircadb1/3Dircadb1.1/MASKS_DICOM/portalvein/image_116"
# slice = pydicom.read_file(test_slice_path)
# slice_array = slice.pixel_array
#
# # preprocessing
# props = regionprops(slice_array)
# print(props[0].bbox)
# import os
#
# ROOT_DIR = os.path.abspath("../../")
# a = "../data"
# print(a)
# print(ROOT_DIR)
#
# ############################################################################
# from unet_seg import load_dicom

# import os
#
# # base_dir = "F:/IRCAD/3Dircadb1/"
# #
# # train_patient_id_list = list(range(1, 10))
# # slice_path_list, mask_path_list = load_dicom.get_slice_mask_path(base_dir,
# #                                                                  patient_id_list=train_patient_id_list,
# #                                                                  shuffle=True)
# # slice_with_liver, mask_with_liver, _ = load_dicom.filter_useless_data(slice_path_list, mask_path_list)
# # # print(slice_with_liver)
# # # slice_path = slice_with_liver[0]
# # #
# # #
#
# # slice_path = info["path"]
# slice_path = "F:/IRCAD/3Dircadb1/3Dircadb1.1/PATIENT_DICOM/image_71"
# patient_id = os.path.split(slice_path)[0].split("/")[3].split(".")[1]
# print(patient_id)
# print(os.path.split(slice_path)[0].split("/"))
# # patient_image_id = os.path.split(slice_path)[1]
# # image_id = "p" + patient_id + "_" + patient_image_id
# # print(image_id)
# # print(os.path.split(slice_path))
# mask_name = os.path.split(slice_path)[1]
# mask_dir = os.path.join(os.path.dirname(os.path.dirname(slice_path)), "MASKS_DICOM/liver")
# mask_path = os.path.join(mask_dir, mask_name)
# print(mask_path)
##############################################################################################################
from unet_seg import load_dicom
import pydicom
import numpy as np


def filter_useless_data(slice_path_list, mask_path_list):
    x_with_mask = []
    y_with_mask = []
    mask_num = 0
    for image_path in mask_path_list:
        image_file = pydicom.dcmread(image_path)
        image_array = image_file.pixel_array
        if np.sum(image_array) > 0:
            mask_num += 1
            idx = mask_path_list.index(image_path)
            y_with_mask.append(image_path)
            x_with_mask.append(slice_path_list[idx])
    return x_with_mask, y_with_mask, mask_num


# base_dir = "/home/guest/notebooks/datasets/3Dircadb"
base_dir = "F:/IRCAD/3Dircadb1"

train_patient_id_list = list(range(1, 21))
train_slice_path_list, train_mask_path_list = load_dicom.get_slice_mask_path(base_dir,
                                                                             patient_id_list=train_patient_id_list,
                                                                             shuffle=True)
train_x_with_vessel, train_y_with_vessel, train_vessel_num = filter_useless_data(train_slice_path_list,
                                                                                 train_mask_path_list)
print(train_vessel_num)

sum_arr = np.int64(0)
for image_path in train_x_with_vessel:
    image_file = pydicom.dcmread(image_path)
    image_array = image_file.pixel_array
    image_array[image_array < -1024] = -1024
    image_array[image_array > 1024] = 1024

    sum_arr += np.sum(image_array)
print(sum_arr)
print(sum_arr/train_vessel_num/512/512)


# image_path = "F:/IRCAD/3Dircadb1/3Dircadb1.4/PATIENT_DICOM/image_1"
# image_file = pydicom.dcmread(image_path)
# image_array = image_file.pixel_array
# print(np.sum(image_array))
