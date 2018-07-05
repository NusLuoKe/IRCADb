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
