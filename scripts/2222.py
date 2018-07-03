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


import pydicom
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

img_generator = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3
)

# load dicom文件
x_path = "D:/x.dcm"
y_path = "D:/y.dcm"
x_file = pydicom.dcmread(x_path)
y_file = pydicom.dcmread(y_path)

x = x_file.pixel_array
y = y_file.pixel_array
print(x.shape)

x = np.expand_dims(x, axis=2)
y = np.expand_dims(x, axis=2)
print(x.shape)

gen = img_generator.flow(x, y, batch_size=1)

# # 读成array后画原图
# plt.imshow(x, cmap=plt.cm.gray)
# plt.show()

plt.figure()
for i in range(3):
    plt.imshow(x)
    plt.show()
    break