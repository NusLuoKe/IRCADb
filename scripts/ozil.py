#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/5 14:49
# @File    : ozil.py
# @Author  : NUS_LuoKe


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取图片并转为数组
oly_ring = Image.open("D:/1.jpeg").convert("L")
# ozil.show()

oly_ring_arr = np.array(oly_ring)

# plt.hist(oly_ring_arr.flatten(), bins=200, color="c")
# plt.show()

# black
# oly_ring_arr[oly_ring_arr > 50] = 255

oly_ring_arr[oly_ring_arr < 50] = 255
oly_ring_arr[oly_ring_arr > 90] = 255
# oly_ring_arr[oly_ring_arr < 100 or oly_ring_arr > 115] = 255
# oly_ring_arr[oly_ring_arr < 115 or oly_ring_arr > 150] = 255
# oly_ring_arr[oly_ring_arr < 150] = 255

# plt.hist(oly_ring_arr.flatten(), bins=80, color="c")
# plt.show()

im = Image.fromarray(oly_ring_arr)
im.save("D:/black.jpeg")
haha = Image.open("D:/black.jpeg")
haha.show()
