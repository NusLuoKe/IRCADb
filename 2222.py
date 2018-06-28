#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/25 10:24
# @File    : 2222.py
# @Author  : NUS_LuoKe


import numpy as np

a = list(range(27))
sum = 0
for i in a:
    sum += i
print(sum)
aa = np.asarray(a)
b = aa.reshape((3, 3, 3))
print(b)
print(np.sum(b))
