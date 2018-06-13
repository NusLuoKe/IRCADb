#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 15:29
# @File    : 1.py
# @Author  : NUS_LuoKe

# for batch_x_y in get_batch(slice_path=training_set[0], liver_path=training_set[1], batch_size=4):
#     step += 1


def a():
    for i in range(10):
        yield i


for j in range(5):
    aa = next(a())
    print(aa)
    bb = next(a())
    print(bb)
    cc = next(a())
    print(cc)
