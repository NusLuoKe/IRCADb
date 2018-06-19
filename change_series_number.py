from __future__ import print_function

import pydicom

series_number = 1
for folder in ["D:/livertumor01/SERIES_0", "D:/livertumor01/SERIES_1", "D:/livertumor01/SERIES_2"]:
    instance = 0
    for file_name in [folder+'/image_{}.dcm'.format(i) for i in range(129)]:
        print(file_name)
        ds = pydicom.read_file(file_name)
        ds.SeriesNumber = series_number
        ds.InstanceNumber = instance
        print(ds.SeriesNumber, ds.InstanceNumber)
        ds.save_as(file_name)
        instance += 1
    series_number += 1
