# -*- coding: utf-8 -*-
import os
import sys
import cv2
import glob
import numpy as np

# args = sys.argv

size = (227,227)
#size = (256,256)
in_dir_path = "/home/ubuntu/hogehoge/bike.jpg"

try:
    img = cv2.imread(in_dir_path)
    print(img.shape)
    height, width = img.shape[:2]
    img = cv2.resize(img, size)
    cv2.imwrite(in_dir_path, img)
    print(img.shape)

    # while True:
    #     cv2.imshow("data",img)
    #     a = cv2.waitKey(10)
    #     if a>0:
    #         if a == 27:
    #             break

except Exception as e:
    print("exception args:", e.args)






