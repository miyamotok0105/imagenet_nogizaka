# -*- coding: utf-8 -*-                                                                                                                                     
import os
import cv2
import sys
import glob
import numpy as np
from PIL import Image


dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nogizaka/akimoto/*.png")
print(dir_path)

for name in glob.glob(dir_path):
    print(name)
    image = np.asarray(Image.open(name))
    print(Image.open(name))
    print(image.shape)
#    break
