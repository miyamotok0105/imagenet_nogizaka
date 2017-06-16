# -*- coding: utf-8 -*-
import os
import sys
import cv2
import glob
import numpy as np
from PIL import Image

in_dir_path = ""
out_dir_path = ""
size = (256,256)
#size = (227,227)
#size = (1024,512)

dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images/*png")
print(dir_path)

for name in glob.glob(dir_path):
    print(name)

    try:
        img = cv2.imread(name)
        height, width = img.shape[:2]
        img = cv2.resize(img, size)
        cv2.imwrite(name, img)
        #image = Image.open(name).convert('P')
        #image.save(name)
    except Exception as e:
        print("exception args:", e.args)
        
