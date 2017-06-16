# -*- coding: utf-8 -*-
import os
import cv2
import glob
import numpy as np
from PIL import Image, ImageTk
#ubuntu:
#sudo apt-get install python-imaging-tk

for name in glob.glob("/home/ubuntu/hogehoge/*.png"):
    print(name)

    img = Image.open(name)
    if img.mode != "RGB":
        img = img.convert("RGB")
    os.remove(name)
    img.save(name.replace(".png", ".jpg"))





