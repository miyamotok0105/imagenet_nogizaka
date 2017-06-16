import os, csv, time
import glob

listFile = 'image_list.txt'

f = open(listFile, 'w')
for filename in glob.glob("/home/ubuntu/hogehoge/*.png"):
    print(filename)
    f.write(filename+" 0"+"\n")
f.close()
