import cv2
import os
import numpy as np
from PIL import Image
import glob

path = "Data_Extracted_4"

for aniname in glob.glob(path + '/*'):
    os.mkdir("Data_Extracted_5" + aniname[16:])
    for scenename in glob.glob(aniname + '/*'):
        os.mkdir("Data_Extracted_5" + scenename[16:])
        for filename in glob.glob(scenename + '/*jpg'):
            im = Image.open(filename)
            im = im.convert('L')
            im.save("Data_Extracted_5" + filename[16:])
