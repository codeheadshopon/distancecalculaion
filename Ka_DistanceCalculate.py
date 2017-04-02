from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from os import listdir
import numpy as np
import cv2
from skimage.feature import hog
from skimage import data, color, exposure
from PIL import Image

def calculateDistance(i1, i2):
    return np.sum((i1-i2)**2)


def HOG_FEATURE(path):
    image = cv2.imread(path)
    image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(7, 7),
                        cells_per_block=(1, 1),  visualise=True)

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    return fd


def TakeHandwritenKA():
    Name2 = []

    for filename in listdir("HandwrittenKa"):
        if filename.endswith(".png"):
            Name2.append("HandwrittenKa/" + filename)
        Name2.sort()
    return Name2


def findDistance():
    Names=TakeHandwritenKA()

    HOG_Original=HOG_FEATURE('ka.jpg')

    Index=0
    for i in Names:
        Image_Image.append(np.asarray(Image.open(i)))
        HOG_HandwritenImage=HOG_FEATURE(i)
        Image_Value.append(calculateDistance(HOG_Original,HOG_HandwritenImage))
        Index+=1


Image_Image = [] # Image that is in i'th index
Image_Value = [] # Resultant Distance between original image and i'th index
findDistance()

print(Image_Value[3])
print(Image_Value[1])