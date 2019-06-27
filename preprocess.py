""" 
Preprocess Script: Resize the images

"""

import cv2
import numpy as np
import os

""" Variables """
HEIGHT = 160
WIDTH = 160
PATH_TO_RAW_TRAIN = 'database/CollectionA/train'
PATH_TO_RAW_VAL = 'database/CollectionA/test'
PATH_TO_RAW_TEST = 'database/CollectionB/'
     
""" Preprocess all images in all subdirectories from the given path """
def preprocess(input_path):
  for path, subdirs, files in os.walk(input_path):
    for name in sorted(files):
        if name[-3:] != 'png':
          continue  
        img_path = os.path.join(path, name)
        print('processing ' + img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        cv2.imwrite(img_path, img)
                  
""" Main """
def main():
  
  preprocess(input_path = PATH_TO_RAW_TRAIN)
  preprocess(input_path = PATH_TO_RAW_VAL)
  preprocess(input_path = PATH_TO_RAW_TEST)
  
if __name__ == "__main__":
  main()
