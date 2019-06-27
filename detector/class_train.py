"""
Train Script : Train the model

"""

import numpy as np
import cv2
import os
import tensorflow as tf
from class_network import ClassNetwork
from data_manager import DataManager

""" Variables """
PATH_TO_TRAIN = '../database/CollectionA/train'
PATH_TO_VAL = '../database/CollectionA/test'
NUM_EPOCHS = 6000
    
""" Main """
def main():
  
  data = DataManager()
  
  data.load_data(path = PATH_TO_TRAIN, data_type = 'train')
  data.load_data(path = PATH_TO_VAL, data_type = 'val')
  
  class_net = ClassNetwork(data.WIDTH, data.HEIGHT, data.CHANNELS)
  val_data, val_landmarks = data.get_val()
  val_data = np.vectorize(data.normalize_img)(val_data) 
  
  val_labels = []
  val_objs = []
  val_no_objs = []
  for landmarks in val_landmarks:
    label, obj, no_obj = data.get_bounding_box_map(landmarks)
    val_labels.append(label)
    val_objs.append(obj)
    val_no_objs.append(no_obj)
    
  val_labels = np.array(val_labels)
  val_objs = np.array(val_objs)
  val_no_objs = np.array(val_no_objs)
  best_acc = 0
  
  for epoch in range(1, NUM_EPOCHS+1):
    data.initialize_epoch()
    
    while True:
      X, Y, OBJ, NO_OBJ = data.next_batch()
      if len(X) == 0:
        break
      loss = class_net.train(X, OBJ)
    output = class_net.predict(val_data)
    acc = 0
    tot = 0
    print_one = 0
    for y, y_ in zip(val_objs, output):
      for i in range(10):
        for j in range(10):
          if y_[i,j] >= 0.5 or y[i,j] == 1.:
            tot += 1
          if y[i,j] == 1. and y_[i,j] >= 0.5:
            acc += 1 
    print(acc, tot)
    acc = acc/tot         
    if acc > best_acc:
      best_acc = acc
      class_net.save_model('model')    
    print('Epoch ' + str(epoch) + '\nPrecision = ' + str(acc) + '\nLOSS = ' + str(loss) + '\nBEST = ' + str(best_acc))
    
    class_net.update_board(acc = acc, loss = loss, idx = epoch)
  
  print('train_finished')
  
if __name__== '__main__':
  main()  
