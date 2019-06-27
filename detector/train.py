"""
Train Script : Train the model

"""

import numpy as np
import cv2
import os
import tensorflow as tf
from network import Network
from data_manager import DataManager

""" Variables """
PATH_TO_TRAIN = '../database/CollectionA/train'
PATH_TO_VAL = '../database/CollectionA/test'
NUM_EPOCHS = 8000
    
""" Main """
def main():
  
  data = DataManager()
  
  data.load_data(path = PATH_TO_TRAIN, data_type = 'train')
  data.load_data(path = PATH_TO_VAL, data_type = 'val')
  
  net = Network(data.WIDTH, data.HEIGHT, data.CHANNELS)
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
  #exit(0)
  best_iou = 0
  for epoch in range(1, NUM_EPOCHS+1):
    data.initialize_epoch()
    
    while True:
      #print('next batch')
      X, Y, OBJ, NO_OBJ = data.next_batch()
      if len(X) == 0:
        break
      loss = net.train(X, Y, OBJ)
      #print(loss)
    output = net.predict(val_data)
    N = val_data.shape[0]
    iou = 0
    acc = 0
    precision = 0
    recall = 0
    f1score = 0
    for y, no_obj, y_ in zip(val_labels, val_no_objs, output):
      ears = 0
      cur_acc = 0
      cur_iou = 0
      cur_precision = 0
      cur_recall = 0
      cur_f1score = 0
      for i in range(data.NUM_AREAS):
        for j in range(data.NUM_AREAS):
          if no_obj[i,j]:
            continue

          ears+=1
          orig_bb = data.original_box(y[i,j])
          out_bb = data.original_box(y_[i,j,:4])
            
          metrics = data.calculate_metrics(orig_bb, out_bb)
          cur_iou += metrics[0]
          cur_acc += metrics[1]
          cur_precision += metrics[2]
          cur_recall += metrics[3]
          cur_f1score += metrics[4]
          
      acc += cur_acc/ears
      iou += cur_iou/ears
      precision += cur_precision/ears
      recall += cur_recall/ears
      f1score += cur_f1score/ears 
             
    if iou > best_iou:
      best_iou = iou
      net.save_model('model')    
    print('Epoch ' + str(epoch) + '\nIOU = ' + str(iou/N) + '\nACC = ' + str(acc/N) + '\nPrecision = ' + str(precision/N) + '\nRecall = ' + str(recall/N) + '\nF1 Score = ' + str(f1score/N) + '\nLOSS = ' + str(loss) + '\nBEST = ' + str(best_iou/N))
    
    net.update_board(iou = iou/N, acc = acc/N, precision = precision/N, recall = recall/N, f1 = f1score/N, loss = loss, idx = epoch)
  
  print('train_finished')
  print(data.area_counter)
  
if __name__== '__main__':
  main()  
