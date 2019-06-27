import numpy as np
import sys
import cv2
import os
import tensorflow as tf
from network import Network
from class_network import ClassNetwork
from data_manager import DataManager
import matplotlib.pyplot as plt
from scipy.interpolate import spline

""" Returns list of metrics for 2 boxes [IoU, Accuracy, Precision, Recall] """
def calculate_pixel_metrics(data, boxes, gt_boxes):
  predict_map = np.zeros((data.HEIGHT, data.WIDTH), dtype = np.int32)
  gt_map = np.zeros((data.HEIGHT, data.WIDTH), dtype = np.int32)
  for box in boxes:
    for i in range(int(box[0]), int(box[2])+1):
      for j in range(int(box[1]), int(box[3])+1):
        predict_map[i,j] = 1
  for box in gt_boxes:
    for i in range(int(box[0]), int(box[2])+1):
      for j in range(int(box[1]), int(box[3])+1):
        gt_map[i,j] = 1
  
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  for i in range(data.WIDTH):
    for j in range(data.HEIGHT):
      if predict_map[i,j] == 1 and gt_map[i,j] == 1:
        TP += 1
      elif predict_map[i,j] != 1 and gt_map[i,j] == 1:
        FN += 1
      elif predict_map[i,j] == 1 and gt_map[i,j] != 1:
        FP += 1
      else:
        TN += 1
        
  metrics = np.zeros(4)
  if TP+FN+FP != 0:
    metrics[0] = TP/(TP+FN+FP)
  metrics[1] = (TP + TN)/(TP + TN + FP + FN)
  if TP+FP != 0:
    metrics[2] = TP/(TP + FP)
  if TP+FN != 0:
    metrics[3] = TP/(TP + FN)

  return metrics
    
def main():

  if len(sys.argv) < 2:
    print('Error: Give an Confidence Score Threshold between [0,100]\nErro: Indique um Limiar de Confiança entre [0,100]')
    exit(0)
  CONFIDENCE_THRES = float(sys.argv[1])
  if CONFIDENCE_THRES > 100 or CONFIDENCE_THRES < 0:
    print('Error: Give an Confidence Score Threshold between [0,100]\nErro: Indique um Limiar de Confiança entre [0,100]')
    exit(0)
    
  data = DataManager()
  data.load_data(path = '../database/CollectionB', data_type = 'test')
  net_graph = tf.Graph()
  class_net_graph = tf.Graph()
  with net_graph.as_default():
    net = Network(data.WIDTH, data.HEIGHT, data.CHANNELS)
    net.restore_model('model')
  with class_net_graph.as_default():
    class_net = ClassNetwork(data.WIDTH, data.HEIGHT, data.CHANNELS)
    class_net.restore_model('model')
  X,Y = data.get_test()
  
  """ Call the network prediction procedure for all images and store the results """
  counter = -1
  pixelwise_file = open('pixel_wise_results.txt', 'w')
  pixel_results = [[], [], [], []]
  for img, landmarks in zip(X, Y):
    counter += 1
    print('Processing image ' + str(counter))
    x = img
    x_pred = data.normalize_img(x)
    output = net.predict(np.reshape(x_pred, (1, 160,160,3)))[0]
    confidence = class_net.predict(np.reshape(x_pred, (1, 160,160,3)))[0]
    label, ear_center, _ = data.get_bounding_box_map(landmarks)
    boxes = []
    gt_boxes = []
    """ Get all boxes """
    for i in range(data.NUM_AREAS):
      for j in range(data.NUM_AREAS):
        
        """ Get ground truth boxes """
        if ear_center[i,j]:
          gt_boxes.append(data.original_box(label[i,j]))
        
        """ Get predicted boxes """
        if confidence[i,j]*100 >= CONFIDENCE_THRES:
          boxes.append((confidence[i,j], data.original_box(output[i,j])))
    
    boxes = data.non_maximal_suppression(boxes)
    
    """ Calculate Pixel-Wise Metrics """
    metrics = calculate_pixel_metrics(data, boxes, gt_boxes)
    for k in range(4):
      pixel_results[k].append(metrics[k])
      
    for box in gt_boxes:
      cv2.rectangle(x, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,255,0), 1)
    for box in boxes:
      cv2.rectangle(x, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)
    cv2.imwrite('test_outputs/out' + str(counter) + '_{:.4f}'.format(metrics[0]) + '.png', x)
  
  pixelwise_file.write('With Confidence Score >= ' + str(CONFIDENCE_THRES) + ':\n')
  pixelwise_file.write('IOU = {:.4f} ± {:.4f}\n'.format(np.mean(np.array(pixel_results[0])), np.std(np.array(pixel_results[0]))))
  pixelwise_file.write('Accuracy = {:.4f} ± {:.4f}\n'.format(np.mean(np.array(pixel_results[1])), np.std(np.array(pixel_results[1]))))
  pixelwise_file.write('Precision = {:.4f} ± {:.4f}\n'.format(np.mean(np.array(pixel_results[2])), np.std(np.array(pixel_results[2]))))
  pixelwise_file.write('Recall = {:.4f} ± {:.4f}\n'.format(np.mean(np.array(pixel_results[3])), np.std(np.array(pixel_results[3]))))
  pixelwise_file.write('------------------------------------------------------------------------------------' + '\n')
  pixelwise_file.close()
        
if __name__== '__main__':
  main() 


