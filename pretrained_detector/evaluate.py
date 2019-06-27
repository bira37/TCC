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
        
def main():
  IOU_THRES_LIST = [0.0001, 10, 20, 30, 40, 50]
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
  X, Y = data.get_test()
  
  """ Call the network prediction procedure for all images and store the results """
  box_outputs = [] #list of predictions for each image
  confidence_outputs = [] #list of confidence scores for each image
  labels = [] #list of ground truth boxes for each image
  locations = [] #list of ground truth locations of center of an ear for each image
  counter = 0
  for img, landmarks in zip(X, Y):
    print('Processing image ' + str(counter))
    counter += 1
    x = img
    x_pred = data.normalize_img(x)
    output = net.predict(np.reshape(x_pred, (1, 160,160,3)))[0]
    confidences = class_net.predict(np.reshape(x_pred, (1, 160,160,3)))[0]
    y, location, _ = data.get_bounding_box_map(landmarks)
    box_outputs.append(np.copy(output))
    confidence_outputs.append(np.copy(confidences))
    labels.append(np.copy(y))
    locations.append(np.copy(location))    
  
  fdr_fnr_file = open('fdr_fnr_results.txt', 'w')
  fdr_fnr_graph = plt.figure(1)
  for IOU_THRES in IOU_THRES_LIST:
    plot_fdr = []
    plot_fnr = []
    for CONFIDENCE_THRES in range(0, 101, 1):
      TP = 0
      FP = 0
      FN = 0
      counter = 0
      print('CONFIDENCE THRES = ' + str(CONFIDENCE_THRES))
      for output, confidence, label, ear_center in zip(box_outputs, confidence_outputs, labels, locations):
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
          
        """ Calculate TP, FP and FN """
        available = np.ones(len(boxes), dtype = bool)
        FP += len(boxes)
        for gt_box in gt_boxes:
          chosen = -1
          best_iou = -1
          for k in range(len(boxes)):
            if available[k] == False:
              continue
            cur_iou = data.IoU(boxes[k], gt_box)
            if cur_iou*100 >= IOU_THRES and cur_iou > best_iou:
              best_iou = cur_iou
              chosen = k
          
          if chosen == -1:
            FN += 1
          else:
            TP += 1
            FP -= 1
            available[chosen] = False
      
      """ Calculate FDR, FNR and Mean Pixel-Wise Metrics """      
      print(TP, FP, FN)
      FDR = 0
      FNR = 0
      if FP+TP > 0:
        FDR = (FP)/(FP+TP)
      if FN+TP > 0:
        FNR = (FN)/(FN+TP)
      print(FNR, FDR)
      plot_fdr.append(FDR)
      plot_fnr.append(FNR)
      
      """ Write Numeric Results on Files """
      fdr_fnr_file.write('With Confidence Score >= ' + str(CONFIDENCE_THRES) + ' and IOU >= ' + str(IOU_THRES) + ':\n')
      fdr_fnr_file.write('TP = ' + str(TP) + '\n')  
      fdr_fnr_file.write('FP = ' + str(FP) + '\n')
      fdr_fnr_file.write('FN = ' + str(FN) + '\n')
      fdr_fnr_file.write('FNR = {:.4f}'.format(FNR) + '\n')
      fdr_fnr_file.write('FDR = {:.4f}'.format(FDR) + '\n')
      fdr_fnr_file.write('------------------------------------------------------------------------------------' + '\n')
      
    plt.plot(plot_fdr, plot_fnr)
          
  plt.xlabel('Taxa de Falsa Descoberta (FDR)')
  plt.ylabel('Taxa de Falso Negativo (FNR)')  
  plt.legend(['> 0 IOU', '>= 0.1 IOU', '>= 0.2 IOU', '>= 0.3 IOU', '>= 0.4 IOU', '>= 0.5 IOU'], loc='upper right')
  fdr_fnr_graph.savefig('FDRxFNR_Graph' + '.png')
  fdr_fnr_file.close()    
        
if __name__== '__main__':
  main() 


