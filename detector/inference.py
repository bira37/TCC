import numpy as np
import cv2
import os
import tensorflow as tf
from network import Network
from class_network import ClassNetwork
from data_manager import DataManager
import sys
  
def run(net, class_net, data, img, CONFIDENCE_THRES):
    
  width = img.shape[1]
  height = img.shape[0]
    
  x = cv2.resize(img, (160,160))
  
  x_pred = data.normalize_img(x)
  predictions = net.predict(np.reshape(x_pred, (1, 160,160,3)))
  confidences = class_net.predict(np.reshape(x_pred, (1, 160, 160, 3)))
  
  boxes = []
  for i in range(data.NUM_AREAS):
    for j in range(data.NUM_AREAS):
      if confidences[0, i, j]*100. >= CONFIDENCE_THRES:  
        boxes.append((confidences[0, i, j], data.original_box(predictions[0, i, j])))
  
  boxes = data.non_maximal_suppression(boxes)
  
  for box in boxes: 
    orig_box = np.copy(box)
    orig_box[0] *= width/160
    orig_box[1] *= height/160
    orig_box[2] *= width/160
    orig_box[3] *= height/160
    cv2.rectangle(img, (int(orig_box[0]), int(orig_box[1])), (int(orig_box[2]), int(orig_box[3])), (0,0,255))
  return img
  
  
def main():
  if len(sys.argv) < 3:
    print('Error: missing path to images and/or confidence threshold\nErro: ausencia do diretorio das imagens e/ou do limiar de confiança')
    exit(0)
  CONFIDENCE_THRES = float(sys.argv[2]) 
  if CONFIDENCE_THRES > 100 or CONFIDENCE_THRES < 0:
    print('Error: Give an Confidence Score Threshold between [0,100]\nErro: Indique um Limiar de Confiança entre [0,100]')
    exit(0)  
     
  data = DataManager()
  net_graph = tf.Graph()
  class_net_graph = tf.Graph()
  with net_graph.as_default():
    net = Network(data.WIDTH, data.HEIGHT, data.CHANNELS)
    net.restore_model('model')
  with class_net_graph.as_default():
    class_net = ClassNetwork(data.WIDTH, data.HEIGHT, data.CHANNELS)
    class_net.restore_model('model')
    
  for rel_path, subdirs, files in os.walk(sys.argv[1]):
    for name in sorted(files):
      if name[-3:] != 'png':
        continue
      full_path = os.path.join(rel_path, name)
      print('processing ' + full_path)
      img = cv2.imread(full_path, cv2.IMREAD_COLOR)
      img = run(net, class_net, data, img, CONFIDENCE_THRES)
      cv2.imwrite(full_path, img)
    
if __name__== '__main__':
  main() 




