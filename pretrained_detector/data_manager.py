""" 
Data Manager : Manage the data used in network training/test
""" 

import numpy as np
import cv2
import os
import math
import imgaug as ia #imgaug augmenter
from imgaug import augmenters as iaa #imgaug augmenter
import random

class DataManager:
  
  """ Constructor """
  def __init__(self):
  
    """ Variables """
    self.HEIGHT = 160
    self.WIDTH = 160
    self.AREA_SIZE = 16
    self.NUM_AREAS = 10
    self.NUM_LANDMARKS = 4
    self.CHANNELS = 3
    self.train_imgs = np.empty(0)
    self.train_landmarks = np.empty(0)
    self.val_imgs = np.empty(0)
    self.val_landmarks = np.empty(0)
    self.test_imgs = np.empty(0)
    self.test_landmarks = np.empty(0)
    self.BATCH_SIZE = 8
    self.ITERATOR = np.empty(0)
    self.index_permutation = np.empty(0)
    self.area_counter = np.zeros((10,10))
  
  """ Read pts file with landmarks """
  def read_pts(self, path):
    min_x = self.WIDTH-1
    max_x = 0
    min_y = self.HEIGHT-1
    max_y = 0
    landmarks = []
    f = open(path, 'r')
    f = f.readlines()
    reading = False
    for line in f:
      if line == '{\n':
        reading = True
        min_x = self.WIDTH-1
        max_x = 0
        min_y = self.HEIGHT-1
        max_y = 0
      elif line == '}\n':
        reading = False
        landmarks.append([])
        landmarks[-1].append([])
        landmarks[-1].append([])
        landmarks[-1][0].append(min_x)
        landmarks[-1][1].append(min_y)
        landmarks[-1][0].append(max_x)
        landmarks[-1][1].append(max_y)
        landmarks[-1][0].append(min_x)
        landmarks[-1][1].append(max_y)
        landmarks[-1][0].append(max_x)
        landmarks[-1][1].append(min_y)
      elif reading == True:
        x,y = [float(x) for x in line.split()]
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        
    return np.array(landmarks)   
    
  """ Load data from given path """
  def load_data(self, path, data_type):
    imgs = []
    landmarks = []
    for rel_path, subdirs, files in os.walk(path):
      for name in sorted(files):
        full_path = os.path.join(rel_path, name)
        if name[-3:] == 'png':
          imgs.append(cv2.imread(full_path, cv2.IMREAD_COLOR))
        else :
          landmarks.append(self.read_pts(path = full_path))
      
    if data_type == 'train':
      num_imgs = len(imgs)
      for i in range(num_imgs):
        flip_img, flip_lands = self.flip_image(imgs[i], landmarks[i])
        imgs.append(flip_img)
        landmarks.append(flip_lands)
      self.train_imgs = np.array(imgs)
      self.train_landmarks = landmarks.copy()
    elif data_type == 'val':
      num_imgs = len(imgs)
      for i in range(num_imgs):
        flip_img, flip_lands = self.flip_image(imgs[i], landmarks[i])
        imgs.append(flip_img)
        landmarks.append(flip_lands)
      self.val_imgs = np.array(imgs)
      self.val_landmarks = landmarks.copy()
    elif data_type == 'test':
      self.test_imgs = np.array(imgs)
      self.test_landmarks = landmarks.copy()
    else:
      print('ERROR in load_data: name = {} doesn\'t exist, type \'train\' or \'val\' or \'test\''.format(name))
      print('Aborting')
      exit(0)
  
  """ Returns the printable image of size (WIDTH,HEIGHT) """
  def original_img(self, img):
    return img * 255.
    
  """ Normalize image """
  def normalize_img(self, img):
    return img / 255.

  """ Get the bounding box from landmarks """
  def get_box(self, landmarks, width = 160, height = 160):
    min_x = max(0, min(width-1, min(landmarks[0])))
    max_x = max(0, min(width-1, max(landmarks[0])))
    min_y = max(0, min(height-1, min(landmarks[1])))
    max_y = max(0, min(height-1, max(landmarks[1])))
    return np.array([min_x, min_y, max_x, max_y])

  """ Convert box format (x1,y1,x2,y2) to (x,y,w,h) and normalize to interval [0,1] """
  def normalize_box(self, box):
    ret_box = np.zeros(4)
    ret_box[0] = box[0] / self.WIDTH
    ret_box[1] = box[1] / self.HEIGHT
    ret_box[2] = abs(box[0] - box[2]) / self.WIDTH
    ret_box[3] = abs(box[1] - box[3]) / self.HEIGHT
    return ret_box
  
  """ Convert box to original values """
  def original_box(self, box):
    ret_box = np.zeros(4)
    ret_box[0] = max(0, min(self.WIDTH-1, box[0]*self.WIDTH))
    ret_box[1] = max(0, min(self.HEIGHT-1, box[1]*self.HEIGHT))
    ret_box[2] = max(0, min(self.WIDTH-1, (box[0] + box[2])*self.WIDTH))
    ret_box[3] = max(0, min(self.HEIGHT-1, (box[1] + box[3])*self.HEIGHT))
    return ret_box
  
  """ Shows image on screen. Pressing ESC stop the execution of the script """
  def show_img(self, img, name):

    cv2.imshow(name, img)
    if cv2.waitKey(0) == 27:
      exit(0)
    cv2.destroyAllWindows()  
  
  """ Draws ears in image """
  def draw_rect(self, img, landmarks):
    ret_img = np.copy(img)
    for landmark_id in range(len(landmarks)):
      bb = self.get_box(landmarks[landmark_id])
      cv2.rectangle(ret_img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,0,255), 1)
    
    return ret_img
        
  """ Resize image and adjust its landmarks """
  def resize_and_adjust_landmarks(self, img, landmarks, final_width, final_height):
    orig_width = img.shape[1]
    orig_height = img.shape[0]
    out_landmarks = np.copy(landmarks)
    out_img = cv2.resize(img, (final_width, final_height))
    for landmark_id in range(out_landmarks.shape[0]):
      out_landmarks[landmark_id][0] = out_landmarks[landmark_id][0]*final_width/orig_width
      out_landmarks[landmark_id][1] = out_landmarks[landmark_id][1]*final_height/orig_height
    return out_img, out_landmarks
  
  """ Create a flipped image and flip landmarks """
  def flip_image(self, img, landmarks):
    orig_width = img.shape[1]
    orig_height = img.shape[0]
    flip_img = np.copy(img)
    flip_img = np.flip(flip_img, axis=1)
    out_landmarks = np.copy(landmarks)
    for i in range(out_landmarks.shape[0]):
      out_landmarks[i][0] = (orig_width - 1.) - out_landmarks[i][0]
    
    return np.array(flip_img), out_landmarks
        
  """ Reset batch iterator and shuffle the indexes """
  def initialize_epoch(self):
    self.index_permutation = np.random.permutation(len(self.train_imgs))
    self.ITERATOR = 0
  
  """ Picture Collage """
  def collage_augmentation(self, image_set, landmark_set, order, offsets, img_redim_size, shape_order):
    if len(image_set) == 1:
      return image_set[0], landmark_set[0]
    
    ret_img = np.zeros((160, 160, 3))
    ret_landmarks = []
    
    black_image = np.zeros((img_redim_size,img_redim_size,3), dtype=np.uint8)
    rows = []
    
    order_iterator = 0
    
    image_set2 = []
    landmark_set2 = []
    
    for i in range(len(image_set)):
      dummy_img, dummy_land = self.resize_and_adjust_landmarks(image_set[i], landmark_set[i], img_redim_size, img_redim_size)
      image_set2.append(dummy_img)
      landmark_set2.append(dummy_land)
        
    for i in offsets:
      for j in offsets:
        if order[order_iterator] < len(image_set2):
          for land in range(len(landmark_set2[order[order_iterator]])):
            landmark_set2[order[order_iterator]][land][0] += j
            landmark_set2[order[order_iterator]][land][1] += i
            ret_landmarks.append(landmark_set2[order[order_iterator]][land])
            
        order_iterator += 1
        
    order = np.reshape(order, (shape_order,shape_order))
    
    for order_row in order:
      cols = []
      for order_id in order_row:
        if order_id < len(image_set2):
          cols.append(image_set2[order_id])
        else:
          cols.append(black_image)
      
      rows.append(np.hstack(cols))
      
    ret_img = np.vstack(rows)
    
    return ret_img, ret_landmarks 
  
  """ Crop Smaller Area of Image that contains the ear given """
  def crop_augmentation(self, img, landmarks):
    max_width = img.shape[1]-1
    max_height = img.shape[0]-1
    ear_lx = int(min(landmarks[0]))
    ear_ly = int(min(landmarks[1]))
    ear_rx = int(max(landmarks[0]))
    ear_ry = int(max(landmarks[1]))
    dx = abs(ear_rx - ear_lx)
    dy = abs(ear_ry - ear_ly)
    
    crop_lx = max(0, ear_lx - int(random.uniform(0,1)*ear_lx))
    crop_ly = max(0, ear_ly - int(random.uniform(0,1)*ear_ly))
    crop_rx = min(max_width-1, ear_rx + int(random.uniform(0,1)*(max_width - 1 - ear_rx + 1)))
    crop_ry = min(max_height-1, ear_ry + int(random.uniform(0,1)*(max_height - 1 - ear_ry + 1)))
    
    box = np.array([crop_lx, crop_ly, crop_rx, crop_ry])
    
    cropped_img = img[box[1]:box[3]+1, box[0]:box[2]+1]
    out_landmarks = np.copy(landmarks)
    out_landmarks[0] = out_landmarks[0] - box[0]
    out_landmarks[1] = out_landmarks[1] - box[1]

    return cropped_img, out_landmarks
  
  """ Augmentation """
  def augmentation(self, x, y):
    #if random.randint(1,2) == 1:
    #  return x, y
    color_aug = iaa.Sequential([iaa.Grayscale(alpha=(0.0, 1.0)), iaa.Multiply((0.5, 1.5))])
    color_aug = color_aug.to_deterministic()
    x = color_aug.augment_images([x])[0]
    
    aug_seq = iaa.Sequential([iaa.Affine(scale={"x": (0.75, 1.25), "y": (0.75, 1.25)}), iaa.Affine(rotate = (-45, 45)), iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)})])
    aug_seq = aug_seq.to_deterministic()  
    
    img = x
    new_landmarks = []
    img = aug_seq.augment_images([img])[0]
    
    for landmark_id in range(len(y)):
      aug_landmarks = np.zeros(y[landmark_id].shape)
      no_fix_bb = np.array([512, 512, -512, -512], dtype = np.float32)
      keypoints_list = []
      for i in range(y[landmark_id].shape[1]):
        keypoints_list.append(ia.Keypoint(x = y[landmark_id][0][i], y = y[landmark_id][1][i]))
      keypoints = ia.KeypointsOnImage(keypoints_list, shape = img.shape)
      orig_bb = self.get_box(y[landmark_id])
      keypoints = aug_seq.augment_keypoints([keypoints])[0]
      for i in range(aug_landmarks.shape[1]):
        aug_landmarks[0][i] = max(0, min(keypoints.keypoints[i].x, 159))
        aug_landmarks[1][i] = max(0, min(keypoints.keypoints[i].y, 159))
        no_fix_bb[0] = min(keypoints.keypoints[i].x, no_fix_bb[0])
        no_fix_bb[1] = min(keypoints.keypoints[i].y, no_fix_bb[1])
        no_fix_bb[2] = max(keypoints.keypoints[i].x, no_fix_bb[2])
        no_fix_bb[3] = max(keypoints.keypoints[i].y, no_fix_bb[3])
        
      bb = self.get_box(aug_landmarks)
      #print(no_fix_bb, bb)
      if self.box_area(bb)/self.box_area(no_fix_bb) < 0.9:
        return x, y
        
      new_landmarks.append(np.copy(aug_landmarks))
    
    return img, new_landmarks
      
                
  """ Process and returns the next batch for train """
  def next_batch(self):
    imgs = []
    labels = []
    no_objs = []
    objs = [] 
    if self.ITERATOR + self.BATCH_SIZE > len(self.train_imgs):
      return np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    
    batch = self.index_permutation[self.ITERATOR : self.ITERATOR + self.BATCH_SIZE]
       
    for remaining in range(self.BATCH_SIZE-1, -1, -1):
      
      collage_type = random.randint(1,2)
      num_images = 0
      order = []
      offsets = []
      img_redim_size = 0
      shape_order = 0
      #1 - One image // 2 - 4 Images // 3 - 16 Images 
      if collage_type == 1:
        num_images = 1
        order = np.arange(0,1)
        offsets = [0]
        img_redim_size = 160
        shape_order = 1
      elif collage_type == 2:
        num_images = random.randint(1,4)
        order = np.arange(0, 4)
        offsets = [0, 80]
        img_redim_size = 80
        shape_order = 2
     
      if self.ITERATOR + num_images + remaining > len(self.train_imgs):
        num_images = 1
      
      image_set = []
      landmark_set = []
      for dummy_iterator in range(num_images):
        push_img = np.copy(self.train_imgs[self.index_permutation[self.ITERATOR]])
        push_landmarks = np.copy(self.train_landmarks[self.index_permutation[self.ITERATOR]])
        push_img, push_landmarks = self.augmentation(push_img, push_landmarks)
        if len(push_landmarks) == 1 and random.randint(1,2) == 1:
          push_img, push_landmarks[0] = self.crop_augmentation(push_img, push_landmarks[0])
          push_img, push_landmarks = self.resize_and_adjust_landmarks(push_img, push_landmarks, 160, 160)
        image_set.append(push_img)
        landmark_set.append(push_landmarks)
        self.ITERATOR+=1
         
      order = np.random.permutation(order)
      x, y = self.collage_augmentation(image_set, landmark_set, order, offsets, img_redim_size, shape_order)
      
      label, obj, no_obj = self.get_bounding_box_map(y)
      
      x = self.normalize_img(x)
      imgs.append(np.copy(x))
      labels.append(np.copy(label))
      objs.append(np.copy(obj))
      no_objs.append(np.copy(no_obj))
      
    return np.array(imgs), np.array(labels), np.array(objs), np.array(no_objs)
  
  """ Convert landmark format to bounding box map """
  def get_bounding_box_map(self, y):
    label = np.zeros((self.NUM_AREAS, self.NUM_AREAS,4))
    no_obj = np.ones((self.NUM_AREAS, self.NUM_AREAS))
    obj = np.zeros((self.NUM_AREAS, self.NUM_AREAS)) 
    for landmark_id in range(len(y)):
      aux_box = self.get_box(y[landmark_id])
      center_x, center_y = self.get_center(aux_box)
      aux_box = self.normalize_box(aux_box)
      label[center_y, center_x] = aux_box
      no_obj[center_y, center_x] = 0
      obj[center_y, center_x] = 1
      self.area_counter[center_y, center_x]+=1
      
    return label, obj, no_obj
    
  """ Get center of box """
  def get_center(self, box):
    center_x = int((box[0] + box[2])/2)
    center_y = int((box[1] + box[3])/2)
    return int(center_x//self.AREA_SIZE), int(center_y//self.AREA_SIZE)
    
  """ Returns the intersection box of 2 bounding boxes """
  def intersection_box(self, box1, box2):
    box3 = np.zeros(4)
    box3[0] = max(box1[0], box2[0])
    box3[1] = max(box1[1], box2[1])
    box3[2] = min(box1[2], box2[2])
    box3[3] = min(box1[3], box2[3])
    if box3[0] > box3[2] or box3[1] > box3[3]:
      box3 = np.zeros(4)
    return box3
  
  """ Returns the union box of 2 bounding boxes """  
  def union_box(self, box1, box2):
    box3 = np.zeros(4)
    box3[0] = min(box1[0], box2[0])
    box3[1] = min(box1[1], box2[1])
    box3[2] = max(box1[2], box2[2])
    box3[3] = max(box1[3], box2[3])
    return box3
  
  """ Returns bounding box area """
  def box_area(self, box):
    dx = box[2] - box[0]
    dy = box[3] - box[1]
    dx = max(0, dx)
    dy = max(0, dy)
    return dx*dy
    
  """ Returns list of metrics for 2 boxes [IoU, Accuracy, Precision, Recall, F1 Score] """
  def calculate_metrics(self, y_, y):
    inter = self.box_area(self.intersection_box(y_,y))
    union = self.box_area(y_) + self.box_area(y) - inter
    TP = inter
    FP = self.box_area(y_) - inter
    FN = self.box_area(y) - inter
    TN = (self.WIDTH*self.HEIGHT - self.box_area(y)) - (self.box_area(y_) - inter)
    
    metrics = np.zeros(5)
    
    """ Divide into Cases """
    if TP == 0 and FP == 0 and FN == 0:
      metrics[0] = inter/union
      metrics[1] = (TP + TN)/(TP + TN + FP + FN)
      metrics[2] = 1
      metrics[3] = 1
      metrics[4] = 1
    elif TP == 0:
      metrics[0] = inter/union
      metrics[1] = (TP + TN)/(TP + TN + FP + FN)
      metrics[2] = 0
      metrics[3] = 0
      metrics[4] = 0
    else :
      metrics[0] = inter/union
      metrics[1] = (TP + TN)/(TP + TN + FP + FN)
      metrics[2] = TP/(TP + FP)
      metrics[3] = TP/(TP + FN)
      metrics[4] = 2*(metrics[2]*metrics[3])/(metrics[2] + metrics[3])
    return metrics
  
  """ Calculate IOU given 2 boxes"""
  def IoU(self, box1, box2):
    inter = self.box_area(self.intersection_box(box1,box2))
    union = self.box_area(box1) + self.box_area(box2) - inter
    return inter/union
    
  """ Suppress redundant boxes. Be sure to give boxes list in format (confidence_score, box) """
  def non_maximal_suppression(self, boxes):
    ret_boxes = []
    available = np.ones(len(boxes), dtype = np.int32)
    boxes.sort(reverse=True, key=lambda x: x[0])
    for i in range(len(boxes)):
      if available[i] == 0:
        continue
      conf, box = boxes[i]
      ret_boxes.append(np.copy(box))
      available[i] = 0
      for j in range(i+1, len(boxes)):
        if available[j] == 0:
          continue
        aux_conf, aux_box = boxes[j]
        if self.IoU(box, aux_box) >= 0.25:
          available[j] = 0
    return np.array(ret_boxes)
            
  """ Returns Validation Data """
  def get_val(self):
    return self.val_imgs, self.val_landmarks
  
  """ Returns Train Data """
  def get_train(self):
    return self.train_imgs, self.train_landmarks
  
  """ Returns Test Data """
  def get_test(self):
    return self.test_imgs, self.test_landmarks
    
if __name__ == "__main__":
  """  DEBUGS """
  
  data = DataManager()
  data.load_data(path = '../database/train', data_type = 'train')
  train_data, train_landmarks = data.get_train()
  data.initialize_epoch()
  
  while True:
    imgs, labels, objs, no_objs = data.next_batch()
    if imgs.shape[0] == 0:
      print('finished batch')
      exit(0)
    
    print('new batch')  
    for k in range(imgs.shape[0]):
      for i in range(data.NUM_AREAS):
        for j in range(data.NUM_AREAS):
          if objs[k, i, j]:
            orig_box = data.original_box(labels[k,i,j])
            cv2.rectangle(imgs[k], (int(orig_box[0]), int(orig_box[1])), (int(orig_box[2]), int(orig_box[3])), (0,0,255), 1)  
      data.show_img(imgs[k], 'image')
  
  exit(0)
  
  img = imgs[0]
  label = labels[0]
  obj = objs[0]
  no_obj = no_objs[0]
  
  for i in range(data.NUM_AREAS):
    for j in range(data.NUM_AREAS):
      if obj[i, j]:
        print('object found in ' + str(i) + ', '+ str(j))
        orig_box = data.original_box(label[i,j])
        cv2.rectangle(img, (int(orig_box[0]), int(orig_box[1])), (int(orig_box[2]), int(orig_box[3])), (0,0,255), 1)
  
  print(label)
  print('')
  print('')
  print(obj)
  print('')
  print('')
  print(no_obj)      
  data.show_img(img, 'image')
  exit(0)
  
  
