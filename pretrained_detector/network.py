""" 
Network Architecture
  
"""

import numpy as np
import cv2
import os
import tensorflow as tf

class Network:
  
  """ Constructor """
  def __init__(self, img_width, img_height, img_channels):
    
    """ Placeholders/Input """
    self.X = tf.placeholder(dtype = tf.float32, shape = (None, img_height, img_width, img_channels), name = 'X')
    self.Y = tf.placeholder(dtype = tf.float32, shape = (None, 10, 10, 4), name = 'Y')
    self.OBJ = tf.placeholder(dtype = tf.float32, shape = (None, 10, 10), name = 'OBJ')
    self.training_flag = tf.placeholder(dtype = tf.bool, name = 'training_flag')
    self.BATCH_SIZE = tf.constant(8, dtype = tf.float32)
    """ Convolution 1 + Max Pooling """
    for i in range(1):
      self.out = tf.layers.conv2d(inputs = self.X, filters = 32, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu)
    self.out = tf.layers.max_pooling2d(inputs = self.out, pool_size = (2,2), strides = (2,2), padding = 'valid')
    
    """ Convolution 2 + Max Pooling """
    for i in range(2):
      self.out = tf.layers.conv2d(inputs = self.out, filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu)
    self.out = tf.layers.max_pooling2d(inputs = self.out, pool_size = (2,2), strides = (2,2), padding = 'valid')
    
    """ Convolution 3 + Max Pooling """
    for i in range(3):
      self.out = tf.layers.conv2d(inputs = self.out, filters = 128, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu)
    self.out = tf.layers.max_pooling2d(inputs = self.out, pool_size = (2,2), strides = (2,2), padding = 'valid')
    
    """ Convolution 4 + Max Pooling """
    for i in range(3):
      self.out = tf.layers.conv2d(inputs = self.out, filters = 256, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu)
    self.out = tf.layers.max_pooling2d(inputs = self.out, pool_size = (2,2), strides = (2,2), padding = 'valid')
    
    """ Convolution 5 + Max Pooling """
    for i in range(5):
      self.out = tf.layers.conv2d(inputs = self.out, filters = 512, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu)
    self.out = tf.layers.max_pooling2d(inputs = self.out, pool_size = (2,2), strides = (2,2), padding = 'valid')

    """ Flattening """
    self.out = tf.reshape(self.out, (-1, self.out.shape[1]*self.out.shape[2]*self.out.shape[3]))
    
    """ Fully Connected 1 + Dropout """
    self.out = tf.layers.dense(self.out, 1024, activation = tf.nn.relu)
    self.out = tf.layers.dropout(self.out, rate = 0.1, training = self.training_flag)

    """ Fully Connected 2 - Output Bounding Boxes + Output Confidences """
    self.out = tf.layers.dense(self.out, 400, activation = tf.nn.sigmoid)
    
    """ Reshape to bounding box map format """
    self.Y_ = tf.reshape(self.out, (-1, 10, 10, 4))
        
    """ Loss Function """
    self.loss = tf.divide(self.calculate_loss(self.Y, self.Y_, self.OBJ), self.BATCH_SIZE)
    
    """ Training Operation """
    self.train_op = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(self.loss)
    
    """ Tensorboard Summary """
    self.iou_placeholder = tf.placeholder(dtype = tf.float32)
    self.accuracy_placeholder = tf.placeholder(dtype = tf.float32)
    self.precision_placeholder = tf.placeholder(dtype = tf.float32)
    self.recall_placeholder = tf.placeholder(dtype = tf.float32)
    self.f1_score_placeholder = tf.placeholder(dtype = tf.float32)
    self.loss_placeholder = tf.placeholder(dtype = tf.float32)
    self.loss_summary = tf.summary.scalar('loss', self.loss_placeholder)
    self.iou_summary = tf.summary.scalar('IoU', self.iou_placeholder)
    self.accuracy_summary = tf.summary.scalar('Accuracy', self.accuracy_placeholder)
    self.precision_summary = tf.summary.scalar('Precision', self.precision_placeholder)
    self.recall_summary = tf.summary.scalar('Recall', self.recall_placeholder)
    self.f1_score_summary = tf.summary.scalar('F1 Score', self.f1_score_placeholder)
    self.merged_summaries = tf.summary.merge_all()
    
    """ Tensorflow Session Initialization + Saver """
    self.sess = tf.Session()
    self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver(save_relative_paths=True)
  
  """ Loss Function """
  def calculate_loss(self, y, y_, obj):
    coords_y = y[..., :, :, :2]
    coords_y_ = y_[..., :, :, :2]
    dims_y = y[..., :, :, 2:]
    dims_y_ = y_[..., :, :, 2:]
    
    coord_loss = tf.reduce_sum(tf.square(coords_y - coords_y_), axis = -1)
    coord_loss = tf.multiply(coord_loss, obj)
    coord_loss = tf.reduce_sum(coord_loss, axis = None)
    dims_loss = tf.reduce_sum(tf.square(tf.sqrt(dims_y) - tf.sqrt(dims_y_)), axis = -1)
    dims_loss = tf.multiply(dims_loss, obj)
    dims_loss = tf.reduce_sum(dims_loss, axis = None)
    
    return tf.add(coord_loss, dims_loss)
        
  """ Calculates IoU with TF functions """
  def TF_IOU(self, box1, box2):
    leftA = box1[..., :2]
    leftB = box2[..., :2]
    rightA = tf.add(box1[..., :2], box1[..., 2:])
    rightB = tf.add(box2[..., :2],box2[..., 2:])
    pt1 = tf.maximum(leftA, leftB)
    pt2 = tf.minimum(rightA, rightB)
    inter_area = tf.reduce_prod(tf.maximum(tf.subtract(pt2, pt1), 0), axis = -1)
    areaA = tf.reduce_prod(tf.maximum(tf.subtract(rightA, leftA), 0), axis = -1)
    areaB = tf.reduce_prod(tf.maximum(tf.subtract(rightB, leftB), 0), axis = -1)
    union_area = tf.subtract(tf.add(areaA, areaB), inter_area)
    union_area = tf.add(union_area, 1e-12)
    return tf.divide(inter_area, union_area)
       
  """ Train Model"""  
  def train(self, X, Y, OBJ):
    _, loss = self.sess.run([self.train_op, self.loss], feed_dict = {self.X: X, self.Y: Y, self.OBJ: OBJ, self.training_flag: True})
    return loss
  
  """ Predict Boxes (Reshape array if only one image is given) """
  def predict(self, x):
    y = self.sess.run(self.Y_, feed_dict = {self.X: x, self.training_flag: False})
    return np.array(y)
  
  """ Update Tensorboard """
  def update_board(self, iou, acc, precision, recall, f1, loss, idx):
    summary = self.sess.run(self.merged_summaries, feed_dict = {self.iou_placeholder: iou, self.accuracy_placeholder: acc, self.precision_placeholder: precision, self.recall_placeholder: recall, self.f1_score_placeholder: f1, self.loss_placeholder: loss})
    self.writer.add_summary(summary, idx)
    self.writer.flush()
    
  """ Save Model"""
  def save_model(self, name):
    self.saver.save(self.sess, 'model/' + name)
  
  """ Restore Model"""
  def restore_model(self, name):
    self.saver.restore(self.sess, 'model/' + name)
