""" 
Class Network Architecture
  
"""

import numpy as np
import cv2
import os
import tensorflow as tf

class ClassNetwork:
  
  """ Constructor """
  def __init__(self, img_width, img_height, img_channels):
    
    """ Placeholders/Input """
    self.X = tf.placeholder(dtype = tf.float32, shape = (None, img_height, img_width, img_channels), name = 'X')
    self.OBJ = tf.placeholder(dtype = tf.float32, shape = (None, 10, 10), name = 'OBJ')
    self.training_flag = tf.placeholder(dtype = tf.bool, name = 'training_flag')
    self.BATCH_SIZE = tf.constant(8, dtype = tf.float32)
    """ Convolution 1 + Max Pooling """
    for i in range(1):
      self.out = tf.layers.conv2d(inputs = self.X, filters = 32, kernel_size = (5,5), strides = (1,1), padding = 'same', activation = tf.nn.relu)
    self.out = tf.layers.max_pooling2d(inputs = self.out, pool_size = (2,2), strides = (2,2), padding = 'valid')
    
    """ Convolution 2 + Max Pooling """
    for i in range(1):
      self.out = tf.layers.conv2d(inputs = self.out, filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu)
    self.out = tf.layers.max_pooling2d(inputs = self.out, pool_size = (2,2), strides = (2,2), padding = 'valid')
    
    """ Convolution 3 + Max Pooling """
    for i in range(1):
      self.out = tf.layers.conv2d(inputs = self.out, filters = 128, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu)
    self.out = tf.layers.max_pooling2d(inputs = self.out, pool_size = (2,2), strides = (2,2), padding = 'valid')
    
    """ Convolution 4 + Max Pooling """
    for i in range(1):
      self.out = tf.layers.conv2d(inputs = self.out, filters = 256, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu)
    self.out = tf.layers.max_pooling2d(inputs = self.out, pool_size = (2,2), strides = (2,2), padding = 'valid')
    
    """ Convolution 5 + Max Pooling """
    for i in range(1):
      self.out = tf.layers.conv2d(inputs = self.out, filters = 512, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu)
    self.out = tf.layers.max_pooling2d(inputs = self.out, pool_size = (2,2), strides = (2,2), padding = 'valid')

    """ Flattening """
    self.out = tf.reshape(self.out, (-1, self.out.shape[1]*self.out.shape[2]*self.out.shape[3]))
    
    """ Fully Connected 1 + Dropout """
    self.out = tf.layers.dense(self.out, 512, activation = tf.nn.relu)
    self.out = tf.layers.dropout(self.out, rate = 0.1, training = self.training_flag)

    """ Fully Connected 2 - Output Bounding Boxes + Output Confidences """
    self.out = tf.layers.dense(self.out, 100, activation = tf.nn.sigmoid)
    
    """ Reshape to bounding box map format """
    self.Y_ = tf.reshape(self.out, (-1, 10, 10))
        
    """ Loss Function """
    self.logits = tf.where(tf.equal(self.OBJ, tf.ones_like(self.OBJ)), self.Y_, 1 - self.Y_, name = 'tf_cond_op') 
    self.loss = tf.divide(self.focal_loss(self.logits), self.BATCH_SIZE)
    
    """ Training Operation """
    self.train_op = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(self.loss)
    
    """ Tensorboard Summary """
    self.accuracy_placeholder = tf.placeholder(dtype = tf.float32)
    self.loss_placeholder = tf.placeholder(dtype = tf.float32)
    self.loss_summary = tf.summary.scalar('loss', self.loss_placeholder)
    self.accuracy_summary = tf.summary.scalar('Accuracy', self.accuracy_placeholder)
    self.merged_summaries = tf.summary.merge([self.loss_summary, self.accuracy_summary])
    
    """ Tensorflow Session Initialization + Saver """
    self.sess = tf.Session()
    self.writer = tf.summary.FileWriter('./class_logs', self.sess.graph)
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver(save_relative_paths=True)
  
  """ Loss Function """
  def focal_loss(self, p):
    gamma = 1
    return tf.reduce_sum((-(1-p)**gamma)*tf.log(p), axis = None)
       
  """ Train Model"""  
  def train(self, X, OBJ):
    _, loss = self.sess.run([self.train_op, self.loss], feed_dict = {self.X: X, self.OBJ: OBJ, self.training_flag: True})
    return loss
  
  """ Predict Boxes (Reshape array if only one image is given) """
  def predict(self, x):
    y = self.sess.run(self.Y_, feed_dict = {self.X: x, self.training_flag: False})
    return np.array(y)
  
  """ Update Tensorboard """
  def update_board(self, acc, loss, idx):
    summary = self.sess.run(self.merged_summaries, feed_dict = {self.accuracy_placeholder: acc, self.loss_placeholder: loss})
    self.writer.add_summary(summary, idx)
    self.writer.flush()
    
  """ Save Model"""
  def save_model(self, name):
    self.saver.save(self.sess, 'class_model/' + name)
  
  """ Restore Model"""
  def restore_model(self, name):
    self.saver.restore(self.sess, 'class_model/' + name)
