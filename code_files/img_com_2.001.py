# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:31:02 2018

@author: champion
"""

import tensorflow as tf
TF_CPP_MIN_LOG_LEVEL=2
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/mnist', one_hot=True)

import matplotlib.pyplot as plt

def Enocder(real_img):
    with tf.variable_scope("encoder"):
        conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[
                                 2, 2], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[2,2])
        print(conv1.shape)
        conv2 = tf.layers.conv2d(inputs=conv1,  filters=64, kernel_size=[
                                 2, 2], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[2,2])
        print(conv2.shape)
#        conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[
#                                 2,2], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[2,2])
#        print(conv3.shape)
#        conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[
#                                 2, 2], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[2,2])
#        print(conv4.shape)       
#        conv5 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=[
#                                 2, 2], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[2,2])
#        print(conv5.shape)
    
        conv3 = tf.layers.conv2d(inputs=conv2,  filters=128, kernel_size=[
                                 1,1], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[7,7])
        print("Encoded the Image Successfully...")
        return conv3


def Decoder(Z):
    with tf.variable_scope("decoder"):
        
        deconv1 = tf.layers.conv2d_transpose(inputs=Z, filters=128, kernel_size=[
                                             2, 2], padding="same", strides=[2,2])
        print(deconv1.shape)
        deconv2 = tf.layers.conv2d_transpose(inputs=deconv1, filters=128, kernel_size=[
                                             2, 2], padding="same", strides=[2, 2])
        print(deconv2.shape)
        deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, filters=64, kernel_size=[
                                             2, 2], padding="same", strides=[2,2])
        print(deconv3.shape)
        deconv4 = tf.layers.conv2d_transpose(inputs=deconv3, filters=32, kernel_size=[
                                             2, 2], padding="same", strides=[2, 2])
        print(deconv4.shape)
        deconv5 = tf.layers.conv2d_transpose(inputs=deconv3, filters=1, kernel_size=[2,2], padding="same", strides=[2,2])
        return deconv5
    
    
def Decoder_new(Z):
     with tf.variable_scope("decoder"):
        
        deconv1 = tf.layers.conv2d_transpose(inputs=Z, filters=128, kernel_size=[
                                             1,1], padding="same", strides=[7,7])
        print(deconv1.shape)
        deconv2 = tf.layers.conv2d_transpose(inputs=deconv1, filters=64, kernel_size=[
                                             2, 2], padding="same", strides=[2, 2])
        print(deconv2.shape)
        deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, filters=32, kernel_size=[
                                             2, 2], padding="same", strides=[2,2])
    
        print(deconv3.shape)
        deconv4=tf.layers.conv2d_transpose(inputs=deconv3,filters=1, kernel_size=[
                                             1, 1], padding="same", strides=[1,1])
    
        
        
        return deconv4


#takin the real image
real_img = tf.placeholder(dtype=tf.float32, shape=[None, 784])
X = tf.reshape(real_img, [-1, 28, 28, 1])
print(X)
enc_img = tf.placeholder(dtype=tf.float32,shape=[None,1,1,128])


enc = Enocder(X)
print(enc.shape)
#-----------------------Done, checked---------

enc_img=enc

#fig=plt.figure()
#
#a=plt.subplot(1,2,1)
#plt.imshow(real_img)



dec = Decoder_new(enc_img)
print("Shape= %s" % dec.shape)
with tf.variable_scope('loss'):
    cost = tf.reduce_sum(tf.squared_difference(X,dec))
#print(cost.shape)
learning_rate=0.001
with tf.variable_scope('trainer'):
    trainer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    f_name="compressed_image_"
    for epoch in range(2):
        batch_x,_ = mnist.train.next_batch(64)
       # _, error,comp_img = sess.run([enc],feed_dict={real_img:batch_x})
#        comp_img = sess.run(enc,feed_dict={real_img:batch_x})
#        print(comp_img.shape)
        _,loss,dec_img = sess.run([trainer,cost,dec],feed_dict={real_img:batch_x})
        print("Epoch {0} and loss {1}".format(epoch,loss))
        
    #decoding the image
    error = sess.run([cost],feed_dict={real_img:mnist.test.images})
    print("Testing cost: {0}".format(error))

