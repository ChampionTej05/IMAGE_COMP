# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:47:56 2018

@author: champion
"""

#importing the necessary modules and importing the dataset from MNIST

import tensorflow as tf
import numpy as np
TF_CPP_MIN_LOG_LEVEL=2
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/mnist', one_hot=True)

import matplotlib.pyplot as plt
"""
@param : Real_img -> Input image of dimmensions 28x28 
@return :  conv3 -> Encoded version of dimension 1x128

This functions applies 5 convolutional filters to the original image each of different size and different strides.
 Activation function used to remove the negative values is Leaky relu. 

"""

def Enocder(real_img):
    with tf.variable_scope("encoder"):
        #1st Layer
        conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[
                                 4, 4], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[2,2])
        print(conv1.shape)

        #2nd Layer
        conv2 = tf.layers.conv2d(inputs=conv1,  filters=64, kernel_size=[
                                 4, 4], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[2,2])
        print(conv2.shape)

        #3rd layer
        conv3 = tf.layers.conv2d(inputs=conv2,  filters=128, kernel_size=[
                                 4, 4], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[2,2])
        print(conv3.shape)

        #4th layer
        conv4 = tf.layers.conv2d(inputs=conv3,  filters=128, kernel_size=[
                                 4, 4], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[2,2])
        print(conv4.shape)
        
        #5th Layer
        conv5 = tf.layers.conv2d(inputs=conv4,  filters=128, kernel_size=[
                                 4,4], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[2,2])
        print("Encoded the Image Successfully...")
        return conv5
"""
@param: Z-> Encoded/ compressed version of image with dimmension 1x128
@return: deconv4 -> Reconstructed Image of dimmension 28x28

@work: This function applies reverse convolutional to 128 batch size image pixels and tries to create the   
        information from 3rd dimmension to the 1st and 2nd dimmension. We are doing exactly reverse of what we 
        we were doing in Encoder() module.

"""


def Decoder_new(Z):
     with tf.variable_scope("decoder"):
        
        #1st Layer
        deconv1 = tf.layers.conv2d_transpose(inputs=Z, filters=128, kernel_size=[
                                             3, 3], padding="same", strides=[2,2])
        print(deconv1.shape)
        
        #2nd Layer
        deconv2 = tf.layers.conv2d_transpose(inputs=deconv1, filters=64, kernel_size=[
                                             3, 3], padding="same", strides=[2, 2])
        print(deconv2.shape)
        
        #3rd layer
        deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, filters=32, kernel_size=[
                                             7, 7], padding="same", strides=[7,7])
        print(deconv3.shape)

        #final layer
        deconv4=tf.layers.conv2d_transpose(inputs=deconv3,filters=1, kernel_size=[
                                             3, 3], padding="same", strides=[1,1])
        print(deconv4.shape)
        # # final layer
        # deconv5 = tf.layers.conv2d_transpose(inputs=deconv4, filters=32, kernel_size=[
        #     3, 3], padding="same", strides=[2, 2])
        # print(deconv5.shape)
        # # final layer
        # deconv6 = tf.layers.conv2d_transpose(inputs=deconv5, filters=1, kernel_size=[
        #     3, 3], padding="same", strides=[1, 1])
        # print(deconv6.shape)
        # deconv7 = np.reshape(deconv6, (None,28,28,1))
        return deconv4


"""
This is the main script which runs and trains the model and produces output in the form of session

"""


#creating placeholders for real image
real_img = tf.placeholder(dtype=tf.float32, shape=[None, 784])
#reshaping 1x784 arrays avaialble in the database to 28x28
X = tf.reshape(real_img, [-1, 28, 28, 1])

#Placeholder for our encoded image
enc_img = tf.placeholder(dtype=tf.float32,shape=[None,1,1,128])

#encoding the image
enc = Enocder(X)
print(enc.shape)
enc_img=enc
print(enc_img)

#decoding the image
dec = Decoder_new(enc_img)
print("Shape= %s" % dec.shape)

#calculating the loss between reconstructed, and original image
with tf.variable_scope('loss'):
    cost = tf.reduce_sum(tf.squared_difference(X,dec))


learning_rate=0.001

#optimizing the cost function, to reduce the loss
with tf.variable_scope('trainer'):
    trainer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)

#Creating the saver object to save the session
saver=tf.train.Saver()

#running the session
loss_array=[]
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    
    #initialize all the variables
    sess.run(tf.global_variables_initializer())
    f_name="compressed_image_"
    count=2000
    #Runiing the module for 'count' epochs
    
    for epoch in range(count):
        #taking the batch of 8 and feeding it to our dataset
        batch_x,_ = mnist.train.next_batch(8)
        
        #running the decoder whihc has dependency on encoder part 
        _,loss,dec_img = sess.run([trainer,cost,dec],feed_dict={real_img:batch_x})
        print("Epoch {0} and loss {1}".format(epoch,loss))
        
        loss_array.append(loss)
        
    #saving the model
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path) 
    
    
    l=loss_array[1900:]
    print(l)
    plt.plot(np.arange(100),l)
    plt.show()
    
    #Calculating the error
    error = sess.run([cost],feed_dict={real_img:mnist.test.images})
    print("Testing cost: {0}".format(error))

