# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:22:55 2018

@author: champion
"""

"""
This is the standalone script which decodes the image,of dimmension 1x128. 
This script produces the reconstucted image of dimmension 28x28 

i/p: JSON file created during Encoding_img.py 

O/P: Reconstructed image in the same directory.

"""


import tensorflow as tf
TF_CPP_MIN_LOG_LEVEL=2
import numpy as np
import cv2
import matplotlib.pyplot as plt


def Decoder_new(Z):
    with tf.variable_scope("decoder"):
        # 1st Layer
        deconv1 = tf.layers.conv2d_transpose(inputs=Z, filters=128, kernel_size=[
            3, 3], padding="same", strides=[2, 2])
        print(deconv1.shape)

        # 2nd Layer
        deconv2 = tf.layers.conv2d_transpose(inputs=deconv1, filters=64, kernel_size=[
            3, 3], padding="same", strides=[2, 2])
        print(deconv2.shape)

        # 3rd layer
        deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, filters=32, kernel_size=[
            7, 7], padding="same", strides=[7, 7])
        print(deconv3.shape)

        # final layer
        deconv4 = tf.layers.conv2d_transpose(inputs=deconv3, filters=1, kernel_size=[
            3, 3], padding="same", strides=[1, 1])
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
    
    
real1_img = tf.placeholder(dtype=tf.float32, shape=[None, 784])
X1 = tf.reshape(real1_img, [-1, 28, 28, 1])
enc1_img = tf.placeholder(dtype=tf.float32,shape=[None,1,1,128])


dec = Decoder_new(enc1_img)

saver=tf.train.Saver()

with tf.Session() as sess:
    #loadign the encoded image from disk
    import json
    with open("encoOne.json","r") as f:
            data = json.loads(f.read())  
    
    #making it ndarray
    a=np.asarray(data)
    
    #a has shape of 1x128 we would try to expand its dimmnensions to 1x1x1x128, which is the expected input for our decoded image
    a=np.expand_dims(a,axis=0)
    a=np.expand_dims(a,axis=0)
    print(a.shape)
    
    #restoring the session of trainning model
    saver.restore(sess, "/tmp/model.ckpt")
    
    f_name="compressed_image_"
    
    #creating the dummy array, for real_img variable as we can't keep any placeholder empty
    tp = np.asarray([[i for i in range(784)]])
    print(tp.shape)
    print(a.shape)
    
    #decoding the image
    output_img=sess.run(dec,feed_dict={real1_img:tp,enc1_img:a}) 
    print("Decoded the image")
    print(output_img.shape)
    final_img=np.reshape(output_img,(28,28))
    
    #print(final_img)
    
    
   

    fig=plt.figure()
    img1=cv2.imread(r"C:\Users\champion\Desktop\Image Compression Pro\four.png",0)
    i=fig.add_subplot(1,2,1)
    i.set_title("Before")
    plt.imshow(img1)
    
    d=fig.add_subplot(1,2,2)
    d.set_title("After")
    plt.imshow(final_img)
    
    cv2.imwrite("recons_four_8.png",final_img)