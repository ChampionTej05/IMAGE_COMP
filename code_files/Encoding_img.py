# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:12:26 2018

@author: champion
"""

"""
THis is standalone script which is used to enocde the input images which we would provide.
This scripts uses the session that we have trained using MNIST dataset and tries to encode the image

i/p: Please give the png image and make sure the path of image is specified correctly

o/p: JSON file will be created in the same directory where you put this code file
"""


import tensorflow as tf
import numpy as np
import cv2
TF_CPP_MIN_LOG_LEVEL=2



"""
This function performs the same Task as it was in the original image
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

        conv4 = tf.layers.conv2d(inputs=conv3,  filters=128, kernel_size=[
                                 4, 4], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[2,2])
        print(conv4.shape)

        conv5 = tf.layers.conv2d(inputs=conv4,  filters=128, kernel_size=[
                                 4,4], use_bias=True, padding="same", activation=tf.nn.leaky_relu,strides=[2,2])
        print("Encoded the Image Successfully...")
        return conv5


real_img = tf.placeholder(dtype=tf.float32, shape=[None, 784])
X = tf.reshape(real_img, [-1, 28, 28, 1])
enc_img = tf.placeholder(dtype=tf.float32,shape=[None,1,1,128])



#Giving input to the model
path_of_image=r"C:\Users\champion\Desktop\Image Compression Pro\four.png"

img=cv2.imread(path_of_image,0)

#resizing image to 28x28
img=cv2.resize(img, (28,28))


#reshaping it to 1x784 as our models expects flattened images
flat_img=np.reshape(img,(1,784))
print(flat_img.shape)

enc = Enocder(X)
print(enc.shape)
enc_img=enc
print(enc_img)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)

saver=tf.train.Saver()

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    #restoring the session
    saver.restore(sess, "/tmp/model.ckpt")
    f_name="compressed_image_"
    
    #here we would pass, our input image as the paramater to the Encoder function
    comp_img = sess.run(enc,feed_dict={real_img:flat_img})
    print(comp_img.shape)
    print(type(comp_img))
    
    #THe shape of comp_img is 1x1x1x128, we are reshapping it to the 1x128 and saving it in the JSON file
    r_comp=np.reshape(comp_img,(1,128))
    img_list = r_comp.tolist()
    
    #saving the encoded image in the disk
    import json
    with open("encoOne.json","w") as f:
        f.write(json.dumps(img_list))

    
