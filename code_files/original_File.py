import tensorflow as tf
TF_CPP_MIN_LOG_LEVEL=2
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/mnist', one_hot=True)

def Enocder(real_img):
    with tf.variable_scope("encoder"):
        conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[
                                 5, 5], use_bias=True, padding="same", activation=tf.nn.leaky_relu)
        conv2 = tf.layers.conv2d(inputs=conv1,  filters=64, kernel_size=[
                                 5, 5], use_bias=True, padding="same", activation=tf.nn.leaky_relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[
                                 5, 5], use_bias=True, padding="same", activation=tf.nn.leaky_relu)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[
                                 5, 5], use_bias=True, padding="same", activation=tf.nn.leaky_relu)       
        conv5 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=[
                                 5, 5], use_bias=True, padding="same", activation=tf.nn.leaky_relu)
        return conv5


def Decoder(Z):
    with tf.variable_scope("decoder"):
        deconv1 = tf.layers.conv2d_transpose(inputs=Z, filters=64, kernel_size=[
                                             5, 5], padding="same", strides=[1, 1])
        deconv2 = tf.layers.conv2d_transpose(inputs=deconv1, filters=64, kernel_size=[
                                             5, 5], padding="same", strides=[1, 1])
        deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, filters=32, kernel_size=[
                                             5, 5], padding="same", strides=[1, 1])
        deconv4 = tf.layers.conv2d_transpose(inputs=deconv3, filters=1, kernel_size=[1,1], padding="same", strides=[1,1])

        return deconv4

real_img = tf.placeholder(dtype=tf.float32, shape=[None, 784])
X = tf.reshape(real_img, [-1, 28, 28, 1])
enc = Enocder(X)
dec = Decoder(enc)
with tf.variable_scope('loss'):
    cost = tf.reduce_sum(tf.squared_difference(X,dec))

with tf.variable_scope('trainer'):
    trainer = tf.train.AdamOptimizer(0.001).minimize(cost)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    f_name="compressed_image_"
    for epoch in range(500):
        batch_x,_ = mnist.train.next_batch(64)
        _, error,comp_img = sess.run([trainer,cost,enc],feed_dict={real_img:batch_x}) 
        print("Epoch {0} Loss {1}".format(epoch,error))
    #error = sess.run([cost],feed_dict={real_img:mnist.test.images})
    print("Testing cost: {0}".format(error))

