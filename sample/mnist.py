#!/usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def xavier_init(n_inputs, n_outputs, uniform=True):
    # Xavier Glorot and Yoshua Bengio (2010)
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('dropout_rate', 0.9, 'Dropout rate')
flags.DEFINE_integer('training_repeat', 1, 'Training repeat count')
flags.DEFINE_integer('batch_size', 100, 'Batch size per loop')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
tf.histogram_summary("images", x)

y_ = tf.placeholder(tf.float32, [None, 10])
tf.histogram_summary("label", y_)

dropout_rate = tf.placeholder(tf.float32)
batch_set_size = tf.placeholder(tf.int32)

cw0 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
cL0r = tf.nn.relu(tf.nn.conv2d(x, cw0, strides=[1, 1, 1, 1], padding='SAME')) # strides = [1, strideX, strideY, 1], padding=SAME or VALID
cL0p = tf.nn.max_pool(cL0r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
cL0 = tf.nn.dropout(cL0p, dropout_rate)

cw1 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
cL1r = tf.nn.relu(tf.nn.conv2d(cL0, cw1, strides=[1, 1, 1, 1], padding='SAME')) # strides = [1, strideX, strideY, 1], padding=SAME or VALID
cL1p = tf.nn.max_pool(cL1r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
cL1 = tf.nn.dropout(cL1p, dropout_rate)

cw2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
cL2r = tf.nn.relu(tf.nn.conv2d(cL1, cw2, strides=[1, 1, 1, 1], padding='SAME')) # strides = [1, strideX, strideY, 1], padding=SAME or VALID
cL2p = tf.nn.max_pool(cL2r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
cL2 = tf.nn.dropout(cL2p, dropout_rate)

# Create the model
with tf.name_scope("input_layer"):
    w0 = tf.get_variable("input_layer_weight", shape=[4*4*128, 256], initializer=xavier_init(4*4*128, 256))
    #w0 = tf.get_variable("input_layer_weight", shape=[28*28, 256], initializer=xavier_init(28*28, 256))
    #tf.histogram_summary("input_layer_weight", w0)    
    b0 = tf.Variable(tf.random_normal([256]))
    tf.histogram_summary("input_layer_bias", b0)
    cL2 = tf.reshape(cL2, tf.pack([batch_set_size, -1]))
    _L0 = tf.nn.relu(tf.matmul(cL2, w0) + b0)
    #_L0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    L0 = tf.nn.dropout(_L0, dropout_rate)

with tf.name_scope("hidden_layer"):
    with tf.name_scope("layer_1"):
        w1 = tf.get_variable("layer_1_weight", shape=[256, 256], initializer=xavier_init(256, 256))
        #tf.histogram_summary("weight", w1)
        b1 = tf.Variable(tf.random_normal([256]))
        tf.histogram_summary("layer_1_bias", b1)

        _L1 = tf.nn.relu(tf.matmul(L0, w1) + b1) # Hidden layer with RELU activation
        L1 = tf.nn.dropout(_L1, dropout_rate)

    with tf.name_scope("layer_2"):
        w2 = tf.get_variable("layer_2_weight", shape=[256, 256], initializer=xavier_init(256, 256))
        #tf.histogram_summary("weight", w2)
        b2 = tf.Variable(tf.random_normal([256]))
        tf.histogram_summary("layer_2_bias", b2)

        _L2 = tf.nn.relu(tf.matmul(L1, w2) + b2)
        L2 = tf.nn.dropout(_L2, dropout_rate)

    with tf.name_scope("layer_3"):
        w3 = tf.get_variable("layer_3_weight", shape=[256, 256], initializer=xavier_init(256, 256))
        #tf.histogram_summary("weight", w3)
        b3 = tf.Variable(tf.random_normal([256]))
        tf.histogram_summary("layer_3_bias", b3)

        _L3 = tf.nn.relu(tf.matmul(L2, w3) + b3) 
        L3 = tf.nn.dropout(_L3, dropout_rate)

with tf.name_scope("output_layer"):
    w4 = tf.get_variable("output_layer_weight", shape=[256, 10], initializer=xavier_init(256, 10))
    #tf.histogram_summary("weight", w4)    
    b4 = tf.Variable(tf.random_normal([10]))
    tf.histogram_summary("output_layer_bias", b4)
    
    hypothesis = tf.matmul(L3, w4) + b4
    prediction = tf.nn.softmax(hypothesis, name="prediction")
    tf.histogram_summary("prediction", prediction)

with tf.name_scope("activation"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, y_), name="loss")
    tf.scalar_summary("loss", loss)

with tf.name_scope("train"):
    # Define loss and optimizer
    # AdadeltaOptimizer, AdagradOptimizer, MomentumOptimizer, FtrlOptimizer ...
    # train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    train_step = tf.train.RMSPropOptimizer(FLAGS.learning_rate, 0.9).minimize(loss)

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph)

# Train
tf.initialize_all_variables().run()

total_batch = int(mnist.train.num_examples / FLAGS.batch_size)

for repeats in range(FLAGS.training_repeat):

    #avg_loss = 0.0

    for batch_set in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        batch_xs = batch_xs.reshape(FLAGS.batch_size, 28, 28, 1)
        train_step.run({x: batch_xs, y_: batch_ys, dropout_rate: FLAGS.dropout_rate, batch_set_size: FLAGS.batch_size})
        # compute average loss
        #avg_loss += loss.eval({x: batch_xs, y_: batch_ys, dropout_rate: FLAGS.dropout_rate}) / total_batch
        if batch_set % 20 == 0:
            print("Epoch:", "{:04d}".format(repeats+1), "{:.2f}%".format(batch_set/total_batch*100.0))
            summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys, dropout_rate: FLAGS.dropout_rate, batch_set_size: FLAGS.batch_size})
            writer.add_summary(summary, batch_set)

    
    #print("Epoch:", "{:04d}".format(repeats+1), "loss=", "{:.9f}".format(avg_loss))
    #summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys, dropout_rate: FLAGS.dropout_rate})
    #writer.add_summary(summary, repeats)

images = mnist.test.images
images = images.reshape(images.shape[0], 28, 28, 1)
    
# Test trained model
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: images, y_: mnist.test.labels, dropout_rate: 1, batch_set_size: images.shape[0]}))
