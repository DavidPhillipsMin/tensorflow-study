#!/usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

# Import data
import numpy as np
import tensorflow as tf
import trainingset as ts

def xavier_init(n_inputs, n_outputs, uniform=True):
    # Xavier Glorot and Yoshua Bengio (2010)
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(5.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('dropout_rate', 0.9, 'Dropout rate')
flags.DEFINE_integer('training_repeat', 1000, 'Training repeat count')
flags.DEFINE_integer('batch_size', 50, 'Batch size per loop')

sess = tf.InteractiveSession()

chart = tf.placeholder(tf.float32, [None, ts.CHART_SIZE, 1, 5])
tf.histogram_summary("chart", chart)

#print(ts.CHART_SIZE)

ratio = tf.placeholder(tf.float32, [None, 61])
tf.histogram_summary("label", ratio)

dropout_rate = tf.placeholder(tf.float32)
batch_set_size = tf.placeholder(tf.int32)

cw0 = tf.Variable(tf.random_normal([3, 1, 5,32], stddev=0.01))
cL0r = tf.nn.relu(tf.nn.conv2d(chart, cw0, strides=[1, 1, 1, 1], padding='SAME')) # strides = [1, strideX, strideY, 1], padding=SAME or VALID
cL0p = tf.nn.max_pool(cL0r, ksize=[1, 5, 1, 1], strides=[1, 5, 1, 1], padding='SAME')
cL0 = tf.nn.dropout(cL0r, dropout_rate)

#print(cL0)

cw1 = tf.Variable(tf.random_normal([3, 1, 32, 64], stddev=0.01))
cL1r = tf.nn.relu(tf.nn.conv2d(cL0, cw1, strides=[1, 1, 1, 1], padding='SAME')) # strides = [1, strideX, strideY, 1], padding=SAME or VALID
cL1p = tf.nn.max_pool(cL1r, ksize=[1, 5, 1, 1], strides=[1, 5, 1, 1], padding='SAME')
cL1 = tf.nn.dropout(cL1p, dropout_rate)

#print(cL1)

cw2 = tf.Variable(tf.random_normal([3, 1, 64, 128], stddev=0.01))
cL2r = tf.nn.relu(tf.nn.conv2d(cL1, cw2, strides=[1, 1, 1, 1], padding='SAME')) # strides = [1, strideX, strideY, 1], padding=SAME or VALID
cL2p = tf.nn.max_pool(cL2r, ksize=[1, 5, 1, 1], strides=[1, 5, 1, 1], padding='SAME')
cL2 = tf.nn.dropout(cL2r, dropout_rate)

#print(cL2)

cw3 = tf.Variable(tf.random_normal([3, 1, 128, 128], stddev=0.01))
cL3r = tf.nn.relu(tf.nn.conv2d(cL2, cw3, strides=[1, 1, 1, 1], padding='SAME')) # strides = [1, strideX, strideY, 1], padding=SAME or VALID
cL3p = tf.nn.max_pool(cL3r, ksize=[1, 5, 1, 1], strides=[1, 5, 1, 1], padding='SAME')
cL3 = tf.nn.dropout(cL3p, dropout_rate)

#print(cL3)

# Create the model
with tf.name_scope("input_layer"):
    w0 = tf.get_variable("input_layer_weight", shape=[867*128, 256], initializer=xavier_init(867*128, 256))
    #tf.histogram_summary("input_layer_weight", w0)    
    b0 = tf.Variable(tf.random_normal([256]))
    tf.histogram_summary("input_layer_bias", b0)
    cL2 = tf.reshape(cL3, tf.pack([batch_set_size, -1]))
    L0r = tf.nn.relu(tf.matmul(cL2, w0) + b0)
    L0 = tf.nn.dropout(L0r, dropout_rate)

with tf.name_scope("hidden_layer"):
    with tf.name_scope("layer_1"):
        w1 = tf.get_variable("layer_1_weight", shape=[256, 256], initializer=xavier_init(256, 256))
        #tf.histogram_summary("weight", w1)
        b1 = tf.Variable(tf.random_normal([256]))
        tf.histogram_summary("layer_1_bias", b1)

        L1r = tf.nn.relu(tf.matmul(L0, w1) + b1) # Hidden layer with RELU activation
        L1 = tf.nn.dropout(L1r, dropout_rate)

    with tf.name_scope("layer_2"):
        w2 = tf.get_variable("layer_2_weight", shape=[256, 256], initializer=xavier_init(256, 256))
        #tf.histogram_summary("weight", w2)
        b2 = tf.Variable(tf.random_normal([256]))
        tf.histogram_summary("layer_2_bias", b2)

        L2r = tf.nn.relu(tf.matmul(L1, w2) + b2)
        L2 = tf.nn.dropout(L2r, dropout_rate)

    with tf.name_scope("layer_3"):
        w3 = tf.get_variable("layer_3_weight", shape=[256, 256], initializer=xavier_init(256, 256))
        #tf.histogram_summary("weight", w3)
        b3 = tf.Variable(tf.random_normal([256]))
        tf.histogram_summary("layer_3_bias", b3)

        L3r = tf.nn.relu(tf.matmul(L2, w3) + b3) 
        L3 = tf.nn.dropout(L3r, dropout_rate)

with tf.name_scope("output_layer"):
    w4 = tf.get_variable("output_layer_weight", shape=[256, 61], initializer=xavier_init(256, 61))
    #tf.histogram_summary("weight", w4)    
    b4 = tf.Variable(tf.random_normal([61]))
    tf.histogram_summary("output_layer_bias", b4)
    
    hypothesis = tf.matmul(L3, w4) + b4
    prediction = tf.nn.softmax(hypothesis, name="prediction")
    tf.histogram_summary("prediction", prediction)

with tf.name_scope("activation"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, ratio), name="loss")
    tf.scalar_summary("loss", loss)

with tf.name_scope("train"):
    # Define loss and optimizer
    # AdadeltaOptimizer, AdagradOptimizer, MomentumOptimizer, FtrlOptimizer ...
    # train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    train_step = tf.train.RMSPropOptimizer(FLAGS.learning_rate, 0.9).minimize(loss)

def checkAccuracy():
    #correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(batch_labels, 1))
    correct_prediction = tf.less_equal(tf.abs(tf.argmax(prediction, 1) - tf.argmax(ratio, 1)), 0)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("accuracy= {:.9f}".format(accuracy.eval({chart: test_charts, ratio: test_labels, dropout_rate: 1, batch_set_size: 50})))
    print
    
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/trainer_logs", sess.graph)

# Train
startTime = time.time()

tf.initialize_all_variables().run()

ts.shuffle_batch_set()

np.set_printoptions(threshold=np.nan, linewidth=200)
#batch_charts, batch_labels = ts.next_batch(FLAGS.batch_size)
#print(batch_labels[:,30:])

# Test trained model
test_charts, test_labels = ts.next_test_batch(50)

total_batch = int(ts.num_set / FLAGS.batch_size)
for repeats in range(FLAGS.training_repeat):

    avg_loss = 0.0

    print("Epoch:", "{:04d} Training...".format(repeats+1))
    #for batch_set in range(total_batch):
    for batch_set in range(1):
        batch_charts, batch_labels = ts.next_batch(FLAGS.batch_size, True)
        train_step.run({chart: batch_charts, ratio: batch_labels, dropout_rate: FLAGS.dropout_rate, batch_set_size: FLAGS.batch_size})
        # compute average loss
        avg_loss += loss.eval({chart: batch_charts, ratio: batch_labels, dropout_rate: FLAGS.dropout_rate, batch_set_size: FLAGS.batch_size}) / 1 #total_batch
        if batch_set % 10 == 0 and batch_set > 2: 
            print("...{:.2f}%".format((batch_set+1)/total_batch*100.0))
            summary = sess.run(merged, feed_dict={chart: batch_charts, ratio: batch_labels, dropout_rate: FLAGS.dropout_rate, batch_set_size: FLAGS.batch_size})
            writer.add_summary(summary, repeats*5+batch_set)
    
    print("loss=", "{:.9f}".format(avg_loss))
    #checkAccuracy()
    #summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys, dropout_rate: FLAGS.dropout_rate})
    #writer.add_summary(summary, repeats)

checkAccuracy()

print("Elapsed time: {:.6f}".format(time.time() - startTime))
