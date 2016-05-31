#!/usr/bin/python2

import sys
import tensorflow as tf
import numpy as np

data = np.loadtxt('train.txt', unpack=True, dtype='float32')

input_data  = np.transpose(data[0:3])
label_data  = np.transpose(data[3:])

DATA            = tf.placeholder('float', [None, 3])
LABEL           = tf.placeholder('float', [None, 3])
# model weight
WEIGHT          = tf.Variable(tf.zeros([3, 3]))

# construct model
hypothesis = tf.nn.softmax(tf.matmul(DATA, WEIGHT)) #Softmax algorithm (matrix shape must be [batch_size, classes num])

# cross entropy
element_sum = -tf.reduce_sum(LABEL * tf.log(hypothesis), reduction_indices = 1)
cost = tf.reduce_mean(element_sum)

# gradient Descent
learning_rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(10001):
    sess.run(optimizer, feed_dict={DATA:input_data, LABEL:label_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={DATA:input_data, LABEL:label_data})#, sess.run(WEIGHT)

sess.close()
