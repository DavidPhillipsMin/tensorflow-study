#!/usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
flags.DEFINE_float('dropout_rate', 1, 'Dropout rate')
flags.DEFINE_integer('training_repeat', 15, 'Training repeat count')
flags.DEFINE_integer('batch_size', 100, 'Batch size per loop')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
tf.histogram_summary("images", x)

y_ = tf.placeholder(tf.float32, [None, 10])
tf.histogram_summary("label", y_)

dropout_rate = tf.placeholder(tf.float32)

# Create the model
with tf.name_scope("input_layer"):
    w0 = tf.get_variable("input_layer_weight", shape=[784, 256], initializer=xavier_init(784, 256))
    #tf.histogram_summary("input_layer_weight", w0)    
    b0 = tf.Variable(tf.random_normal([256]))
    tf.histogram_summary("input_layer_bias", b0)

    _L0 = tf.nn.relu(tf.matmul(x, w0) + b0)
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
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)


merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph)

# Train
tf.initialize_all_variables().run()

total_batch = int(mnist.train.num_examples / FLAGS.batch_size)

for repeats in range(FLAGS.training_repeat):

    avg_loss = 0.0

    for batch_set in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_step.run({x: batch_xs, y_: batch_ys, dropout_rate: FLAGS.dropout_rate})
        # compute average loss
        avg_loss += loss.eval({x: batch_xs, y_: batch_ys, dropout_rate: FLAGS.dropout_rate}) / total_batch
    
    print("Epoch:", "{:04d}".format(repeats+1), "loss=", "{:.9f}".format(avg_loss))        
    summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys, dropout_rate: FLAGS.dropout_rate})
    writer.add_summary(summary, repeats)

    
# Test trained model
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, dropout_rate: 1}))
