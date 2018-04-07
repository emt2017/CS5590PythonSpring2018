""" Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from
the number of fire in the city of Chicago
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

#https://raw.githubusercontent.com/tensorflow/models/master/official/wide_deep/wide_deep_test.csv
DATA_FILE = 'data/wide_deep_test.csv'
#https://web.stanford.edu/class/cs20si/2017/lectures/notes_03.pdf


import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#load the MNIST dataset
MNIST = input_data.read_data_sets("/data/mnist", one_hot=True)

# the arrays carry the parameters for each run
lr = [0.01,0.15, 1, 0.01, 0.01, 0.01, 0.01, 0.15, 0.15]
bs = [128, 128, 128, 10, 200, 128, 128, 10, 128]
ne = [25, 25, 25, 25, 25, 1, 100, 100, 100]
#clear the file for next run
open('scratch111111111111', 'w').close()
#open file to write in order to store results for each run
with open('scratch111111111111', 'w') as f:
    #the for loop iterates through the parameter arrays and runs logistic regression on each set
    for i in range(0,9):
        # the parameters are defined by iterating through the array of numbers
        learn_rate = lr[i]
        batch_siz = bs[i]
        epoch = ne[i]
        #this creates the placeholders for features
        #each one of this pictures are held in an array of 1x784 known as a tensor
        X = tf.placeholder(tf.float32, [batch_siz, 784])
        Y = tf.placeholder(tf.float32, [batch_siz, 10])
        # Creates and initializes the weights and the bias's for the logistic regression
        w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
        b = tf.Variable(tf.zeros([1, 10]), name="bias")

        # predicts the model
        logits = tf.matmul(X, w) + b
        # defines the loss function using softmax cross entropy
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=Y)
        loss = tf.reduce_mean(entropy) #computes mean cross entropy
        #Trains the model
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            batch = int(MNIST.train.num_examples/batch_siz)
            for i in range(epoch):
                for _ in range(batch):
                    X_batch, Y_batch = MNIST.train.next_batch(batch_siz)
                    sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})

        # test the model
            batch = int(MNIST.test.num_examples/batch_siz)
            total_correct_preds = 0
            for i in range(batch):
                X_batch, Y_batch = MNIST.test.next_batch(batch_siz)
                _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
        feed_dict={X: X_batch, Y:Y_batch})
                preds = tf.nn.softmax(logits_batch)
                correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
                accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
                total_correct_preds += sess.run(accuracy)

        #calculate the accuracy
        finalAccuracy = format(total_correct_preds/MNIST.test.num_examples)

        #write accuracy to the file
        f.write('Run: ' + finalAccuracy + '\n    learn_rate = '
                + str(learn_rate) + '\n    batch_siz = ' + str(batch_siz)
                + '\n    epoch = ' + str(epoch)+'\n')
