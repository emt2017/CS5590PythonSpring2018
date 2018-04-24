#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from numpy import array

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
#tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")


tf.flags.DEFINE_string("positive_data_file", "./data/textCatCollection/all-exchanges-strings.lc.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/textCatCollection/all-orgs-strings.lc.txt", "Data source for the negative cool data.")
tf.flags.DEFINE_string("positive_data_file1", "./data/textCatCollection/all-people-strings.lc.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file2", "./data/textCatCollection/all-places-strings.lc.txt", "Data source for the negative cool data.")
tf.flags.DEFINE_string("positive_data_file3", "./data/textCatCollection/all-topics-strings.lc.txt", "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 207, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
#assigns classes
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file,FLAGS.positive_data_file1, FLAGS.negative_data_file2,FLAGS.positive_data_file3)
f = open("xtraintext",'w')#make sure getting correct matrices
f.write(str(x_text))
f.close()
# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]



# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
f = open("xtraintext", 'w')  # make sure getting correct matrices
f.write(str(x_shuffled))
f.close()

x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

a=np.asarray(x_train)
np.savetxt("xtraintext.csv",a,delimiter=",")

x_train = x_train[:,:]
y_train = y_train[:,:]
print(x_train)
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================


#y_train = np.transpose(y_train[:,:])

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],#changed this !!!!!!!
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)#ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """

            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)



        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }

            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        #x_train = np.transpose(x_train[:, :])

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        #x_train = np.transpose(x_train[:, :])
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            #x_batch = ([6, 0, 0, 0],[37,  0,  0,  0], [67,  0,  0,  0],[68,  0,  0,  0], [31,  0,  0,  0], [82,  0,  0,  0],[24,  0,  0,  0], [40, 41,  0,  0],[21,  0,  0,  0], [15,  0,  0,  0],[18,  0,  0,  0],[55,  0,  0,  0],[83,  0,  0,  0], [77,  0,  0,  0],[32,  0,  0,  0],[96,  0,  0,  0],[53,  0,  0,  0],[54,  0,  0,  0], [44,  0,  0,  0],[84, 86,  0,  0], [20,  0,  0,  0], [60,  0,  0,  0], [94,  0,  0,  0],[73,  0,  0,  0], [47,  0,  0,  0],[13,  0,  0,  0],[64,  0,  0,  0], [91,  0,  0,  0], [84, 85,  0,  0], [95,  0,  0,  0], [28,  0,  0,  0], [3, 0, 0, 0], [45,  0,  0,  0], [34,  0,  0,  0],[46,  0,  0,  0], [43,  0,  0,  0], [52,  0,  0,  0], [81,  0,  0,  0], [51,  0,  0,  0], [23,  0,  0,  0], [39,  0,  0,  0],[58,  0,  0,  0],[38,  0,  0,  0],[22,  0,  0,  0],[8, 0, 0, 0],[48,  0,  0,  0],[19,  0,  0,  0],[65,  0,  0,  0],[57,  0,  0,  0],[12,  0,  0,  0],[70, 72,  0,  0],[49,  0,  0,  0],[79,  0,  0,  0],[40, 42,  0,  0],[92,  0,  0,  0],[5, 0, 0, 0],[59,  0,  0,  0], [17,  0,  0,  0],[4, 0, 0, 0], [26,  0,  0,  0], [7, 0, 0, 0],[33,  0,  0,  0], [74,  0,  0,  0], [25,  0,  0,  0])
            #x_batch = array(x_batch).reshape(1,64,4)

            #x_train = np.transpose(x_train[:, :])
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
