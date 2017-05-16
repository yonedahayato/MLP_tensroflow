# -*- coding: utf-8 -*-

import tensorflow as tf
import sys

# data mnist
train_size, test_size = 6000, 1000
from mnist import download_mnist, load_mnist, key_file
download_mnist()
X_train = load_mnist(key_file["train_img"])[8:train_size+8, :]
X_test = load_mnist(key_file["test_img"], )[8:test_size+8,:]
y_train = load_mnist(key_file["train_label"], 1)[:train_size,0]
y_test = load_mnist(key_file["test_label"], 1)[:test_size,0]


labels = tf.placeholder(tf.int64, [None])
y_ = tf.one_hot(labels, depth=10, dtype=tf.float32)
print(y_train[:10])
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(y_, feed_dict={labels: y_train[:10]}))
