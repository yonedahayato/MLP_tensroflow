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

def MLP(alpha, lr, layer1, layer2, layer3):
    X = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.int32, [None, ])
    y_ = tf.one_hot(label, depth=10, dtype=tf.float32)

    w_0 = tf.Variable(tf.random_normal([784, int(layer1)], mean=0.0, stddev=0.05))
    b_0 = tf.Variable(tf.zeros([int(layer1)]))
    h_0 = tf.sigmoid(tf.matmul(X, w_0) + b_0)

    w_1 = tf.Variable(tf.random_normal([int(layer1), int(layer2)], mean=0.0, stddev=0.05))
    b_1 = tf.Variable(tf.zeros([int(layer2)]))
    h_1 = tf.sigmoid(tf.matmul(h_0, w_1) + b_1)

    w_2 = tf.Variable(tf.random_normal([int(layer2), int(layer3)], mean=0.0, stddev=0.05))
    b_2 = tf.Variable(tf.zeros([int(layer3)]))
    h_2 = tf.sigmoid(tf.matmul(h_1, w_2) + b_2)

    w_o = tf.Variable(tf.random_normal([int(layer3), 10], mean=0.0, stddev=0.05))
    b_o = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(h_2, w_o) + b_o)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))


    L2_sqr = tf.nn.l2_loss(w_0) + tf.nn.l2_loss(w_1) + tf.nn.l2_loss(w_2)

    loss = cross_entropy + alpha + L2_sqr
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print("Training...")
        for i in range(int(6000/50)-1):
            batch_x, batch_y = X_train[(50*i):(50*(i+1)),:], y_train[(50*i):(50*(i+1))]
            print(batch_x.shape)
            print(batch_y.shape)
            #sys.exit()
            train_step.run({X: batch_x, y_: batch_y})
            if i % int((int(6000/50)-1)/10):
                train_accuracy = accuracy.eval({X: batch_x, y_: batch_y})
                print(" %6d %6.3f" % (i, train_accuracy))
        print("accuracy %6.3f" % accuracy.eval({X: X_test, y_: y_test}))

if __name__ == "__main__":
    MLP(0.001, 0.001, 50, 50, 50)
