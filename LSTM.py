#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 11:19:40 2017

@author: victor
"""
import numpy as np
import tensorflow as tf
from buffer import Buffer
import query_btc as qbtc

window = 1  # window size of the rolling mean used for target
batch_size = 50
n_inputs = 4
n_outputs = 1
n_nodes = 300
n_layers = 1
period = 50
start_learning_rate = 0.01
n_steps = 5000
decay = 0.01
print('starting program...\n')

data_in, data_targ, val_in, val_targ = qbtc.generate_data(window)

#data_in, data_targ, val_in, val_targ = qbtc.gen_fake_data2()

buff = Buffer(data_in, data_targ, val_in, val_targ)

def lstm_cell(n_nodes, keep_prob):
    cell = tf.contrib.rnn.BasicLSTMCell(n_nodes, activation=tf.nn.relu)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    
inputs = tf.placeholder(tf.float32, shape=[batch_size, period, n_inputs])
targets = tf.placeholder(tf.float32, shape=[batch_size, period, n_outputs])
keep_prob = tf.placeholder(tf.float32, shape=[])

stacked_cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell(n_nodes, keep_prob) for _ in range(n_layers)])

W = tf.Variable(tf.random_normal([n_nodes,n_outputs]))
b = tf.Variable(tf.zeros(shape=[n_outputs]))

input_list = tf.unstack(inputs, axis=1)
zero_state = stacked_cell.zero_state(batch_size, dtype=tf.float32)

with tf.variable_scope("LSTM"):
    outputs, state = tf.contrib.rnn.static_rnn(
            stacked_cell, input_list, initial_state=zero_state)

outputs = tf.stack([tf.matmul(x,W)+b for x in outputs], axis=1)

loss = tf.reduce_sum(tf.squared_difference(outputs, targets))

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                           n_steps, decay)

optimizer = tf.train.AdamOptimizer(learning_rate)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad,-1.0,1.0), var) for grad, var, in gvs]
train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

val_list = tf.unstack(buff.v_in
                      .reshape(1,-1, n_inputs)
                      .astype(np.float32), axis=1)

zero_state = stacked_cell.zero_state(1, dtype=tf.float32)

with tf.variable_scope("LSTM", reuse=True):
    val_outputs, val_state = tf.contrib.rnn.static_rnn(
            stacked_cell, val_list, initial_state=zero_state)

val_outputs = tf.stack([tf.matmul(x,W)+b for x in val_outputs], axis=1)

loss_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(n_steps):
        train_in, train_targ = buff.rand_batch(batch_size, period)
        
        feed_dict = {
                inputs: train_in,
                targets: train_targ,
                keep_prob: 1}
        
        _, train_loss = sess.run([train_op, loss], feed_dict=feed_dict)
        
        if i % 100 == 1:
            print('i = {}, error = {}'.format(i,train_loss))
        
        loss_list.append(train_loss)
        
    val_pred = sess.run(val_outputs, feed_dict={keep_prob: 1})
    
print('done!')
        
        
        
        
        
        