from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class RnnModel:
    
    def __init__(self, placeholder, min_step=10, state_dim=156,layer_num=3,keep_prob = 1.0):
        
        self.min_step = min_step
        self.state_dim = state_dim
        self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(state_dim)
        self.rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=self.rnn_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        self.rnn_cell = tf.contrib.rnn.MultiRNNCell([self.rnn_cell] * layer_num)
        
        # Initial state of the LSTM memory.
        # batch_size: # examples in mini-batch train 
        # time range: default time_range for training is 50; for music generation, default time_range is 1.
        # note_input_dim: music input for a certain time step is a vector 1*156
        # Music input dimension batch_size * time_range
        self.music_input = placeholder['music_input']
        self.batch_size = tf.shape(self.music_input)[0]
        self.time_range = tf.shape(self.music_input)[1]
        self.note_input_dim = tf.shape(self.music_input)[2]
        self.note_out_dim = tf.shape(self.music_input)[2]
        self.targets_pitch = self.music_input[:,-1,:]
        self.initial_state = self.rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
        
        self.build()
        self.loss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in self.grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[1]
        for _, var in clipped_grads_and_vars:
            print(var.name)
        # with tf.control_dependencies([self.assign_op]):
        #     self.train_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.train_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def build(self):
        # music input pre-preprocessing, find embedding matrix that could made the input not so sparse
        self.embeddings = tf.get_variable('note_embedding', shape=(self.state_dim, self.state_dim),
                                                 dtype=tf.float32, 
                                                 initializer=tf.contrib.layers.xavier_initializer())
        music_input_reshaped = tf.reshape(self.music_input, [self.batch_size * self.time_range, self.state_dim])
        self.music_input = tf.matmul(music_input_reshaped, self.embeddings)
        self.music_input = tf.reshape(self.music_input, [self.batch_size, self.time_range, self.state_dim])

        # The value of state is updated after processing each batch of notes.
        # The LSTM output can be used to make next pitch predictions
        self.outputs, self.state = tf.nn.dynamic_rnn(self.rnn_cell, self.music_input[:, :-1, :],
                                      initial_state=self.initial_state,
                                      dtype=tf.float32)
        # get output dimension
        self.out_batch_size = tf.shape(self.outputs)[0]
        self.out_time_range = tf.shape(self.outputs)[1]
        self.out_note_input_dim = tf.shape(self.outputs)[2]                              
        
        # connecting the input layer with output layer
        weights = tf.Variable(tf.random_normal([self.state_dim, 156]))
        bias = tf.Variable(tf.zeros([156]))
        
        # reshape the output to match the condition to apply multiplication
        output = tf.reshape(self.outputs,[self.out_batch_size*self.out_time_range,self.out_note_input_dim ])
        self.logits = tf.matmul(output,weights)+ bias
        self.logits = tf.reshape(output,[self.out_batch_size,self.out_time_range,self.out_note_input_dim])
        # prob = tf.nn.sigmoid(self.logits)
        self.logits= self.logits[:, -1, :]
        # self.pred = prob

    def loss(self):
       
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets_pitch, logits=self.logits) 
        #self.loss = tf.losses.hinge_loss(labels=self.targets_pitch,logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        #self.loss = tf.nn.l2_loss(self.targets_pitch - self.pred)
        tf.summary.scalar('loss', self.loss)
