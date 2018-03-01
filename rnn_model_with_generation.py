from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class RnnModel:
    
    ####### one hidden layer
    '''
    def __init__(self, placeholder, min_step=10, state_dim=156):
        self.min_step = min_step
        self.state_dim = state_dim
        self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(state_dim)
        # Initial state of the LSTM memory.
        self.music_input = placeholder['music_input']
        self.batch_size = tf.shape(self.music_input)[0]
        self.time_range = tf.shape(self.music_input)[1]
        self.note_input_dim = tf.shape(self.music_input)[2]
        self.note_out_dim = tf.shape(self.music_input)[2]
        self.targets_pitch = self.music_input[:,-1,:]
        
        self.initial_state = self.rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.build()
        self.loss()
        self.inference()
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in self.grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[1]
        for _, var in clipped_grads_and_vars:
            print(var.name)
        self.train_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
    '''
    
    ###### two hidden layers
    def __init__(self, placeholder, min_step=10, state_dim=156,layer_num=2,keep_prob = 1.0):
        
        self.min_step = min_step
        self.state_dim = state_dim
        self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(state_dim)
        self.rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=self.rnn_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        self.rnn_cell = tf.contrib.rnn.MultiRNNCell([self.rnn_cell] * layer_num)
        # Initial state of the LSTM memory.
        self.music_input = placeholder['music_input']
        self.batch_size = tf.shape(self.music_input)[0]
        self.time_range = tf.shape(self.music_input)[1]
        self.note_input_dim = tf.shape(self.music_input)[2]
        self.note_out_dim = tf.shape(self.music_input)[2]
        self.targets_pitch = self.music_input[:,-1,:]
        self.initial_state = self.rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
        #self.initial_state = placeholder['state_input']
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
        ### 
        self.inference()

    def build(self):

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
        
        self.out_batch_size = tf.shape(self.outputs)[0]
        self.out_time_range = tf.shape(self.outputs)[1]
        self.out_note_input_dim = tf.shape(self.outputs)[2]                              
        #self.W = tf.Variable(tf.random_normal([self.state_dim,128]),validate_shape=False)
        
        #self.b = tf.Variable(tf.zeros([128]))
        
        self.weights = tf.Variable(tf.random_normal([self.state_dim, 156]))
        self.bias = tf.Variable(tf.zeros([156]))
        
        
        output = tf.reshape(self.outputs,[self.out_batch_size*self.out_time_range,self.out_note_input_dim ])
        self.logits = tf.matmul(output,self.weights)+ self.bias
        self.logits = tf.reshape(output,[self.out_batch_size,self.out_time_range,self.out_note_input_dim])
        #prob = tf.nn.sigmoid(self.logits)
        self.logits= self.logits[:, -1, :]
        # self.pred = prob


    def loss(self):
        
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets_pitch, logits=self.logits) 
        self.loss = tf.reduce_mean(loss)
        #self.loss = tf.nn.l2_loss(self.targets_pitch - self.pred)
        tf.summary.scalar('loss', self.loss)

    def inference(self,n=100,time_range=1,note_size=156,k=5):
        
        # run after build() in init
        # if inference() is run, music_input is of dimension 
        # [batch_size x existing time steps x note_input_dim]
        # the music output would be n+1 length

        # encode existing notes
        curr_state = self.initial_state
        
        for i in range(time_range):
            
            music_input_reshaped = tf.reshape(self.music_input, [self.batch_size * self.time_range, self.state_dim])
            self.music_input = tf.matmul(music_input_reshaped, self.embeddings)
            self.music_input = tf.reshape(self.music_input, [self.batch_size, self.time_range, self.state_dim])
            output, curr_state = self.rnn_cell(self.music_input[:,i,:],state=curr_state) # some other params for cell
            output = tf.reshape(output,[self.batch_size * self.time_range,self.note_input_dim])
            logits = tf.matmul(output,self.weights)+ self.bias
            logits = tf.reshape(logits,[self.batch_size,self.time_range,self.note_input_dim])
            output = tf.nn.sigmoid(logits)   # multiple keys could be pressed at the same time
            ## code for at most 5 note
            '''
            k_index = tf.nn.top_k(output[-1], k, name=None)
            indices = k_index.indices
            values=[1.0]
            shape = [1,1,128]
            delta = tf.SparseTensor(indices, values, shape)
            c = tf.constant(0.0,shape=[1,1,128])
            output = c + tf.sparse_tensor_to_dense(delta)
            '''
            '''
            ## code for one note 
            index = (tf.argmax(output[-1], 1))
            indices = [[0,0,index[0]]]
            values = [1.0]
            shape = [1,1,156]
            delta = tf.SparseTensor(indices, values, shape)
            c = tf.constant(0.0,shape=[1,1,156])
            output = c + tf.sparse_tensor_to_dense(delta)
            '''
            ### set a probability threshold
            c = tf.constant(0.999999,shape = [1,1,156])
            output = tf.ceil(output - c)
            
        outputs = []
        outputs.append(output)
        # predict next n steps
        # n can be passed in via placeholder, or just a constant
        for i in range(n):
            music_input_reshaped = tf.reshape(self.music_input, [self.batch_size * self.time_range, self.state_dim])
            self.music_input = tf.matmul(music_input_reshaped, self.embeddings)
            self.music_input = tf.reshape(self.music_input, [self.batch_size, self.time_range, self.state_dim])
            output, curr_state = self.rnn_cell(output[:,-1,:], state=curr_state) # some other params
            output = tf.reshape(output,[self.batch_size * self.time_range,self.note_input_dim])
            logits = tf.matmul(output,self.weights)+ self.bias
            logits = tf.reshape(logits,[self.batch_size,self.time_range,self.note_input_dim])
            output = tf.nn.sigmoid(logits)
            # code for one note only
            '''
            index = (tf.argmax(output[-1], 1))
            indices = [[0,0,index[0]]]
            values = [1.0]
            shape = [1,1,156]
            delta = tf.SparseTensor(indices, values, shape)
            c = tf.constant(0.0,shape=[1,1,156])
            output = c + tf.sparse_tensor_to_dense(delta)
            '''
            c = tf.constant(0.999999,shape = [1,1,156])
            output = tf.ceil(output - c)
            outputs.append(output)
            
        self.outputs_vec = outputs   
       
