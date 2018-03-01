from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from rnn_model_2_hidden_layers import RnnModel

import os
import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import math
from random import randint

state_dim = 156

def tensorflow_music_input(music_input, seq_len):
    
    t_music_input = tf.placeholder(tf.float32,shape=[None,seq_len,music_input.shape[2]],
                                   name='music_input')

    state_input = tf.placeholder(tf.float32,shape=[None, state_dim],
                                   name='state_input')

    return {'music_input':t_music_input, 'state_input': state_input}


def matrix_to_input(channelMatrix):
    # note vector which indicate which notes are played in a certain time step
    note_start = 0
    note_end = 128
    note_size= note_end - note_start
    
    # total time step
    time_size = channelMatrix.shape[1]
    #print(time_size)
    curr_vector = np.zeros((5,),dtype=np.int32)
    #print(curr_vector)
    batch_size=channelMatrix.shape[0]
    music_input= np.zeros((batch_size,time_size,note_size),dtype=np.float)
  
    # resize the note vector to 1*note_size
    for m in range(batch_size):
        for i in range(time_size):  
            for j in range(5):  
                curr_vector[j]= channelMatrix[m][i][j].astype(np.int32)
                #print(channelMatrix[m][i][j])
                if(curr_vector[j]==0):
                    continue
                music_input[m][i][curr_vector[j]-note_start-1] = 1.0 
    #print(curr_vector)
    #print(music_input[m][i])
    return music_input 
      
def main(argv=None):  
    originalMatrix = np.load('1750_train_set.npy')
    print(originalMatrix.shape)
    channelMatrix = originalMatrix[:,:,:]
    music_input = channelMatrix
    
    # dev_set
    dev_matrix = np.load('1750_dev_set.npy')
    dev_set = dev_matrix[:,:,:]
    dev_input = dev_set
    
    # parameters initialization
    num_epoch = 10
    step = 0
    seq_len = 50
    placeholder = tensorflow_music_input(music_input, seq_len)
    model = RnnModel(placeholder) 
    config = tf.ConfigProto(log_device_placement=False)
    initial_flag_train = True
    # saver
    saver = tf.train.Saver()
    
    # start tensorflow session
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    save_path = saver.save(sess, "/tmp/model.ckpt")

    #print( tf.trainable_variables())
    merged = tf.summary.merge_all()
    max_value = originalMatrix.shape[1] - originalMatrix.shape[1] % 5000 - 1
    #print(music_input.shape[2])

    # first state
    state = np.zeros((music_input.shape[0], state_dim))
    ############################ training set ###########################3
    for epoch in range(num_epoch):
        print ('++++++++++++++++')
        print ("Current epoch: ", epoch)
        # saver = tf.train.import_meta_graph('3_model.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('./'))
        saver.restore(sess, "/tmp/model.ckpt")
        # random a starting point
        i = randint(0, 10)
        print (i)

        for iter in range(2000):
            # check the starting point
            if i > channelMatrix.shape[1] - seq_len:
                break
            batch_input = music_input[:,i:i+seq_len,:]

            # update i for next step
            i += seq_len
    
            # run session 
            val,_, loss, grad, input, initial_state = sess.run([merged, model.train_op, model.loss, model.grad,
                    model.music_input, model.initial_state], feed_dict={placeholder['music_input']: batch_input,
                                                                placeholder['state_input']: state})

            #summary_writer.add_summary(val,step)
            # step += 1
            if iter % 1000 == 0 or i >= channelMatrix.shape[1] - seq_len:
                if initial_flag_train:
                    # summary_writer.add_summary(val,0)
                    initial_flag_train = False
                print('Iter ', iter, ': loss=', loss)
                #print(initial_state)
                curr_grad = LA.norm(grad)
                print(curr_grad)
        
        save_path = saver.save(sess, "/tmp/model.ckpt")  # save the path for generation
        
        #################### dev set loss calculation ###########################
        
        print ('%-------%')
        total_loss = 0
        i = randint(0, 100)    # choose a random start point
        for iter in range(2000):  # feed a sequence of length seq into the placeholder
            if i > dev_input.shape[1] - seq_len:
                break
            batch_input = dev_input[:,i:i+seq_len,:]

            i += seq_len
            # _,_, loss, grad, pred, input = sess.run([merged, model.train_op, model.loss, model.grad,
            #         model.pred, model.music_input], feed_dict={placeholder['music_input']: batch_input})
            val,loss, grad, input = sess.run([merged, model.loss, model.grad,
                    model.music_input], feed_dict={placeholder['music_input']: batch_input,placeholder['state_input']: state})
            total_loss += loss
            if iter % 1000 == 0 or i >= dev_input.shape[1] - seq_len:
                print('Iter ', iter, ': loss=', loss)
                curr_grad = LA.norm(grad)
                print(curr_grad)
                avg_loss = total_loss
                if iter > 0:
                    avg_loss = total_loss / float(iter)
                #dev_writer.add_summary(val,epoch+1)
                print ("avg_loss is: ", avg_loss)
        
    
    print("Done Training")
    
    
if __name__ == "__main__":
    tf.app.run()
