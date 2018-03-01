from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from rnn_model_1_hidden_layer import RnnModel

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
   
    # TODO: fix batch size
    batch_size=channelMatrix.shape[0]
    music_input= np.zeros((batch_size,time_size,note_size),dtype=np.float)
    
    #print(music_input.shape)
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
    # channel 0, timestep from 1:50
    originalMatrix = np.load('1750_train_set.npy')
    print(originalMatrix.shape)
    channelMatrix = originalMatrix[:,:,:]
    music_input = channelMatrix
    
    # dev_set
    dev_matrix = np.load('1750_dev_set.npy')
    dev_set = dev_matrix[:,:,:]
    dev_input = dev_set
    
    # 
    # tranpose the matrix, now its dimension is timestep*note_size
    #channelMatrix  = np.transpose(channelMatrix )
    #print(channelMatrix.shape)
    
    #expand dimension, now batch_size*time*note_size
    #channelMatrix = np.expand_dims(channelMatrix , axis=0)
    #print(channelMatrix.shape)
    
    #expand note_size from 5*1 to 128*1
    # music_input = matrix_to_input(channelMatrix)
    #print(music_input.shape)
    #print((music_input.shape)[0])
    #print((music_input.shape)[1])
    #print((music_input.shape)[2])
    
    seq_len = 50
    placeholder = tensorflow_music_input(music_input, seq_len)
    model = RnnModel(placeholder) 
    config = tf.ConfigProto(log_device_placement=False)
    

    # saver
    saver = tf.train.Saver()

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save_path = os.path.join(save_dir, 'epoch_validation')
    # save_file = 'model.ckpt'

    # start tensorflow session
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver.save(sess, './test_model')

    print( tf.trainable_variables())
    merged = tf.summary.merge_all()
    #summary_writer
    max_value = originalMatrix.shape[1] - originalMatrix.shape[1] % 5000 - 1
    '''
    for i in range(18):
        left = 5000*i
        right = 5000*(i+1)
        channelMatrix = originalMatrix[:,left:right,:]
        music_input = matrix_to_input(channelMatrix)
        _,_, loss,accuracy = sess.run([merged,model.train_op, model.loss, model.accuracy], feed_dict={placeholder['music_input']: music_input})
    
        print(loss)
        print(accuracy)
    '''
    # summary_writer = tf.summary.FileWriter('/Users/yuxing/Desktop/Stanford/Academic/2017-2018/CS\ 229/plots/1', sess.graph)
    #dev_writer = tf.summary.FileWriter('/Users/yuxing/Desktop/Stanford/Academic/2017-2018/CS\ 229/plots/2', sess.graph) 
   
    num_epoch = 30
    initial_flag_train = True
    # initial_flag_dev = True

    step = 0

    print(music_input.shape[2])

    # min_loss = 2

    # first state
    state = np.zeros((music_input.shape[0], state_dim))

    for epoch in range(num_epoch):
        print ('++++++++++++++++')
        print ("Current epoch: ", epoch)
        saver = tf.train.import_meta_graph('test_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        # random a starting point
        i = randint(0, 100)
        print (i)

        for iter in range(2000):
            # check the starting point
            if i > channelMatrix.shape[1] - seq_len:
                break
            batch_input = music_input[:,i:i+seq_len,:]

            # update i for next step
            i += seq_len
    
            # _,_, loss, grad, pred, input = sess.run([merged, model.train_op, model.loss, model.grad,
            #         model.pred, model.music_input], feed_dict={placeholder['music_input']: batch_input})
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
        saver.save(sess, './test_model')
        
        print ('%-------%')
        total_loss = 0
        i = randint(0, 100)
        for iter in range(2000):
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
    

            #print(accuracy)
        print("Loss for epoch %d = %f" % (epoch,loss)) #use this if we wanna generate a plot of loss vs. epoch
    # print (min_loss)
    # saver.save(sess, './hinge_model')
    print("Done Training")
    
    
if __name__ == "__main__":
    tf.app.run()
