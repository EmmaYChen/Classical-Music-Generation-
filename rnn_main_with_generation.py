from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from rnn_model_with_generation import RnnModel

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import math
from random import randint


def tensorflow_music_input(music_input, seq_len):
    
    t_music_input = tf.placeholder(tf.float32,shape=[None,None,music_input.shape[2]],
                                   name='music_input')
    return {'music_input':t_music_input}


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
    return music_input 
    
    
def main(argv=None):  
    # channel 0, timestep from 1:50
    originalMatrix = np.load('training_set.npy')
    channelMatrix = originalMatrix[:,:,:]
    music_input = channelMatrix
    
    
    seq_len = 50
    placeholder = tensorflow_music_input(music_input, seq_len)
    model = RnnModel(placeholder) 
    config = tf.ConfigProto(log_device_placement=False)
    
    # start tensorflow session
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print( tf.trainable_variables())
    merged = tf.summary.merge_all()
    
    #summary_writer
    max_value = originalMatrix.shape[1] - originalMatrix.shape[1] % 5000 - 1
    
    saver = tf.train.Saver(tf.global_variables())
    
    #################################### train ########################################
    '''

    num_epoch =4
    print(music_input.shape[2])
    for epoch in range(num_epoch):
        for iter in range(200000):
            i = randint(0, channelMatrix.shape[1] - seq_len)
            batch_input = music_input[:,i:i+seq_len,:]
            # _,_, loss, grad, pred, input = sess.run([merged, model.train_op, model.loss, model.grad,
            #         model.pred, model.music_input], feed_dict={placeholder['music_input']: batch_input})
            _,_, loss, grad, input = sess.run([merged, model.train_op, model.loss, model.grad,
                    model.music_input], feed_dict={placeholder['music_input']: batch_input})
            if iter % 1000 == 0:
                print('Iter ', iter, ': loss=', loss)
                curr_grad = LA.norm(grad)
                print(curr_grad)
                if curr_grad < 1e-6:
                    break

            #print(accuracy)
        print("Loss for epoch %d = %f" % (epoch,loss)) #use this if we wanna generate a plot of loss vs. epoch
    # Save model
    saver.save(sess, './my_model')
    print("Done Training")
    
    '''
    ##################################### generate #########################################
    # music generation
    # test initialization
    seq_len = 1
    state_dim= 156   
    batch_size = 1
    time_range = 1
    note_input_dim = 156
    
    # n is the length of result music for each iteration
    n = 1
    
    # restore the trained model
    saver = tf.train.import_meta_graph('3_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    
    # initialize, given note
    seq = np.zeros((n,note_input_dim))
    seq[0][72] = 1.0
    seq_len = 1
    start = np.expand_dims(seq[0], axis=0)
    start = np.expand_dims(start, axis=0)
    
    output = sess.run([model.outputs_vec], feed_dict={placeholder['music_input']: start})
    print(output)
    print('Done, start rescaling!')
    # turn output into a vector of  1 * 5 * time)step

    result = np.zeros((101,156))
    i = 0
    for out in output:
        for arr in out:
            result[i,:] = np.squeeze(arr, axis = 0)
            i += 1
    np.save('predict', result)
    
if __name__ == "__main__":
    tf.app.run()
