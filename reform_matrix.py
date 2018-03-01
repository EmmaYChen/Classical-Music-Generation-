'''
--- reform_matrix.py
main():
    Goal: we want to convert midi file into a numpy array that can be fed into our RNN model
    Note: 1. we use `midiToNoteStateMatrix` imported from `midi_to_statematrix.py`
            `midi_to_statematrix.py` is obtained from:
                https://github.com/hexahedria/biaxial-rnn-music-composition
          2. `midi_to_statematrix.py` will generate a list for dimension (time_step, 78, 2) for
            each midi file. We need to flatten this list into numpy array and concatenate it with 
            other numpy arrays we've obtained
          3. the new numpy array is of shape (num_of_files, time_steps, 156)
          4. it will be stored as "dev.npy" or "train.dev" or any other name we want
'''

import numpy as np
from midi_to_statematrix import midiToNoteStateMatrix

import os
##### lowbound pitch :24
##### upperbound pitch: 102
##### pitch range = upperbound_pitch- lowerbound_pitch
PITCH_RANGE = 78
path = './dev/'


def append_zeros(d, max_len, num_file):
    result = np.zeros((num_file, max_len,2*PITCH_RANGE))
    ctr = 0
    for key, lst in d.iteritems():
        old_array = np.array(lst)
        result[ctr,0:old_array.shape[0],:] = old_array
        ctr += 1
    return result        

def main():
    entire_dict = {}
    max_len = 0
    file_count = 0
    for fname in os.listdir(path):
        file_count += 1
        if fname[-4:] not in ('.mid','.MID'):
            continue

        name = fname[:-4]
        reformed_list = []
        state_matrix = midiToNoteStateMatrix(os.path.join(path, fname))
        print "Loaded {}".format(name)
        for state in state_matrix:
            sub_list = []
            for i in range(PITCH_RANGE):
                sub_list.append(state[i][0])
                sub_list.append(state[i][1])
            reformed_list.append(sub_list)
        print (np.array(reformed_list).shape[0])
        if np.array(reformed_list).shape[0] > max_len:
            max_len = np.array(reformed_list).shape[0]
        entire_dict[name] = reformed_list
    print (max_len)
    reformed_matrix = append_zeros(entire_dict, max_len, file_count)
    np.save('dev_set', reformed_matrix)
    print "Successfully build training set with {} examples".format(file_count)

if __name__ == "__main__":
    main()
