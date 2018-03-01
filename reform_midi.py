'''
--- reform_midi.py
main():
    Goal: we want to reconstruct midi file from numpy array of shape (time_step+1)*156*1
    Note: we use `noteStateMatrixToMidi` imported from `midi_to_statematrix.py`
        `midi_to_statematrix.py` is obtained from:
                https://github.com/hexahedria/biaxial-rnn-music-composition
    Approach: we first convert the numpy array of shape ((time_step+1), 156, 1) to a numpy
                array of shape ((time_step+1), 78, 2), which then becomes readable by
                the `noteStateMatrixToMidi` function
              the new midi file will be stored as "xxx.mid" in the working directory
'''
import numpy as np
from midi_to_statematrix import noteStateMatrixToMidi

time_step = 100 # change this value based on the shape of the input array

# Music generated by RNN is stored as 'predict.npy'
# Transfer the numpy array back a matrix that could be processed by noteStateMatrixToMidi
def main():
    input_array = np.load('predict.npy')
    print (input_array.shape)
    # convert the numpy array of dimension (101 * 156 *1) to (101 * 78 * 2)
    result = np.zeros((time_step+1,78,2)) 
    for j in range(time_step+1):
        i = 0
        while i < 156:
            if input_array[j,i] == 1:
                result[j,i/2,0] = 1
                if input_array[j,i+1] == 1:
                    result[j,i/2,1] = input_array[j,i+1]
            i = i + 2
    noteStateMatrixToMidi(result, name="ex6") # transform back to midi-file

if __name__ == "__main__":
    main()