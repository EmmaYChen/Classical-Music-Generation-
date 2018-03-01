import os
import tensorflow as tf
import numpy as np
import operator
import magenta
from magenta.music import note_sequence_io
from magenta.music import encoder_decoder
from magenta.music import testing_lib
from magenta.protobuf import music_pb2

'''
    after converting array to sequence, use sequence_proto_to_midi_file to get midi file
'''

def compute_time(total_time, time_step, start_pos, end_pos):
    start_time = total_time * (float(start_pos) / time_step)
    end_time = total_time * (float(end_pos) / time_step)
    return (start_time, end_time)

def store_in_midi(input_left, input_right):
    '''
        By our assumption: batch_size should be 1, note_size should be 5
        ? check time_step of two channels, they should be the same
    '''
    _, time_step = input_left.shape
    print(time_step)

    # TODO: replace velocity and sequence_id with more reasonable values?
    qpm = 120
    velocity = 60
    sequence_id = 1
    sequence = music_pb2.NoteSequence()
    
    # time_step = total_time/(1/qpm)
    total_time = time_step/qpm
    sequence.total_time = total_time
    sequence.id = str(sequence_id)

    # insert tempo into sequence.tempos, since qpm doesn't change over time, 
    # there should be only one tempo in tempos
    tempo = sequence.tempos.add()
    tempo.qpm = qpm
    tempo.time = 0.0
    
    # insert notes
    # form a list of tuples
    # (pitch, velocity, start_time, end_time)
    notes_left = []
    notes_right = []
    for layer in range(5):
        music_left = input_left[layer,:]
        music_right = input_right[layer,:]
        # left
        pos = 0
        prev_pitch = music_left[0]
        for i in range(1,time_step):
            curr_pitch = music_left[i]
            if curr_pitch != prev_pitch:
                if prev_pitch != 0: # append only when the pich is non-zero
                    start_time, end_time = compute_time(total_time, time_step, pos, i)
                    notes_right.append((int(prev_pitch), velocity, start_time, end_time))
                # reset tracking numbers
                pos = i
            prev_pitch = curr_pitch
        if prev_pitch != 0:
            start_time, end_time = compute_time(total_time, time_step, pos, time_step)
            notes_right.append((int(prev_pitch), velocity, start_time, end_time))
        # right
        pos = 0
        prev_pitch = music_right[0]
        for i in range(1,time_step):
            curr_pitch = music_right[i]
            if curr_pitch != prev_pitch:
                if prev_pitch != 0: # append only when the pich is non-zero
                    start_time, end_time = compute_time(total_time, time_step, pos, i)
                    notes_right.append((int(prev_pitch), velocity, start_time, end_time))
                # reset tracking numbers
                pos = i
            prev_pitch = curr_pitch
        if prev_pitch != 0:
            start_time, end_time = compute_time(total_time, time_step, pos, time_step)
            notes_right.append((int(prev_pitch), velocity, start_time, end_time))

    # sort two lists by start_time
    # notes_left.sort(key=lambda x: x[2])
    # notes_right.sort(key=lambda x: x[2])
    notes_left.sort(key=operator.itemgetter(2))
    notes_right.sort(key=operator.itemgetter(2))

    # concatenate two channels
    notes = notes_left + notes_right

    # add
    testing_lib.add_track_to_sequence(sequence, 0, notes)

    midi_filename = '4_0.mid'
    output_dir = os.getcwd()
    midi_path = os.path.join(output_dir, midi_filename)
    magenta.music.sequence_proto_to_midi_file(sequence, midi_path)

def main():
    # import .npy
    input_left = np.load('4.npy')
    input_right = np.load('4.npy')
    store_in_midi(input_left, input_right)
    # input_left = np.load('500.npy')
    #print(input_left[0,:])

if __name__ == "__main__":
    main()
