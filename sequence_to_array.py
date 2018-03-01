import tensorflow as tf
import numpy as np
from magenta.music import note_sequence_io
from magenta.protobuf import music_pb2


def process_tempo_index(qpm_dict, keys):
    timestart_to_qpm_idx = {}
    prev_idx = 0
    for i in range(len(keys)-1):
        start_time = keys[i]
        end_time = keys[i+1]
        idx_inc = (end_time - start_time) * qpm_dict[start_time]
        if abs(idx_inc - round(idx_inc)) > 0.01:
            print(idx_inc)
        timestart_to_qpm_idx[start_time] = (qpm_dict[start_time], int(round(prev_idx)))
        prev_idx += idx_inc
    end_idx = prev_idx
    timestart_to_qpm_idx[keys[-1]] = (qpm_dict[keys[-1]], int(round(end_idx)))

    return timestart_to_qpm_idx, int(round(end_idx))

def store_in_array(file):
    '''
    Data structure: 
        First store the pitches in channel_0 in a list of lists. Each sublist represents a time track.
        Then store the pitches in channel_1 in a list of lists. Each sublist represents a time track.
        Finally, convert each of them to a 3D numpy array of shape (?, H, W). Here we set 
            W corresponds to the length.
            H corresponds to the max num of different pitches at a time.
            ? depends on how many pieces we feed in
        *** REMINDER: SET H AS CONSTANT 5 ***
    Rules:
        0. # of repeated pitch is determined by (end_time-start_time)/(1/qpm)
        1. If more than one note occurs at the same time: record in different time track.
        2. Gap: set the pitch value as 0.
    '''
    sequence = music_pb2.NoteSequence()
    
    '''
    Use a 3D numpy array to store all info
    '''
    all_channel0 = None
    all_channel1 = None

    for i, sequence in enumerate(note_sequence_io.note_sequence_record_iterator(file)):
        '''
        * create a list of tuples, which stores the values of time and tempo:
            the first element in each tuple: tempo.time
            the second element in each tuple: tempo.qpm
        ** let curr_tempo_ptr point to the first element in the list
        '''
        total_time = sequence.total_time        
        # use a dictionray to store the relationship between time and qpm
        # use a numpy array to store the keys
        time_qpm = []
        keys = []
        for tempo in sequence.tempos:
            time_qpm.append((tempo.time,tempo.qpm))
            keys.append(tempo.time)
        qpm_keys = np.array(keys)
        qpm_dict = dict(time_qpm)

        # construct dictionary and two channel arrays
        my_dict, end_len = process_tempo_index(qpm_dict, keys)
        channel_0 = np.zeros((5,end_len))
        channel_1 = np.zeros((5,end_len))
        channel_flag = 0
        # construct an array to keep track of available slots
        available_slot = np.zeros((end_len,))

        prev_end_time = 0
        prev_start_time = 0
        for note in sequence.notes:
            # get current pitch value, start time, and end time
            curr_pitch = note.pitch
            curr_start_time = note.start_time
            curr_end_time = note.end_time

            # switch the channel if needed
            if prev_start_time > curr_start_time and prev_end_time == total_time:
                channel_flag = 1
                # clean up the tracking status
                prev_end_time = 0
                available_slot = np.zeros((end_len,))

            prev_end_time = curr_end_time
            prev_start_time = curr_start_time
            
            # get the starting and ending key indices
            start_ind = (np.abs(qpm_keys-curr_start_time)).argmin()
            if (qpm_keys[start_ind]) > curr_start_time:
                start_ind = start_ind - 1
            end_ind = (np.abs(qpm_keys-curr_end_time)).argmin()
            if (qpm_keys[end_ind]) > curr_end_time:
                end_ind = end_ind - 1
            # get the starting and ending position
            start_key = qpm_keys[start_ind]
            start_base = my_dict[start_key][1]
            start_pos = int(start_base + round((curr_start_time - start_key)*qpm_dict[start_key]))
            end_key = qpm_keys[end_ind]
            end_base = my_dict[end_key][1]
            end_pos = int(end_base + round((curr_end_time - end_key)*qpm_dict[end_key]))

            # update values
            for idx in range(start_pos, min(end_pos,end_len)):
                row = int(available_slot[idx])
                if row < 5:
                    if channel_flag == 0:
                        channel_0[row,idx] = curr_pitch
                        available_slot[idx] = row + 1
                    else:
                        channel_1[row,idx] = curr_pitch
                        available_slot[idx] = row + 1
        
        if i == 0:
            all_channel0 = np.transpose(channel_0)
            all_channel1 = np.transpose(channel_1)
            all_channel0 = np.expand_dims(all_channel0, axis=0)
            all_channel1 = np.expand_dims(all_channel1, axis=0)
        else:
            prev_max_len = all_channel0.shape[1]
            if prev_max_len > end_len:
                temp0 = np.zeros((5,prev_max_len))
                temp0[:,:end_len] = channel_0 
                temp1 = np.zeros((5,prev_max_len))
                temp1[:,:end_len] = channel_1
                temp0 = np.expand_dims(np.transpose(temp0),axis=0)
                temp1 = np.expand_dims(np.transpose(temp1),axis=0)
                all_channel0 = np.concatenate((all_channel0, temp0), axis=0)
                all_channel1 = np.concatenate((all_channel1, temp1), axis=0)
            elif end_len > prev_max_len:
                a,b,c = all_channel0.shape
                pad0 = np.zeros((a,end_len,c))
                pad1 = np.zeros((a,end_len,c))
                pad0[:,:b,:] = all_channel0
                pad1[:,:b,:] = all_channel1
                channel_0 = np.expand_dims(np.transpose(channel_0),axis=0)
                channel_1 = np.expand_dims(np.transpose(channel_1),axis=0)
                all_channel0 = np.concatenate((pad0, channel_0), axis=0)
                all_channel1 = np.concatenate((pad1, channel_1), axis=0)

    return all_channel0, all_channel1
            

def write_to_file(file):
    fp = open("pitch_mond2.txt","w+")
    ft = open("tempo_mond2.txt","w+")
    fp.write('pitch,start_time,end_time,time_diff,note_len\n')
    # fp = open("velocity_mond2.txt","w+")
    # fp.write('pitch,start_time,end_time,velocity\n')
    sequence = music_pb2.NoteSequence()
    for i, sequence in enumerate(note_sequence_io.note_sequence_record_iterator(file)):
        for tempo in sequence.tempos:
            ft.write('time:{}\n'.format(tempo.time))
            ft.write('qpm:{}\n'.format(tempo.qpm))
        for note in sequence.notes:
            fp.write('{},{},{},{},{}/{}\n'.format(note.pitch,note.start_time,note.end_time,note.end_time - note.start_time,note.numerator,note.denominator))
            # fp.write('{},{},{},{}\n'.format(note.pitch,note.start_time,note.end_time,note.velocity))


def main():
    file_name = "new_dev.tfrecord"
    channel0, channel1 = store_in_array(file_name)
    np.save('newdev_0.npy',channel0)
    np.save('newdev_1.npy',channel1)
    #write_to_file(file_name)
    # file_name = "newnotes.tfrecord"
    # channel0, channel1 = store_in_array(file_name)
    # np.save('new_channel0',channel0)
    # np.save('new_channel1',channel1)

if __name__ == "__main__":
    main()
