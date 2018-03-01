# Classical-Music-Generation-

In our project, we are going to build an application that can generate a full piece of classical music from a single passage of rhythm and melody given by the potential users. Our training data come from the existing music work from 18th to 20th century. Without being given the labels of when the training samples were composed, the trained model will also be able to classify the era that the music pieces belong to, and generate a list of work of the similar composing pattern for reference, so that our users could get a better sense of which categories of music they are working with and better matching to their composing expectations. 

0. midi_to_statematrix.py is obtained from https://github.com/hexahedria/biaxial-rnn-music-composition

1. reform_matrix.py
    convert midi files into numpy arrays

2. reform_midi.py
    convert numpy array to midi files

3. rnn_main_1_hidden_layer.py
4. rnn_model_1_hidden_layer.py
    LSTM RNN model with 1 hidden layer

5. rnn_main_2_hidden_layers.py
6. rnn_model_2_hidden_layer.py
    LSTM RNN model with 2 hidden layers

--------

7. array_to_midi.py (NOT USING, but method is mentioned in report)
8. sequence_to_array.py (NOT USING, but method is mentioned in report)
    Old way of reading info from midi files and generating input arrays
    Old way of converting numpy arrays back to midi file
