import numpy as np
from keras import layers
from keras import models


max_train_pair = 2700
max_test_pair = 900
batch_size = 256
num_of_epoch = 10
max_sequence_len = 120
max_word = 10000
START = "["
END = "]"
LSTM_SIZE = 256
BATCH_SIZE = 256
NUM_EPOCHS = 10

classes = {'neutral' : 1, 'contradiction' : 2, 'entailment' : 3}




###--------------------.....................---------------------------.........................----------------------###
###Train
train_input_sentence = []
train_target_sentence = []

print('Reading training data...')

with open('/home/utpal/Project/multinli_1.0/multinli_1.0_train.txt') as f:
    next(f)
    l = 1
    num_train_pair = {}
    num_train_pair['neutral'] = 0
    num_train_pair['contradiction'] = 0
    num_train_pair['entailment'] = 0
    for line in f:
        a = line.split()
        gl = a[0]
        a = a[1:]
        k = 0
        j = 0
        s = ''
        for word in a:
            if word is '(':
                j = j + 1
            elif word is ')':
                j = j - 1
            else:
                s = s + word
            if j == 0:
                break;
            k = k + 1
        if gl in classes:
            if  num_train_pair[gl] < max_train_pair/3:
                train_input_sentence.append(s)
                num_train_pair[gl] = num_train_pair[gl] + 1
            else:
                continue
        a = a[k+1:]
        j = 0
        s = ''
        for word in a:
            if word is '(':
                j = j + 1
            elif word is ')':
                j = j - 1
            else:
                s = s + word
            if j == 0:
                break;
        s = START + s + END
        if gl in classes:
            train_target_sentence.append(s)
            l = l + 1
        if l > max_train_pair:
            break;

print('Found %s sentence pair in training data.' % len(train_input_sentence))
print('\n')

train_input_chars = set([c for c in ''.join(train_input_sentence)])
train_target_chars = set([c for c in ''.join(train_target_sentence)])

train_input_char2index = dict([(c, i) for i, c in enumerate(train_input_chars, start=1)])
train_input_char2index['_'] = 0
train_input_index2char = dict([(i, c) for c, i in train_input_char2index.items()])

train_target_char2index = dict([(c, i) for i, c in enumerate(train_target_chars, start=1)])
train_target_char2index['_'] = 0
train_target_index2char = dict([(i, c) for c, i in train_target_char2index.items()])

max_len_train_input_sentence = max([len(input_sentence) for input_sentence in train_input_sentence])
max_len_train_target_sentence = max([len(target_sentence) for target_sentence in train_target_sentence])
###--------------------.....................---------------------------.........................----------------------###







###--------------------.....................---------------------------.........................----------------------###
###Matched
matched_input_sentence = []
matched_target_sentence = []

print('Reading matched data for testing...')

with open('/home/utpal/Project/multinli_1.0/multinli_1.0_dev_matched.txt') as f:
    next(f)
    l = 1
    num_test_pair = {}
    num_test_pair['neutral'] = 0
    num_test_pair['contradiction'] = 0
    num_test_pair['entailment'] = 0
    for line in f:
        a = line.split()
        gl = a[0]
        a = a[1:]
        k = 0
        j = 0
        s = ''
        for word in a:
            if word is '(':
                j = j + 1
            elif word is ')':
                j = j - 1
            else:
                s = s + word
            if j == 0:
                break;
            k = k + 1
        if gl in classes:
            if  num_test_pair[gl] < max_test_pair/3:
                matched_input_sentence.append(s)
                num_test_pair[gl] = num_test_pair[gl] + 1
            else:
                continue
        a = a[k+1:]
        j = 0
        s = ''
        for word in a:
            if word is '(':
                j = j + 1
            elif word is ')':
                j = j - 1
            else:
                s = s + word
            if j == 0:
                break;
        s = START + s + END
        if gl in classes:
            matched_target_sentence.append(s)
            l = l + 1
        if l > max_test_pair:
            break;

print('Found %s sentence pair in matched testing data.' % len(matched_input_sentence))
print('\n')

matched_input_chars = set([c for c in ''.join(matched_input_sentence)])
matched_target_chars = set([c for c in ''.join(matched_target_sentence)])

matched_input_char2index = dict([(c, i) for i, c in enumerate(matched_input_chars, start=1)])
matched_input_char2index['_'] = 0
matched_input_index2char = dict([(i, c) for c, i in matched_input_char2index.items()])

matched_target_char2index = dict([(c, i) for i, c in enumerate(matched_target_chars, start=1)])
matched_target_char2index['_'] = 0
matched_target_index2char = dict([(i, c) for c, i in matched_target_char2index.items()])

max_len_matched_input_sentence = max([len(input_sentence) for input_sentence in matched_input_sentence])
max_len_matched_target_sentence = max([len(target_sentence) for target_sentence in matched_target_sentence])
###--------------------.....................---------------------------.........................----------------------###







###--------------------.....................---------------------------.........................----------------------###
###Mismatched
mismatched_input_sentence = []
mismatched_target_sentence = []

print('Reading mismatched data for testing...')

with open('/home/utpal/Project/multinli_1.0/multinli_1.0_dev_mismatched.txt') as f:
    next(f)
    l = 1
    num_test_pair = {}
    num_test_pair['neutral'] = 0
    num_test_pair['contradiction'] = 0
    num_test_pair['entailment'] = 0
    for line in f:
        a = line.split()
        gl = a[0]
        a = a[1:]
        k = 0
        j = 0
        s = ''
        for word in a:
            if word is '(':
                j = j + 1
            elif word is ')':
                j = j - 1
            else:
                s = s + word
            if j == 0:
                break;
            k = k + 1
        if gl in classes:
            if  num_test_pair[gl] < max_test_pair/3:
                mismatched_input_sentence.append(s)
                num_test_pair[gl] = num_test_pair[gl] + 1
            else:
                continue
        a = a[k+1:]
        j = 0
        s = ''
        for word in a:
            if word is '(':
                j = j + 1
            elif word is ')':
                j = j - 1
            else:
                s = s + word
            if j == 0:
                break;
        s = START + s + END
        if gl in classes:
            mismatched_target_sentence.append(s)
            l = l + 1
        if l > max_test_pair:
            break;

print('Found %s sentence pair in mismatched testing data.' % len(mismatched_input_sentence))
print('\n')

mismatched_input_chars = set([c for c in ''.join(mismatched_input_sentence)])
mismatched_target_chars = set([c for c in ''.join(mismatched_target_sentence)])

mismatched_input_char2index = dict([(c, i) for i, c in enumerate(mismatched_input_chars, start=1)])
mismatched_input_char2index['_'] = 0
mismatched_input_index2char = dict([(i, c) for c, i in mismatched_input_char2index.items()])

mismatched_target_char2index = dict([(c, i) for i, c in enumerate(mismatched_target_chars, start=1)])
mismatched_target_char2index['_'] = 0
mismatched_target_index2char = dict([(i, c) for c, i in mismatched_target_char2index.items()])

max_len_mismatched_input_sentence = max([len(input_sentence) for input_sentence in mismatched_input_sentence])
max_len_mismatched_target_sentence = max([len(target_sentence) for target_sentence in mismatched_target_sentence])
###--------------------.....................---------------------------.........................----------------------###

max_input_char_length = max(len(train_input_char2index)+1, len(matched_input_char2index)+1, len(mismatched_input_char2index)+1)
max_target_char_length = max(len(train_target_char2index)+1, len(matched_target_char2index)+1, len(mismatched_target_char2index)+1)

###--------------------.....................---------------------------.........................----------------------###
###Train
train_encoder_input_data = np.zeros((len(train_input_sentence), max_len_train_input_sentence, max_input_char_length), dtype='float32')
train_decoder_input_data = np.zeros((len(train_input_sentence), max_len_train_target_sentence, max_target_char_length), dtype='float32')
train_decoder_target_data = np.zeros((len(train_input_sentence), max_len_train_target_sentence, max_target_char_length), dtype='float32')
for i, (input_text, target_text) in enumerate(zip(train_input_sentence, train_target_sentence)):
    for t, c in enumerate(input_text):
        if train_input_char2index[c] < max_input_char_length:
            train_encoder_input_data[i][t][train_input_char2index[c]] = 1
    for t, c in enumerate(target_text):
        if train_target_char2index[c] < max_target_char_length:
            train_decoder_input_data[i][t][train_target_char2index[c]] = 1
            if t > 0:
                train_decoder_target_data[i][t-1][train_target_char2index[c]] = 1

print(train_encoder_input_data.shape)
print(train_decoder_input_data.shape)
print(train_decoder_target_data.shape)
###--------------------.....................---------------------------.........................----------------------###



###--------------------.....................---------------------------.........................----------------------###
###Matched
matched_encoder_input_data = np.zeros((len(matched_input_sentence), max_len_matched_input_sentence, max_input_char_length), dtype='float32')
matched_decoder_input_data = np.zeros((len(matched_input_sentence), max_len_matched_target_sentence, max_target_char_length), dtype='float32')
matched_decoder_target_data = np.zeros((len(matched_input_sentence), max_len_matched_target_sentence, max_target_char_length), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(matched_input_sentence, matched_target_sentence)):
    for t, c in enumerate(input_text):
        if matched_input_char2index[c] < max_input_char_length:
            matched_encoder_input_data[i][t][matched_input_char2index[c]] = 1
    for t, c in enumerate(target_text):
        if matched_target_char2index[c] < max_target_char_length:
            matched_decoder_input_data[i][t][matched_target_char2index[c]] = 1
            if t > 0:
                matched_decoder_target_data[i][t-1][matched_target_char2index[c]] = 1

print(matched_encoder_input_data.shape)
print(matched_decoder_input_data.shape)
print(matched_decoder_target_data.shape)
###--------------------.....................---------------------------.........................----------------------###



###--------------------.....................---------------------------.........................----------------------###
###Mismatched
mismatched_encoder_input_data = np.zeros((len(mismatched_input_sentence), max_len_mismatched_input_sentence, max_input_char_length), dtype='float32')
mismatched_decoder_input_data = np.zeros((len(mismatched_input_sentence), max_len_mismatched_target_sentence, max_target_char_length), dtype='float32')
mismatched_decoder_target_data = np.zeros((len(mismatched_input_sentence), max_len_mismatched_target_sentence, max_target_char_length), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(mismatched_input_sentence, mismatched_target_sentence)):
    for t, c in enumerate(input_text):
        if mismatched_input_char2index[c] < max_input_char_length:
            mismatched_encoder_input_data[i][t][mismatched_input_char2index[c]] = 1
    for t, c in enumerate(target_text):
        if mismatched_target_char2index[c] < max_target_char_length:
            mismatched_decoder_input_data[i][t][mismatched_target_char2index[c]] = 1
            if t > 0:
                mismatched_decoder_target_data[i][t-1][mismatched_target_char2index[c]] = 1

print(matched_encoder_input_data.shape)
print(matched_decoder_input_data.shape)
print(matched_decoder_target_data.shape)
###--------------------.....................---------------------------.........................----------------------###




###--------------------.....................---------------------------.........................----------------------###
encoder_input = layers.Input(shape=(None, max_input_char_length), name="input_encoder")
encoder = layers.LSTM(LSTM_SIZE,
                          return_state=True,
                          name="encoder")
_, encoder_state_h, encoder_state_c = encoder(encoder_input)
encoder_states = [encoder_state_h, encoder_state_c]

decoder_input = layers.Input(shape=(None, max_target_char_length), name="input_decoder")
decoder = layers.LSTM(LSTM_SIZE,
                          return_sequences=True,
                          return_state=True,
                          name="decoder")
decoder_outputs, _, _ = decoder(decoder_input, initial_state=encoder_states)
decoder_dense = layers.Dense(max_target_char_length, activation='softmax', name="target_text")
decoder_outputs = decoder_dense(decoder_outputs)

model = models.Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([train_encoder_input_data, train_decoder_input_data], train_decoder_target_data,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              verbose=2)

matched_score, matched_acc = model.evaluate([matched_encoder_input_data, matched_decoder_input_data], matched_decoder_target_data,
                            batch_size=batch_size)

print('Test accuracy for matched data:', matched_acc)

mismatched_score, mismatched_acc = model.evaluate([mismatched_encoder_input_data, mismatched_decoder_input_data], mismatched_decoder_target_data, batch_size=batch_size)

print('Test accuracy for matched data:', mismatched_acc)
###--------------------.....................---------------------------.........................----------------------###




###--------------------.....................---------------------------.........................----------------------###
'''encoder_model = models.Model(inputs=encoder_input, outputs=encoder_states)

decoder_state_input_h = layers.Input(shape=(LSTM_SIZE,))
decoder_state_input_c = layers.Input(shape=(LSTM_SIZE,))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder(decoder_input, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = models.Model(inputs=[decoder_input] + decoder_state_inputs,
                                 outputs=[decoder_outputs] + decoder_states)

for seq_index in range(max_test_pair):
    input_data = matched_encoder_input_data[seq_index: seq_index + 1]
    states_value = encoder_model.predict(input_data)
    target_data = np.zeros((1, 1, len(train_target_char2index)+1), dtype='float32')
    target_data[0, 0, matched_target_char2index[START]] = 1
    decoded_sentence = ''
    while True:
        output_chars, state_h, state_c = decoder_model.predict([target_data] + states_value)
        decoded_sentence += matched_target_index2char[np.argmax(output_chars[0, -1, :])]

        if decoded_sentence[-1] == END or len(decoded_sentence) > max_len_matched_target_sentence:
            break

        target_data = np.zeros((1, 1, len(train_target_char2index)+1), dtype='float32')
        target_data[0, 0, matched_target_char2index[decoded_sentence[-1]]] = 1
        states_value = [state_h, state_c]
    correct = 0;
    if matched_encoder_input_data[seq_index] == decoded_sentence:
        correct = correct + 1

accuracy = correct/max_test_pair


print('Test accuracy for matched data:', accuracy)'''
