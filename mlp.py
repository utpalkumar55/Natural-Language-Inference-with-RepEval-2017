from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Activation


max_train_pair = 50000
batch_size = 128
num_of_epoch = 5
max_sequence_len = 120
max_word = 10000

classes = {'neutral' : 1, 'contradiction' : 2, 'entailment' : 3}

train_class = []
train_sentence = []

print('Reading training data...')

with open('/home/utpal/Project/multinli_1.0/multinli_1.0_train.txt') as f:
    next(f)
    l = 1
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
                s = s + word + ' '
            if j == 0:
                break;
            k = k + 1
        a = a[k+1:]
        j = 0
        for word in a:
            if word is '(':
                j = j + 1
            elif word is ')':
                j = j - 1
            else:
                s = s + word + ' '
            if j == 0:
                break;

        if gl in classes:
            train_sentence.append(s)
            train_class.append(classes[gl])
            l = l + 1
        if l > max_train_pair:
            break;


print('Found %s sentence pair in training data.' % len(train_sentence))


tokenizer = Tokenizer(num_words=max_word)
tokenizer.fit_on_texts(train_sentence)
train_sequences = tokenizer.texts_to_sequences(train_sentence)
train_sequences = tokenizer.sequences_to_matrix(train_sequences,mode='binary')

word_index = tokenizer.word_index

train_labels = to_categorical(train_class,4)

print('Shape of training sequences: ',train_sequences.shape)
print('Shape of training labels: ',train_labels.shape)
print('\n')



matched_class = []
matched_sentence = []

print('Reading matched data for testing...')

with open('/home/utpal/Project/multinli_1.0/multinli_1.0_dev_matched.txt') as f:
    next(f)
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
                s = s + word + ' '
            if j == 0:
                break;
            k = k + 1
        a = a[k+1:]
        j = 0
        for word in a:
            if word is '(':
                j = j + 1
            elif word is ')':
                j = j - 1
            else:
                s = s + word + ' '
            if j == 0:
                break;

        if gl in classes:
            matched_sentence.append(s)
            matched_class.append(classes[gl])


print('Found %s sentence pair in matched testing data.' % len(matched_sentence))


tokenizer = Tokenizer(num_words=max_word)
tokenizer.fit_on_texts(matched_sentence)
matched_sequences = tokenizer.texts_to_sequences(matched_sentence)
matched_sequences = tokenizer.sequences_to_matrix(matched_sequences,mode='binary')

matched_labels = to_categorical(matched_class,4)

print('Shape of matched testing sequences: ',matched_sequences.shape)
print('Shape of matched testing labels: ',matched_labels.shape)
print('\n')



mismatched_class = []
mismatched_sentence = []

print('Reading mismatched data for testing...')

with open('/home/utpal/Project/multinli_1.0/multinli_1.0_dev_mismatched.txt') as f:
    next(f)
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
                s = s + word + ' '
            if j == 0:
                break;
            k = k + 1
        a = a[k+1:]
        j = 0
        for word in a:
            if word is '(':
                j = j + 1
            elif word is ')':
                j = j - 1
            else:
                s = s + word + ' '
            if j == 0:
                break;

        if gl in classes:
            mismatched_sentence.append(s)
            mismatched_class.append(classes[gl])


print('Found %s sentence pair in matched testing data.' % len(mismatched_sentence))


tokenizer = Tokenizer(num_words=max_word)
tokenizer.fit_on_texts(mismatched_sentence)
mismatched_sequences = tokenizer.texts_to_sequences(mismatched_sentence)
mismatched_sequences = tokenizer.sequences_to_matrix(mismatched_sequences,mode='binary')

mismatched_labels = to_categorical(mismatched_class,4)

print('Shape of mismatched testing sequences: ',mismatched_sequences.shape)
print('Shape of mismatched testing labels: ',mismatched_labels.shape)
print('\n')




print('Model Definition...')

model = Sequential()
model.add(Dense(512, input_shape=(max_word,)))
model.add(Activation('relu'))

model.add(Dense(64, input_dim=784))
model.add(Activation('softmax'))

model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


print('Training...')

model.fit(train_sequences, train_labels,
          batch_size=batch_size,
          epochs=num_of_epoch,
          verbose=2,
          validation_split=0.2)

matched_score, matched_acc = model.evaluate(matched_sequences, matched_labels,
                            batch_size=batch_size)

print('Test accuracy for matched data:', matched_acc)

print('\n')

mismatched_score, mismatched_acc = model.evaluate(mismatched_sequences, mismatched_labels,
                            batch_size=batch_size)

print('Test accuracy for mismatched data:', mismatched_acc)
