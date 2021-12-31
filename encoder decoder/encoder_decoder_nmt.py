# Heavily inspired by keras/examples/lstm_seq2seq.py

import io
import argparse
import numpy as np
import pprint

from keras import layers
from keras import models

START = "["
END = "]"

MAX_INSTANCES = 10000

LSTM_SIZE = 256
BATCH_SIZE = 64
NUM_EPOCHS = 100

def main(file_name):
    input_texts, target_texts = [], []
    with io.open(file_name, "r", encoding="utf-8") as f:
        for line in f.readlines()[:MAX_INSTANCES]:
            line = line.rstrip()

            input_text, target_text = line.split("\t")
            target_text = START + target_text + END
            input_texts.append(input_text)
            target_texts.append(target_text)

    input_chars = set([c for c in ''.join(input_texts)])
    target_chars = set([c for c in ''.join(target_texts)])
    ichar2index = dict([(c, i) for i, c in enumerate(input_chars, start=1)])
    ichar2index['_'] = 0
    index2ichar = dict([(i, c) for c, i in ichar2index.iteritems()])
    tchar2index = dict([(c, i) for i, c in enumerate(target_chars, start=1)])
    tchar2index['_'] = 0
    index2tchar = dict([(i, c) for c, i in tchar2index.iteritems()])
    max_len_input_text = max([len(input_text) for input_text in input_texts])
    max_len_target_text = max([len(target_text) for target_text in target_texts])

    assert len(input_texts) == len(target_texts)
    print "Read %s pairs" % len(input_texts)
    print "Input vocabulary: [%s] %s" % (len(input_chars), ''.join((sorted(input_chars))))
    print "Target vocabulary: [%s] %s" % (len(target_chars), ''.join(sorted(target_chars)))
    print "Input maximum length: %s" % max_len_input_text
    print "Target maximum length: %s" % max_len_target_text

    encoder_input_data = np.zeros((len(input_texts), max_len_input_text, len(ichar2index)), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_len_target_text, len(tchar2index)), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_len_target_text, len(tchar2index)), dtype='float32')
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, c in enumerate(input_text):
            encoder_input_data[i][t][ichar2index[c]] = 1
        for t, c in enumerate(target_text):
            decoder_input_data[i][t][tchar2index[c]] = 1
            if t > 0:
                decoder_target_data[i][t-1][tchar2index[c]] = 1

    print encoder_input_data.shape
    print decoder_input_data.shape
    print decoder_target_data.shape
    return

    # print "|" + ''.join(([index2ichar[c.argmax()] for c in encoder_input_data[0]])) + "|"
    # print "|" + ''.join(([index2tchar[c.argmax()] for c in decoder_input_data[0]])) + "|"
    # print "|" + ''.join(([index2tchar[c.argmax()] for c in decoder_target_data[0]])) + "|"

    encoder_input = layers.Input(shape=(None, len(ichar2index)), name="input_encoder")
    encoder = layers.LSTM(LSTM_SIZE,
                          return_state=True,
                          name="encoder")
    _, encoder_state_h, encoder_state_c = encoder(encoder_input)
    encoder_states = [encoder_state_h, encoder_state_c]

    decoder_input = layers.Input(shape=(None, len(tchar2index)), name="input_decoder")
    decoder = layers.LSTM(LSTM_SIZE,
                          return_sequences=True,
                          return_state=True,
                          name="decoder")
    decoder_outputs, _, _ = decoder(decoder_input, initial_state=encoder_states)
    decoder_dense = layers.Dense(len(tchar2index), activation='softmax', name="target_text")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = models.Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    print "\n\n"
    model.summary()

    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_split=0.2)
    model.save('model.keras')

    encoder_model = models.Model(inputs=encoder_input, outputs=encoder_states)
    print "\n\n"
    encoder_model.summary()

    decoder_state_input_h = layers.Input(shape=(LSTM_SIZE,))
    decoder_state_input_c = layers.Input(shape=(LSTM_SIZE,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder(decoder_input, initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = models.Model(inputs=[decoder_input] + decoder_state_inputs,
                                 outputs=[decoder_outputs] + decoder_states)
    print "\n\n"
    decoder_model.summary()

    def decode_sequence(input_seq):
        states_value = encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1, len(tchar2index)), dtype='float32')
        target_seq[0, 0, tchar2index[START]] = 1

        decoded_sentence = ''
        while True:
            output_chars, state_h, state_c = decoder_model.predict([target_seq] + states_value)
            decoded_sentence += index2tchar[np.argmax(output_chars[0, -1, :])]

            if decoded_sentence[-1] == END or len(decoded_sentence) > max_len_target_text:
                break

            target_seq = np.zeros((1, 1, len(tchar2index)), dtype='float32')
            target_seq[0, 0, tchar2index[decoded_sentence[-1]]] = 1
            states_value = [state_h, state_c]

        return decoded_sentence

    for seq_index in range(10):
        # Take one sequence (part of the training set) for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)

        print 'Input sentence:', input_texts[seq_index]
        print 'Decoded sentence:', decoded_sentence


if __name__ == '__main__':
    parser =argparse.ArgumentParser(description='Try an encoder-decode for NMT')
    parser.add_argument("CORPUS",
                        help="Parallel corpus to train with")
    args = parser.parse_args()

    main(args.CORPUS)

