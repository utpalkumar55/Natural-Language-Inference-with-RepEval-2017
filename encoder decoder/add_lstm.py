import random
import numpy as np

# from keras.models import Sequential
from keras import layers
from keras import models

NUM_INSTANCES = 10000
NUM_INSTANCES_TR = int(NUM_INSTANCES * 0.8)
NUM_INSTANCES_TE = NUM_INSTANCES - NUM_INSTANCES_TR
INT_MIN = 1
INT_MAX = 999

# If set to true, only two dummy additions will be generated
DUMMY = False

def generate_additions(dummy=False):
    if dummy:
        for a, b, c in [[155, 149, 304], [100, 100, 200]]:
            operands = "%s+%s" % (a, b)
            result = "%s" % c
            yield operands, result
        return

    for i in range(NUM_INSTANCES):
        a = random.randint(INT_MIN, INT_MAX)
        b = random.randint(INT_MIN, INT_MAX)
        operands = "%s+%s" % (a, b)
        result = "%s" % (a + b)
        yield operands, result


def str2digit(i):
    try:
        return int(i)
    except ValueError:
        return 10


def str2onehot(ai):
    onehot = [0] * 11
    onehot[str2digit(ai)] = 1
    return onehot


def encode_instance(operands, results):
    operands_onehot = [str2onehot(i) for i in operands]
    results_onehot = [str2onehot(i) for i in results]
    return np.array(operands_onehot), np.array(results_onehot)


def decode_instance(operands_onehot, results_onehot):
    operands = [o.argmax() if o.argmax() < 10 else '+' for o in operands_onehot]
    results = [r.argmax() for r in results_onehot]
    return operands, results


def learn(instances):
    print "Num train instances: %s. Num test instances: %s" % (NUM_INSTANCES_TR, NUM_INSTANCES_TE)

    instances = [encode_instance(operands, results) for operands, results in instances]
    inputs = [instance[0] for instance in instances]
    outputs = [instance[1] for instance in instances]

    inputs_tr, outputs_tr = np.array(inputs[:NUM_INSTANCES_TR]), np.array(outputs[:NUM_INSTANCES_TR])
    inputs_te, outputs_te = np.array(inputs[NUM_INSTANCES_TR:]), np.array(outputs[NUM_INSTANCES_TR:])
    print inputs_tr.shape, outputs_tr.shape, inputs_te.shape, outputs_te.shape

    inputs = layers.Input(shape=(len(inputs_tr[0]), 11))
    encoder = layers.LSTM(256)(inputs)

    decoder = layers.RepeatVector(len(outputs_tr[0]))(encoder)
    decoder = layers.LSTM(256, return_sequences=True)(decoder)

    predictions = layers.Dense(11, activation='softmax')(decoder)

    model = models.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    model.fit(inputs_tr, outputs_tr,
              batch_size=32,
              epochs=100,
              validation_data=(inputs_te, outputs_te))

    sample_inputs_te, sample_outputs_te = inputs_te[:10], outputs_te[:10]
    predictions = model.predict((sample_inputs_te))
    for inputs, predictions, gold in zip(sample_inputs_te, predictions, sample_outputs_te):
        print inputs
        print decode_instance(inputs, predictions)[0]
        # print predictions
        print decode_instance(inputs, predictions)[1]
        # print gold
        print decode_instance(inputs, gold)[1]

        print

def main():
    instances = list(generate_additions(dummy=DUMMY))
    max_len_operands = max([len(operands) for operands, _ in instances])
    max_len_results = max([len(results) for _, results in instances])
    instances = [("0" * (max_len_operands - len(operands)) + operands,
                  "0" * (max_len_results - len(results)) + results) for operands, results in instances]

    print "Here are all the instances: "
    for operands, results in instances:
        print operands, results
        operands_onehot, results_onehot = encode_instance(operands, results)
        operands2, results2 = decode_instance(operands_onehot, results_onehot)
        # Check that the encoding / decoding works
        assert ''.join(map(str, operands2)) == operands
        assert ''.join(map(str, results2)) == results

    learn(instances)


if __name__ == "__main__":
    main()