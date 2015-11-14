# -*- coding: utf8 -*-
#
# Created by 'myth' on 11/9/15

import os
import time

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from module5.mnist.mnist_basics import load_all_flat_cases

srng = RandomStreams()

# Settings
TRAIN_MAX = 60000
TEST_MAX = 10000
TRAIN_BATCH_SIZE = 512
LEARNING_RATE = 0.001


def vectorize_labels(labels, num_categories):
    if type(labels) == list:
        labels = np.array(labels)
    labels = labels.flatten()
    label_vector = np.zeros((len(labels), num_categories))
    label_vector[np.arange(len(labels)), labels] = 1

    return label_vector

datasets_dir = 'module5'


def float_x(x):
    return np.asarray(x, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(float_x(np.random.randn(*shape) * 0.01))


def rectify(x):
    return T.maximum(x, 0.)


def softmax(x):
    e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def rms_prop(_cost, _params, lr=0.001, rho=0.9, epsilon=1e-6):
    _grads = T.grad(cost=_cost, wrt=_params)
    _updates = []
    for p, g in zip(_params, _grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        _updates.append((acc, acc_new))
        _updates.append((p, p - lr * g))
    return _updates


def dropout(x, p=0.):
    if p > 0:
        retain_prob = 1 - p
        x *= srng.binomial(x.shape, p=retain_prob, dtype=theano.config.floatX)
        x /= retain_prob
    return x


def model(x, _w_h, _w_h2, _w_o, p_drop_input, p_drop_hidden):
    print("MODEL")
    print("x: %s" % x)
    print("_w_w: %s" % _w_h)
    print("_w_h2: %s" % _w_h2)
    print("_w_o: %s" % _w_o)
    x = dropout(x, p_drop_input)
    _h = rectify(T.dot(x, _w_h))

    print("x: %s" % x)
    print("_h: %s" % _h)

    _h = dropout(_h, p_drop_hidden)
    _h2 = rectify(T.dot(_h, _w_h2))

    print("_h: %s" % _h)
    print("_h2: %s" % _h2)

    _h2 = dropout(_h2, p_drop_hidden)
    _py_x = softmax(T.dot(_h2, _w_o))

    print("_h2: %s" % _h2)
    print("_py_x: %s" % _py_x)

    return _h, _h2, _py_x


def debug():
    """
    Performs some actions
    """

    print('Loading data...')

    images, labels = load_all_flat_cases()

    tr_x = np.array(images).astype(float)[:TRAIN_MAX] / 255.
    te_x = np.array(images).astype(float)[:TEST_MAX] / 255.
    tr_y = vectorize_labels(labels[:TRAIN_MAX], 10)
    te_y = vectorize_labels(labels[:TEST_MAX], 10)

    images_matrix = T.fmatrix()
    labels_matrix = T.fmatrix()

    print('Initializing weights')

    w_h = init_weights((784, 784))
    w_h2 = init_weights((784, 625))
    w_o = init_weights((625, 10))

    print(images_matrix)
    print(w_h)
    print(w_h2)
    print(w_o)

    print('Building model')

    noise_h, noise_h2, noise_py_x = model(images_matrix, w_h, w_h2, w_o, 0.2, 0.5)
    print(noise_h)
    print(noise_h2)
    print(noise_py_x)
    h, h2, py_x = model(images_matrix, w_h, w_h2, w_o, 0., 0.)
    print(h)
    print(h2)
    print(py_x)
    y_x = T.argmax(py_x, axis=1)
    print(y_x)

    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, labels_matrix))
    params = [w_h, w_h2, w_o]
    updates = rms_prop(cost, params, lr=LEARNING_RATE)

    train = theano.function(
        inputs=[images_matrix, labels_matrix],
        outputs=cost,
        updates=updates,
        allow_input_downcast=True
    )
    predict = theano.function(inputs=[images_matrix], outputs=y_x, allow_input_downcast=True)

    for i in range(10):
        print('Epoch: %d' % i)
        start_range = range(0, len(tr_x), TRAIN_BATCH_SIZE)
        end_range = range(TRAIN_BATCH_SIZE, len(tr_x), TRAIN_BATCH_SIZE)
        for start, end in zip(start_range, end_range):
            cost = train(tr_x[start:end], tr_y[start:end])

        # for j, img in enumerate(te_x):
        #     print('Predicted: %.4f Actual: %.2f' % (float(np.argmax([te_y[j]], axis=1)), predict([img])))
        print('Correctness: %.4f' % np.average(np.argmax(te_y, axis=1) == predict(te_x)))

    print('Correctness: %.4f' % np.average(np.argmax(te_y[3572:5873], axis=1) == predict(te_x[3572:5873])))
