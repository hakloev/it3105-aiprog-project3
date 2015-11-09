# -*- coding: utf8 -*-
#
# Created by 'myth' on 11/9/15

import os

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

# KOK

datasets_dir = 'module5'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(ntrain=100, ntest=10, onehot=False):
    data_dir = os.path.join(datasets_dir, 'mnist/')
    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tr_x = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tr_y = loaded[8:].reshape((60000,))

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    te_x = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    te_y = loaded[8:].reshape((10000,))

    tr_x /= 255.
    te_x /= 255.

    tr_x = tr_x[:ntrain]
    tr_y = tr_y[:ntrain]

    te_x = te_x[:ntest]
    te_y = te_y[:ntest]

    if onehot:
        tr_y = one_hot(tr_y, 10)
        te_y = one_hot(te_y, 10)
    else:
        tr_y = np.asarray(tr_y)
        te_y = np.asarray(te_y)

    return tr_x, te_x, tr_y, te_y


# END KOK


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
    x = dropout(x, p_drop_input)
    _h = rectify(T.dot(x, _w_h))

    _h = dropout(_h, p_drop_hidden)
    _h2 = rectify(T.dot(_h, _w_h2))

    _h2 = dropout(_h2, p_drop_hidden)
    _py_x = softmax(T.dot(_h2, _w_o))
    return _h, _h2, _py_x


def debug():
    """
    Performs some actions
    """

    print('Loading data...')

    images, labels = load_all_flat_cases()

    # debug_tr_x, debug_te_x, debug_tr_y, debug_te_y = mnist()

    # print(debug_tr_x)
    # print(debug_tr_y)
    # print(debug_te_x)
    # print(debug_te_y)

    tr_x = np.array(images).astype(float)[:TRAIN_MAX]
    te_x = np.array(images).astype(float)[:TEST_MAX]
    tr_y = one_hot(labels[:TRAIN_MAX], 10)
    te_y = one_hot(labels[:TEST_MAX], 10)

    images_matrix = T.fmatrix()
    labels_matrix = T.fmatrix()

    print('Initializing weights')

    w_h = init_weights((784, 625))
    w_h2 = init_weights((625, 625))
    w_o = init_weights((625, 10))

    print('Building model')

    noise_h, noise_h2, noise_py_x = model(images_matrix, w_h, w_h2, w_o, 0.2, 0.5)
    h, h2, py_x = model(images_matrix, w_h, w_h2, w_o, 0., 0.)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, labels_matrix))
    params = [w_h, w_h2, w_o]
    updates = rms_prop(cost, params, lr=0.002)

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
