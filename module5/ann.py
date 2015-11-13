# -*- encoding: utf-8 -*-

import logging

import numpy as np
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from module5.mnist.mnist_basics import load_all_flat_cases

# DEFAULTS
TRAIN_MAX = 60000
TEST_MAX = 10000
TRAIN_BATCH_SIZE = 512
LEARNING_RATE = 0.001

# SETTINGS
DATASET_DIR = 'module5'


class ANN(object):
    """
    Construct an Artificial neural network
    """

    def __init__(self, layer_structure, learning_rate=LEARNING_RATE, train_max=TRAIN_MAX, test_max=TEST_MAX,
                 train_batch_size=TRAIN_BATCH_SIZE):
        self._log = logging.getLogger(__name__)
        self._log.info('Initializing Neural Network...')
        self._lr = learning_rate
        self._train_max = train_max
        self._test_max = test_max
        self._train_batch_size = train_batch_size
        self._log.info(
            'Layers: %s, LR: %.5f, MTR: %d, MTE: %d, TRBS: %d.' % (
                repr(layer_structure), self._lr, self._train_max, self._test_max, self._train_batch_size
            )
        )
        self._srng = RandomStreams()
        self.net = self._generate_network(layer_structure)

    def _generate_network(self, layer_structure):
        return None

    def blind_test(self):
        """
        The method blind test must accept a list of sublists, where each sublist is a vector of length 784 corresponding
        to the raw features of one image: each sublist is a flattened image containing integers in the range [0, 255].
        These raw features come directly from a flat-case file. The list does NOT contain labels, hence the adjective
        blind. Your method must produce a flat list of labels predicted by the ann when given each feature vector as
        input. Items in the labels list must correspond to items in the flattened image list. So if feature sets
        consists of five image vectors, a 7, two 3’s, an 8 and a 2 (in that order), then if your ann classifies them
        correctly, it should return this:

        [7,3,3,8,2]
        :return:
        """

        pass

    @staticmethod
    def vectorize_labels(labels, num_categories):
        if type(labels) == list:
            labels = np.array(labels)
        labels = labels.flatten()
        label_vector = np.zeros((len(labels), num_categories))
        label_vector[np.arange(len(labels)), labels] = 1

        return label_vector
