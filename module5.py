# -*- encoding: utf-8 -*-

import logging
from logging.config import dictConfig

import numpy

from module5.mnist import mnist_basics
from module5.ann import ANN, rectify, softmax, sigmoid

DO_BLIND_TEST = True

LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'default',
            'filename': 'debug.log',
            'maxBytes': 1024 * 1024 * 10,
            'backupCount': 1
        }
    },
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s %(name)s.%(funcName)s:%(lineno)d %(message)s'
        }
    },
    'loggers': {
        '': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}

if __name__ == "__main__":

    # Set up logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)

    # Network structure
    # Structure: [input_layer, hidden_layer, hidden_layer ... , output_layer]
    # Example: [784, 620, 100, 10]
    layer_structure = [784, 620, 10]
    # Example: [rectify, rectify, softmax]
    activation_functions = [rectify, rectify, softmax]
    cfg = {'learning_rate': 0.003}

    # Create a network using the default parameters
    a = ANN(layer_structure, activation_functions, config=cfg)
    a.load_input_data()

    train_data_cache = a.train_input_data
    train_labels_cache = a.train_correct_labels
    test_data_cache = a.test_input_data
    test_labels_cache = a.test_correct_labels

    # Train a bit and perform blind test
    a.train(epochs=5, include_test_set=True)

    if DO_BLIND_TEST:
        mnist_basics.minor_demo(a)

    """
    feature_sets, feature_labels = mnist_basics.gen_flat_cases(digits=numpy.arange(10), type='testing')
    feature_labels = feature_labels[:10]

    blind_test_result = a.blind_test(feature_sets[:10])
    # Can't do intersection, since order of list matter
    correct_result = [i for i, j in zip(blind_test_result, feature_labels) if i == j]

    log.debug("Returned: %s Actual %s " % (blind_test_result, feature_labels))
    log.info("Number of correct guesses: %i" % len(correct_result))
    """

    """
    # MULTIPLE NETWORK GENERATION AND TESTING

    # Create new net
    layer_structure = [784, 620, 10]
    activation_functions = [rectify, rectify, softmax]
    cfg = {'learning_rate': 0.0003}
    a = ANN(layer_structure, activation_functions, config=cfg)

    a.train_input_data = train_data_cache
    a.train_correct_labels = train_labels_cache
    a.test_input_data = test_data_cache
    a.test_correct_labels = test_labels_cache

    # Train current net
    a.train(epochs=50, include_test_set=True)
    """
