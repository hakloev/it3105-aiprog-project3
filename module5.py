# -*- encoding: utf-8 -*-

import logging
from logging.config import dictConfig
from configuration import LOG_CONFIG
import numpy as np

from module5.mnist import mnist_basics
from module5.ann import ANN, rectify, softmax, sigmoid

DO_BLIND_TEST = False

ANN_CONFIGURATIONS = [
    {
        'layer_structure': [784, 620, 10],
        'activation_functions': [rectify, rectify, softmax],
        'config': {
            'learning_rate': 0.001
        }
    },
    {
        'layer_structure': [784, 1568, 1176, 10],
        'activation_functions': [rectify, rectify, rectify, softmax],
        'config': {
            'learning_rate': 0.001
        }
    },
    {
        'layer_structure': [784, 620, 10],
        'activation_functions': [sigmoid, sigmoid, rectify],
        'config': {
            'learning_rate': 0.001
        }
    },
    {
        'layer_structure': [784, 320, 10],
        'activation_functions': [sigmoid, rectify, softmax],
        'config': {
            'learning_rate': 0.005
        }
    },
    {
        'layer_structure': [784, 620, 310, 10],
        'activation_functions': [rectify, rectify, rectify, softmax],
        'config': {
            'learning_rate': 0.001,
            'error_function': 'SSE'
        }
    },
    {
        'layer_structure': [784, 620, 310, 10],
        'activation_functions': [rectify, rectify, rectify, softmax],
        'config': {
            'learning_rate': 0.001,
            'error_function': 'Crossentropy'
        }
    }
]


def do_ann_analysis(configurations, **kwargs):
    for config in configurations:
        do_single_ann_analysis(config, **kwargs)


def do_single_ann_analysis(config, epochs=20, do_welch_test=False, write_statistics=False):
    layer_struct = config['layer_structure']
    af = config['activation_functions']
    conf = config['config']

    a = ANN(layer_struct, af, config=conf)
    a.load_input_data()
    errors, correctness = a.train(epochs=epochs, visualize=False)
    log.debug('ANN average error: %.4f' % np.mean(errors))

    training_correctness = np.mean(np.argmax(a.train_correct_labels, axis=1) == a.predict(a.train_input_data))
    testing_correctness = np.mean(np.argmax(a.test_correct_labels, axis=1) == a.predict(a.test_input_data))

    log.info('ANN correctness on training data: %.4f' % training_correctness)
    log.info('ANN correctness on testing data: %.4f' % testing_correctness)

    if write_statistics:
        with open('analysis.txt', 'a') as file:
            statistics = '%s\n%.4f\n%.4f\n%s\n%s\n-\n' % (
                str(a),
                training_correctness,
                testing_correctness,
                repr(correctness),
                repr(errors)
            )
            file.write(statistics)

    if do_welch_test:
        mnist_basics.minor_demo(a)

if __name__ == "__main__":

    # Set up logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)

    # Do analysis on all but the last ann configuration
    do_ann_analysis(ANN_CONFIGURATIONS[:-1], epochs=20, do_welch_test=True, write_statistics=True)

    # Do analysis on same net, but with SSE vs Crossentropy
    # do_single_ann_analysis(ANN_CONFIGURATIONS[-2])
    # do_single_ann_analysis(ANN_CONFIGURATIONS[-1])

    """
    # Network structure
    # Structure: [input_layer, hidden_layer, hidden_layer ... , output_layer]
    # Example: [784, 620, 100, 10]
    layer_structure = [784, 620, 10]
    # Example: [rectify, rectify, softmax]
    activation_functions = [rectify, rectify, softmax]
    cfg = {'learning_rate': 0.002}

    # Create a network using the default parameters
    a = ANN(layer_structure, activation_functions, config=cfg)
    a.load_input_data()

    train_data_cache = a.train_input_data
    train_labels_cache = a.train_correct_labels
    test_data_cache = a.test_input_data
    test_labels_cache = a.test_correct_labels

    # Train a bit and perform blind test
    a.train(epochs=10, include_test_set=False, visualize=False)

    if DO_BLIND_TEST:
        mnist_basics.minor_demo(a)
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
