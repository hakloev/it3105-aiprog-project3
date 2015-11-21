# -*- encoding: utf-8 -*-

import logging
from logging.config import dictConfig
from configuration import LOG_CONFIG
import numpy as np

from module5.mnist import mnist_basics
from module5.ann import ANN, rectify, softmax, sigmoid, softplus

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
        'layer_structure': [784, 620, 10],
        'activation_functions': [sigmoid, softplus, softmax],
        'config': {
            'learning_rate': 0.001
        }
    },
    {
        'layer_structure': [784, 392, 10],
        'activation_functions': [softplus, softplus, softmax],
        'config': {
            'learning_rate': 0.005
        }
    },
    {
        'layer_structure': [784, 112, 10],
        'activation_functions': [rectify, sigmoid, softmax],
        'config': {
            'learning_rate': 0.005
        }
    },
    {
        'layer_structure': [784, 1568, 1176, 10],
        'activation_functions': [rectify, rectify, rectify, softmax],
        'config': {
            'learning_rate': 0.0001
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


def get_ann_network_from_config(config):
    layer_struct = config['layer_structure']
    af = config['activation_functions']
    conf = config['config']

    return ANN(layer_struct, af, config=conf)


def do_ann_analysis(configurations, **kwargs):
    for config in configurations:
        do_single_ann_analysis(get_ann_network_from_config(config), **kwargs)


def do_single_ann_analysis(ann, epochs=20, do_welch_test=False, write_statistics=False):
    ann.load_input_data(normalize=True)
    errors, correctness = ann.train(epochs=epochs, visualize=False)
    log.debug('ANN average error: %.4f' % np.mean(errors))

    training_correctness = np.mean(np.argmax(ann.train_correct_labels, axis=1) == ann.predict(ann.train_input_data))
    testing_correctness = np.mean(np.argmax(ann.test_correct_labels, axis=1) == ann.predict(ann.test_input_data))

    log.info('ANN correctness on training data: %.4f' % training_correctness)
    log.info('ANN correctness on testing data: %.4f' % testing_correctness)

    if write_statistics:
        with open('analysis.txt', 'a') as file:
            statistics = '%s\n%.4f\n%.4f\n%s\n%s\n-\n' % (
                str(ann),
                training_correctness,
                testing_correctness,
                repr(correctness),
                repr(errors)
            )
            file.write(statistics)

    if do_welch_test:
        mnist_basics.minor_demo(ann)


if __name__ == "__main__":

    # Set up logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)

    """
    # Do analysis on all but the last ann configuration
    # do_ann_analysis(ANN_CONFIGURATIONS[:-1], epochs=20, do_welch_test=True, write_statistics=True)
    do_ann_analysis(
        ANN_CONFIGURATIONS,
        epochs=20,
        do_welch_test=True,
        write_statistics=True
    )
    """

    #  Analyse 20 runs of demo100 set
    data100, labels100 = mnist_basics.load_flat_text_cases('demo100_text.txt')
    correctness = []
    for run in range(5):
        a = get_ann_network_from_config(ANN_CONFIGURATIONS[0])
        a.load_input_data(normalize=True)
        a.train(epochs=20, include_test_set=False, visualize=False)
        results = a.blind_test(data100) # Automatic normalize
        correct_result = [i for i, j in zip(results, labels100) if i == j]
        log.info('Run %i gave correctness of %i' % (run + 1, len(correct_result)))
        correctness.append(len(correct_result) / 100)

    log.info('Average correctness: %.4f' % np.mean(correctness))

    """
    # Create a network using the default parameters
    a = get_ann_network_from_config(ANN_CONFIGURATIONS[0])
    a.load_input_data()

    train_data_cache = a.train_input_data
    train_labels_cache = a.train_correct_labels
    test_data_cache = a.test_input_data
    test_labels_cache = a.test_correct_labels

    # Train a bit and perform blind test
    a.train(epochs=20, include_test_set=False, visualize=False)

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
