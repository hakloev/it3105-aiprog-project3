# -*- encoding: utf-8 -*-

import logging
import os
from logging.config import dictConfig

import numpy as np

from config.configuration import LOG_CONFIG
from module5.ann import ANN, rectify, softmax, sigmoid, softplus, ERROR_FUNCTIONS
from module5.mnist import mnist_basics
from module5.mnist.mnist_basics import __mnist_path__
from module5.mnist import mnistdemo2


DO_BLIND_TEST = False
__analysis_path__ = os.path.realpath(os.path.dirname(__name__)) + '/module5/analysis/'

ANN_CONFIGURATIONS = [
    {
        'name': 'ANN_1',
        'layer_structure': [784, 1156, 784, 10],
        'activation_functions': [rectify, rectify, rectify, softmax],
        'config': {
            'learning_rate': 0.001,
            'error_function': ERROR_FUNCTIONS[1]
        }
    },
    {
        'name': 'ANN_2',
        'layer_structure': [784, 1568, 784, 10],
        'activation_functions': [rectify, rectify, rectify, softmax],
        'config': {
            'learning_rate': 0.001,
            'error_function': ERROR_FUNCTIONS[1]
        }
    },
    {
        'name': 'ANN_3',
        'layer_structure': [784, 620, 10],
        'activation_functions': [rectify, rectify, softmax],
        'config': {
            'learning_rate': 0.001,
            'error_function': ERROR_FUNCTIONS[1]
        }
    },
    {
        'name': 'ANN_4',
        'layer_structure': [784, 620, 10],
        'activation_functions': [sigmoid, softplus, softmax],
        'config': {
            'learning_rate': 0.001,
            'error_function': ERROR_FUNCTIONS[1]
        }
    },
    {
        'name': 'ANN_5',
        'layer_structure': [784, 392, 10],
        'activation_functions': [softplus, softplus, softmax],
        'config': {
            'learning_rate': 0.005,
            'error_function': ERROR_FUNCTIONS[1]
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

    points = []
    if do_welch_test:
        points = mnist_basics.minor_demo_hakloev(ann)
    log.debug('Complete demo: %s' % repr(points))

    if write_statistics:
        with open(__analysis_path__ + 'analysis.txt', 'a') as file:
            statistics = '%s\nTrain: %.4f\nTest: %.4f\nComplete demo: %s\n%s\n%s\n-\n' % (
                str(ann),
                training_correctness,
                testing_correctness,
                repr(points),
                repr(correctness),
                repr(errors)
            )
            file.write(statistics)


def do_demo_procedure(r, ann_config=2, epochs=20, d=__mnist_path__):
    ann = get_ann_network_from_config(ANN_CONFIGURATIONS[ann_config])
    ann.load_input_data(normalize=True)
    ann.train(epochs)
    print('------- READY TO RUN DEMO ------')
    print('1: Run demo')
    print('2: Exit')
    choice = input('Enter command: ')
    if choice == '1':
        # mnist_basics.minor_demo(ann)
        mnistdemo2.major_demo(ann, r, d)


if __name__ == "__main__":

    # Set up logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)

    do_demo_procedure(63, ann_config=2, epochs=25, d=__mnist_path__ + 'demo/')

    """
    # Do analysis on all but the last ann configuration
    # do_ann_analysis(ANN_CONFIGURATIONS[:-1], epochs=20, do_welch_test=True, write_statistics=True)
    do_ann_analysis(
        [ANN_CONFIGURATIONS[0]],
        epochs=15,
        do_welch_test=True,
        write_statistics=False
    )
    """
    """
    #  Analyse 20 runs of demo100 set
    data100, labels100 = mnist_basics.load_flat_text_cases('demo100_text.txt')

    total_correctness = []
    for config in [ANN_CONFIGURATIONS[0], ANN_CONFIGURATIONS[2]]:
        correctness = []
        for run in range(50):
            a = get_ann_network_from_config(config)
            a.load_input_data(normalize=True)
            a.train(epochs=20, include_test_set=False, visualize=False)
            results = a.blind_test(data100)  # Automatic normalize
            correct_result = [i for i, j in zip(results, labels100) if i == j]
            log.debug('%s run %i correctness is %i' % (config['name'], run + 1, len(correct_result)))
            correctness.append(len(correct_result) / 100)
        average_correctness = np.mean(correctness)
        total_correctness.append(average_correctness)
        log.debug('Correctness for all runs of %s: %s' % (config['name'], repr(correctness)))
        log.info('Avg correctness for all runs of %s: %.4f' % (config['name'], average_correctness))
    log.info('Correctness for all ANN configs %s' % repr(total_correctness))
    log.debug('Avg correctness for all ANN configs %s' % np.mean(total_correctness))
    """

    """
    # Running two configurations for 15 runs, each of 20 epoch, and doing test
    for config in [ANN_CONFIGURATIONS[0], ANN_CONFIGURATIONS[2]]:
        correctness = []
        for run in range(15):
            a = get_ann_network_from_config(config)
            a.load_input_data(normalize=True)
            a.train(epochs=20, include_test_set=False, visualize=False)
            results = mnist_basics.minor_demo_hakloev(a)
            correctness.append(results)
        log.info('Correctness for all runs of %s: %s' % (config['name'], repr(correctness)))
    """

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
