# -*- encoding: utf-8 -*-

import logging
from logging.config import dictConfig

from theano.tensor.nnet import sigmoid

from module5.mnist import mnist_basics
from module5.ann import ANN, rectify, softmax, relu

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
    activation_functions = [rectify, rectify, sigmoid]

    # Create a network using the default parameters
    a = ANN(layer_structure, activation_functions)
    a.load_input_data()

    train_data_cache = a.train_input_data
    train_labels_cache = a.train_correct_labels
    test_data_cache = a.test_input_data
    test_labels_cache = a.test_correct_labels

    # Train a bit and perform blind test
    a.train(epochs=5)
    feature_sets, labels = mnist_basics.load_all_flat_cases(type="testing")
    print(a.blind_test(feature_sets[:10]))

    # MULTIPLE NETWORK GENERATION AND TESTING

    # Create new net
    layer_structure = [784, 620, 10]
    activation_functions = [relu, relu, softmax]
    a = ANN(layer_structure, activation_functions)

    a.train_input_data = train_data_cache
    a.train_correct_labels = train_labels_cache
    a.test_input_data = test_data_cache
    a.test_correct_labels = test_labels_cache

    # Train current net
    a.train(epochs=10)

    # Create new net
    layer_structure = [784, 784, 620, 10]
    activation_functions = [relu, relu, relu, relu]
    a = ANN(layer_structure, activation_functions, config={'learning_rate': 0.030})

    a.train_input_data = train_data_cache
    a.train_correct_labels = train_labels_cache
    a.test_input_data = test_data_cache
    a.test_correct_labels = test_labels_cache

    # Train current net
    a.train(epochs=10)

    # Create new net
    layer_structure = [784, 784, 620, 10]
    activation_functions = [rectify, relu, relu, relu]
    a = ANN(layer_structure, activation_functions, config={'learning_rate': 0.025})

    a.train_input_data = train_data_cache
    a.train_correct_labels = train_labels_cache
    a.test_input_data = test_data_cache
    a.test_correct_labels = test_labels_cache

    # Train current net
    a.train(epochs=10)

    # Create new net
    layer_structure = [784, 512, 128, 10]
    activation_functions = [rectify, rectify, softmax, softmax]
    a = ANN(layer_structure, activation_functions, config={'learning_rate': 0.015})

    a.train_input_data = train_data_cache
    a.train_correct_labels = train_labels_cache
    a.test_input_data = test_data_cache
    a.test_correct_labels = test_labels_cache

    # Train current net
    a.train(epochs=10)

    # Create new net
    layer_structure = [784, 320, 10]
    activation_functions = [rectify, rectify, softmax]
    a = ANN(layer_structure, activation_functions)

    a.train_input_data = train_data_cache
    a.train_correct_labels = train_labels_cache
    a.test_input_data = test_data_cache
    a.test_correct_labels = test_labels_cache

    # Train current net
    a.train(epochs=10)

    # Create new net
    layer_structure = [784, 620, 10]
    activation_functions = [relu, relu, softmax]
    a = ANN(layer_structure, activation_functions)

    a.train_input_data = train_data_cache
    a.train_correct_labels = train_labels_cache
    a.test_input_data = test_data_cache
    a.test_correct_labels = test_labels_cache

    # Train current net
    a.train(epochs=10)

