# -*- coding: utf8 -*-
#
# Created by 'myth' on 11/16/15

import logging
from logging.config import dictConfig

from theano.tensor.nnet import sigmoid, hard_sigmoid

from module5.ann import ANN, rectify, softmax, softmax_biased
from module6.storage import Games

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
    games = Games()

    # Use the following to parse 512 games from scratch, that have at least reached 2048,
    # then add an additional filter of min_maxtile 4096, flatten to Nx16 np.ndArray and gzip pickle

    games.parse_from_raw_game_data(num_games=2048, only_successful=True)
    boards, labels = games.flatten()
    print('Total labels: %d' % len(labels))
    print('Total board states: %d' % len(boards))
    games.save()

    """
    # Use the following to load already parsed data
    games.load()
    boards, labels = games.flatten()

    # Network structure
    # Structure: [input_layer, hidden_layer, hidden_layer ... , output_layer]
    # Example: [784, 620, 100, 10]
    layer_structure = [16, 32, 16, 4]
    # Example: [rectify, rectify, softmax]
    activation_functions = [rectify, rectify, rectify, softmax_biased]
    # Remeber to change num_labels to 4, since we have Up, Right, Down, Left
    # Also we normalize the values. Don't know if it will affect anything,
    # but not taking any chances.
    cfg = {
        'learning_rate': 0.00001,
        'num_labels': 4,
        'normalize_max_value': 1,
        'training_batch_size': 512,
    }

    # Create a network using the default parameters
    a = ANN(layer_structure, activation_functions, config=cfg)
    a.load_input_data(normalize=True, module6_file=True)

    train_data_cache = a.train_input_data
    train_labels_cache = a.train_correct_labels
    test_data_cache = a.test_input_data
    test_labels_cache = a.test_correct_labels

    a.train(epochs=1000, include_test_set=False)
    """
