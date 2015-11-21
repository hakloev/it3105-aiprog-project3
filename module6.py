# -*- coding: utf8 -*-
#
# Created by 'myth' on 11/16/15

import logging
import sys
from logging.config import dictConfig

from theano.tensor.nnet import sigmoid

from module5.ann import ANN, rectify, softmax
from module6.storage import Games
from module6.points import calculate_points, create_run_lists
from module6.control.browser import BrowserController, BrowserControllerRandom
from module6.control.java_client_adapter import JavaAdapter
from module6.demo.ai2048demo import welch


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

log = logging.getLogger(__name__)


def load_raw_and_save(alternate=False, num_games=16, only_successful=False):
    """
    Loads raw data from game_data.txt, parses, transforms to alternate repr if set to True,
    then flattens the data, and pickles and saves the data to disk.
    """

    # Set up logging
    dictConfig(LOG_CONFIG)
    games = Games()

    # Use the following to parse 512 games from scratch, that have at least reached 2048,
    # then add an additional filter of min_maxtile 4096, flatten to Nx16 np.ndArray and gzip pickle

    games.parse_from_raw_game_data(num_games=num_games, only_successful=only_successful)
    if alternate:
        games.transform_to_alternate_representation()
    boards, labels = games.flatten()
    print('Total labels: %d' % len(labels))
    print('Total board states: %d' % len(boards))
    games.save()


def load_train_and_run():
    # Set up logging
    dictConfig(LOG_CONFIG)
    games = Games()

    # Use the following to load already parsed data
    games.load()
    boards, labels = games.flatten()

    # Network structure
    # Structure: [input_layer, hidden_layer, hidden_layer ... , output_layer]
    # Example: [784, 620, 100, 10]
    layer_structure = [16, 32, 4]
    # Example: [rectify, rectify, softmax]
    activation_functions = [rectify, rectify, softmax]
    # Remeber to change num_labels to 4, since we have Up, Right, Down, Left
    # Also we normalize the values. Don't know if it will affect anything,
    # but not taking any chances.
    cfg = {
        'learning_rate': 0.0008,
        'num_labels': 4,
        'normalize_max_value': 1,
        'training_batch_size': 256,
    }

    # Create a network using the default parameters
    a = ANN(layer_structure, activation_functions, config=cfg)
    a.load_input_data(normalize=False, module6_file=True)

    a.train(epochs=1000, include_test_set=False)


def load_train_and_play_game(epochs=500):
    # Set up logging
    dictConfig(LOG_CONFIG)
    games = Games()

    # Use the following to load already parsed data
    games.load()
    boards, labels = games.flatten()

    # Network structure
    # Structure: [input_layer, hidden_layer, hidden_layer ... , output_layer]
    # Example: [784, 620, 100, 10]
    layer_structure = [16, 32, 16, 4]
    # Example: [rectify, rectify, softmax]
    activation_functions = [rectify, rectify, rectify, softmax]
    # Remeber to change num_labels to 4, since we have Up, Right, Down, Left
    # Also we normalize the values. Don't know if it will affect anything,
    # but not taking any chances.
    cfg = {
        'learning_rate': 0.00001,
        'num_labels': 4,
        'training_batch_size': 1,
    }

    # Create a network using the default parameters
    a = ANN(layer_structure, activation_functions, config=cfg)
    a.load_input_data(normalize=False, module6_file=True)

    a.train(epochs=epochs, include_test_set=False)

    BrowserController(sys.argv[1:], a, gui_update_interval=0)


def load_train_and_store_stats(random=False, epochs=1000):
    # Set up logging
    dictConfig(LOG_CONFIG)
    games = Games()

    # Use the following to load already parsed data
    games.load()
    boards, labels = games.flatten()

    # Network structure
    # Structure: [input_layer, hidden_layer, hidden_layer ... , output_layer]
    # Example: [784, 620, 100, 10]
    layer_structure = [16, 32, 4]
    # Example: [rectify, rectify, softmax]
    activation_functions = [rectify, rectify, softmax]
    # Remeber to change num_labels to 4, since we have Up, Right, Down, Left
    # Also we normalize the values. Don't know if it will affect anything,
    # but not taking any chances.
    cfg = {
        'learning_rate': 0.003,
        'num_labels': 4,
        'training_batch_size': 512,
    }

    # Create a network using the default parameters
    a = ANN(layer_structure, activation_functions, config=cfg)
    a.load_input_data(normalize=False, module6_file=True)

    a.train(epochs=epochs, include_test_set=False)

    if random:
        j = JavaAdapter(a, stats_filename='random_statistics.txt')
    else:
        j = JavaAdapter(a, stats_filename='ann_statistics.txt')

    j.connect()
    j.play(random=random)


if __name__ == "__main__":
    """
    results = []

    for i in range(50):
        load_train_and_store_stats(random=False, epochs=25)
        load_train_and_store_stats(random=True, epochs=1)
        result = calculate_points()
        print('Current iteration score: %d' % result)
        results.append(result)

    print(results)
    """
    load_train_and_play_game(epochs=1000)

    # load_raw_and_save(alternate=True, only_successful=False, num_games=2)

    """
    load_train_and_store_stats(random=False, epochs=500)
    load_train_and_store_stats(random=True, epochs=1)
    # According to the task description, the lists must be in order: random, ann
    result = welch(*create_run_lists())
    print('%s' % result)
    """
