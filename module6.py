# -*- coding: utf8 -*-
#
# Created by 'myth' on 11/16/15

import logging
import sys
from logging.config import dictConfig

from config.configuration import LOG_CONFIG
from module5.ann import ANN, rectify, softmax
from module6.control.browser import BrowserController
from module6.control.java_client_adapter import JavaAdapter
from module6.demo.ai2048demo import welch
from module6.points import create_run_lists
from module6.storage import Games


def load_raw_and_save(alternate=False, num_games=16, only_successful=False, discrete=False, vectorlength=16):
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
    boards, labels = games.flatten(vectorlength=vectorlength)
    print('Total labels: %d' % len(labels))
    print('Total board states: %d' % len(boards))
    games.save()


def load_train_and_run(vectorlength=16):
    # Set up logging
    dictConfig(LOG_CONFIG)
    games = Games()

    # Use the following to load already parsed data
    games.load()
    boards, labels = games.flatten(vectorlength=vectorlength)

    # Network structure
    # Structure: [input_layer, hidden_layer, hidden_layer ... , output_layer]
    # Example: [784, 620, 100, 10]
    layer_structure = [vectorlength, 32, 4]
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


def load_train_and_play_game(epochs=500, vectorlength=16):
    # Set up logging
    dictConfig(LOG_CONFIG)
    games = Games()

    # Use the following to load already parsed data
    games.load()
    boards, labels = games.flatten(vectorlength=vectorlength)

    # Network structure
    # Structure: [input_layer, hidden_layer, hidden_layer ... , output_layer]
    # Example: [784, 620, 100, 10]
    layer_structure = [vectorlength, 192, 4]
    # Example: [rectify, rectify, softmax]
    activation_functions = [rectify, rectify, softmax]
    # Remeber to change num_labels to 4, since we have Up, Right, Down, Left
    # Also we normalize the values. Don't know if it will affect anything,
    # but not taking any chances.
    cfg = {
        'learning_rate': 0.00005,
        'num_labels': 4,
        'training_batch_size': 512,
    }

    # Create a network using the default parameters
    a = ANN(layer_structure, activation_functions, config=cfg)
    a.load_input_data(normalize=False, module6_file=True)

    a.train(epochs=epochs, include_test_set=False)

    BrowserController(sys.argv[1:], a, gui_update_interval=0, vectorlength=vectorlength)


def load_train_and_store_stats(epochs=1000, vectorlength=16, runs=10):
    # Set up logging
    dictConfig(LOG_CONFIG)
    games = Games()

    # Use the following to load already parsed data
    games.load()
    boards, labels = games.flatten(vectorlength=vectorlength)

    # Network structure
    # Structure: [input_layer, hidden_layer, hidden_layer ... , output_layer]
    # Example: [784, 620, 100, 10]
    layer_structure = [vectorlength, 128, 96, 64, 4]
    # Example: [rectify, rectify, softmax]
    activation_functions = [rectify, rectify, rectify, rectify, softmax]
    # Remeber to change num_labels to 4, since we have Up, Right, Down, Left
    # Also we normalize the values. Don't know if it will affect anything,
    # but not taking any chances.
    cfg = {
        'learning_rate': 0.000001,
        'num_labels': 4,
        'training_batch_size': 512,
    }

    # Create a network using the default parameters
    a = ANN(layer_structure, activation_functions, config=cfg)
    a.load_input_data(normalize=False, module6_file=True)

    a.train(epochs=epochs, include_test_set=False)

    results = []
    for z in range(runs):
        j = JavaAdapter(a, stats_filename='random_statistics.txt')
        j.connect()
        j.play(random=True, vectorlength=vectorlength)

        j = JavaAdapter(a, stats_filename='ann_statistics.txt')
        j.connect()
        j.play(random=False, vectorlength=vectorlength)

        # According to the task description, the lists must be in order: random, ann
        result = welch(*create_run_lists())
        print('%s' % result)
        results.append(result)

    for r in results:
        print(r)


if __name__ == "__main__":
    log = logging.getLogger(__name__)

    # load_train_and_play_game(epochs=10, vectorlength=64)

    # load_train_and_play_game(epochs=100, vectorlength=16)
    # load_raw_and_save(alternate=True, only_successful=False, num_games=2048, vectorlength=64)

    load_train_and_store_stats(epochs=50, vectorlength=64, runs=20)

    # load_train_and_store_stats(epochs=2, vectorlength=16, runs=20)

