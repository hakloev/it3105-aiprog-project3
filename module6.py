# -*- coding: utf8 -*-
#
# Created by 'myth' on 11/16/15

import logging
import sys
from logging.config import dictConfig

from config.configuration import LOG_CONFIG
from module5.ann import ANN, rectify, softmax, ERROR_FUNCTIONS
from module6.control.browser import BrowserController
from module6.control.java_client_adapter import JavaAdapter
from module6.demo.ai2048demo import welch
from module6.points import create_run_lists
from module6.storage import Games


def load_raw_and_save(alternate=False, num_games=16, only_successful=False, vectorlength=16):
    """
    Loads raw data from game_data.txt, parses, transforms to alternate repr if set to True,
    then flattens the data, and pickles and saves the data to disk.
    """

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


def load_and_train(epochs=50, vectorlength=16):
    # Set up logging
    dictConfig(LOG_CONFIG)
    games = Games()

    # Use the following to load already parsed data
    games.load()
    games.flatten(vectorlength=vectorlength)

    # Network structure
    # Structure: [input_layer, hidden_layer, hidden_layer ... , output_layer]
    # Example: [784, 620, 100, 10]
    layer_structure = [vectorlength, 192, 128, 96, 4]
    # Example: [rectify, rectify, softmax]
    activation_functions = [rectify, rectify, rectify, rectify, softmax]
    # Remeber to change num_labels to 4, since we have Up, Right, Down, Left
    # Also we normalize the values. Don't know if it will affect anything,
    # but not taking any chances.
    cfg = {
        'learning_rate': 0.00001,
        'num_labels': 4,
        'training_batch_size': 512,
    }

    # Create a network using the default parameters
    net = ANN(layer_structure, activation_functions, config=cfg)
    net.load_input_data(normalize=False, module6_file=True)

    net.train(epochs=epochs, include_test_set=False)

    return net


def load_train_and_play_game(net, **kwargs):
    BrowserController(sys.argv[1:], net, gui_update_interval=0, **kwargs)


def load_train_and_store_stats(net, runs=10, **kwargs):
    results = []
    for z in range(runs):
        j = JavaAdapter(net, stats_filename='random_statistics.txt')
        j.connect()
        j.play(random=True, **kwargs)

        j = JavaAdapter(net, stats_filename='ann_statistics.txt')
        j.connect()
        j.play(random=False, **kwargs)

        # According to the task description, the lists must be in order: random, ann
        result = welch(*create_run_lists())
        print('Result run %d:' % z)
        print('%s' % result)
        results.append(result)

    print('--- Summary (%d runs) ---' % len(results))
    for r in results:
        print(r)


if __name__ == "__main__":
    # Set up logging
    dictConfig(LOG_CONFIG)
    log = logging.getLogger(__name__)

    # Setup
    veclength = 48
    # load_raw_and_save(alternate=False, only_successful=False, num_games=2048, vectorlength=veclength)
    a = load_and_train(epochs=30, vectorlength=veclength)

    # Start main loop
    while True:
        print('--- 2048 Neural Network --------------------')
        print("1: Perform 2048 3x50 runs and Welch's 2-Test")
        print("2: Play the 2048 game in the browser")
        print("3: Exit")
        print('')
        choice = input('Enter your choice: ')

        if choice == '1':
            load_train_and_store_stats(a, runs=20, vectorlength=veclength)
        elif choice == '2':
            load_train_and_play_game(a, vectorlength=veclength)
        else:
            sys.exit(0)
