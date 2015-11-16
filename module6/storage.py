# -*- coding: utf8 -*-
#
# Created by 'myth' on 11/16/15

from datetime import datetime
import glob
import gzip
import logging
import numpy as np
from os import path
import pickle
import time

__data_directory__ = path.join('.', 'module6', 'data')


def load_pickle(filename=None):
    """
    Loads a state from GZipped pickled binary dumps
    """

    if not filename:
        data_directory = path.join('.', 'module6', 'data')
        filename = max(glob.glob(path.join(data_directory, '*.2048.gz')), key=path.getctime)

    logging.getLogger(__name__).debug('Loading file: %s' % filename)

    try:
        with gzip.open(filename) as f:
            return pickle.load(f)
    except OSError as e:
        logging.getLogger(__name__).error('Could not open file: %s (%s)' % (filename, e))
        return None


def save_pickle(payload, filename=None):
    """
    Saves the state of payload in GZipped pickled binary
    """

    if not filename:
        filename = '%s%d.2048.gz' % (datetime.now().strftime('%Y-%m-%d'), time.clock())
        filename = path.join(__data_directory__, filename)

    try:
        with gzip.open(filename, 'wb') as f:
            pickle.dump(payload, f)
    except OSError as e:
        logging.getLogger(__name__).error('Could not open file: %s (%s)' % (filename, e))

    logging.getLogger(__name__).debug('Saved file: %s' % filename)

    return filename


def save_training(training_set, filename=None):
    """
    Saves a training set as gzipped pickle
    :param training_set: A numpy matrix representing the training data
    :param filename: A filename to store the training set as
    :return: The path to the file that was saves
    """

    if not filename:
        filename = path.join(__data_directory__, 'training.2048.gz')

    return save_pickle(training_set, filename=filename)


def load_training(filename=None):
    """
    Loads a training set from gzipped pickle
    :param filename: A filename to the training set
    :return: A tuple of boards and correct moves
    """

    if not filename:
        filename = path.join(__data_directory__, 'training.2048.gz')

    # Just in case the file does not exist, we need to have None for data and None for labels
    data = load_pickle(filename=filename)
    if not data:
        return None, None
    return data


def save_test(test_set, filename=None):
    """
    Saves a training set as gzipped pickle
    :param test_set: A numpy matrix representing the training data
    :param filename: A filename to store the training set as
    :return: The path to the file that was saves
    """

    if not filename:
        filename = path.join(__data_directory__, 'test.2048.gz')

    return save_training(test_set, filename=filename)


def load_test(filename=None):
    """
    Loads a pickled test set
    :param filename: The filename to the testset
    :return: A tuple of boards and correct moves
    """

    if not filename:
        filename = path.join(__data_directory__, 'test.2048.gz')

    # Just in case the file does not exist, we need to have None for data and None for labels
    data = load_training(filename)
    if not data:
        return None, None
    return data


def load_raw_game_data(games=1, only_successful=True):
    """
    Loads the raw game_data from file as a generator
    :return: A tuple containing a list of board state lists, and corresponding correct moves
    """

    log = logging.getLogger(__name__)
    log.info(
        'Loading raw game data, this may take a while... [Games: %d Only successful: %s]' % (games, only_successful)
    )

    def finalize(gl, os=only_successful):
        """
        Performs final checks on game_list before returning
        """

        if os:
            gl = list(filter(lambda game: max(game[0][-1]) >= 2048, gl))
        log.info('Parsing complete, returning %d games' % len(gl))
        return gl

    with open(path.join(__data_directory__, 'game_data.txt')) as f:
        game_list = []

        # Start with clean game-internal structures
        boards = []
        moves = []
        maximum = 0

        # For each line in the raw data file
        for line in f:
            line = line.strip()

            # If we have EOF or game separator character
            if not line or line == '-':
                game_list.append((boards, moves))

                if len(game_list) == games:
                    return finalize(game_list)
                # Clear game-internal structures
                boards = []
                moves = []
                maximum = 0

            else:
                values = list(map(int, line.split(',')))
                correct_move = values.pop()
                current_max_tile = max(values)

                # If we have a sudden drop in max tile (Ctrl+c and restarting)
                if current_max_tile < maximum:
                    game_list.append((boards, moves))
                    if len(game_list) == games:
                        return finalize(game_list)
                    # Clear game-internal structures
                    boards = []
                    moves = []
                    maximum = 0

                # Regular new board state and correct move prediction
                else:
                    boards.append(values)
                    moves.append(correct_move)
                    maximum = current_max_tile

    return finalize(game_list)


class Games(object):
    """
    Data container for games
    """

    def __init__(self, games=None):
        """
        Constructs a Game container.
        :param games: A dictionary containing a list of lists of board vectors
        :return: A list of lists of correct moves
        """
        if games:
            self.games = games
        else:
            self.games = {
                'boards': [],
                'moves': []
            }

        self.games_flat = None
        self.labels_flat = None

    def filter(self, lowest_max_tile=2048):
        """
        Filters through current dataset and removes games not meeting threshold
        :param lowest_max_tile: The minimum max-tile threshold for a game
        """

        log = logging.getLogger(__name__)
        log.info('Filtering dataset, lower bound: %d' % lowest_max_tile)

        boards = []
        moves = []
        for i, game in enumerate(self.games['boards']):
            if max(game[-1]) >= lowest_max_tile:
                boards.append(self.games['boards'][i])
                moves.append(self.games['moves'][i])

        log.info('Remaining games: %d' % len(moves))

        self.games['boards'] = boards
        self.games['moves'] = moves

    def parse_from_raw_game_data(self, num_games=1, only_successful=True):
        """
        Reads raw game data from txt file, and inserts lists of board vectors and corresponding moves
        into this object
        """

        logging.getLogger(__name__).info('Parsing raw game data...')

        for boards, labels in load_raw_game_data(games=num_games, only_successful=only_successful):
            assert len(boards) == len(labels)
            logging.getLogger(__name__).debug(
                'Adding game with %d moves and highest score: %d' % (len(labels), max(boards[-1]))
            )
            self.games['boards'].append(boards)
            self.games['moves'].append(labels)

    def games(self):
        """
        Returns a generator of game (baord) lists and label lists
        :return: A board and label list generator
        """

        for i, game in enumerate(self.games['boards']):
            yield game, self.games['moves'][i]

    def flatten(self):
        """
        Returns a list of flattened 16-element board vectors and a list of correct moves corresponding
        to the board states.
        :return: A list of board vectors and a list of corresponding correct moves
        """

        log = logging.getLogger(__name__)
        log.debug('Flattening datastructure...')

        if self.games_flat is None and self.labels_flat is None:
            out_games = np.array([], dtype='uint32')
            for game in self.games['boards']:
                out_games = np.append(out_games, np.array([b for b in game], dtype='uint32'))

            self.games_flat = out_games.reshape((len(out_games) / 16, 16))
            out_labels = np.array([], dtype='uint32')
            for label_set in self.games['moves']:
                out_labels = np.append(out_labels, np.array(label_set))
            out_labels.flatten()
            self.labels_flat = out_labels

            # Final check to see that we have matching board and move vectors
            assert len(self.games_flat) == len(self.labels_flat)

        return self.games_flat, self.labels_flat

    def load(self, test=False):
        """
        Loads existing pickled training set
        """

        log = logging.getLogger(__name__)
        start_time = time.time()
        log.info('Loading datastructure... [Test: %s]' % test)

        if test:
            self.games_flat, self.labels_flat = load_test()
        else:
            self.games_flat, self.labels_flat = load_training()

        log.info('Datastructure decompressed in %.2fs' % (time.time() - start_time))

    def save(self, test=False):
        """
        Saves the current state of the Games object
        """

        log = logging.getLogger(__name__)
        log.info('Compressing datastructure... [Test: %s] (This may take a while depending on set size!' % test)
        start_time = time.time()

        if test:
            filename = save_training((self.games_flat, self.labels_flat))
        else:
            filename = save_training((self.games_flat, self.labels_flat))

        log.info('Datastructure saved as %s in %.2fs' % (filename, time.time() - start_time))
