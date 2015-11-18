# -*- coding: utf8 -*-
#
# Created by 'myth' on 11/17/15

import numpy as np
import logging
import os
from random import randint
import socket

from module6.control.browser import valid_move

RECV_BYTE_SIZE = 1024
DEFAULT_FILENAME = 'play_statistics.txt'
HOST = 'localhost'
PORT = 57315


class JavaAdapter(object):
    """
    Socket communication channel for Java version of 2048
    """

    def __init__(self, network, stats_filename=None):
        self._log = logging.getLogger(__name__)
        self._log.info('Initiating 2048 Game JavaAdapter...')
        self.sock = socket.socket()
        self.net = network
        if not stats_filename:
            stats_filename = DEFAULT_FILENAME
        self.stats_filename = stats_filename
        self.statistics = []

    def connect(self, host=HOST, port=PORT):
        self.sock.connect((host, port))
        self._log.info('Connected to %s:%d' % (host, port))

    def play(self, runs=50, random=False):
        """
        Starts the main run loop for playing the game using the network.
        """

        self._log.info('Starting play loop with %d runs...' % runs)

        # Clear the file
        with open(self.stats_filename, 'w') as f:
            f.write('')

        for i in range(runs):
            last_board = None
            while True:
                board = self.sock.recv(RECV_BYTE_SIZE).decode('utf-8')
                board = board.strip()
                if board == 'END':
                    max_tile = max(last_board[0])
                    self._log.debug('Game ended. Max tile was: %d' % max_tile)
                    with open(os.path.join('.', self.stats_filename), 'a') as f:
                        f.write("%d\n" % max_tile)
                    break
                board = self.transform_board(board)
                last_board = board
                if not random:
                    results = self.net.predict_all(board)
                    direction = determine_best_move(results[0], board[0].reshape(4, 4))
                else:
                    direction = randint(0, 3)
                self.sock.send(bytes('%d\n' % direction, encoding='utf-8'))

        self.sock.close()

    def reset(self):
        """
        Reinitiate the socket
        """

        self.sock = socket.socket()

    @staticmethod
    def transform_board(board):
        tiles = board.split(',')
        tiles = map(int, tiles)
        return np.array([np.array(list(tiles))])


def determine_best_move(moves, m):
    """
    Returnes the move with the highest probability, where the direction is possible
    :param moves: Moves to check validity of
    :param m: The board as a 2D-matrix
    :return: The direction to move
    """
    moves = [(i, p) for i, p in enumerate(moves)]
    best = max(filter(lambda x: valid_move(x[0], m), moves), key=lambda x: x[1])

    return best[0]