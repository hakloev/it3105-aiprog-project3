#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Browser connection for displaying the GUI of 2048.
Freely edited version of https://raw.githubusercontent.com/nneonneo/2048-ai/master/2048.py
"""

from __future__ import print_function
import random
import time
import logging
import sys

from module6.storage import transform


class BrowserController(object):

    def __init__(self, argv, net, gui_update_interval=0.5, vectorlength=16):
        if net is None:
            raise ValueError("Need positional argument 'net' to be set")

        self._log = logging.getLogger(__name__)
        self._args = self._parse_args(argv)
        self._GUI_UPDATE_INTERVAL = gui_update_interval
        self._NET = net
        self._vec_len = vectorlength

        if self._args.browser == 'firefox':
            from module6.control.ffctrl import FirefoxRemoteControl
            if self._args.port is None:
                self._args.port = 32000
            ctrl = FirefoxRemoteControl(self._args.port)

        if self._args.ctrlmode == 'keyboard':
            from module6.control.gamectrl import Keyboard2048Control
            game_ctrl = Keyboard2048Control(ctrl)
        elif self._args.ctrlmode == 'fast':
            from module6.control.gamectrl import Fast2048Control
            game_ctrl = Fast2048Control(ctrl)
        elif self._args.ctrlmode == 'hybrid':
            from module6.control.gamectrl import Hybrid2048Control
            game_ctrl = Hybrid2048Control(ctrl)

        if game_ctrl.get_status() == 'ended':
            game_ctrl.restart_game()

        self._play_game(game_ctrl)

    def find_best_move(self, m):
        board = self._to_ann_board(m)
        # Connect to ANN here
        # best_move = self._NET.predict([board])  # Returns the best move
        # self._log.debug("Best move is %i" % best_move[0])

        if self._vec_len != 16:
            board = transform(board)

        best_move_all = self._NET.predict_all([board])  # Returns a list of all moves and their probability
        self._log.debug("All moves probability %s" % best_move_all)
        direction = self.determine_best_move(best_move_all[0], m)

        return direction

    def determine_best_move(self, moves, m):
        """
        Returnes the move with the highest probability, where the direction is possible
        :param moves: Moves to check validity of
        :param m: The board as a 2D-matrix
        :return: The direction to move
        """
        moves = [(i, p) for i, p in enumerate(moves)]
        best = max(filter(lambda x: valid_move(x[0], m), moves), key=lambda x: x[1])
        self._log.debug("Best move is %s with a value of %s" % (best[0], best[1]))

        return best[0]

    def _play_game(self, game_ctrl):
        move_no = 0
        start = time.time()
        while 1:
            time.sleep(self._GUI_UPDATE_INTERVAL)
            state = game_ctrl.get_status()
            if state == 'ended':
                game_ctrl.restart_game()
                move_no = 0
                continue
            elif state == 'won':
                b = game_ctrl.get_board()
                max_tile = max(max(l) for l in b)
                if max_tile > 1024:
                    cont = input('2048 reached! Continue? yes/no')
                    if cont == 'no':
                        sys.exit(0)
                time.sleep(2)
                game_ctrl.continue_game()

            move_no += 1
            board = game_ctrl.get_board()
            move = self.find_best_move(board)
            if move < 0:
                break
            self._log.debug("%010.6f: Score %i, Move %i: %s" % (
                time.time() - start,
                game_ctrl.get_score(),
                move_no,
                self.move_name(move)
            ))

            game_ctrl.execute_move(move)

        score = game_ctrl.get_score()
        board = game_ctrl.get_board()
        # The following line will fail, as it is not implemented yet
        max_val = max(max(row) for row in board)
        self._log.info("Game over. Final score %d; highest tile %d." % (score, max_val))

    def print_board(self, m):
        for row in m:
            self._log.debug('%i' % row, end=', ')

    def to_val(self, m):
        return [[self._to_val(c) for c in row] for row in m]

    @staticmethod
    def _to_val(c):
        if c == 0:
            return 0
        return 2**c

    @staticmethod
    def _to_ann_board(m):
        board = []
        # print(m)
        for row in m:
            board.extend(row)
        return board

    @staticmethod
    def move_name(move):
        return ['up', 'right', 'down', 'left'][move]

    @staticmethod
    def _parse_args(argv):
        import argparse
        parser = argparse.ArgumentParser(description="Use the AI to play 2048 via browser control")
        parser.add_argument('-p', '--port', help="Port number to control on (default: 32000 for Firefox, 9222 for Chrome)", type=int)
        parser.add_argument('-b', '--browser', help="Browser you're using. Only Firefox with the Remote Control extension, and Chrome with remote debugging, are supported right now.", default='firefox', choices=('firefox', 'chrome'))
        parser.add_argument('-k', '--ctrlmode', help="Control mode to use. If the browser control doesn't seem to work, try changing this.", default='hybrid', choices=('keyboard', 'fast', 'hybrid'))

        return parser.parse_args(argv)


def valid_move(dir, m):
    """
    Test if a move in the given direction is possible
    U: 0, R: 1: D: 2: L: 3
    :param dir: Direction to check
    :param m: The board to check on
    :return: A boolean telling if a move is valid or not
    """
    size = range(0, 4)
    if dir == 3 or dir == 1:
        for x in size:
            col = m[x]
            for y in size:
                if y < 4 - 1 and col[y] == col[y + 1] and col[y] != 0:
                    return True
                if dir == 1 and y > 0 and col[y] == 0 and col[y - 1] != 0:
                    return True
                if dir == 3 and y < 4 - 1 and col[y] == 0 and col[y + 1] != 0:
                    return True

    if dir == 0 or dir == 2:
        for y in size:
            line = get_column(y, m)
            for x in size:
                if x < 4 - 1 and line[x] == line[x + 1] and line[x] != 0:
                    return True
                if dir == 2 and x > 0 and line[x] == 0 and line[x - 1] != 0:
                    return True
                if dir == 0 and x < 4 - 1 and line[x] == 0 and line[x + 1] != 0:
                    return True
    return False


def get_column(y, m):
    return [m[i][y] for i in range(4)]


class BrowserControllerRandom(BrowserController):
    """
    Plays randomly
    """

    def __init__(self, *args, **kwargs):
        super(BrowserControllerRandom, self).__init__(*args, **kwargs)

    def find_best_move(self, m):
        return random.randint(0, 3)

    def _play_game(self, game_ctrl):
        move_no = 0
        start = time.time()
        while 1:
            time.sleep(self._GUI_UPDATE_INTERVAL)
            state = game_ctrl.get_status()
            if state == 'ended':
                self._log.debug('Random player ended. Board state: %s' % repr(game_ctrl.get_board()))
                self._log.debug('Highest tile was: %d' % max(self._to_ann_board(game_ctrl.get_board())))
            elif state == 'won':
                time.sleep(0.75)
                game_ctrl.continue_game()

            move_no += 1
            board = game_ctrl.get_board()
            move = self.find_best_move(board)
            if move < 0:
                break
            self._log.debug("%010.6f: Score %i, Move %i: %s" % (
                time.time() - start,
                game_ctrl.get_score(),
                move_no,
                self.move_name(move)
            ))

            game_ctrl.execute_move(move)

        score = game_ctrl.get_score()
        board = game_ctrl.get_board()
        # The following line will fail, as it is not implemented yet
        max_val = max(max(row) for row in board)
        self._log.info("Game over. Final score %d; highest tile %d." % (score, max_val))
