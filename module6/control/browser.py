#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Browser connection for displaying the GUI of 2048.
Freely edited version of https://raw.githubusercontent.com/nneonneo/2048-ai/master/2048.py
"""

from __future__ import print_function
import time
import logging


class BrowserController(object):

    def __init__(self, argv, net, gui_update_interval=2):
        if net is None:
            raise ValueError("Need positional argument 'net' to be set")

        self._log = logging.getLogger(__name__)
        self._args = self._parse_args(argv)
        self._GUI_UPDATE_INTERVAL = gui_update_interval
        self._NET = net

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
        # print(board)
        # Connect to ANN here
        best_move = self._NET.predict([board])
        self._log.debug("Best move is %i" % best_move[0])
        return best_move

    def _play_game(self, game_ctrl):
        move_no = 0
        start = time.time()
        while 1:
            time.sleep(self._GUI_UPDATE_INTERVAL)
            state = game_ctrl.get_status()
            if state == 'ended':
                break
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
        max_val = max(max(row) for row in to_val(board))
        self._log.info("Game over. Final score %d; highest tile %d." % (score, max_val))

    def print_board(self, m):
        for row in m:
            self._log.debug('%i' % row, end=', ')

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


"""
def _to_val(c):
    if c == 0: return 0
    return 2**c


def to_val(m):
    return [[_to_val(c) for c in row] for row in m]


def _to_score(c):
    if c <= 1:
        return 0
    return (c-1) * (2**c)


def to_score(m):
    return [[_to_score(c) for c in row] for row in m]
"""