#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Browser connection for displaying the GUI of 2048.
Taken from https://raw.githubusercontent.com/nneonneo/2048-ai/master/2048.py
"""

from __future__ import print_function

import time

NET = None


def to_ann_board(m):
    board = []
    print(m)
    for row in m:
        board.extend(row)
    return board


def print_board(m):
    for row in m:
        print('%i' % row, end=', ')

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

"""
if MULTITHREAD:
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(4)

    def score_toplevel_move(args):
        print(args)
        return ailib.score_toplevel_move(*args)

    def find_best_move(m):
        board = to_ann_board(m)
        print_board(board)

        # Connect to ANN here and get move for state

        best_move = NET.predict([board])
        print(""best_move)

        scores = pool.map(score_toplevel_move, [(board, move) for move in range(4)])
        bestmove, bestscore = max(enumerate(scores), key=lambda x:x[1])

        if bestscore == 0:
            return -1
        return best_move
else:
"""


def find_best_move(m):
    board = to_ann_board(m)
    print(board)
    # Connect to ANN here
    best_move = NET.predict([board])
    print("Best move is %i" % best_move[0])
    return best_move


def move_name(move):
    return ['up', 'right', 'down', 'left'][move]


def play_game(gamectrl):
    moveno = 0
    start = time.time()
    while 1:
        time.sleep(2)
        state = gamectrl.get_status()
        if state == 'ended':
            break
        elif state == 'won':
            time.sleep(0.75)
            gamectrl.continue_game()

        moveno += 1
        board = gamectrl.get_board()
        move = find_best_move(board)
        if move < 0:
            break
        print("%010.6f: Score %d, Move %d: %s" % (time.time() - start, gamectrl.get_score(), moveno, move_name(move)))
        gamectrl.execute_move(move)

    score = gamectrl.get_score()
    board = gamectrl.get_board()
    max_val = max(max(row) for row in to_val(board))
    print("Game over. Final score %d; highest tile %d." % (score, max_val))


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(description="Use the AI to play 2048 via browser control")
    parser.add_argument('-p', '--port', help="Port number to control on (default: 32000 for Firefox, 9222 for Chrome)", type=int)
    parser.add_argument('-b', '--browser', help="Browser you're using. Only Firefox with the Remote Control extension, and Chrome with remote debugging, are supported right now.", default='firefox', choices=('firefox', 'chrome'))
    parser.add_argument('-k', '--ctrlmode', help="Control mode to use. If the browser control doesn't seem to work, try changing this.", default='hybrid', choices=('keyboard', 'fast', 'hybrid'))

    return parser.parse_args(argv)


def main(argv, net):
    if net is None:
        raise ValueError("Need positional argument 'net' to be set")

    global NET
    NET = net

    args = parse_args(argv)

    if args.browser == 'firefox':
        from module6.control.ffctrl import FirefoxRemoteControl
        if args.port is None:
            args.port = 32000
        ctrl = FirefoxRemoteControl(args.port)

    if args.ctrlmode == 'keyboard':
        from module6.control.gamectrl import Keyboard2048Control
        game_ctrl = Keyboard2048Control(ctrl)
    elif args.ctrlmode == 'fast':
        from module6.control.gamectrl import Fast2048Control
        game_ctrl = Fast2048Control(ctrl)
    elif args.ctrlmode == 'hybrid':
        from module6.control.gamectrl import Hybrid2048Control
        game_ctrl = Hybrid2048Control(ctrl)

    if game_ctrl.get_status() == 'ended':
        game_ctrl.restart_game()

    play_game(game_ctrl)