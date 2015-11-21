# -*- coding: utf-8 -*-
import logging
import re
import time
import json
import random
import numpy as np


class Generic2048Control(object):
    def __init__(self, ctrl):
        self._log = logging.getLogger(__name__)
        self.ctrl = ctrl
        self.setup()

    def setup(self):
        raise NotImplementedError()

    def execute(self, cmd):
        return self.ctrl.execute(cmd)

    def get_status(self):
        # Check if the game is in an unusual state.
        return self.execute('''
            var messageContainer = document.querySelector(".game-message");
            if(messageContainer.className.search(/game-over/) !== -1) {"ended"}
            else if(messageContainer.className.search(/game-won/) !== -1) {"won"}
            else {"running"}
            ''')

    def restart_game(self):
        self.send_key_event('keydown', 82)
        time.sleep(0.1)
        self.send_key_event('keyup', 82)

        self.send_key_event('keydown', 32)
        time.sleep(0.1)
        self.send_key_event('keyup', 32)

    def continue_game(self):
        # Continue the game. Only works if the game is in the 'won' state.
        self.execute('document.querySelector(".keep-playing-button").click();')

    def send_key_event(self, action, key):
        # Use generic events for compatibility with Chrome,
        # which (for inexplicable reasons) doesn't support setting keyCode on KeyboardEvent objects.
        # See http://stackoverflow.com/questions/8942678/keyboardevent-in-chrome-keycode-is-0.
        return self.execute('''
        var keyboardEvent = document.createEventObject ? document.createEventObject() : document.createEvent("Events");
        if(keyboardEvent.initEvent)
            keyboardEvent.initEvent("%(action)s", true, true);
        keyboardEvent.keyCode = %(key)s;
        keyboardEvent.which = %(key)s;
        var element = document.body || document;
        element.dispatchEvent ? element.dispatchEvent(keyboardEvent) : element.fireEvent("on%(action)s", keyboardEvent);
        ''' % locals())


class Fast2048Control(Generic2048Control):
    """
    Control 2048 by hooking the GameManager and executing its move() function.

    This is both safer and faster than the keyboard approach, but it is less compatible with clones.
    """

    def setup(self):
        # Obtain the GameManager instance by triggering a fake restart.
        self.ctrl.execute(
            '''
            var _func_tmp = GameManager.prototype.isGameTerminated;
            GameManager.prototype.isGameTerminated = function() {
                GameManager._instance = this;
                return true;
            };
            ''')

        # Send an "up" event, which will trigger our replaced isGameTerminated function
        self.send_key_event('keydown', 38)
        time.sleep(0.1)
        self.send_key_event('keyup', 38)

        self.execute('GameManager.prototype.isGameTerminated = _func_tmp;')

    def get_status(self):
        """
        Check if the game is in an unusual state.
        """
        return self.execute('''
            if(GameManager._instance.over) {"ended"}
            else if(GameManager._instance.won && !GameManager._instance.keepPlaying) {"won"}
            else {"running"}
            ''')

    def get_score(self):
        return self.execute('GameManager._instance.score')

    def get_board(self):
        # Chrome refuses to serialize the Grid object directly through the debugger.
        grid = json.loads(self.execute('JSON.stringify(GameManager._instance.grid)'))

        board = [[0]*4 for _ in range(4)]
        for row in grid['cells']:
            for cell in row:
                if cell is None:
                    continue
                pos = cell['x'], cell['y']
                board[pos[1]][pos[0]] = cell['value']

        return board

    def execute_move(self, move):
        # Move in an URDL manner, as 2048
        self._log.debug('Executing move in direction %i' % move)
        self.execute('GameManager._instance.move(%d)' % move)


class Keyboard2048Control(Generic2048Control):
    """
    Control 2048 by accessing the DOM and using key events.

    This is relatively slow, and may be prone to race conditions if your
    browser is slow. However, it is more generally compatible with various
    clones of 2048.
    """

    def setup(self):
        self.execute(
            '''
            var elems = document.getElementsByTagName('div');
            for(var i in elems)
                if(elems[i].className == 'tile-container') {
                    tileContainer = elems[i];
                    break;
                }
            ''')

    def get_score(self):
        score = self.execute('''
            var scoreContainer = document.querySelector(".score-container");
            var scoreText = '';
            var scoreChildren = scoreContainer.childNodes;
            for(var i = 0; i < scoreChildren.length; ++i) {
                if(scoreChildren[i].nodeType == Node.TEXT_NODE) {
                    scoreText += scoreChildren[i].textContent;
                }
            }
            scoreText;
            ''')

        return int(score)

    def get_board(self):
        res = self.execute(
            '''
            var res = [];
            var tiles = tileContainer.children;
            for(var i=0; i<tiles.length; i++)
                res.push(tiles[i].className);
            res
            ''')
        board = [[0]*4 for _ in range(4)]
        for tile in res:
            tval = pos = None
            for k in tile.split():
                m = re.match(r'^tile-(\d+)$', k)
                if m:
                    tval = int(m.group(1))
                m = re.match(r'^tile-position-(\d+)-(\d+)$', k)
                if m:
                    pos = int(m.group(1)), int(m.group(2))
            board[pos[1]-1][pos[0]-1] = tval

        return board

    def execute_move(self, move):
        # Ordered as 2048 in an URDL manner
        key = [38, 39, 40, 37][move]
        self._log.debug("Key %i Move %i" % (key, move))
        self.send_key_event('keydown', key)
        time.sleep(0.005)
        self.send_key_event('keyup', key)
        time.sleep(0.005)


class Hybrid2048Control(Fast2048Control, Keyboard2048Control):
    """
    Control 2048 by hooking the GameManager and using keyboard inputs.
    This is safe and fast, and correctly generates keyboard events for compatibility.
    """

    setup = Fast2048Control.setup
    get_status = Keyboard2048Control.get_status
    get_score = Fast2048Control.get_score
    get_board = Fast2048Control.get_board
    execute_move = Keyboard2048Control.execute_move

# GENERICS FOR MOVE STATE EVALUATION

# Directions, DO NOT MODIFY
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

# Offsets for computing tile indices in each direction.
# DO NOT MODIFY this dictionary.
OFFSETS = {UP: (1, 0),
           DOWN: (-1, 0),
           LEFT: (0, 1),
           RIGHT: (0, -1)}


def zero_to_right(line):

    """ Helper function for merge() that put all non-zero term
    to the left with no space. i.e. zero's to the right"""

    length = len(line)
    result = [0] * length
    idx = 0
    for num in line:
        if num != 0:
            result[idx] = num
            idx += 1

    return result


def next_occ(seq, idx):
    """find the index of next value that is the same and to the right of current index """

    if seq[idx + 1:].count(seq[idx]) > 0:
        new_idx = seq[idx + 1:].index(seq[idx])
        return new_idx + idx + 1

    else:
        return -9999


def check_gap(seq, idx1, idx2):
    """check if there are any non-zero entries in between idx1 and idx2"""

    for num in range(idx1 + 1, idx2):
        if seq[num] != 0:
            return True

    return False


def merge(line):
    """
    Helper function that merges a single row or column in 2048
    """

    result_list = [0] * len(line)
    merged = []

    for num in range(len(line)):
        second_occurence = next_occ(line, num)
        first_bool = not(check_gap(line, num, second_occurence))
        second_bool = not(num in merged)

        if second_occurence != -9999 and first_bool and second_bool:

            result_list[second_occurence] = 2 * line[num]
            merged.append(second_occurence)
        elif second_bool:
            result_list[num] = line[num]

    return zero_to_right(result_list)


def choose(lst, height, width):
    """choose a random zero tile"""
    while True:
        row = random.randrange(height)
        col = random.randrange(width)
        if lst[row][col] == 0:
            return row, col


def insert_row(seq, des_row, row):
    """
    insert a row to a given destination row number in the sequence that represent the grid
    return the new modified grid representation
    """
    seq[des_row] = row
    return seq


def insert_col(seq, des_col, col):
    """
    insert a column to a given destination column number in the sequence that represent the grid
    return the new modified grid representation
    """

    num = 0

    for row in seq:
        row[des_col] = col[num]
        num += 1

    return seq


def init_tiles(seq, direction):
    """
    return a list of indices of tiles whose value will be first passed into the merged function
    with respect to the direction chosen.
    """
    row = len(seq)
    col = len(seq[0])
    dir_dict = {
        UP: [[0, i] for i in range(col)],
        DOWN: [[row - 1, i] for i in range(col)],
        LEFT: [[i, 0] for i in range(row)],
        RIGHT: [[i, col - 1] for i in range(row)]
    }
    return dir_dict[direction]


def lines_for_insert(seq, direction):
    """
    to determine all the lines of values that are to be inserted into the grid
    """
    # this is a list of list that contains list of lines
    lines = []
    initial = init_tiles(seq, direction)
    offsets = OFFSETS[direction]

    line_len = 0
    if direction == UP or direction == DOWN:
        line_len = len(seq)
    elif direction == LEFT or direction == RIGHT:
        line_len = len(seq[0])

    for tile in initial:
        # this is a list of value of one single individual line
        line = []
        for num in range(line_len):
            row = tile[0] + num * offsets[0]
            col = tile[1] + num * offsets[1]
            line.append(seq[row][col])

        lines.append(line)

    return lines


def apply_move(seq, direction):
    """
    this helper function will exercute the move method in place of the class below
    """
    grid = seq[:]
    idx = 0
    lines = lines_for_insert(grid, direction)
    for line in lines:
        if direction == UP:
            insert_col(grid, idx, merge(line))

        elif direction == DOWN:
            insert_col(grid, idx, merge(line)[::-1])

        elif direction == LEFT:
            insert_row(grid, idx, merge(line))

        elif direction == RIGHT:
            insert_row(grid, idx, merge(line)[::-1])

        idx += 1

    npgrid = np.array(grid).reshape((16,))

    return npgrid
