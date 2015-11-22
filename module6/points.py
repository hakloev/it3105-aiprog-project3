# -*- coding: utf8 -*-
#
# Created by 'myth' on 11/17/15

import math
import os
from scipy.stats import ttest_ind

ANN_RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'statistics/', 'ann_statistics.txt')
RANDOM_RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'statistics/', 'random_statistics.txt')


def create_run_lists():
    ann_stats = []
    random_stats = []
    with open(ANN_RUNS) as ann:
        for line in ann:
            ann_stats.append(int(line.strip()))
    with open(RANDOM_RUNS) as rnd:
        for line in rnd:
            random_stats.append(int(line.strip()))

    shortest = min(len(ann_stats), len(random_stats))
    ann_stats = ann_stats[:shortest]
    random_stats = random_stats[:shortest]

    # Need to be returned in this order according to the task description
    return random_stats, ann_stats


def calculate_points():
    random_stats, ann_stats = create_run_lists()

    def points(p):
        return max(0, min(7, math.ceil(-math.log(p, 10))))

    return points(ttest_ind(random_stats, ann_stats, equal_var=False).pvalue)


