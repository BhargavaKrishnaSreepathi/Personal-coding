from __future__ import division

import json
import pandas as pd
import multiprocessing
import time
import numpy as np
from deap import base
from deap import creator
from deap import tools
from pulp import *
import numpy as np
import random

__author__ = "Sreepathi Bhargava Krishna"
__copyright__ = "Copyright 2015, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Sreepathi Bhargava Krishna"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daren Thomas"
__email__ = "thomas@arch.ethz.ch"
__status__ = "Production"


# Creates a list of the Ingredients
Ingredients = ['CHICKEN', 'BEEF', 'MUTTON', 'RICE', 'WHEAT', 'GEL']

lift_time = 30 # 30 secs
unloading_time = 50 # 50 secs
loading_time = 50 # 50 secs
time_multiple_of_distance = 10 # so if the distance is 20m, the time is 10*20 secs

origin_point = [0, 0, 0] # x-coordinate, y-coordinate and level of floor

shop_0 = [20, 20, 0, 3600, 3]
shop_1 = [40, 20, 0, 6000, 3]
shop_2 = [60, 20, 0, 1000, 10]
shop_3 = [30, 40, 0, 3600, 3]
shop_4 = [60, 40, 0, 6000, 4]

shop_5 = [20, 20, 1, 3600, 3]
shop_6 = [40, 20, 1, 6000, 3]
shop_7 = [60, 20, 1, 1000, 10]
shop_8 = [30, 40, 1, 3600, 3]
shop_9 = [60, 40, 1, 6000, 4]

shop_10 = [20, 20, 2, 6000, 4]
shop_11 = [40, 20, 2, 3600, 3]
shop_12 = [60, 20, 2, 1000, 10]
shop_13 = [30, 40, 2, 3600, 3]
shop_14 = [60, 40, 2, 6000, 5]

shop_15 = [20, 20, 3, 6000, 5]
shop_16 = [40, 20, 3, 6000, 4]
shop_17 = [60, 20, 3, 3600, 3]
shop_18 = [30, 40, 3, 1000, 10]
shop_19 = [60, 40, 3, 4500, 5]

shop_list = np.array([shop_0, shop_1, shop_2, shop_3, shop_4, shop_5, shop_6, shop_7, shop_8, shop_9, shop_10, shop_11,
                       shop_12, shop_13, shop_14, shop_15, shop_16, shop_17, shop_18, shop_19])
len_of_individual = 3+3+10+3+4+3+3+10+3+4+4+3+10+3+5+5+4+3+10+5

max_robo_load = 2

cost = 0

individual = [18,15,5,5,2,5,2,0,6,5,0,0,6,5,5,5,1,0,0,10,7,5,8,0,8,10,8,5,1,5,13,10,9,10,12,0,4,0,3,5,2,0,4,0,8,5,7,5,2,0,3,1,4,0,6,5,13,0,9,5,9,5,1,0,0,0,1,0,5,5,4,0,12,
              10,11,5,14,15,4,0,10,10,1,0,1,0,1,5,0,0,8,15,2,0,11,15,4,5]
robo_load = max_robo_load
for i in range(len_of_individual):

    if robo_load == max_robo_load:

        cost = cost + unloading_time + \
                    (shop_list[individual[i]][0] + shop_list[individual[i]][1]) * time_multiple_of_distance + (
               shop_list[individual[i]][2]) * lift_time
        robo_load = robo_load - 1

    else:
        if shop_list[individual[i]][2] == shop_list[individual[i - 1]][2]:
            cost = cost + unloading_time + (
                    shop_list[individual[i]][0] + shop_list[individual[i]][1] - shop_list[individual[i - 1]][
                0] - shop_list[individual[i - 1]][1]) * time_multiple_of_distance
            robo_load = robo_load - 1
        else:
            cost = cost + unloading_time + (
                    shop_list[individual[i]][0] + shop_list[individual[i]][1] - shop_list[individual[i - 1]][
                0] - shop_list[individual[i - 1]][1]) * time_multiple_of_distance
            robo_load = robo_load - 1
            cost = cost + abs(shop_list[individual[i]][2] - shop_list[individual[i - 1]][2]) * lift_time

    if robo_load == 0:
        cost = cost + (shop_list[individual[i]][0] + shop_list[individual[i]][1]) * time_multiple_of_distance + (
        shop_list[individual[i]][2]) * lift_time
        cost = cost + loading_time
        robo_load = max_robo_load

print(cost)