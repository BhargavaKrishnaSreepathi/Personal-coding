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

lift_time = 300  # 30 secs
unloading_time = 1000  # 50 secs
loading_time = 5000  # 50 secs
time_multiple_of_distance = 10  # so if the distance is 20m, the time is 10*20 secs

origin_point = [0, 0, 0]  # x-coordinate, y-coordinate and level of floor

shop_0 = [20, 20, 0, 3600, 3, 0]
shop_1 = [40, 20, 0, 6000, 3, 0]
shop_2 = [60, 20, 0, 1000, 10, 0]
shop_3 = [30, 40, 0, 3600, 3, 0]
shop_4 = [60, 40, 0, 6000, 4, 0]

shop_5 = [20, 20, 1, 3600, 3, 0]
shop_6 = [40, 20, 1, 6000, 3, 0]
shop_7 = [60, 20, 1, 1000, 10, 0]
shop_8 = [30, 40, 1, 3600, 3, 0]
shop_9 = [60, 40, 1, 6000, 4, 0]

shop_10 = [20, 20, 2, 6000, 4, 0]
shop_11 = [40, 20, 2, 3600, 3, 0]
shop_12 = [60, 20, 2, 1000, 10, 0]
shop_13 = [30, 40, 2, 3600, 3, 0]
shop_14 = [60, 40, 2, 6000, 5, 0]

shop_15 = [20, 20, 3, 6000, 5, 0]
shop_16 = [40, 20, 3, 6000, 4, 0]
shop_17 = [60, 20, 3, 3600, 3, 0]
shop_18 = [30, 40, 3, 1000, 10, 0]
shop_19 = [60, 40, 3, 4500, 5, 0]

shop_list = np.array([shop_0, shop_1, shop_2, shop_3, shop_4, shop_5, shop_6, shop_7, shop_8, shop_9, shop_10, shop_11,
                      shop_12, shop_13, shop_14, shop_15, shop_16, shop_17, shop_18, shop_19])
len_of_individual = 3 + 3 + 10 + 3 + 4 + 3 + 3 + 10 + 3 + 4 + 4 + 3 + 10 + 3 + 5 + 5 + 4 + 3 + 10 + 5
individual_pool = [0, 0, 0,
                   1, 1, 1,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   3, 3, 3,
                   4, 4, 4, 4,
                   5, 5, 5,
                   6, 6, 6,
                   7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                   8, 8, 8,
                   9, 9, 9, 9,
                   10, 10, 10, 10,
                   11, 11, 11,
                   12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                   13, 13, 13,
                   14, 14, 14, 14, 14,
                   15, 15, 15, 15, 15,
                   16, 16, 16, 16,
                   17, 17, 17,
                   18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
                   19, 19, 19, 19, 19]

max_robo_load = 2
penalty = 100000

genCP = 0
npop = 500

PROBA = 0.9
SIGMAP = 0.2
ngen = 2000

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()


def generate_main(len_of_individual):
    individual = np.random.choice(individual_pool, len_of_individual, replace=False)

    return individual


def objective_function(individual):
    # cost function
    cost = 0
    robo_load = max_robo_load
    for i in range(len_of_individual):

        if robo_load == max_robo_load:

            cost = cost + unloading_time + (
                        shop_list[individual[i]][0] + shop_list[individual[i]][1]) * time_multiple_of_distance + (
                   shop_list[individual[i]][2]) * lift_time
            robo_load = robo_load - 1

            if shop_list[individual[i]][5] < shop_list[individual[i]][3]:
                cost = cost + penalty

            for q in range(0, 20):
                if q == individual[i]:
                    shop_list[q][5] = 0
                else:
                    shop_list[q][5] = shop_list[q][5] + unloading_time + (
                                shop_list[individual[i]][0] + shop_list[individual[i]][
                            1]) * time_multiple_of_distance + (shop_list[individual[i]][2]) * lift_time


        else:
            if shop_list[individual[i]][2] == shop_list[individual[i - 1]][2]:
                cost = cost + unloading_time + (
                        shop_list[individual[i]][0] + shop_list[individual[i]][1] - shop_list[individual[i - 1]][
                    0] - shop_list[individual[i - 1]][1]) * time_multiple_of_distance
                robo_load = robo_load - 1

                if shop_list[individual[i]][5] < shop_list[individual[i]][3]:
                    cost = cost + penalty

                for q in range(0, 20):
                    if q == individual[i]:
                        shop_list[q][5] = 0
                    else:
                        shop_list[q][5] = shop_list[q][5] + unloading_time + (
                        shop_list[individual[i]][0] + shop_list[individual[i]][1] - shop_list[individual[i - 1]][
                    0] - shop_list[individual[i - 1]][1]) * time_multiple_of_distance

            else:
                cost = cost + unloading_time + (
                        shop_list[individual[i]][0] + shop_list[individual[i]][1] - shop_list[individual[i - 1]][
                        0] - shop_list[individual[i - 1]][1]) * time_multiple_of_distance
                robo_load = robo_load - 1
                cost = cost + abs(shop_list[individual[i]][2] - shop_list[individual[i - 1]][2]) * lift_time

                if shop_list[individual[i]][5] < shop_list[individual[i]][3]:
                    cost = cost + penalty

                for q in range(0, 20):
                    if q == individual[i]:
                        shop_list[q][5] = 0
                    else:
                        shop_list[q][5] = shop_list[q][5] + unloading_time + (
                        shop_list[individual[i]][0] + shop_list[individual[i]][1] - shop_list[individual[i - 1]][
                        0] - shop_list[individual[i - 1]][1]) * time_multiple_of_distance

        if robo_load == 0:
            cost = cost + (shop_list[individual[i]][0] + shop_list[individual[i]][1]) * time_multiple_of_distance + (
            shop_list[individual[i]][2]) * lift_time
            cost = cost + loading_time
            robo_load = max_robo_load

    for j in range(0, 20):
        if individual.count(j) != shop_list[j][4]:
            cost = cost + penalty
    # print(cost)
    return cost, cost


def mutShuffle(individual, proba, len_of_individual):
    """
    Swap with probability *proba*
    :param individual: list of all parameters corresponding to an individual configuration
    :param proba: mutation probability
    :type individual: list
    :type proba: float
    :return: mutant list
    :rtype: list
    """
    mutant = toolbox.clone(individual)

    # Swap buildings

    for i in range(len_of_individual):
        if random.random() < proba:
            iswap = random.randint(0, len_of_individual-2)
            if iswap >= i:
                iswap += 1
            rank = i
            irank = iswap
            mutant[rank], mutant[irank] = mutant[irank], mutant[rank]

    del mutant.fitness.values

    return mutant


def cxUniform(ind1, ind2, proba, len_of_individual):
    """
    Performs a uniform crossover between the two parents.
    Each segments is swapped with probability *proba*

    :param ind1: a list containing the parameters of the parent 1
    :param ind2: a list containing the parameters of the parent 2
    :param proba: Crossover probability
    :type ind1: list
    :type ind2: list
    :type proba: float
    :return: child1, child2
    :rtype: list, list
    """
    child1 = toolbox.clone(ind1)
    child2 = toolbox.clone(ind2)

    # Swap functions
    def swap(inda, indb, n):
        inda[n], indb[n] = indb[n], inda[n]

    # Swap DHN and DCN, connected buildings
    for i in range(len_of_individual):
        if random.random() < proba:
            swap(child1, child2, i)

    del child1.fitness.values
    del child2.fitness.values

    return child1, child2


def non_dominated_sorting_genetic_algorithm():
    # SET-UP EVOLUTIONARY ALGORITHM
    # Contains 3 minimization objectives : Costs, CO2 emissions, Primary Energy Needs
    # this part of the script sets up the optimization algorithm in the same syntax of DEAP toolbox

    toolbox.register("generate", generate_main, len_of_individual)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.generate)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective_function)
    toolbox.register("select", tools.selNSGA2)



    columns_of_saved_files = ['costs', 'costs_1']
    for i in range(len_of_individual):  # DHN
        columns_of_saved_files.append(str(i) + ' DHN')

    pop = toolbox.population(n=npop)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]

    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    # linking every individual with the corresponding fitness, this also keeps a track of the number of function
    # evaluations. This can further be used as a stopping criteria in future
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))  # assigning crowding distance

    zero_data = np.zeros(shape=(len(invalid_ind), len(columns_of_saved_files)))
    saved_dataframe_for_each_generation = pd.DataFrame(zero_data, columns=columns_of_saved_files)

    for i, ind in enumerate(invalid_ind):
        for j in range(len(columns_of_saved_files) - 2):
            saved_dataframe_for_each_generation[columns_of_saved_files[j + 2]][i] = ind[j]
        saved_dataframe_for_each_generation['costs'][i] = ind.fitness.values[0]
        saved_dataframe_for_each_generation['costs_1'][i] = ind.fitness.values[1]

    saved_dataframe_for_each_generation.to_csv(r'C:\Users\krish\Desktop\IMDA/' + str(0) + '.csv')

    proba, sigmap = PROBA, SIGMAP

    # Evolution starts !

    g = genCP

    while g < ngen:

        g += 1

        offspring = list(pop)
        # Apply crossover and mutation on the pop
        for ind1, ind2 in zip(pop[::2], pop[1::2]):
            child1, child2 = cxUniform(ind1, ind2, proba, len_of_individual)
            offspring += [child1, child2]

        for mutant in pop:
            mutant = mutShuffle(mutant, proba, len_of_individual)
            offspring.append(mutant)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # Evaluate the individuals with an invalid fitness
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        # linking every individual with the corresponding fitness, this also keeps a track of the number of function
        # evaluations. This can further be used as a stopping criteria in future
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        zero_data = np.zeros(shape=(len(invalid_ind), len(columns_of_saved_files)))
        saved_dataframe_for_each_generation = pd.DataFrame(zero_data, columns=columns_of_saved_files)

        for i, ind in enumerate(invalid_ind):
            for j in range(len(columns_of_saved_files) - 2):
                saved_dataframe_for_each_generation[columns_of_saved_files[j + 2]][i] = ind[j]
            saved_dataframe_for_each_generation['costs'][i] = ind.fitness.values[0]
            saved_dataframe_for_each_generation['costs_1'][i] = ind.fitness.values[1]

        saved_dataframe_for_each_generation.to_csv(r'C:\Users\krish\Desktop\IMDA/' + str(g) + '.csv')

        selection = toolbox.select(pop + invalid_ind, npop)  # assigning crowding distance
        pop[:] = selection
        zero_data = np.zeros(shape=(len(invalid_ind), len(columns_of_saved_files)))
        selected_dataframe_for_each_generation = pd.DataFrame(zero_data, columns=columns_of_saved_files)
        for i, ind in enumerate(selection):
            for j in range(len(columns_of_saved_files) - 2):
                saved_dataframe_for_each_generation[columns_of_saved_files[j + 2]][i] = ind[j]
            saved_dataframe_for_each_generation['costs'][i] = ind.fitness.values[0]
            saved_dataframe_for_each_generation['costs_1'][i] = ind.fitness.values[1]

        selected_dataframe_for_each_generation.to_csv(r'C:\Users\krish\Desktop\IMDA/' + str(g) + '_selected.csv')

        for i in range(len(selection)):
            print (selection[i].fitness)
        print ('the generation number is: ' + str(g))
    if g == ngen:
        print
        "Final Generation reached"
    else:
        print
        "Stopping criteria reached"

    # Dataframe with all the individuals whose objective functions are calculated, gathering all the results from
    # multiple generations
    df = pd.read_csv(r'C:\Users\krish\Desktop\IMDA/' + str(0) + '.csv')
    for i in range(ngen):
        df = df.append(pd.read_csv(r'C:\Users\krish\Desktop\IMDA/' + str(i + 1) + '.csv'))
    df.to_csv(r'C:\Users\krish\Desktop\IMDA/All.csv')

    return pop


if __name__ == "__main__":
    non_dominated_sorting_genetic_algorithm()
