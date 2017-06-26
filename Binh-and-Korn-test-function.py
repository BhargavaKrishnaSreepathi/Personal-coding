# Aim: To test various multi-objective optimization algorithms present in python
# Author: Sreepathi Bhargava Krishna
# Date: 26.06.2017
# Problem formulation
# Name: Binh and Korn function
# Objectives: Two
# Constraints: Two

# from PIL import Image
# from StringIO import StringIO
# import urllib
#
# image = Image.open(StringIO(urllib.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Binh_and_Korn_function.pdf/page1-796px-Binh_and_Korn_function.pdf.jpg").read()))
#
# image.show()

import array, random
from deap import creator, base, algorithms, tools


creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):
    return sum(individual)


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=300)

NGEN = 40
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = offspring