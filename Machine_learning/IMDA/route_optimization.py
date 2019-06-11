"""
The Full Whiskas Model Python Formulation for the PuLP Modeller

Authors: Antony Phillips, Dr Stuart Mitchell  2007
"""

# Import PuLP modeler functions
from pulp import *
import random

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

len_of_individual = 3+3+10+3+4+3+3+10+3+4+4+3+10+3+5+5+4+3+10+5

individual = []

for i in range(len_of_individual):
    a = random.randint(0,19)
    individual.append(a)


#cost function
cost = 0
for i in range(len_of_individual):
    b = individual[i]
    if b == 0:
        cost = cost + loading_time + unloading_time + (shop_0[0] + shop_0[1]) * 2 * time_multiple_of_distance
    elif b == 1:
        cost = cost + loading_time + unloading_time + (shop_1[0] + shop_1[1]) * 2 * time_multiple_of_distance
    elif b == 2:
        cost = cost + loading_time + unloading_time + (shop_2[0] + shop_2[1]) * 2 * time_multiple_of_distance
    elif b == 3:
        cost = cost + loading_time + unloading_time + (shop_3[0] + shop_3[1]) * 2 * time_multiple_of_distance
    elif b == 4:
        cost = cost + loading_time + unloading_time + (shop_4[0] + shop_4[1]) * 2 * time_multiple_of_distance
    elif b == 5:
        cost = cost + loading_time + unloading_time + (shop_5[0] + shop_5[1]) * 2 * time_multiple_of_distance
    elif b == 6:
        cost = cost + loading_time + unloading_time + (shop_6[0] + shop_6[1]) * 2 * time_multiple_of_distance
    elif b == 7:
        cost = cost + loading_time + unloading_time + (shop_7[0] + shop_7[1]) * 2 * time_multiple_of_distance
    elif b == 8:
        cost = cost + loading_time + unloading_time + (shop_8[0] + shop_8[1]) * 2 * time_multiple_of_distance
    elif b == 9:
        cost = cost + loading_time + unloading_time + (shop_9[0] + shop_9[1]) * 2 * time_multiple_of_distance
    elif b == 10:
        cost = cost + loading_time + unloading_time + (shop_10[0] + shop_10[1]) * 2 * time_multiple_of_distance
    elif b == 11:
        cost = cost + loading_time + unloading_time + (shop_11[0] + shop_11[1]) * 2 * time_multiple_of_distance
    elif b == 12:
        cost = cost + loading_time + unloading_time + (shop_12[0] + shop_12[1]) * 2 * time_multiple_of_distance
    elif b == 13:
        cost = cost + loading_time + unloading_time + (shop_13[0] + shop_13[1]) * 2 * time_multiple_of_distance
    elif b == 14:
        cost = cost + loading_time + unloading_time + (shop_14[0] + shop_14[1]) * 2 * time_multiple_of_distance
    elif b == 15:
        cost = cost + loading_time + unloading_time + (shop_15[0] + shop_15[1]) * 2 * time_multiple_of_distance
    elif b == 16:
        cost = cost + loading_time + unloading_time + (shop_16[0] + shop_16[1]) * 2 * time_multiple_of_distance
    elif b == 17:
        cost = cost + loading_time + unloading_time + (shop_17[0] + shop_17[1]) * 2 * time_multiple_of_distance
    elif b == 18:
        cost = cost + loading_time + unloading_time + (shop_18[0] + shop_18[1]) * 2 * time_multiple_of_distance
    elif b == 19:
        cost = cost + loading_time + unloading_time + (shop_19[0] + shop_19[1]) * 2 * time_multiple_of_distance


print (cost)
print (individual)
# A dictionary of the costs of each of the Ingredients is created
costs = {'CHICKEN': 0.013,
         'BEEF': 0.008,
         'MUTTON': 0.010,
         'RICE': 0.002,
         'WHEAT': 0.005,
         'GEL': 0.001}

# A dictionary of the protein percent in each of the Ingredients is created
proteinPercent = {'CHICKEN': 0.100,
                  'BEEF': 0.200,
                  'MUTTON': 0.150,
                  'RICE': 0.000,
                  'WHEAT': 0.040,
                  'GEL': 0.000}

# A dictionary of the fat percent in each of the Ingredients is created
fatPercent = {'CHICKEN': 0.080,
              'BEEF': 0.100,
              'MUTTON': 0.110,
              'RICE': 0.010,
              'WHEAT': 0.010,
              'GEL': 0.000}

# A dictionary of the fibre percent in each of the Ingredients is created
fibrePercent = {'CHICKEN': 0.001,
                'BEEF': 0.005,
                'MUTTON': 0.003,
                'RICE': 0.100,
                'WHEAT': 0.150,
                'GEL': 0.000}

# A dictionary of the salt percent in each of the Ingredients is created
saltPercent = {'CHICKEN': 0.002,
               'BEEF': 0.005,
               'MUTTON': 0.007,
               'RICE': 0.002,
               'WHEAT': 0.008,
               'GEL': 0.000}

# Create the 'prob' variable to contain the problem data
prob = LpProblem("The Whiskas Problem", LpMinimize)

# A dictionary called 'ingredient_vars' is created to contain the referenced Variables
ingredient_vars = LpVariable.dicts("Ingr",Ingredients,0)

# The objective function is added to 'prob' first
prob += lpSum([costs[i]*ingredient_vars[i] for i in Ingredients]), "Total Cost of Ingredients per can"

# The five constraints are added to 'prob'
prob += lpSum([ingredient_vars[i] for i in Ingredients]) == 100, "PercentagesSum"
prob += lpSum([proteinPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 8.0, "ProteinRequirement"
prob += lpSum([fatPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 6.0, "FatRequirement"
prob += lpSum([fibrePercent[i] * ingredient_vars[i] for i in Ingredients]) <= 2.0, "FibreRequirement"
prob += lpSum([saltPercent[i] * ingredient_vars[i] for i in Ingredients]) <= 0.4, "SaltRequirement"


# The problem data is written to an .lp file
prob.writeLP("WhiskasModel.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)

# The optimised objective function value is printed to the screen
print("Total Cost of Ingredients per can = ", value(prob.objective))