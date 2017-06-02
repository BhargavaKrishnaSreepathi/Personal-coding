"""
=======================================
Selection of Pareto Optimal individuals
=======================================

"""

from __future__ import division
from deap import tools
import pandas as pd
import numpy as np

__author__ = "Sreepathi Bhargava Krishna"


def identify_pareto_observations(book):

    pareto_frontier_RESp = []
    pareto_frontier_GFA = []
    pareto_frontier_PDG = []
    pareto_frontier_d = []
    pareto_frontier_w = []
    pareto_frontier_r = []
    pareto_frontier_RES = []
    pareto_frontier_COM = []
    pareto_frontier_OFF = []
    fitness_values = book.sort_values(['RES%', 'GFA-m2', 'PDG-Kwh'],
                                      ascending=[False, False, True])
    # fitness_values = pd.DataFrame({'RES%':[1,1,1], 'GFA-m2': [0,1,-1], 'PDG-Kwh':[1,1,-1]})
    print (fitness_values)
    a = fitness_values.idxmin(axis = 0)
    b = a['RES%']
    print (fitness_values['RES%'][b])
    pareto_frontier_RESp.append(fitness_values['RES%'][b])
    pareto_frontier_GFA.append(fitness_values['GFA-m2'][b])
    pareto_frontier_PDG.append(fitness_values['PDG-Kwh'][b])
    # pareto_frontier_d.append(fitness_values['d'][b])
    # pareto_frontier_w.append(fitness_values['w'][b])
    # pareto_frontier_r.append(fitness_values['r'][b])
    # pareto_frontier_RES.append(fitness_values['RES'][b])
    # pareto_frontier_COM.append(fitness_values['COM'][b])
    # pareto_frontier_OFF.append(fitness_values['OFF'][b])

    for row in xrange(0, len(fitness_values)):
        dominated = False
        for j in xrange(0, len(pareto_frontier_RESp)):
            if (pareto_frontier_RESp[j] <= fitness_values['RES%'][row]):
                if (pareto_frontier_GFA[j] <= fitness_values['GFA-m2'][row]):
                    if (pareto_frontier_PDG[j] >= fitness_values['PDG-Kwh'][row]):
                        print (pareto_frontier_PDG[j], pareto_frontier_GFA[j], pareto_frontier_RESp[j])
                        print (fitness_values['PDG-Kwh'][row], fitness_values['GFA-m2'][row],fitness_values['RES%'][row])
                        dominated = True
                        break
        if not dominated:
            pareto_frontier_RESp.append(fitness_values['RES%'][row])
            pareto_frontier_GFA.append(fitness_values['GFA-m2'][row])
            pareto_frontier_PDG.append(fitness_values['PDG-Kwh'][row])
            # pareto_frontier_d.append(fitness_values['d'][row])
            # pareto_frontier_w.append(fitness_values['w'][row])
            # pareto_frontier_r.append(fitness_values['r'][row])
            # pareto_frontier_RES.append(fitness_values['RES'][row])
            # pareto_frontier_COM.append(fitness_values['COM'][row])
            # pareto_frontier_OFF.append(fitness_values['OFF'][row])

    print (len(pareto_frontier_RESp))
    print (len(pareto_frontier_GFA))
    print (len(pareto_frontier_PDG))

    # print (type(pareto_frontier_RES))
    # pareto_frontier = pd.DataFrame({'RES%': pareto_frontier_RESp, 'GFA': pareto_frontier_GFA, 'PDG': pareto_frontier_PDG,
    #                                 'r': pareto_frontier_r, 'w': pareto_frontier_w, 'd': pareto_frontier_d,
    #                                 'RES': pareto_frontier_RES, 'COM': pareto_frontier_COM, 'OFF': pareto_frontier_OFF})
    pareto_frontier = pd.DataFrame(
        {'RES%': pareto_frontier_RESp, 'GFA': pareto_frontier_GFA, 'PDG': pareto_frontier_PDG})
    pareto_frontier.to_csv('C:\Users\Bhargava\Downloads\\abc.csv')
    print (pareto_frontier)
    return pareto_frontier_RES



if __name__ == '__main__':

    pop = []
    book = pd.read_excel(open('C:\Users\Bhargava\Downloads\Solar_DDD_5.xlsx', 'rb'))
    # del book['d']
    # del book['w']
    # del book['r']
    # del book['RES']
    # del book['COM']
    # del book['OFF']
    # del book['PVE-Kwh']
    # del book['PDG-Kwh']
    # del book['Unnamed: 6']
    # RES = book['RES%']
    # GFA = book['GFA-m2']
    # PDG-Kwh = book['PDG-Kwh']
    # print (RES, GFA, PDG-Kwh)

    identify_pareto_observations(book)















