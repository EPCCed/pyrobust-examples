"""Pyrobust 1d TP1 (Branke&Fei 2016) example 
                                                                                
PYTHON version based on source code from University of Exeter                   
J. Fieldsend, K. Alayahya, K. Doherty                                           
                                                                                
Authors: N. Banglawala, EPCC, 2019                                              
License: MIT                                                                      
                                                                                
"""                                                                             
                                                                                
###############################################################################

from pyrobust import robust                                            
from pyrobust import GECCO2017                                                                             

def objective_func(x):
    dims = 1                                             

    tmp = 0
    if x < 8:                                                                   
        tmp = - (8 - x)**0.1 * np.exp(-0.2 * (8 - x))                          
 
    return 0.9 + tmp 


def main():

    # set GECCO2017 problem parameters here
    dims = 1 
    bounds = [0, 10] 
    disturbance_bounds = [-1, 1]
    # evolution parameters
    crossover_rate =  0.8
    mutation_width = 0.1 * (bounds[1] - bounds[0]) # for Gaussian mutation
    mutation_rate = 0.5

    # set robust optimiser parameters here 
    max_pop_size     = 2500 # max population size   
    max_evalations   = 2500 # max number of evaluations of objective function
    initial_pop_size = 10   # size of initial population

    # set robust optimisation options here
    use_history = None          # ['individual', 'neighbour']
    resample = 'wasserstein'    #

    # create problem
    MyProblem = GECCO2017.GECCO2017(bounds, disturbance_bounds, 
                                    objective_func, crossover_rate,
                                    mutation_rate, mutation_width)

    # create optimiser
    optimiser = robust.robust(max_pop_size, MyProblem, use_history=use_history,
                              resample=resample)

    # create initial population
    optimiser.create_initial_population(initial_pop_size)

    # run optimiser
    optimiser.run(max_evaluations)

    # get best solutions
    #best_results =  optimiser.get_best() 


if __name__ == "__main__":

    main() 
