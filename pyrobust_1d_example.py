"""pyrobust 1d example using TP1 function from:

Branke J., Fei X. (2016) 
Efficient Sampling When Searching for Robust Solutions. 
In: Handl J., Hart E., Lewis P., López-Ibáñez M., Ochoa G., Paechter B. (eds) 
Parallel Problem Solving from Nature – PPSN XIV. PPSN 2016. 
Lecture Notes in Computer Science, vol 9921. Springer, Cham
DOI: 10.1007/978-3-319-45823-6 22) 
                                                                                
                                                                                
Authors: N. Banglawala, EPCC, 2019                                              
License: MIT                                                                      
                                                                                
"""                                                                             
                                                                                
###############################################################################

import numpy as np


from pyrobust import robust                 
from pyrobust import robust_default_problem as rdp 


def fitness_func(x):
    """Fitness or objective function to minimise. TP1 function (1d) from Branke & Fei 
    2016."""

    tmp = 0
    if x < 8:                                                                   
        tmp = - (8 - x)**0.1 * np.exp(-0.2 * (8 - x))                          
 
    return 0.9 + tmp 


def main():

    # set default problem parameters here
    bounds = [[0], [10]] 
    disturbance_bounds = [[-1], [1]]

    # set robust optimiser parameters here 
    max_pop_size     = 2500 # max population size 
    max_evaluations   = 2500 # max number of evaluations of objective function
    initial_pop_size = 10   # size of initial population

    # set robust optimisation options here
    use_history = False
    update_history = False
    resample = 'wasserstein'

    # create problem
    MyProblem = rdp.RobustDefaultProblem(bounds, disturbance_bounds, 
                                         fitness_func) 

    # create optimiser
    optimiser = robust.Robust(max_pop_size, MyProblem, use_history=use_history,
                              update_history=update_history, resample=resample)

    # create initial population
    optimiser.create_initial_population(initial_pop_size)

    # run optimiser, returns True if all runs ok
    optimiser.run(max_evaluations)

    # get best solutions: (x, robust fitness estimate)
    num_best = 3
    x, robust_fitness_estimate =  optimiser.get_best(num_best) 
    final_results = zip(x.tolist(), robust_fitness_estimate.tolist())

    print("top {} results (x, robust_fitness_estimate):\n {} \n".format(
                                               num_best, list(final_results) ))


if __name__ == "__main__":

    main() 
