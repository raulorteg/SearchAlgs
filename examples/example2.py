"""
Simulated annealing for the travelling salesman.
@author Raul Ortega
Created 26/06/2021

Example 2: Working with a cost matrix. Comparing different methods
of initialization and cooling laws.
"""
import pandas as pd
import matplotlib.pyplot as plt
from SimulatedAnnealing import Simulated_Annealing

def main():
    fname = "cost.csv"
    cost_matrix = pd.read_csv(fname, dtype='int').to_numpy()
    
    # run variations of the method to compare initializations and T laws
    for init_method in ['greedy', 'random']:
        for temp_type in ['sqrt', 'log', 'exp']:
            simulation = Simulated_Annealing(cost_matrix=cost_matrix, temp_type=temp_type, init_method=init_method, max_iter=1000)
            path, path_cost, cost_list = simulation.run()
    
            plt.plot(cost_list, label=f"({init_method}, {temp_type})")
    plt.legend()
    plt.ylabel("Cost of path (distance)")
    plt.xlabel("Iterations")
    plt.show()

if __name__ == "__main__":
    main()

    