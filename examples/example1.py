"""
Simulated annealing for the travelling salesman.
@author Raul Ortega
Created 12/06/2021

Example 1: Working with a cost matrix
"""
import pandas as pd
import matplotlib.pyplot as plt
from SimulatedAnnealing import Simulated_Annealing

def main():
    fname = "cost.csv"
    cost_matrix = pd.read_csv(fname, dtype='int').to_numpy()

    simulation = Simulated_Annealing(cost_matrix=cost_matrix, temp_type="sqrt", init_method="random", max_iter=1000)
    path, path_cost, cost_list = simulation.run()
    
    # print results in command line
    print(f"Path solution: {path}")
    print(f"Cost of solution path: {path_cost}")

    # plot the cost evolution vs iteration
    plt.plot(cost_list)
    plt.ylabel("Cost of path (distance)")
    plt.xlabel("Iterations")
    plt.show()

if __name__ == "__main__":
    main()