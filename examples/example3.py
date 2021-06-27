"""
Simulated annealing for the travelling salesman.
@author Raul Ortega
Created 26/06/2021

Example 3: Working with the cost matrix from the Circle Map
"""
from maps import Circle_Map
from SimulatedAnnealing import Simulated_Annealing

def main():

    # Lets test it in the Circle map
    circle_map = Circle_Map(num_cities=20)
    cost_matrix = circle_map.get_distances()

    simulation = Simulated_Annealing(cost_matrix=cost_matrix, temp_type='log', init_method='random', max_iter=200)
    path, path_cost, cost_list = simulation.run(map=circle_map, animate=False)

    # print results in command line
    print(f"Path solution: {path}")
    print(f"Cost of solution path: {path_cost}")

    # plot the resulting solution
    circle_map.plot(path)

if __name__ == "__main__":
    main()