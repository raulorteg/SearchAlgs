"""
Simulated annealing for the travelling salesman.
@author Raul Ortega
Created 27/06/2021

Example 5: Working with the cost matrix from the Random Clusters Map
"""
from maps import Random_Clusters_Map
from SimulatedAnnealing import Simulated_Annealing

def main():
    
    # Lets test it in the Random Clusters map
    random_map = Random_Clusters_Map(num_cities=25, num_clusters=4)
    cost_matrix = random_map.get_distances()

    simulation = Simulated_Annealing(cost_matrix=cost_matrix, temp_type='log', init_method='random', max_iter=1000)
    path, path_cost, cost_list = simulation.run(map=random_map, animate=True)

    # print results in command line
    print(f"Path solution: {path}")
    print(f"Cost of solution path: {path_cost}")

    # plot the solution
    random_map.plot(path)

if __name__ == "__main__":
    main()