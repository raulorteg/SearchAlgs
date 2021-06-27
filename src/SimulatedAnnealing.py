"""
Simulated annealing for the travelling salesman.
@author Raul Ortega
Created 12/06/2021
"""

import numpy as np
from math import sqrt, log, cos, sin
import math, random
import matplotlib.pyplot as plt
import glob, os

class Simulated_Annealing:
    """ Performs Simulated Annealing using Temperature law <temp_type> with initialization <init_method> on a given map
    using the 2d coordinates <coordinates=True> of each node or the cost matrix (coordinates=False)"""

    def __init__(self, cost_matrix=[], temp_type="sqrt", T=1, max_iter=100, T_threshold=0.001, init_method="greedy", alpha=0.99, coordinates=False):
        assert temp_type in ["log", "sqrt", "exp"], "temperature function type not implemented. Try <sqrt>, <log>, <exp>"
        assert init_method in ["greedy", "random"], "initialization method not implemented. Try <greedy>, <random>"
        
        self.cost_matrix = cost_matrix
        if coordinates:
            self.compute_cost_matrix(coordinates)

        self.num_nodes = len(cost_matrix)   # number of nodes in the map
        self.nodes = range(self.num_nodes)  # list of nodes
        self.temp_type = temp_type          # temperature law
        self.T = T                          # Initial temperature (T=1)
        self.max_iter = max_iter            # Max number of iterations (stopping criteria I)
        self.T_threshold = T_threshold      # Lowest temperature (stopping criteria II)
        self.init_method = init_method      # method of initialization of the solution
        self.alpha = alpha                  # exponent of the exponential temperature law T_{k+1} = T_{k}^{\alpha}

    def compute_cost_matrix(self, coordinates):
        """
        Fill in the matrix of distances, Compute the distance (euclidean) 
        between all pairs of cities.
        Note: Since distance from A to B is the same as B to A, the number of operations
        done here could be cut in half, but given the low number of cities its ok to repeat calculations.
        """
        for city in coordinates:
            for next_city in coordinates:
                distance = self.euclidean_distance(city, next_city)
                self.cost_matrix.append(distance)

        self.cost_matrix = np.array(self.cost_matrix).reshape((self.num_nodes, self.num_nodes))

    def euclidean_distance(self, origin, dest):
        x0, y0 = origin
        x1, y1 = dest
        return math.sqrt((x0-x1)**2 + (y0-y1)**2)

    def compute_segment_cost(self, node_start, node_end):
        """ Computes the distance/cost between two nodes"""

        assert (node_start in self.nodes) and (node_end in self.nodes), 'selected nodes are not within the node list'
        return self.cost_matrix[node_start,node_end]

    def compute_cost(self, path):
        """ Computes the distance/cost of the whole path """
        path_cost = 0
        for idx in range((len(path)-1)):
            path_cost += self.compute_segment_cost(path[idx], path[idx+1])
        return path_cost

    def p_accept(self, path_cost):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        return math.exp(-abs(path_cost - self.path_cost) / self.T)

    def accept(self, path):
        """
        Accept the new candidate always if better than old candidate
        if new candidate worse than previous then accept it with a probability
        that depends on how worse this path is compared to the previous one p_accept()
        """
        path_cost = self.compute_cost(path)
        if path_cost < self.path_cost:
            self.path_cost, self.path = path_cost, path
        else:
            if np.random.uniform(low=0.0, high=1.0) < self.p_accept(path_cost):
                self.path_cost, self.path = path_cost, path

    def update_T(self, iter_):
        """ Simulate the cooling according to the selected temperature law"""
        if self.temp_type == "sqrt":
            self.T = 1/sqrt(1+iter_)

        elif self.temp_type == "log":
            self.T = -log(iter_/self.max_iter)

        else: # exponential
            self.T = self.T**self.alpha

    def initialize(self):
        """
        Generate an initial solution.

        Greedy: Pick a random node, the the path of the salesman 
        is from one node to the closest other non-visited node.
        
        Random: get a random permutation of the
        nodes to be the path and compute its cost
        """
        if self.init_method == "greedy":
            path, path_cost = [], 0
            curr_node = random.randint(0, self.num_nodes - 1)
            path.append(curr_node)

            while len(path) < self.num_nodes:
                best_candidate, cost_best_candidate = None, np.inf
                for candidate in self.nodes:
                    if (candidate != curr_node) and (candidate not in path):
                        candidate_cost = self.compute_segment_cost(curr_node, candidate)
                        if candidate_cost < cost_best_candidate:
                            cost_best_candidate = candidate_cost
                            best_candidate = candidate
                path.append(best_candidate)
                path_cost += cost_best_candidate

            self.path = np.array(path)
            self.path_cost = path_cost
        
        else: # random initialization
            self.path = list(self.nodes)
            random.shuffle(self.path)
            self.path_cost = self.compute_cost(self.path)

    def run(self, map=None, animate=False):
        """ execute the simulation """

        cost_list, iter_ = [], 1
        self.initialize() # initial path, path_cost
        cost_list.append(self.path_cost)

        while (self.T >= self.T_threshold) and (iter_ < self.max_iter):
            path = list(self.path)
            delta_idx= random.randint(2, self.num_nodes-1)
            idx = random.randint(0, self.num_nodes-1)
            path[idx:(idx+delta_idx)] = reversed(path[idx:(idx+delta_idx)])
            self.accept(path)

            cost_list.append(self.path_cost)

            if map and animate: # to save plots for animation
                map.save_progress(path=path, iter_=iter_, iter_max=self.max_iter)

            iter_ += 1
            self.update_T(iter_)

        if map and animate:
            map.animate()
        return self.path, self.path_cost, cost_list