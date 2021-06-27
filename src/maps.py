"""
Example Maps for Simulated Annealing: Circle Map, Random Map.
@author Raul Ortega
Created 26/06/2021
"""
import matplotlib.pyplot as plt
from math import sqrt, log, cos, sin
import numpy as np
import os, glob, math
from PIL import Image

class Circle_Map:
    def __init__(self, num_cities=20, radius=200, equispaced=True):
        self.num_cities = num_cities    # number of cities to spawn
        self.radius = radius            # radius of the circle on which the cities are created
        self.equispaced = equispaced    # (not usable yet)
        self.cities = []                # list of cities spawned (starts empty) [(coord_x, coord_y), ....]
        self.distance_matrix = []       # cost matrix, pairwise distance between cities
        self.create_map()               # creates the cities. fills in self.cities
        self.compute_distances()        # uses the cities to fill in the cost matrix (distance_matrix)

    def create_map(self):
        """ Only equispaced option implemented. Fills in the cities list,
        a city is defined as a pair (x_coord, y_coord), where both coords are ints """
        if self.equispaced: 
            delta_theta = 2*np.pi/self.num_cities
            thetas = np.cumsum(delta_theta*np.ones(self.num_cities))

            for theta in thetas:
                x, y = self.radius*math.cos(theta), self.radius*math.sin(theta)
                self.cities.append((x,y))

    def compute_distances(self):
        """
        Fill in the matrix of distances, Compute the distance (euclidean) 
        between all pairs of cities.
        Note: Since distance from A to B is the same as B to A, the number of operations
        done here could be cut in half, but given the low number of cities its ok to repeat calculations.
        """
        for city in self.cities:
            for next_city in self.cities:
                distance = self.euclidean_distance(city, next_city)
                self.distance_matrix.append(distance)

        self.distance_matrix = np.array(self.distance_matrix).reshape((self.num_cities, self.num_cities))

    def euclidean_distance(self, origin, dest):
        x0, y0 = origin
        x1, y1 = dest
        return math.sqrt((x0-x1)**2 + (y0-y1)**2)

    def plot(self, path=None):
        """
        Given a path plot the route of the salesman and the cities.
        """
        coord_path_x, coord_path_y = [], []
        for city in self.cities:
            plt.plot(*city, "xr") # plot the cities

        if path:
            past_city = self.cities[path[0]]
            plt.plot(*past_city, "og") # plot the starting point
            for i in range(1,self.num_cities):
                city_idx = path[i]
                coord_path_x.append([past_city[0], self.cities[city_idx][0]])
                coord_path_y.append([past_city[1], self.cities[city_idx][1]])
                past_city = self.cities[city_idx]

            for idx in range(len(coord_path_x)):
                plt.plot(coord_path_x[idx], coord_path_y[idx], "b")

        plt.title("Route of the Salesman on Circle Map")
        plt.show()

    def save_progress(self, iter_, iter_max, path=None):
        """
        Given a path plot the route of the salesman and the cities.
        """

        # folder where images are saved for animation
        if not os.path.exists("temp_folder_animation"):
            os.makedirs("temp_folder_animation")

        coord_path_x, coord_path_y = [], []
        for city in self.cities:
            plt.plot(*city, "xr") # plot the cities

        if path:
            past_city = self.cities[path[0]]
            plt.plot(*past_city, "og") # plot the starting point
            for i in range(1,self.num_cities):
                city_idx = path[i]
                coord_path_x.append([past_city[0], self.cities[city_idx][0]])
                coord_path_y.append([past_city[1], self.cities[city_idx][1]])
                past_city = self.cities[city_idx]

            for idx in range(len(coord_path_x)):
                plt.plot(coord_path_x[idx], coord_path_y[idx], "b")

            plt.title(f"Route of the Salesman on Circle Map, iter={iter_}")
            magnitude = math.floor(math.log10(iter_))
            max_magnitude = math.floor(math.log10(iter_max))
            plt.savefig("temp_folder_animation/circle"+ "0"*(max_magnitude-magnitude) + f"{iter_}.png", format="png")
            plt.close()

            print(f"Generating figure {iter_}/{iter_max}")

    def animate(self):
        # Create the frames
        frames = []
        imgs = glob.glob("temp_folder_animation/circle"+"*.png")
        imgs = sorted(imgs)
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)
    
        # folder where images are saved for animation
        if not os.path.exists("results"):
            os.makedirs("results")

        print("Generating animation...")
        # Save into a GIF file that loops forever
        frames[0].save("results"+'/circle.gif', format='GIF', append_images=frames[1:], save_all=True, duration=30, loop=0)

        print("Cleaning temporary images for animation...")
        # remove the images
        for img in imgs:
            os.remove(img)

    def get_cities(self): # if want to work with the coordinates of cities
        return self.cities

    def get_distances(self): # if want to work with the computed distances
        return self.distance_matrix

class Random_Map:
    def __init__(self, max_x=100, max_y=100, num_cities=20):
        self.max_x = max_x              # define the rectangle on which to spawn the cities (x-axis)
        self.max_y = max_y              # define the rectangle on which to spawn the cities (y-axis)
        self.num_cities = num_cities    # number of cities to spawn
        self.cities = []                # list of cities spawned (starts empty) [(coord_x, coord_y), ....]
        self.distance_matrix = []       # cost matrix, pairwise distance between cities
        self.create_map()               # creates the cities. fills in self.cities
        self.compute_distances()        # uses the cities to fill in the cost matrix (distance_matrix)

    def create_map(self):
        """ Create cities on random locations (int, int) inside the rectangle area defined
        by max_x, max_y """
        u1 = [np.random.randint(0,self.max_x) for _ in range(self.num_cities)]
        u2 = [np.random.randint(0,self.max_y) for _ in range(self.num_cities)]
        self.cities = [city for city in zip(u1, u2)]

    def compute_distances(self):
        """
        Fill in the matrix of distances, Compute the distance (euclidean) 
        between all pairs of cities.
        Note: Since distance from A to B is the same as B to A, the number of operations
        done here could be cut in half, but given the low number of cities its ok to repeat calculations.
        """
        for city in self.cities:
            for next_city in self.cities:
                distance = self.euclidean_distance(city, next_city)
                self.distance_matrix.append(distance)

        self.distance_matrix = np.array(self.distance_matrix).reshape((self.num_cities, self.num_cities))

    def euclidean_distance(self, origin, dest):
        x0, y0 = origin
        x1, y1 = dest
        return math.sqrt((x0-x1)**2 + (y0-y1)**2)

    def plot(self, path=None):
        """
        Given a path plot the route of the salesman and the cities.
        """
        coord_path_x, coord_path_y = [], []
        for city in self.cities:
            plt.plot(*city, "xr") # plot the cities

        if path:
            past_city = self.cities[path[0]]
            plt.plot(*past_city, "og") # plot the starting point
            for i in range(1,self.num_cities):
                city_idx = path[i]
                coord_path_x.append([past_city[0], self.cities[city_idx][0]])
                coord_path_y.append([past_city[1], self.cities[city_idx][1]])
                past_city = self.cities[city_idx]

            for idx in range(len(coord_path_x)):
                plt.plot(coord_path_x[idx], coord_path_y[idx], "b")

        plt.title("Route of the Salesman on Random Map")
        plt.show()

    def save_progress(self, iter_, iter_max, path=None):
        """
        Given a path plot the route of the salesman and the cities.
        """
        # folder where images are saved for animation
        if not os.path.exists("temp_folder_animation"):
            os.makedirs("temp_folder_animation")

        coord_path_x, coord_path_y = [], []
        for city in self.cities:
            plt.plot(*city, "xr") # plot the cities

        if path:
            past_city = self.cities[path[0]]
            plt.plot(*past_city, "og") # plot the starting point
            for i in range(1,self.num_cities):
                city_idx = path[i]
                coord_path_x.append([past_city[0], self.cities[city_idx][0]])
                coord_path_y.append([past_city[1], self.cities[city_idx][1]])
                past_city = self.cities[city_idx]

            for idx in range(len(coord_path_x)):
                plt.plot(coord_path_x[idx], coord_path_y[idx], "b")

            plt.title(f"Route of the Salesman on Random Map, iter={iter_}")
            magnitude = math.floor(math.log10(iter_))
            max_magnitude = math.floor(math.log10(iter_max))
            plt.savefig("temp_folder_animation/random"+ "0"*(max_magnitude-magnitude) + f"{iter_}.png", format="png")
            plt.close()
            print(f"Generating figure {iter_}/{iter_max}")

    def animate(self):
        # Create the frames
        frames = []
        imgs = glob.glob("temp_folder_animation/random"+"*.png")
        imgs = sorted(imgs)
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)
        
        # folder where images are saved for animation
        if not os.path.exists("results"):
            os.makedirs("results")

        print("Generating animation...")
        # Save into a GIF file that loops forever
        frames[0].save("results"+'/random.gif', format='GIF', append_images=frames[1:], save_all=True, duration=30, loop=0)

        print("Cleaning temporary images for animation...")
        # remove the images
        for img in imgs:
            os.remove(img)

    def get_cities(self): # if want to work with the coordinates of cities
        return self.cities

    def get_distances(self): # if want to work with the computed distances
        return self.distance_matrix


class Random_Clusters_Map:
    def __init__(self, max_x=100, max_y=100, max_X=1000, max_Y=1000, num_cities=25, num_clusters=4):
        self.max_X = max_X              # define the greater rectangle on which to spawn the clusters (x-axis)
        self.max_Y = max_Y              # define the rectangle on which to spawn the clusters (y-axis)
        self.num_clusters=num_clusters  # number of clusters of cities to spawn

        self.max_x = max_x              # define the rectangle on which to spawn the cities (x-axis)
        self.max_y = max_y              # define the rectangle on which to spawn the cities (y-axis)
        self.num_cities = num_cities    # number of cities to spawn
        self.cities = []                # list of cities spawned (starts empty) [(coord_x, coord_y), ....]
        self.distance_matrix = []       # cost matrix, pairwise distance between cities
        self.create_map()               # creates the cities. fills in self.cities
        self.compute_distances()        # uses the cities to fill in the cost matrix (distance_matrix)

    def create_map(self):
        """ Create cities on random locations (int, int) inside the rectangle area defined
        by max_X, max_Y (clusters area) and then each city inside the rectangle max_x, max_y (city area) """
        for _ in range(self.num_clusters):
            U1 = np.random.randint(0, self.max_X)
            U2 = np.random.randint(0, self.max_Y)

            u1 = [np.random.randint(0,self.max_x) + U1 for _ in range(self.num_cities)]
            u2 = [np.random.randint(0,self.max_y) + U2 for _ in range(self.num_cities)]

            for city in zip(u1, u2):
                self.cities.append(city)

    def compute_distances(self):
        """
        Fill in the matrix of distances, Compute the distance (euclidean) 
        between all pairs of cities.
        Note: Since distance from A to B is the same as B to A, the number of operations
        done here could be cut in half, but given the low number of cities its ok to repeat calculations.
        """
        for city in self.cities:
            for next_city in self.cities:
                distance = self.euclidean_distance(city, next_city)
                self.distance_matrix.append(distance)

        self.distance_matrix = np.array(self.distance_matrix).reshape((self.num_cities*self.num_clusters, self.num_cities*self.num_clusters))

    def euclidean_distance(self, origin, dest):
        x0, y0 = origin
        x1, y1 = dest
        return math.sqrt((x0-x1)**2 + (y0-y1)**2)

    def plot(self, path=None):
        """
        Given a path plot the route of the salesman and the cities.
        """
        coord_path_x, coord_path_y = [], []
        for city in self.cities:
            plt.plot(*city, "xr") # plot the cities

        if path:
            past_city = self.cities[path[0]]
            plt.plot(*past_city, "og") # plot the starting point
            for i in range(1,len(self.cities)):
                city_idx = path[i]
                coord_path_x.append([past_city[0], self.cities[city_idx][0]])
                coord_path_y.append([past_city[1], self.cities[city_idx][1]])
                past_city = self.cities[city_idx]

            for idx in range(len(coord_path_x)):
                plt.plot(coord_path_x[idx], coord_path_y[idx], "b")

        plt.title("Route of the Salesman on Random Map")
        plt.show()

    def save_progress(self, iter_, iter_max, path=None):
        """
        Given a path plot the route of the salesman and the cities.
        """
        # folder where images are saved for animation
        if not os.path.exists("temp_folder_animation"):
            os.makedirs("temp_folder_animation")

        coord_path_x, coord_path_y = [], []
        for city in self.cities:
            plt.plot(*city, "xr") # plot the cities

        if path:
            past_city = self.cities[path[0]]
            plt.plot(*past_city, "og") # plot the starting point
            for i in range(1,self.num_cities):
                city_idx = path[i]
                coord_path_x.append([past_city[0], self.cities[city_idx][0]])
                coord_path_y.append([past_city[1], self.cities[city_idx][1]])
                past_city = self.cities[city_idx]

            for idx in range(len(coord_path_x)):
                plt.plot(coord_path_x[idx], coord_path_y[idx], "b")

            plt.title(f"Route of the Salesman on Random Cluster Map, iter={iter_}")
            magnitude = math.floor(math.log10(iter_))
            max_magnitude = math.floor(math.log10(iter_max))
            plt.savefig("temp_folder_animation/random_cluster"+ "0"*(max_magnitude-magnitude) + f"{iter_}.png", format="png")
            plt.close()
            print(f"Generating figure {iter_}/{iter_max}")

    def animate(self):
        # Create the frames
        frames = []
        imgs = glob.glob("temp_folder_animation/random_cluster"+"*.png")
        imgs = sorted(imgs)
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)
        
        # folder where images are saved for animation
        if not os.path.exists("results"):
            os.makedirs("results")

        print("Generating animation...")
        # Save into a GIF file that loops forever
        frames[0].save("results"+'/random_cluster.gif', format='GIF', append_images=frames[1:], save_all=True, duration=30, loop=0)

        print("Cleaning temporary images for animation...")
        # remove the images
        for img in imgs:
            os.remove(img)

    def get_cities(self): # if want to work with the coordinates of cities
        return self.cities

    def get_distances(self): # if want to work with the computed distances
        return self.distance_matrix