
# Import Statements
import os
import topogenesis as tg
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import cycle



class PerformanceLattice:
    def __init__(self, base_lattice):
        self.base_lattice = base_lattice+1
        self.lattice = base_lattice
        self.north_distance = np.max(base_lattice.centroids.T[1])
        self.south_distance = np.min(base_lattice.centroids.T[1])
        self.east_distance = np.max(base_lattice.centroids.T[0])
        self.west_distance = np.min(base_lattice.centroids.T[0])

    def create_distance_lattice(self, points):
        """
        Create a distance based lattice  on the point sources and initial lattice.
        The resulting distance is arg min values from the closest point
        """
        point_lattice =np.array([[pt.X, pt.Y, pt.Z] for pt in points])
        # Create availability lattice
        init_lattice = self.base_lattice
        voxel_coordinates = init_lattice.centroids

        # Flatten the initial lattice
        flattened_lattice = self.base_lattice.flatten()

        # Compute Euclidean distances
        eucledian_distance = sc.spatial.distance.cdist(point_lattice, voxel_coordinates)
        distance_from_each_source = eucledian_distance.T

        # Compute the average quiteness values
        average_distance_indexing = np.argmin(distance_from_each_source, axis=1)
        average_distance_values_full = np.array([branch[index] for branch, index in zip(distance_from_each_source, average_distance_indexing)])
        
        # Flat list of distance values
        average_distance_values = average_distance_values_full.reshape(self.base_lattice.shape)[self.lattice].flatten()

        # Convert to lattice format
        distance_lattice = average_distance_values_full.reshape(self.base_lattice.shape)

        return average_distance_values , distance_lattice
    
    def create_facade_lattice(self, direction:str) -> None:
        # Flatten the initial lattice
        flattened_lattice = self.base_lattice.flatten()

        if direction == 'north':
            evaluation_value = self.north_distance
            distance_lattice_full = evaluation_value- self.base_lattice.centroids.T[1]
            distance_lattice_np = distance_lattice_full.reshape(self.base_lattice.shape)
            distance_lattice = distance_lattice_np[self.lattice].flatten()

        elif direction == 'south':
            evaluation_value = self.south_distance
            distance_lattice_full = np.abs(evaluation_value-self.base_lattice.centroids.T[1])
            distance_lattice_np = distance_lattice_full.reshape(self.base_lattice.shape)
            distance_lattice = distance_lattice_np[self.lattice].flatten()

        elif direction == 'west':
            evaluation_value = self.west_distance
            distance_lattice_full = np.abs(evaluation_value-self.base_lattice.centroids.T[0])
            distance_lattice_np = distance_lattice_full.reshape(self.base_lattice.shape)
            distance_lattice = distance_lattice_np[self.lattice].flatten()

        elif direction == 'east':
            evaluation_value = self.east_distance
            distance_lattice_full = evaluation_value-self.base_lattice.centroids.T[0]
            distance_lattice_np = distance_lattice_full.reshape(self.base_lattice.shape)
            distance_lattice = distance_lattice_np[self.lattice].flatten()

        return distance_lattice,distance_lattice_np
    
    def createcolormap(self,lattice,gradient):
        colormap = cm.get_cmap(f'{gradient}')
        # Step 3: Normalize values between 0 and 1
        def normalize_lattice(lattice):
            """
            Normalize a NumPy lattice (array) to the range [0,1].
            
            :param lattice: NumPy array
            :return: Normalized NumPy array
            """
            min_val = np.min(lattice)
            max_val = np.max(lattice)
            
            # Avoid division by zero
            if max_val - min_val == 0:
                return np.zeros_like(lattice)
            
            normalized_lattice = (lattice - min_val) / (max_val - min_val)
            return normalized_lattice
        normalized_lattice = normalize_lattice(lattice)
        colour_list = (colormap(normalized_lattice)[:, :3] * 255).astype(int) 
        return colour_list
    
    def distance_from_outermostvoxel (self):
        return self.base_lattice