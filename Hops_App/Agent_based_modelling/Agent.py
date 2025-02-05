# Import Statements
import os
import topogenesis as tg
import trimesh as tm
import numpy as np


# Definition for an agent class
class agent():
    def __init__(self, origin, stencil, id):

        # define the origin attribute of the agent and making sure that it is an intiger
        self.origin = np.array(origin).astype(int)
        # define old origin attribute and assigning the origin to it as the initial state
        self.old_origin = self.origin
        # define stencil of the agent
        self.stencil = stencil
        #define agent id
        self.id = id
        #Number of voxels left 
        self.voxelsoccupied = 1
        
   
    def rectangularish_occupy(self, env):
        # retrieve the list of neighbours of the agent based on the stencil
        #print(self.origin)
        neighs = env.availibility.find_neighbours_masked(self.stencil, loc = self.origin)
        #print(neighs)

        unravelled_neighs = np.array(np.unravel_index(neighs, env.availibility.shape)).T
        #print(unravelled_neighs)
        # find availability of neighbours
        neighs_availibility = env.availibility.flatten()[neighs]  

        
        # find env values of all neighbours
        all_neighs_value = env.value.flatten()[neighs]
        all_neighs_value_mod = np.copy(all_neighs_value)
        
        
        #finding number of neighbours and bumping the values based on adjacency for a 9 neighbourhood
        """
        This method is assuming that you will have a 9 neighbourhood everytime but it is not the case so try to addd if functions etc to make it
        work on neighbourhoods with smaller number of neighbours as well
        """

        #print(neighbourhood_details)
        one = neighs_availibility[4] + neighs_availibility[5] 
        two = neighs_availibility[4] + neighs_availibility[6] 
        three = neighs_availibility[5] + neighs_availibility[7] 
        four = neighs_availibility[6] + neighs_availibility[7] 
        five = neighs_availibility[1] + neighs_availibility[0] 
        six = neighs_availibility[0] + neighs_availibility[2] 
        seven = neighs_availibility[1] + neighs_availibility[3] 
        eight = neighs_availibility[2] + neighs_availibility[3] 
        neighbourhood_details = [one, two, three, four, five, six, seven, eight]


        #print(neighbourhood_details)
        for detail in range(len(neighs_availibility)-1):
            neighbourhood_condition = neighbourhood_details[detail] 
            #print(neighbourhood_condition)
            if neighbourhood_condition == self.id or neighbourhood_condition == self.id+1:
                all_neighs_value_mod[detail]= all_neighs_value_mod[detail] + 1
                #print("One neigh  found")
            elif neighbourhood_condition == self.id*2:
                all_neighs_value_mod[detail]= all_neighs_value_mod[detail] + 2
                #print("Two neigh  found")
            else:
                all_neighs_value_mod[detail] = all_neighs_value_mod[detail] 

        #print(all_neighs_value_mod)

        neighs_value_flattened = env.value.flatten()
        
        
        for neigh, val_mod in zip(neighs, all_neighs_value_mod):
             neighs_value_flattened[neigh] = val_mod
        
        # separate available neighbours
        free_neighs = neighs[neighs_availibility==1]

        avail_index = np.array(np.where(env.availibility == 1))
        neighs_availibility_full_floor = np.ravel_multi_index(avail_index, env.availibility.shape)

        if len(free_neighs)== 0 :
            free_neighs = neighs_availibility_full_floor
            #print("Full floor considered")
        else: 
            free_neighs= free_neighs

        # retrieve the value of each neighbour
        free_neighs_value = neighs_value_flattened[free_neighs]
        #print(free_neighs)
        #print(free_neighs_value)

        # find the neighbour with maximum my value
        selected_neigh = free_neighs[np.argmax(free_neighs_value)]
        #print(selected_neigh)
        # update information
        ####################
        self.voxelsoccupied = np.sum(env.availibility.flatten()== self.id)
        # set the current origin as the ol origin
        self.old_origin = self.origin
        # update the current origin with the new selected neighbour
        self.origin = np.array(np.unravel_index(selected_neigh, env.availibility.shape)).flatten()
        #print(self.origin)

    def squarish_occupy(self,env, iter):
        num_iter = env.number_of_iterations
        current_step = iter
        def generate_pattern(total_numbers):
            # Initialize an empty list to store the result
            pattern = []

            # We need to keep track of the number of digits added so far
            current_count = 0
            i = 1  # Start with row 1 (initially number 3)

            while current_count < total_numbers:
                if i % 2 != 0:  # Odd rows (e.g., row 1, 3, 5...)
                    row = [int(digit) for digit in '1' * i] + [int(digit) for digit in '3' * i]
                else:  # Even rows (e.g., row 2, 4, 6...)
                    row = [int(digit) for digit in '2' * i] + [int(digit) for digit in '0' * i]

                # Check if adding this row exceeds the total number of digits
                if current_count + len(row) > total_numbers:
                    row = row[:total_numbers - current_count]  # Trim the row if needed
                
                pattern.append(row)
                current_count += len(row)
                i += 1  # Increment the row number
            
            # Flatten the pattern into a 1D NumPy array
            flattened_pattern = np.concatenate(pattern)
            
            return flattened_pattern
        # Generate the flattened 1D NumPy array
        result = generate_pattern(iter+10)
        neighs = env.availibility.find_neighbours_masked(self.stencil, loc = self.origin)
        occupy_id = neighs[result[iter]]

        neighs_availibility = env.availibility.flatten()[neighs]

        self.old_origin = self.origin
        # update the current origin with the new selected neighbour
        self.origin = np.array(np.unravel_index(occupy_id, env.availibility.shape)).flatten()

        self.voxelsoccupied = np.sum(env.availibility.flatten()== self.id)
        #print(self.remainingvoxels)

        if neighs_availibility[result[iter]] == 1 :
            return True
        num_iter = env.number_of_iterations
        current_step = iter
        def generate_pattern(total_numbers):
            # Initialize an empty list to store the result
            pattern = []

            # We need to keep track of the number of digits added so far
            current_count = 0
            i = 1  # Start with row 1 (initially number 3)

            while current_count < total_numbers:
                if i % 2 != 0:  # Odd rows (e.g., row 1, 3, 5...)
                    row = [int(digit) for digit in '1' * i] + [int(digit) for digit in '3' * i]
                else:  # Even rows (e.g., row 2, 4, 6...)
                    row = [int(digit) for digit in '2' * i] + [int(digit) for digit in '0' * i]

                # Check if adding this row exceeds the total number of digits
                if current_count + len(row) > total_numbers:
                    row = row[:total_numbers - current_count]  # Trim the row if needed
                
                pattern.append(row)
                current_count += len(row)
                i += 1  # Increment the row number
            
            # Flatten the pattern into a 1D NumPy array
            flattened_pattern = np.concatenate(pattern)
            
            return flattened_pattern
        # Generate the flattened 1D NumPy array
        result = generate_pattern(iter+10)
        neighs = env.availibility.find_neighbours_masked(self.stencil, loc = self.origin)
        occupy_id = neighs[result[iter]]

        neighs_availibility = env.availibility.flatten()[neighs]

        self.old_origin = self.origin
        # update the current origin with the new selected neighbour
        self.origin = np.array(np.unravel_index(occupy_id, env.availibility.shape)).flatten()

        self.voxelsoccupied = np.sum(env.availibility.flatten()== self.id)
        #print(self.remainingvoxels)

        if neighs_availibility[result[iter]] == 1 :
            return True