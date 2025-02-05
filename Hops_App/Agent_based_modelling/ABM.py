# Import Statements
import topogenesis as tg
import numpy as np
from Agent_based_modelling.Agent import agent
from Agent_based_modelling.Enviornment import environment
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# creating neighborhood definition
stencil_von_neumann = tg.create_stencil("von_neumann", 1, 1)
stencil_von_neumann.set_index([0,0,0], 0)

# creating neighborhood definition
stencil_moore = tg.create_stencil("moore", 1, 1)
stencil_moore.set_index([0,0,0], 0)

# creating neighborhood definition 
stencil_squareness_moore = tg.create_stencil("moore", 1, 1)
# Reshaping the moore neighbourhood
stencil_squareness_moore[0,:,:] = 0 
stencil_squareness_moore[2,:,:] = 0
stencil_squareness_moore.set_index([0,0,0], 0)
stencil_squareness_t = np.transpose(stencil_squareness_moore) 

# creating neighborhood definition 
stencil_squareness_von = tg.create_stencil("von_neumann", 1, 1)
# Reshaping the moore neighbourhood
stencil_squareness_von[0,:,:] = 0 
stencil_squareness_von[2,:,:] = 0
stencil_squareness_von.set_index([0,0,0], 0)

# Functions to Define agents
def initialize_agents_random_origin (stencil,avail_lattice,agn_num):
    avail_index = np.array(np.where(avail_lattice == 1))
    avail_index_1d = np.ravel_multi_index(avail_index, avail_lattice.shape)
    select_id = np.random.choice(avail_index_1d, 1)
    agn_origins = np.array(np.unravel_index(select_id[0],avail_lattice.shape)).T
    
    myagent = agent(agn_origins, stencil, agn_num)
    return myagent

def initialize_agents_defined_origin (stencil,avail_lattice,agn_num,origin_id):
    agn_origins = np.array(np.unravel_index(origin_id,avail_lattice.shape)).T
    myagent = agent(agn_origins, stencil, agn_num)
    return myagent

def split_list(flat_list, sizes):

    result = []
    index = 0
    
    for size in sizes:
        result.append(flat_list[index : index + size])  # Extract sublist
        index += size  # Move index forward

    return result

def repeat_list(flat_list, sizes):
    expanded_list = []  # This will store elements repeated as per sizes
    for item, size in zip(flat_list, sizes):
        expanded_list.extend([item] * size)  # Repeat each element based on sizes
    
    result = []  # This will store the final split lists
    index = 0
    
    for size in sizes:
        result.append(expanded_list[index: index + size])  # Extract sublist
        index += size  # Move index forward

    return result

def split_values_into_sublists(values, divisions):
    result = []
    
    for value, div in zip(values, divisions):
        base_value = value // div  # Integer division for base split
        remainder = value % div  # Remaining amount to distribute
        
        sublist = [base_value] * div  # Start with equal splits
        
        # Distribute the remainder evenly across the first few elements
        for i in range(remainder):
            sublist[i] += 1
        
        result.append(sublist)
    
    return result

class abm():
    def __init__(self,base_lattice,desirability_lattices,zonenames,zonesizes,numberofagents,agentbehaviours,numberofiterations):
        self.lattice = base_lattice
        self.desirabilityarrays = desirability_lattices
        self.zonenames = zonenames
        self.zonesizes = zonesizes
        self.numberofagents = numberofagents
        self.agentbehaviours = agentbehaviours
        self.options = numberofiterations
        self.zoneids = []

    def runZones(self):
        total_agents = sum(self.numberofagents)
        Agent_id_flatlist = [random.randint(10,200) for _ in range(len(self.numberofagents))]
        Agentids = repeat_list(Agent_id_flatlist,self.numberofagents)
        behaviourlistforagents = repeat_list(self.agentbehaviours,self.numberofagents)
        target_voxels = split_values_into_sublists(self.zonesizes,self.numberofagents)
        self.zoneids = Agent_id_flatlist
        env_availability_viz=[]
        for a in range(self.options):
            Enviornment_list = []
            occ_lattice_sim = tg.to_lattice(np.copy(self.lattice*1), self.lattice*1)
            for zone in range(len(self.zonenames)):
                for num in range(self.numberofagents[zone]):
                    Agent = initialize_agents_random_origin (stencil_squareness_t,self.lattice*1,Agentids[zone][num])
                    env_details= {"availibility": occ_lattice_sim,"enviornment": self.desirabilityarrays[zone]}
                    env = environment(env_details,Agent,target_voxels[zone][num],behaviourlistforagents[zone][num])
                    Enviornment_list.append(env)

            for e in Enviornment_list:
                a = 0
                while e.voxelsleft > 1 :
                    if e.method_name == "rectangularish_occupy":
                        e.rectangularish_occupy()
                        
                    elif e.method_name == "squarish_occupy" :
                        e.squarish_occupy(a)               
                    a = a+1 
            env_availability_viz.append(e.availibility[self.lattice]-1)
    
        return env_availability_viz
    
    def getValues(self,optionlattices):
        Agent_ids = self.zoneids
        Zone_ids = [x -1 for x in Agent_ids]
        value_outputs = []
        for option in optionlattices:
            masks = [(option == value) for value in Zone_ids]
            valuelattice = np.zeros(option.size)
            for mask,lattice in zip(masks,self.desirabilityarrays):
                # valuelattice.append(lattice[mask])
                desirability = lattice[self.lattice].flatten()
                valuelattice[mask]= desirability[mask]
        
            value_outputs.append(valuelattice)
    
        return value_outputs
        


    