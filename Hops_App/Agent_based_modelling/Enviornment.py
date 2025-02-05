# Definition for an enviornment class
# environment class
class environment():
    def __init__(self, lattices, agents,number_of_iterations,method_name):
        self.availibility = lattices["availibility"]
        self.value = lattices["enviornment"]
        self.agent_origin = self.availibility 
        self.agents = agents
        self.update_agents_rectangularish()
        self.number_of_iterations = number_of_iterations
        self.method_name = method_name
        self.neigh_squareness = []
        #Number of voxels left 
        self.voxelsleft = 100       

    def update_agents_rectangularish(self):
        # making previous position available
      #  self.availibility[tuple(self.agents.old_origin)] = self.availibility[tuple(self.agents.old_origin)] * 0 + 1
        # removing agent from previous position
        self.agent_origin[tuple(self.agents.old_origin)] *= 0+1
        # making the current position unavailable
        self.availibility[tuple(self.agents.origin)] = self.agents.id
        # adding agent to the new position 
        self.agent_origin[tuple(self.agents.origin)] = self.agents.id

    def update_agents_squarish(self,bool_response):
        # making previous position available
        #  self.availibility[tuple(self.agents.old_origin)] = self.availibility[tuple(self.agents.old_origin)] * 0 + 1
        # removing agent from previous position
        self.agent_origin[tuple(self.agents.old_origin)] *= 0+1
        # making the current position unavailable
        if bool_response == True :
            self.availibility[tuple(self.agents.origin)] = self.agents.id
            # adding agent to the new position 
            self.agent_origin[tuple(self.agents.origin)] = self.agents.id        
        
    def rectangularish_occupy(self):
        # iterate over egents and perform the walk
        self.agents.rectangularish_occupy(self)
        # update the agent states in environment
        self.voxelsleft = self.number_of_iterations - self.agents.voxelsoccupied
        self.update_agents_rectangularish()
        #print(self.voxelsleft)
        
    def squarish_occupy(self,iter):
        # iterate over egents and perform the walk
        var = self.agents.squarish_occupy(self,iter)
        self.voxelsleft = self.number_of_iterations - self.agents.voxelsoccupied
        #print(self.voxelsleft)
      
        # update the agent states in environment
        self.update_agents_squarish(var)