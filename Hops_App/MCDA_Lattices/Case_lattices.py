from Performance_matrices.Performance_Lattice import PerformanceLattice
from MCDA_Lattices.Desirability_lattice import DesirabilityLattice
import numpy as np

class mcda_case:
    def __init__(self,base_lattice,zones,mcdaweights,noise,core):
        self.baselattice = base_lattice
        self.zones = zones
        self.mcdaweights = mcdaweights
        self.noisesources = noise
        self.coresources = core

    def createPerformancelattices(self):
        # Create Performance Lattice creator class
        Performance_Creator = PerformanceLattice(self.baselattice+1)
        Quiteness_lattice = Performance_Creator.create_distance_lattice(self.noisesources)[1]
        Coredis_lattice = Performance_Creator.create_distance_lattice(self.coresources)[1]
        East_facade_lattice = Performance_Creator.create_facade_lattice('east')[1]
        West_facade_lattice = Performance_Creator.create_facade_lattice('west')[1]
        North_facade_lattice = Performance_Creator.create_facade_lattice('north')[1]
        South_facade_lattice = Performance_Creator.create_facade_lattice('south')[1]
        array_all_performance_matrices = [Quiteness_lattice.flatten(),Coredis_lattice.flatten(), East_facade_lattice.flatten(), West_facade_lattice.flatten(), North_facade_lattice.flatten(), South_facade_lattice.flatten()]
        performance_matrix = np.array(array_all_performance_matrices).T

        return performance_matrix


    def createDesirabilitylattices(self):
        weightsflatarray  = np.array(self.mcdaweights)
        reshapedweights = weightsflatarray.reshape(-1,6).tolist()
        criteria_array = ["quietness", "coredis", "eastfacade", "westfacade", "northfacade", "southfacade"]
        objectives_array = [max,min,max,max,max,max]

        performance_matrix = self.createPerformancelattices()
        MCDA_generator = DesirabilityLattice(self.baselattice,performance_matrix)

        #Zone1 parameters
        Desirability_lattices = []
        for weights in reshapedweights :
            Desirability_lattices.append(MCDA_generator.topsis(objectives_array, weights, criteria_array)[1])
        
        return Desirability_lattices