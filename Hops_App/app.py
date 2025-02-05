from flask import Flask
import ghhops_server as hs
import os
from topogenesis import geometry
import rhino3dm as rg
import numpy as np
from Voxelation.Rhino_Voxelate import meshvoxelation
from Performance_matrices.Performance_Lattice import PerformanceLattice
from MCDA_Lattices.Case_lattices import mcda_case
from MCDA_Lattices.Desirability_lattice import DesirabilityLattice
from Agent_based_modelling.ABM import abm
import matplotlib.cm as cm



# register hops app as middleware
app = Flask(__name__)
hops = hs.Hops(app)    

@hops.component(
    "/voxelate",
    name="Voxelate",
    description="Get voxels from mesh",
    inputs=[
        hs.HopsMesh("Basemesh", "B_m", "Triangular mesh to voxelate"),
        hs.HopsNumber("xVoxel", "xsize", "Voxelation x size"),
        hs.HopsNumber("yVoxel", "ysize", "Voxelation  y size"),
        hs.HopsNumber("zVoxel", "zsize", "Voxelation z size")
    ],
    outputs=[
        hs.HopsBrep("Voxel Grid", "Gets Voxel Breps", "Get uncoloured voxels")
    ]
)
def voxelate(mesh,vx,vy,vz):

    voxelator = meshvoxelation(mesh,vx,vy,vz)
    voxels = voxelator.createbrepvoxels()  
    return voxels

@hops.component(
    "/performancearray",
    name="Performances",
    description="Generate performance lattices",
    inputs=[
        hs.HopsMesh("Basemesh", "B_m", "Triangular mesh to voxelate"),
        hs.HopsNumber("xVoxel", "xsize", "Voxelation x size"),
        hs.HopsNumber("yVoxel", "ysize", "Voxelation  y size"),
        hs.HopsNumber("zVoxel", "zsize", "Voxelation z size"),
        hs.HopsPoint("Noise Source", "points indicating the source of noise", "Noise Source",access=list),
        hs.HopsPoint("Core Source", "Points showing the location of core", "Distance from core",access=list),
        hs.HopsString("Analysis to display", "Select the analysis to show on grasshopper", "Distance from core")
    ],
    outputs=[
        hs.HopsMesh("Performance Lattice", "P lattices ", "Vizualse the various performance lattices")
    ]
)
def performancearray(mesh,vx,vy,vz,noise,core,analysis):

    voxelator = meshvoxelation(mesh,vx,vy,vz)
    lattice = voxelator.rhino3dmtotglattice()
    # Create Performance Lattice creator class
    Performance_Creator = PerformanceLattice(lattice)
    
    if analysis == "Quiteness":
        Quiteness_lattice = Performance_Creator.create_distance_lattice(noise)[0]
        performance_lattice = Quiteness_lattice
    elif analysis == "Core_distance":
        Coredis_lattice = Performance_Creator.create_distance_lattice(core)[0]
        performance_lattice = Coredis_lattice
    elif analysis == "East":
        East_facade_lattice = Performance_Creator.create_facade_lattice('east')[0]
        performance_lattice = East_facade_lattice
    elif analysis == "West":
        West_facade_lattice = Performance_Creator.create_facade_lattice('west')[0]
        performance_lattice = West_facade_lattice
    elif analysis =="North":
        North_facade_lattice = Performance_Creator.create_facade_lattice('north')[0]
        performance_lattice = North_facade_lattice
    elif analysis == "South":
        South_facade_lattice = Performance_Creator.create_facade_lattice('south')[0]
        performance_lattice = South_facade_lattice
    
    #Closeness to Facade can be equated to daylight availability
    colour_list = Performance_Creator.createcolormap(performance_lattice,'viridis')
    voxels = voxelator.createcoloredmeshvoxels(colour_list)
    return voxels

@hops.component(
    "/desirabilitylattice",
    name="MCDA",
    description="Perform a Multi-Criteria Decision Analysis on the performance lattices",
    inputs=[
        hs.HopsMesh("Basemesh", "B_m", "Triangular mesh to voxelate"),
        hs.HopsNumber("xVoxel", "xsize", "Voxelation x size"),
        hs.HopsNumber("yVoxel", "ysize", "Voxelation  y size"),
        hs.HopsNumber("zVoxel", "zsize", "Voxelation z size"),
        hs.HopsPoint("Noise Source", "Points indicating the source of noise", "Noise Source",access=list),
        hs.HopsPoint("Core Source", "Points showing the location of core", "Distance from core",access=list),
        hs.HopsNumber("MCDA Weights", "Weights for the MCDA process", "MCDA Weights",access=list)
    ],
    outputs=[
        hs.HopsMesh("Desirability Lattice", "MCDA", "MCDA Results shown as a desirability lattice")
    ]
)
def desirabilitylattice(mesh,vx,vy,vz,noise,core,weights):

    voxelator = meshvoxelation(mesh,vx,vy,vz)
    lattice = voxelator.rhino3dmtotglattice()

    # Create Performance Lattice creator class
    Performance_Creator = PerformanceLattice(lattice)
    Quiteness_lattice = Performance_Creator.create_distance_lattice(noise)[1]
    Coredis_lattice = Performance_Creator.create_distance_lattice(core)[1]
    East_facade_lattice = Performance_Creator.create_facade_lattice('east')[1]
    West_facade_lattice = Performance_Creator.create_facade_lattice('west')[1]
    North_facade_lattice = Performance_Creator.create_facade_lattice('north')[1]
    South_facade_lattice = Performance_Creator.create_facade_lattice('south')[1]
    array_all_performance_matrices = [Quiteness_lattice.flatten(),Coredis_lattice.flatten(), East_facade_lattice.flatten(), West_facade_lattice.flatten(), North_facade_lattice.flatten(), South_facade_lattice.flatten()]
    performance_matrix = np.array(array_all_performance_matrices).T

    
    criteria_array = ["quietness", "coredis", "eastfacade", "westfacade", "northfacade", "southfacade"]
    objectives_array = [max,min,max,max,max,max]
    MCDA_generator = DesirabilityLattice(lattice,performance_matrix)

    #Zone1 parameters
    Z1_Desirability_lattice = MCDA_generator.topsis(objectives_array, weights, criteria_array)[0]

    colour_list = Performance_Creator.createcolormap(Z1_Desirability_lattice,'plasma')
    voxels = voxelator.createcoloredmeshvoxels(colour_list)

    return voxels

@hops.component(
    "/ABMSimulation",
    name="ABM",
    description="Perform an Agent Based Simulation to get Zoning outputs",
    inputs=[
        hs.HopsMesh("Basemesh", "B_m", "Triangular mesh to voxelate"),
        hs.HopsNumber("xVoxel", "xsize", "Voxelation x size"),
        hs.HopsNumber("yVoxel", "ysize", "Voxelation  y size"),
        hs.HopsNumber("zVoxel", "zsize", "Voxelation z size"),
        hs.HopsPoint("Noise Source", "Noise Source", "Points indicating the source of noise",access=list),
        hs.HopsPoint("Core Source", "Core Location", "Points showing the location of core",access=list),
        hs.HopsString("Zone names", "Zone names", "Create a list of zone names",access=list),
        hs.HopsString("Zone colors", "Zone colors", "Assign the colors to zones",access=list),
        hs.HopsInteger("Zone size", "Zone size", "Specify the number of zones and make sure that total number of voxels is less than equal to available voxels",access=list),
        hs.HopsInteger("Number of Agents", "Agent number", "Number of Agents per zone",access=list),
        hs.HopsString("Agent Behaviours", "Agent behaviours", "Select the occupy behaviour of the agents. Currently configured as behaviour per zone ",access=list),
        hs.HopsNumber("MCDA Weights", "MCDA Weights", "Weights for each performance matrix the total of all weights added should equal to 1",access = 2 ),
        hs.HopsInteger("Number of options", "No. Options", "Mention number of required zoning options to create"),
    ],
    outputs=[
        hs.HopsMesh("Zoning options", "Zonings", "Voxels outputed as a tree of options", access = 2),
        hs.HopsNumber("Total available voxels", "total voxels", "Total number of voxels that can be occupied"),
        hs.HopsInteger("Agent id", "id of Agents", "Id of each agent assigne din the voxelated grid",access=list),
        hs.HopsNumber("Voxel values", "Values", "Desirability value for each voxel is extracted from the respective desirability lattices of the zones",access=list)
    ]
)
def ABMSimulation(mesh,vx,vy,vz,noise,core,zones,zonecolors,sizes,agentsnumber,agentbehaviours,mcda,options):

    voxelator = meshvoxelation(mesh,vx,vy,vz)
    base_lattice = voxelator.rhino3dmtotglattice()
    mcda_creator = mcda_case(base_lattice,zones,mcda,noise,core)
    desirability_lattices = mcda_creator.createDesirabilitylattices()
    abm_simulator = abm(base_lattice,desirability_lattices,zones,sizes,agentsnumber,agentbehaviours,options)
    rgb_arrays = [list(map(int, s.split(','))) for s in zonecolors]
    rgb_arrays.insert(0,[255,255,255])
    rgb_nparray = np.array(rgb_arrays)

    if len(zones) != len(agentsnumber) :
        raise ValueError("Error: The lengths of 'zones' and 'agentsnumber' lists do not match. You should define number of agents for each zone in the list")
    
    # if len(zonecolors) != len(zones) :
    #     raise ValueError("Error: The lengths of 'zone colours' and 'zones' lists do not match. You should define a color for each zone")
    
    if len(mcda)/len(zones) != 6:
        raise ValueError("Error: The lengths of 'MCDA Weights ' and 'number of zones' lists do not match. All the 6 criterias considered should have weights for each zone ")

    def createcolormap(lattice,gradient):
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
    
    zoneLattices = abm_simulator.runZones()
    voxel_list = [] 
    
    for count, option in enumerate(zoneLattices) :

        colour_list = createcolormap(option,'plasma')
        unique_values = np.unique(option)
        color_map = dict(zip(unique_values, rgb_nparray))
        # Convert the mapping function to work with NumPy arrays
        map_function = np.vectorize(lambda x: color_map[x], signature='()->(n)')
        rgb_array = map_function(option)
        voxels = voxelator.createcoloredmeshvoxels(rgb_array) 
        voxel_list.append(voxels)
    flattened_voxels = [item for sublist in voxel_list for item in sublist]
    flattenedids = [item for sublist in zoneLattices for item in sublist]
 
    zoneValues = abm_simulator.getValues(zoneLattices)
    flattned_values_for_zones = np.ravel(zoneValues).astype(float).tolist()

    # return zoneMeshes , zoneValues
    return flattened_voxels , len(base_lattice.centroids), [int(n) for n in flattenedids],flattned_values_for_zones

    

if __name__ == "__main__":
    app.run()