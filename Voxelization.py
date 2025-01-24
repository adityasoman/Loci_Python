import os
from topogenesis import geometry
import pandas as pd

vs = 0.01
unit = [vs, vs, vs]
tol = 1e-09
mesh_path = os.path.relpath(r"C:\Users\aditya.soman\Documents\Loci_GENARCH\topogenesis\data\bunny_lowpoly.obj")
mesh = geometry.load_mesh(mesh_path)



sample_cloud, ray_origins = geometry.mesh_sampling(mesh, unit, multi_core_process=False, return_ray_origin = True, tol=tol)
lattice = sample_cloud.voxelate(unit, closed=True)
voxel_points = lattice.centroids


