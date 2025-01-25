from flask import Flask
import ghhops_server as hs
import os
from topogenesis import geometry
import pandas as pd
import rhino3dm as rg


# register hops app as middleware
app = Flask(__name__)
hops = hs.Hops(app)    

@hops.component(
    "/pointat",
    name="PointAt",
    description="Get point along curve",
    inputs=[
        hs.HopsNumber("Curve", "C", "Curve to evaluate"),
        hs.HopsNumber("t", "t", "Parameter on Curve to evaluate"),
    ],
    outputs=[
        hs.HopsPoint("P", "P", "Point on curve at t")
    ]
)


def pointat(curve, t):
    vs = 0.02
    unit = [vs, vs, vs]
    tol = 1e-09
    mesh_path = os.path.relpath(r"C:\Users\aditya.soman\Documents\Loci_GENARCH\topogenesis\data\bunny_lowpoly.obj")
    mesh = geometry.load_mesh(mesh_path)
    sample_cloud, ray_origins = geometry.mesh_sampling(mesh, unit, multi_core_process=False, return_ray_origin = True, tol=tol)
    lattice = sample_cloud.voxelate(unit, closed=True)
    voxel_points = lattice.centroids
    curve_points = [rg.Point3d(pt[0], pt[1], pt[2]) for pt in voxel_points]
    return curve_points

@hops.component(
    "/rhino3dmtovoxels",
    name="Voxelate",
    description="Get voxels from mesh",
    inputs=[
        hs.HopsMesh("Basemesh", "B_m", "Triangular mesh to voxelate"),
        hs.HopsNumber("voxelsize", "size", "Voxelation size"),
    ],
    outputs=[
        hs.HopsPoint("P", "P", "Point on curve at t")
    ]
)
def rhino3dmtovoxels(mesh, vs):
    unit = [vs, vs, vs]
    tol = 1e-09
    mesh = geometry.load_rhino3dmmesh(mesh)
    #sample_cloud, ray_origins = geometry.mesh_sampling(mesh, unit, multi_core_process=False, return_ray_origin = False, tol=tol)
    
    #lattice = sample_cloud.voxelate(unit, closed=True)
    #voxel_points = lattice.centroids
    #curve_points = [rg.Point3d(pt[0], pt[1], pt[2]) for pt in voxel_points]
    output = mesh.Faces.TriangleCount()
    return mesh.Vertces

if __name__ == "__main__":
    app.run()