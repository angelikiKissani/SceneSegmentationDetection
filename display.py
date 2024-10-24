import numpy as np
from open3d import *    

def display():
    cloud = open3d.io.read_point_cloud("cloud.ply") 
    open3d.visualization.draw_geometries([cloud]) 
if __name__ == "__main__":
    display()







