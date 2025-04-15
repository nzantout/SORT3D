import numpy as np
import math
 

class GridMapHandler:
    def __init__(self, origin=(0, 0)):
        self.grid_map = None
        self.origin = origin

    def create_2d_map(self, points, freespace, resolution=0.25):
        points_2d = points[:, :2]
        points_2d[:, 1] = -points_2d[:, 1]

        min_x = np.min(points_2d[:, 0])
        max_x = np.max(points_2d[:, 0])
        min_y = np.min(points_2d[:, 1])
        max_y = np.max(points_2d[:, 1])

        buffer = 0  # Buffer of 0.5 meters
        min_x -= buffer
        max_x += buffer
        min_y -= buffer
        max_y += buffer

        height = math.ceil(np.ceil((max_x - min_x) / resolution))
        width = math.ceil(np.ceil((max_y - min_y) / resolution))
        print(f"Grid Map Dimensions: Width={width}, Height={height}")
        self.grid_map = np.ones((height, width))

        print(min_x, max_x, min_y, max_y)

        freespace_normed = np.array([freespace[:, 0], -freespace[:, 1]]).T

        freespace_normed = (freespace_normed - np.array([[min_x, min_y]])) / resolution

        freespace_idxs = np.round(freespace_normed).astype(int)

        # filter freespace points
        height_oob = np.argwhere(freespace_idxs[:, 0] >= height)
        width_oob = np.argwhere(freespace_idxs[:, 1] >= width)
        print("height oob", height_oob)
        print("width oob", width_oob)
        if height_oob.size != 0:
            freespace_idxs = np.delete(freespace_idxs, height_oob, axis=0)
        
        if width_oob.size != 0:
            freespace_idxs = np.delete(freespace_idxs, width_oob, axis=0)

        print("freespace, ", freespace.shape)
        print("freespace idx ", freespace_idxs.shape)
        
        self.grid_map[freespace_idxs[:, 0], freespace_idxs[:, 1]] = 0
    
        self.origin = np.array([min_x, min_y])


    def convert_global_to_grid(self, global_points):
        # Define your grid resolution and origin
        grid_size = (self.grid_map.shape[0], self.grid_map.shape[1])
        print("grid size", grid_size)
        cell_size = 0.1 # resolution, adjust as needed
        grid_points=[]
        for coord in global_points:
                grid_x = int(np.floor((coord[0] - self.origin[0]) / cell_size))
                grid_y = int(np.floor((-coord[1] - self.origin[1]) / cell_size))
                grid_z = int(np.floor(coord[2] / cell_size)) # TODO: adjust as needed for range
                grid_points.append((grid_x, grid_y, grid_z))
        
        print("grid points", grid_points)

        return np.array(grid_points)
