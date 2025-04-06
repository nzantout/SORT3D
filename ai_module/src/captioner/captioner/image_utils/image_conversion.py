import numpy as np

from typing import List
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def sample_pixels(begin: float, end: float, n: int) -> np.ndarray:
    """
    Sample pixel values of an affine function, which is equal
    to "begin" on the far left and "end" on the far right.
    """
    x_coords = 0.5 + np.arange(n, dtype=np.float32)
    return begin + (end-begin) * x_coords / n  # ]begin, end[


def equirec_to_cubemap(equirec: np.ndarray, out_size: int) -> List[np.ndarray]:
    """Convert an equirectangular image to a list of cubefaces (FRBLUD)"""
    height, width = equirec.shape[:2]

    u, v = np.meshgrid(sample_pixels(-1, 1, out_size),
                       sample_pixels(-1, 1, out_size),
                       indexing="ij")
    ones = np.ones((out_size, out_size), dtype=np.float32)

    list_xyz = [
        (v, u, ones),    # FRONT
        (ones, u, -v),   # RIGHT
        (-v, u, -ones),  # BACK
        (-ones, u, v),   # LEFT
    ]

    faces = []
    r = np.sqrt(u**2 + v**2 + 1) # Same values for each face
    for x, y, z in list_xyz:
        # Camera Convention RIGHT_DOWN_FRONT
        phi = np.arcsin(y/r)  # in [-pi/2, pi/2]
        theta = np.arctan2(x, z)  # in [-pi, pi]

        phi_map = (phi/np.pi + 0.5) * height
        theta_map = (theta/(2*np.pi) + 0.5) * width
    
        # Opencv shift
        theta_map -= 0.5
        phi_map -= 0.5
        
        faces.append(cv2.remap(equirec, theta_map, phi_map, cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_WRAP))
    return faces


def undistort_cylindrical_section(image, left_pixel, right_pixel, top_pixel, bottom_pixel, output_size):
    """
    Undistorts a section of a cylindrical image.
    
    Args:
        image (np.ndarray): The input cylindrical image.
        start_angle (float): Starting horizontal angle (in degrees, 0 to 360).
        end_angle (float): Ending horizontal angle (in degrees, 0 to 360).
        vertical_range (tuple): Vertical range in the image (top, bottom) as fractions of the height (0 to 1).
        output_size (tuple): The size of the output image (width, height).
    
    Returns:
        np.ndarray: The undistorted section of the image.
    """
    h, w = image.shape[:2]
    output_width, output_height = output_size
    
    # Convert angles to radians
    left_angle = left_pixel / w * 2 * np.pi
    right_angle = right_pixel / w * 2 * np.pi
    
    # Define the vertical range
    top = top_pixel
    bottom = bottom_pixel
    
    # Create remap coordinates
    x_map = np.zeros((output_height, output_width), dtype=np.float32)
    y_map = np.zeros((output_height, output_width), dtype=np.float32)
    
    for i in range(output_height):
        for j in range(output_width):
            # Calculate the angle and vertical position in the cylindrical image
            angle = left_angle + (right_angle - left_angle) * (j / output_width)
            y = top + (bottom - top) * (i / output_height)
            
            # Convert cylindrical coordinates to image coordinates
            x_map[i, j] = (angle / (2 * np.pi)) * w
            y_map[i, j] = y
    
    # Wrap around the horizontal axis
    x_map = np.mod(x_map, w)
    
    # Remap the image
    undistorted = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR)
    
    return undistorted


def binary_opening_torch(image: torch.Tensor, device):
    kernel = torch.tensor([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    eroded = F.conv2d(image.unsqueeze(0).to(torch.float32), kernel, padding='same') >= kernel.sum()
    dilated = F.conv2d(eroded.to(torch.float32), kernel, padding='same') > 0
    return dilated.squeeze(0)