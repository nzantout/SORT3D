import os
import colorsys
import rospy
import rosbag
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
import time

import os.path as osp
from pathlib import Path
import csv

from scipy.spatial.transform import Rotation as R

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, TransformStamped
from tf2_msgs.msg import TFMessage

import tf
import numpy as np
from std_msgs.msg import Header
import struct

def generate_colors(n, is_int=False):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        if is_int:
            rgb = [int(round(channel * 255)) for channel in rgb]
        colors.append(rgb)
    return colors


def create_odom_msg(odom, stamp, frame_id="map"):
    ros_odom = Odometry()
    ros_odom.header.stamp = rospy.Time.from_sec(stamp)
    ros_odom.header.frame_id = frame_id
    ros_odom.pose.pose.position.x = odom['position'][0]
    ros_odom.pose.pose.position.y = odom['position'][1]
    ros_odom.pose.pose.position.z = odom['position'][2]
    ros_odom.pose.pose.orientation.x = odom['orientation'][0]
    ros_odom.pose.pose.orientation.y = odom['orientation'][1]
    ros_odom.pose.pose.orientation.z = odom['orientation'][2]
    ros_odom.pose.pose.orientation.w = odom['orientation'][3]
    return ros_odom

def create_tf_msg(odom, stamp, frame_id="map", child_frame_id="sensor"):
    """
    Callback to handle incoming Odometry messages and publish them as a TF transform.
    """
    # Create a TransformStamped message
    transform = TransformStamped()
    
    # Set header information
    transform.header.stamp = rospy.Time.from_sec(stamp)
    transform.header.frame_id = frame_id  # e.g., "world"
    transform.child_frame_id = child_frame_id    # e.g., "base_link"
    
    # Set the translation from odometry
    transform.transform.translation.x = odom['position'][0]
    transform.transform.translation.y = odom['position'][1]
    transform.transform.translation.z = odom['position'][2]

    # Set the rotation from odometry
    transform.transform.rotation.x = odom['orientation'][0]
    transform.transform.rotation.y = odom['orientation'][1]
    transform.transform.rotation.z = odom['orientation'][2]
    transform.transform.rotation.w = odom['orientation'][3]

    tf_msg = TFMessage()
    tf_msg.transforms.append(transform)

    return tf_msg

def create_point_cloud(points: np.ndarray, stamp, frame_id="map"):
    header = Header()
    header.stamp = rospy.Time.from_sec(stamp)
    header.frame_id = frame_id

    # Define fields for x, y, z, and rgb
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]

    cloud_data = points.astype(np.float32)
    data = cloud_data.tobytes()

    # Create PointCloud2 message
    point_cloud = PointCloud2()
    point_cloud.header = header
    point_cloud.height = 1  # Unordered point cloud
    point_cloud.width = len(points)
    point_cloud.fields = fields
    point_cloud.is_bigendian = False
    point_cloud.point_step = 12  # 3 fields (x, y, z) * 4 bytes
    point_cloud.row_step = point_cloud.point_step * len(points)
    point_cloud.is_dense = True
    point_cloud.data = data

    return point_cloud

# @profile
def create_colored_point_cloud(points: np.ndarray, colors: np.ndarray, stamp, frame_id="map"):
    """
    Create a PointCloud2 message with colors.

    :param points: List of points, where each point is [x, y, z].
    :param colors: List of colors, where each color is [r, g, b].
    :param frame_id: Frame ID for the point cloud.
    :return: A PointCloud2 message.
    """
    header = Header()
    header.stamp = rospy.Time.from_sec(stamp)
    header.frame_id = frame_id

    # Define fields for x, y, z, and rgb
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.FLOAT32, 1),
    ]

    if colors.max() <= 1:
        colors = colors * 255.0
    rgb_colors = colors.astype(np.uint32)
    rgb_colors = (rgb_colors[:, 0].astype(np.uint32) << 16) | \
                (rgb_colors[:, 1].astype(np.uint32) << 8) | \
                rgb_colors[:, 2].astype(np.uint32)
    rgb_colors = rgb_colors.view(np.float32)
    cloud_data = np.concatenate((points, rgb_colors[:, None]), axis=1).astype(np.float32)

    # # Create a structured array with point and color data
    # cloud_data = []
    # for point, color in zip(points, colors):
    #     x, y, z = point
    #     r, g, b = color
    #     # Pack RGB into a single float using struct
    #     rgb = struct.unpack("f", struct.pack("I", (int(r) << 16) | (int(g) << 8) | int(b)))[0]
    #     cloud_data.append([x, y, z, rgb])

    # cloud_data = np.array(cloud_data, dtype=np.float32)

    # Convert to byte array
    data = cloud_data.tobytes()

    # Create PointCloud2 message
    point_cloud = PointCloud2()
    point_cloud.header = header
    point_cloud.height = 1  # Unordered point cloud
    point_cloud.width = len(points)
    point_cloud.fields = fields
    point_cloud.is_bigendian = False
    point_cloud.point_step = 16  # 4 fields (x, y, z, rgb) * 4 bytes
    point_cloud.row_step = point_cloud.point_step * len(points)
    point_cloud.is_dense = True
    point_cloud.data = data

    return point_cloud

def create_wireframe_marker_from_corners(corners, ns, box_id, color, stamp, frame_id="map"):
    marker = Marker()
    
    # Set the frame ID and marker type
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.from_sec(stamp)
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.id = int(box_id)
    marker.ns = ns

    # Set the color (red wireframe in this case)
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3] if len(color) == 4 else 0.8

    # Set the scale of the lines (thickness)
    marker.scale.x = 0.05  # Line thickness
    
    # Set the pose
    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Define the 12 edges of the box by connecting the corners
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom square
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top square
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Add the points defining the edges to the marker
    for edge in edges:
        p1 = Point(*corners[edge[0]])
        p2 = Point(*corners[edge[1]])
        marker.points.append(p1)
        marker.points.append(p2)

    return marker

def get_3d_box(center, box_size, heading_angle):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (length, wide, height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def rotz(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  s,  0],
                         [-s,  c,  0],
                         [0,  0,  1]])
    
    R = rotz(heading_angle)
    l,w,h = box_size
    x_corners = [-l/2,l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2]
    y_corners = [-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2,w/2]
    z_corners = [-h/2,-h/2,-h/2,-h/2,h/2,h/2,h/2,h/2]
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d.tolist()

def create_wireframe_marker(center, extent, yaw, ns, box_id, color, stamp, frame_id="map"):
    # Compute the corners of the bounding box
    corners = get_3d_box(center, extent, yaw)
    return create_wireframe_marker_from_corners(corners, ns, box_id, color=color, stamp=stamp, frame_id=frame_id)

def create_point_marker(center, box_id=0, frame_id="world", color=(1.0, 0.0, 0.0, 0.8)):
    marker = Marker()
    
    # Set the frame ID and marker type
    marker.header.frame_id = frame_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.id = box_id
    marker.ns = "points"

    # Set the color (red wireframe in this case)
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    # Set the scale of the points (size)
    marker.scale.x = 0.5  # Point size
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    
    # Set the pose
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1

    # Add the point to the marker
    p = Point(*center)
    marker.points.append(p)

    return marker

def create_text_marker(center, marker_id, text, color, text_height, frame_id="map"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "text"
    marker.id = int(marker_id)  # Unique ID
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    
    # Set position
    marker.pose.position.x = center[0]
    marker.pose.position.y = center[1]
    marker.pose.position.z = center[2]
    
    # Text properties
    marker.scale.z = text_height  # Text height

    marker.color.r = color[0]
    marker.color.b = color[1]
    marker.color.g = color[2]
    marker.color.a = color[3]  # Fully visible
    marker.text = text

    return marker

def create_box_marker(center, extent, yaw, ns, box_id, color, stamp, frame_id="map"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = int(box_id)
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    # Set position (center of the cube)
    marker.pose.position.x = center[0]
    marker.pose.position.y = center[1]
    marker.pose.position.z = center[2]

    # Convert yaw to quaternion (assuming rotation about Z-axis)
    quat = R.from_euler('xyz', [0, 0, yaw]).as_quat()
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]

    # Set scale (dimensions of the cube)
    marker.scale.x = extent[0]  # Width
    marker.scale.y = extent[1]  # Height
    marker.scale.z = extent[2]  # Depth

    # Set color (RGBA)
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]  # Semi-transparent

    return marker