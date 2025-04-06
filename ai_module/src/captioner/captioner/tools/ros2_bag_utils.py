import os
import rclpy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, TransformStamped
from tf2_msgs.msg import TFMessage

import numpy as np
from scipy.spatial.transform import Rotation as R
import struct

import rosbag2_py
from builtin_interfaces.msg import Time as TimeMsg

def init_bag_writer(output_path: str):
    # Create a StorageOptions object to define where and how the bag is stored
    storage_options = rosbag2_py.StorageOptions(
        uri=output_path,  # Path to store the bag file
        storage_id='sqlite3'  # Use SQLite storage backend
    )

    # Create a ConverterOptions object to define serialization format
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',  # Default serialization format
        output_serialization_format='cdr'
    )

    # Create a writer instance
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    return writer

def add_topic(writer, topic_name, message_type):
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name=topic_name,
            type=message_type,
            serialization_format='cdr'
        )
    )

def create_odom_msg(odom, seconds: int, nanoseconds: int, frame_id="map"):
    ros_odom = Odometry()
    ros_odom.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    ros_odom.header.frame_id = frame_id
    ros_odom.pose.pose.position.x = odom['position'][0]
    ros_odom.pose.pose.position.y = odom['position'][1]
    ros_odom.pose.pose.position.z = odom['position'][2]
    ros_odom.pose.pose.orientation.x = odom['orientation'][0]
    ros_odom.pose.pose.orientation.y = odom['orientation'][1]
    ros_odom.pose.pose.orientation.z = odom['orientation'][2]
    ros_odom.pose.pose.orientation.w = odom['orientation'][3]
    return ros_odom

def create_tf_msg(odom, seconds: int, nanoseconds: int, frame_id="map", child_frame_id="sensor"):
    transform = TransformStamped()
    transform.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    transform.header.frame_id = frame_id
    transform.child_frame_id = child_frame_id
    transform.transform.translation.x = odom['position'][0]
    transform.transform.translation.y = odom['position'][1]
    transform.transform.translation.z = odom['position'][2]
    transform.transform.rotation.x = odom['orientation'][0]
    transform.transform.rotation.y = odom['orientation'][1]
    transform.transform.rotation.z = odom['orientation'][2]
    transform.transform.rotation.w = odom['orientation'][3]

    tf_msg = TFMessage()
    tf_msg.transforms.append(transform)
    return tf_msg

def create_point_cloud(points: np.array, seconds: int, nanoseconds: int, frame_id="map"):
    header = Header()
    header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    header.frame_id = frame_id

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    cloud_data = points.astype(np.float32)
    data = cloud_data.tobytes()

    point_cloud = PointCloud2()
    point_cloud.header = header
    point_cloud.height = 1
    point_cloud.width = len(points)
    point_cloud.fields = fields
    point_cloud.is_bigendian = False
    point_cloud.point_step = 12
    point_cloud.row_step = point_cloud.point_step * len(points)
    point_cloud.is_dense = True
    point_cloud.data = data

    return point_cloud

def create_colored_point_cloud(points: np.array, colors: np.array, seconds: int, nanoseconds: int, frame_id="map"):
    header = Header()
    header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
    header.frame_id = frame_id

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    if colors.max() <= 1:
        colors = colors * 255
    rgb_colors = colors.astype(np.uint32)
    rgb_colors = (rgb_colors[:, 0].astype(np.uint32) << 16) | \
                 (rgb_colors[:, 1].astype(np.uint32) << 8) | \
                 (rgb_colors[:, 2].astype(np.uint32))
    rgb_colors = rgb_colors.view(np.float32)
    cloud_data = np.concatenate((points, rgb_colors[:, None]), axis=1).astype(np.float32)

    data = cloud_data.tobytes()

    point_cloud = PointCloud2()
    point_cloud.header = header
    point_cloud.height = 1
    point_cloud.width = len(points)
    point_cloud.fields = fields
    point_cloud.is_bigendian = False
    point_cloud.point_step = 16
    point_cloud.row_step = point_cloud.point_step * len(points)
    point_cloud.is_dense = True
    point_cloud.data = data

    return point_cloud

def create_wireframe_marker_from_corners(corners, ns: str, box_id: str, color: list|np.ndarray, seconds: int, nanoseconds: int, frame_id="map"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.id = int(box_id)
    marker.ns = ns
    marker.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3] if len(color) == 4 else 0.8

    marker.scale.x = 0.05

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        p1 = Point(x=corners[edge[0]][0], y=corners[edge[0]][1], z=corners[edge[0]][2])
        p2 = Point(x=corners[edge[1]][0], y=corners[edge[1]][1], z=corners[edge[1]][2])
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

def create_wireframe_marker(center, extent, yaw, ns, box_id, color, seconds: int, nanoseconds: int, frame_id="map"):
    # Compute the corners of the bounding box
    corners = get_3d_box(center, extent, yaw)
    return create_wireframe_marker_from_corners(corners, ns, box_id, color, seconds, nanoseconds, frame_id)

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

def create_text_marker(center, marker_id, text, color, text_height,  seconds: int, nanoseconds: int, frame_id="map"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
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

def create_box_marker(center, extent, yaw, ns, box_id, color, seconds: int, nanoseconds:int, frame_id="map"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
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