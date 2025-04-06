import numpy as np
import os
import spacy
import json
import open3d as o3d
import matplotlib.pyplot as plt
import re
import argparse
import sys
from pathlib import Path

import rospy

from geometry_msgs.msg import Pose2D, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.point_cloud2 import read_points_list
from visualization_msgs.msg import MarkerArray, Marker
from time import sleep

from language_planner.prompts import get_prompt
from language_planner.llm_backend.llm_query_langchain import NavQueryRunMode, ObjectQueryType, LanguageModel, SystemMode
from language_planner.language_planner_backend import LanguagePlannerBackend

from captioner.tools import ros1_bag_utils


class LanguagePlanner:

    def __init__(
            self,
            environment_name: str,
            model='mistral',
            run_mode='use_tools',
            object_query_type='llm'
            ):
        
        # Parameters

        self.environment_name = environment_name
        self.log_folder = Path(__file__).resolve().parents[1] / "logs" / "llm_outputs"
        if not os.path.exists(str(self.log_folder)):
            os.makedirs(self.log_folder)

        self.run_mode = NavQueryRunMode(run_mode)
        self.object_query_type = ObjectQueryType(object_query_type)
        self.model = LanguageModel(model)
        self.system_mode = SystemMode.LIVE_NAVIGATION

        # ROS

        rospy.init_node('language_planner_node')

        self.waypoint_pub = rospy.Publisher('/way_point_with_heading', Pose2D, queue_size=5)
        self.object_query_pub = rospy.Publisher('/object_query', String, queue_size=5)
        self.object_marker_pub = rospy.Publisher('/selected_object_markers', Marker, queue_size=5)

        self.pose_sub = rospy.Subscriber('/state_estimation', Odometry, self.handle_pose, queue_size=1)
        self.map_sub = rospy.Subscriber('/global_cloud', PointCloud2, self.handle_map, queue_size=1)
        self.freespace_sub = rospy.Subscriber('/traversable_area', PointCloud2, self.handle_freespace, queue_size=1)

        self.caption_sub = rospy.Subscriber('/queried_captions', String, self.handle_captions, queue_size=1)
        self.planner_query_sub = rospy.Subscriber('/language_planner_query', String, self.handle_language_query, queue_size=1)

        # Variable Initialization

        self.cur_pos = np.array([0., 0., 0.])
        self.cur_vel = np.array([0., 0., 0.])
        self.map_pcl: np.ndarray = None
        self.freespace_pcl: np.ndarray = None

        self.target_waypoints = []
        self.target_ids = []


        # Other

        self.language_planner_backend = LanguagePlannerBackend(
            self.model,
            self.run_mode,
            self.system_mode,
            self.log_info
            )

        self.object_dict = {}
        self.obj_query_response_echo = []

    
    
    def handle_pose(self, pose: Odometry):
        self.cur_pos[0] = pose.pose.pose.position.x
        self.cur_pos[1] = pose.pose.pose.position.y
        self.cur_pos[2] = pose.pose.pose.position.z

    
    def handle_map(self, msg: PointCloud2):
        self.map_pcl = np.array(read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True))


    def handle_freespace(self, msg: PointCloud2):
        self.freespace_pcl = np.array(read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True))
    

    def handle_captions(self, msg: String):
        response_dict = json.loads(msg.data)
        self.object_dict = response_dict["response"]
        self.obj_query_response_echo = response_dict["query"]


    def publish_waypoint(self, waypoint: np.ndarray):
        waypoint_msg = Pose2D()
        waypoint_msg.x = float(waypoint[0]) + 0.0001
        waypoint_msg.y = float(waypoint[1]) + 0.0001
        waypoint_msg.theta = 0.
        self.waypoint_pub.publish(waypoint_msg)
    

    def publish_object_markers(self, object_dict, target_ids):

        timestamp = rospy.Time.now().to_sec()

        for id_sublist in target_ids:
            for i in id_sublist:
                bbox_marker = ros1_bag_utils.create_box_marker(
                    center=object_dict[i]["centroid"],
                    extent=object_dict[i]["dimensions"],
                    yaw=object_dict[i]["heading"],
                    ns="bboxes",
                    box_id=i,
                    color=[0., 0., 1., 0.6],
                    stamp=timestamp
                )

                text_height = 0.35
                text_position = [
                    object_dict[i]["centroid"][0],
                    object_dict[i]["centroid"][1],
                    object_dict[i]["centroid"][2] + object_dict[i]["dimensions"][2] + text_height/2 + 0.05,
                ]

                text_marker = ros1_bag_utils.create_text_marker(
                    center=text_position,
                    marker_id=i,
                    text=object_dict[i]["name"],
                    color=[1., 1., 1., 1.],
                    text_height=text_height
                )

                self.object_marker_pub.publish(bbox_marker)
                self.object_marker_pub.publish(text_marker)


    def recolor_object_markers(self, object_dict, target_ids):
        timestamp = rospy.Time.now().to_sec()

        for i in target_ids:
            bbox_marker = ros1_bag_utils.create_box_marker(
                center=object_dict[i]["centroid"],
                extent=object_dict[i]["dimensions"],
                yaw=object_dict[i]["heading"],
                ns="bboxes",
                box_id=i,
                color=[0., 1., 0., 0.6],
                stamp=timestamp
            )

        self.object_marker_pub.publish(bbox_marker)


    def clear_object_markers(self):

        ns_list = ["text", "bboxes"]
        for id_sublist in self.target_ids:
            for idx in id_sublist:
                for ns in ns_list:
                    marker = Marker()
                    marker.header.frame_id = "map" 
                    marker.header.stamp = rospy.Time.now()
                    marker.id = idx
                    marker.ns = ns
                    marker.type = Marker.CUBE
                    marker.action = Marker.DELETE
                    
                    self.object_marker_pub.publish(marker)

        self.target_ids = []
        self.target_waypoints = []


    def query_objects(self, object_nouns):
        object_json = json.dumps(object_nouns)
        object_msg = String(data=object_json)
        self.object_query_pub.publish(object_msg)


    def log_info(self, msg: str):
        rospy.loginfo(msg)


    def publish_waypoints(self, ids, waypoints, object_dict):
        if not len(waypoints):
            return

        freespace_points = self.freespace_pcl[:, :2]

        for i, (id_sublist, waypoint) in enumerate(zip(ids, waypoints)):
            closest_point_idx = np.linalg.norm(freespace_points - waypoint, axis=-1).argmin()
            closest_point = freespace_points[closest_point_idx]

            self.publish_waypoint(closest_point)

            self.log_info(f"Navigating... ({i+1}/{len(waypoints)})")
            while not rospy.is_shutdown():
                dist_from_waypoint = np.linalg.norm(self.cur_pos[:2] - closest_point)
                if dist_from_waypoint < 1.3:
                    self.recolor_object_markers(object_dict, id_sublist)
                    break
        
        self.log_info(f'Finished navigation.')
    

    def log_llm_output(self, input_statement, output_code):
        timestamp = rospy.Time.now().secs

        with open(str(self.log_folder / f'{timestamp}.txt'), 'w') as f:
            f.write("===============\n")
            f.write("INPUT STATEMENT\n")
            f.write("===============\n")
            f.write(input_statement + '\n')
            f.write("===============\n")
            f.write("OUTPUT CODE\n")
            f.write("===============\n")
            f.write(output_code)


    def handle_language_query(self, msg: String):

        self.log_info(f'Query received: {msg.data}')

        if self.freespace_pcl is None:
            return

        input_statement = msg.data

        self.clear_object_markers()

        obj_query_list = self.language_planner_backend.get_object_references(input_statement)
        sleep(1)

        self.log_info(f'Parsed objects: {obj_query_list}')

        self.object_dict = {}
        self.obj_query_response_echo = None

        if self.object_query_type == ObjectQueryType.CLIP_BASED:
            self.query_objects(obj_query_list)
            self.log_info("Queried objects")
            while not rospy.is_shutdown() and self.obj_query_response_echo != obj_query_list:
                pass
            object_dict = self.object_dict
        elif self.object_query_type == ObjectQueryType.LLM_BASED:
            self.query_objects(["GET_ALL"])
            self.log_info("Queried objects")
            while not rospy.is_shutdown() and self.obj_query_response_echo != ["GET_ALL"]:
                pass
            object_dict = self.object_dict

            filtered_obj_ids = self.language_planner_backend.get_retrieved_objects(obj_query_list, object_dict)
            if '-1' in self.object_dict:
                filtered_obj_ids.append('-1')

            object_dict = {int(obj_id):object_dict[obj_id] for obj_id in filtered_obj_ids}

            
        self.log_info(f'{object_dict}')

        self.target_waypoints, self.target_ids, filtered_objects_out, output_code = self.language_planner_backend.generate_plan(
            self.environment_name,
            input_statement,
            self.map_pcl,
            self.freespace_pcl,
            object_dict,
            self.cur_pos
        )

        self.log_info(output_code)
        self.log_llm_output(input_statement, output_code)

        self.publish_object_markers(object_dict, self.target_ids)
        self.publish_waypoints(self.target_ids, self.target_waypoints, object_dict)

        
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--environment_name', default='a room')
    parser.add_argument('--model', default='mistral')
    parser.add_argument('--run_mode', default='use_tools')
    parser.add_argument('--object_query_type', default='llm')

    argparse_args, other_args = parser.parse_known_args()

    handler = LanguagePlanner(**vars(argparse_args))

    rospy.spin()


if __name__ == "__main__":
    main()