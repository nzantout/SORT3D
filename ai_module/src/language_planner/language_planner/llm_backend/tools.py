from itertools import permutations, product
from langchain_core.tools import tool, StructuredTool, InjectedToolArg
from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated
from pydantic import BaseModel, Field
import numpy as np
from pprint import pprint
import traceback
from textwrap import dedent
from spatial_relations.bbox_utils import *
from llm_backend.enums import SystemMode


class AgentToolbox:
    def __init__(
            self,
            system_mode = SystemMode.LIVE_NAVIGATION):
        
        self.tools = [
            StructuredTool(
                name="find_between",
                func=self.find_between, 
                args_schema=self.FindBetweenSchema, 
                description=dedent(self.FindBetweenSchema.__doc__)),
            StructuredTool(
                name="find_near",
                func=self.find_near, 
                args_schema=self.FindNearSchema, 
                description=dedent(self.FindNearSchema.__doc__)),
            StructuredTool(
                name="find_left",
                func=self.find_left, 
                args_schema=self.FindLeftSchema, 
                description=dedent(self.FindLeftSchema.__doc__)),
            StructuredTool(
                name="find_right",
                func=self.find_right, 
                args_schema=self.FindRightSchema, 
                description=dedent(self.FindRightSchema.__doc__)),
            StructuredTool(
                name="order_left_to_right",
                func=self.order_left_to_right, 
                args_schema=self.OrderLeftToRightSchema, 
                description=dedent(self.OrderLeftToRightSchema.__doc__)),
            StructuredTool(
                name="find_above",
                func=self.find_above, 
                args_schema=self.FindAboveSchema, 
                description=dedent(self.FindAboveSchema.__doc__)),
            StructuredTool(
                name="find_below",
                func=self.find_below, 
                args_schema=self.FindBelowSchema, 
                description=dedent(self.FindBelowSchema.__doc__)),
            StructuredTool(
                name="order_bottom_to_top",
                func=self.order_bottom_to_top, 
                args_schema=self.OrderBottomToTopSchema, 
                description=dedent(self.OrderBottomToTopSchema.__doc__)),
            StructuredTool(
                name="order_smallest_to_largest",
                func=self.order_smallest_to_largest, 
                args_schema=self.OrderSmallestToLargestSchema, 
                description=dedent(self.OrderSmallestToLargestSchema.__doc__)),
            StructuredTool(
                name="find_objects_near_room_corner",
                func=self.find_objects_near_room_corner, 
                args_schema=self.FindObjectsNearRoomCornerSchema, 
                description=dedent(self.FindObjectsNearRoomCornerSchema.__doc__)),
            StructuredTool(
                name="notepad",
                func=self.notepad, 
                args_schema=self.NotepadSchema, 
                description=dedent(self.NotepadSchema.__doc__))
            ]
        
        if system_mode == SystemMode.LIVE_NAVIGATION:
            self.tools.append(
                StructuredTool(
                    name="command_robot",
                    func=self.command_robot, 
                    args_schema=self.CommandRobotSchema, 
                    description=dedent(self.CommandRobotSchema.__doc__))
                )
        else:
            self.tools.append(
                StructuredTool(
                    name="pick_object",
                    func=self.pick_object, 
                    args_schema=self.PickObjectSchema, 
                    description=dedent(self.PickObjectSchema.__doc__))
            )
            
        self.cmd = []
        self.notes = []

        self.object_dict = {}
        self.object_id_map = {}
        self.inv_object_id_map = {}

        self.freespace: np.ndarray = None
        self.pcl: np.ndarray = None

        self.distance_thres = 1
        self.overlap_thres = 0.3
        self.symmetry_thres = 0.5
        self.between_iom = 0.05
        self.vertical_iom = 0.5
        self.near_thres = 1
        self.on_thres = 0.01
        self.under_thres = 0.01


    def set_object_dict(self, object_dict: dict):
        self.object_dict = object_dict
        self.object_id_map = {relative_id: absolute_id for relative_id, absolute_id in enumerate(object_dict.keys()) if absolute_id != -1} 
        self.object_id_map[-1] = -1
        self.inv_object_id_map = {absolute_id: relative_id for relative_id, absolute_id in enumerate(object_dict.keys()) if absolute_id != -1}
        self.inv_object_id_map[-1] = -1


    def clear(self):
        self.cmd = []
        self.notes = []


    class NotepadSchema(BaseModel):
        """Use this tool as a notepad to write down your thoughts step-by-step"""

        thoughts: str = Field(..., description="The step-by-step reasoning / thoughts you have")

    def notepad(self, thoughts: str) -> str:
        self.notes.append(thoughts)
        print(thoughts)
        return "Noted and please proceed with the next step!"
    

    class CommandRobotSchema(BaseModel):
        """Command the robot to move sequentially based on the list of commands, each command should be a tuple of (str, tuple), where the first element, str, is either "go_near" or "go_between" and the second element, tuple, is the object index(es) associated with the command. The object index(es) should be of length 1 for "go_near" and length 2 for "go_between"
        """

        list_of_commands: list[tuple] = Field(..., description="A list of commands to execute sequentially, for example, if you want to command the robot to go near object 1 and then go between object 2 and object 3, the list should be [('go_near', (1,)), ('go_between', (2, 3))]")

    def command_robot(self, list_of_commands: list[tuple]) -> str:
        # input validation
        for cmd in list_of_commands:
            if cmd[0] not in {"go_near", "go_between"}:
                return "Please provide a valid command, it should be either 'go_near' or 'go_between'"
            if cmd[0] == "go_near" and len(cmd[1]) != 1:
                return "For 'go_near' command, please provide only one object index"
            if cmd[0] == "go_between" and len(cmd[1]) != 2:
                return "For 'go_between' command, please provide two object indices"

        self.cmd = list_of_commands
        return "The robot has executed the commands successfully, now say 'done'!"


    class PickObjectSchema(BaseModel):
        """Pick out the ID of the object referred to by the referring statement.
        """

        object_id: int = Field(..., description="The ID of the object referred to by the referring statement.")

    def pick_object(self, object_id: int) -> str:

        self.cmd = [('go_near', (object_id,))]
        return "The object has been picked successfully, now say 'done'!"


    class FindBetweenSchema(BaseModel):
        """
        Find the ID of the object with name 'target_name' between two objects with names 'first_anchor_name' and 'second_anchor_name'. If multiple objects match this criterion, returns the list of all the objects that do. You may find it useful when you want to know, for example, which towel is between two other sofas, or which chair is between two other chairs.
        """

        target_name: str = Field(..., description="The name of the target object.")
        first_anchor_name: str = Field(..., description="The name of the first anchor object.")
        second_anchor_name: str = Field(..., description="The name of the second anchor object.")

    def find_between(
        self, 
        target_name: str, 
        first_anchor_name: str, 
        second_anchor_name: str) -> list[int]:

        try:

            target_objs = [obj_id for obj_id, obj in self.object_dict.items() if target_name in obj["name"]]
            first_anchor_objs = [obj_id for obj_id, obj in self.object_dict.items() if first_anchor_name in obj["name"]]
            second_anchor_objs = [obj_id for obj_id, obj in self.object_dict.items() if second_anchor_name in obj["name"]]

            if not len(target_objs):
                return f'{target_name} is not among the names in the list of objects. Please use a synonym that is found exactly in the list of objects.'
            if not len(first_anchor_objs):
                return f'{first_anchor_name} is not among the names in the list of objects. Please use a synonym that is found exactly in the list of objects.'
            if not len(second_anchor_objs):
                return f'{second_anchor_name} is not among the names in the list of objects. Please use a synonym that is found exactly in the list of objects.'

            between_objs = []
            for target_obj_id, first_anchor_obj_id, second_anchor_obj_id in product(target_objs, first_anchor_objs, second_anchor_objs):

                if target_obj_id == first_anchor_obj_id \
                    or first_anchor_obj_id == second_anchor_obj_id \
                        or target_obj_id == second_anchor_obj_id:
                    continue

                center1 = np.array(self.object_dict[target_obj_id]["centroid"])
                center2 = np.array(self.object_dict[first_anchor_obj_id]["centroid"])
                center3 = np.array(self.object_dict[second_anchor_obj_id]["centroid"])

                bbox1 = np.array(get_bbox_coords_heading_xyzlwh(
                    center1,
                    self.object_dict[target_obj_id]["dimensions"],
                    self.object_dict[target_obj_id]["heading"]))
                bbox2 = np.array(get_bbox_coords_heading_xyzlwh(
                    center2,
                    self.object_dict[first_anchor_obj_id]["dimensions"],
                    self.object_dict[first_anchor_obj_id]["heading"]))
                bbox3 = np.array(get_bbox_coords_heading_xyzlwh(
                    center3,
                    self.object_dict[second_anchor_obj_id]["dimensions"],
                    self.object_dict[second_anchor_obj_id]["heading"]))

                r = (center3 - center2)[:2]
                r /= np.linalg.norm(r, axis=-1, keepdims=True)
                R = np.zeros((3, 3), dtype=np.float64)
                R[0, 0] = r[0]
                R[1, 0] = r[1]
                R[0, 1] = -r[1]
                R[1, 1] = r[0]
                R[2, 2] = 1
                center1_rot = R.T @ center1
                center2_rot = R.T @ center2
                center3_rot = R.T @ center3

                if (center1_rot[0] < center2_rot[0]) or (center1_rot[0] > center3_rot[0]):
                    continue

                # bboxes: 8 x 3 x 1
                # R: 1 x 3 x 3

                bbox1_rot = (R[None, :, :].transpose(0, 2, 1) @ bbox1[:, :, None])[..., 0]
                bbox2_rot = (R[None, :, :].transpose(0, 2, 1) @ bbox2[:, :, None])[..., 0]
                bbox3_rot = (R[None, :, :].transpose(0, 2, 1) @ bbox3[:, :, None])[..., 0]

                # Make sure bboxes are not overlapping by checking single dimensional iom

                max_xy1 = bbox1_rot[..., 0].max(axis=-1)
                min_xy1 = bbox1_rot[..., 0].min(axis=-1)
                max_xy2 = bbox2_rot[..., 0].max(axis=-1)
                min_xy2 = bbox2_rot[..., 0].min(axis=-1)
                max_xy3 = bbox3_rot[..., 0].max(axis=-1)
                min_xy3 = bbox3_rot[..., 0].min(axis=-1)

                iom1_1d_xy = (max_xy1 - min_xy3) / np.minimum(max_xy1 - min_xy1, max_xy3 - min_xy3)
                iom2_1d_xy = (max_xy2 - min_xy1) / np.minimum(max_xy1 - min_xy1, max_xy2 - min_xy2)

                dist_sums_xy = -iom1_1d_xy - iom2_1d_xy

                filter_xy = (iom1_1d_xy < self.overlap_thres) \
                    & (iom2_1d_xy < self.overlap_thres) \
                    & (np.abs(iom1_1d_xy - iom2_1d_xy) < self.symmetry_thres) \
                    & (-iom1_1d_xy < self.distance_thres) \
                    & (-iom2_1d_xy < self.distance_thres) 
                
                if not filter_xy:
                    continue

                bbox1, bbox2 = get_2D_bboxes([1, 2], bbox1_rot, bbox2_rot)
                iom1_xy = calculate_iom_single(bbox1, bbox2)
                # IOU to second anchor
                bbox1, bbox2 = get_2D_bboxes([1, 2], bbox1_rot, bbox3_rot)
                iom2_xy = calculate_iom_single(bbox1, bbox2)

                filt = (iom1_xy > self.between_iom) & (iom2_xy > self.between_iom)

                if filt:
                    between_objs.append(target_obj_id)
                    continue

                # Check z intersection

                between_z = (center1_rot[2] > center2_rot[2]) & (center1_rot[2] < center3_rot[2])
                if not between_z:
                    continue

                max_z1 = bbox1_rot[..., 2].max(axis=-1)
                min_z1 = bbox1_rot[..., 2].min(axis=-1)
                max_z2 = bbox2_rot[..., 2].max(axis=-1)
                min_z2 = bbox2_rot[..., 2].min(axis=-1)
                max_z3 = bbox3_rot[..., 2].max(axis=-1)
                min_z3 = bbox3_rot[..., 2].min(axis=-1)

                iom1_1d_z = (max_z1 - min_z3) / np.minimum(max_z1 - min_z1, max_z3 - min_z3)
                iom2_1d_z = (max_z2 - min_z1) / np.minimum(max_z1 - min_z1, max_z2 - min_z2)

                dist_sums_z = -iom1_1d_z - iom2_1d_z

                filter_z = (iom1_1d_z < self.overlap_thres) & (iom2_1d_z < self.overlap_thres)
                if not filter_z:
                    continue

                # print(bbox1_rot_between.shape)

                iom1_z = calculate_iom_poly(bbox1_rot, bbox2_rot)
                iom2_z = calculate_iom_poly(bbox1_rot, bbox3_rot)

                filt = (iom1_z > self.between_iom) & (iom2_z > self.between_iom)
                if not filt:
                    continue

                between_objs.append(target_obj_id)

            # Mapping back to LLM IDs
            between_objs = list(set(between_objs))
            between_objs = [self.inv_object_id_map[obj] for obj in between_objs]
            if len(between_objs) == 0:
                return f"No {target_name} found between {first_anchor_name} and {second_anchor_name}"
            elif len(between_objs) == 1:
                return f"One {target_name} found between {first_anchor_name} and {second_anchor_name}, the id is {between_objs[0]}"
            else:
                return f"Multiple {target_name} found between {first_anchor_name} and {second_anchor_name}, the ids are {between_objs}"
        
        except Exception as e:
            print(traceback.format_exc())
    

    class FindNearOldSchema(BaseModel):
        """Find the ID of the object with name 'target_name' near an object with name 'anchor_name'. If multiple objects match this criterion, returns the list of all the objects that do.

        Example:
        Corresponding intermediate statement: 'the night stand close to the bed'.
        Object List: [
        [0, 'nightstand'],
        [1, 'nightstand'],
        [2, 'bed'],
        ]
        Python Script:
        object_id = find_near('nightstand', 'bed')
        # Returns [0], meaning nightstand 0 is the nightstand that is near the bed.

        Args:
            target_name (str): The name of the target object.
            anchor_name (str): The name of the anchor object.
        
        Returns:
            list[int]: A list of object IDs that match the criterion.
        """

        target_name: str = Field(..., description="The name of the target object.")
        anchor_name: str = Field(..., description="The name of the anchor object.")

    def find_near_old(self, target_name: str, anchor_name: str) -> list[int]:

        try:

            target_objs = [obj_id for obj_id, obj in self.object_dict.items() if target_name in obj["name"]]
            anchor_objs = [obj_id for obj_id, obj in self.object_dict.items() if anchor_name in obj["name"]]

            near_objs = []
            for target_obj_id, anchor_obj_id in product(target_objs, anchor_objs):

                if target_obj_id == anchor_obj_id:
                    continue

                target_center = np.array(self.object_dict[target_obj_id]["centroid"])
                target_coords = np.array(get_bbox_coords_heading_xyzlwh(
                        target_center,
                        self.object_dict[target_obj_id]["dimensions"],
                        self.object_dict[target_obj_id]["heading"]))

                anchor_center = np.array(self.object_dict[anchor_obj_id]["centroid"])
                anchor_coords = np.array(get_bbox_coords_heading_xyzlwh(
                        anchor_center,
                        self.object_dict[anchor_obj_id]["dimensions"],
                        self.object_dict[anchor_obj_id]["heading"]))
            

                center_dist = np.linalg.norm(np.array(anchor_center)[:2] - np.array(target_center)[:2])

                # check if centers close
                if (center_dist < self.near_thres):
                    near_objs.append(target_obj_id)
                else:
                    dists = []
                    # get distance between each pair of box coordinates
                    for p1 in anchor_coords:
                        d = [np.linalg.norm(p1[:2]-p2[:2]) for p2 in target_coords]
                        dists += d

                    #print("dists", dists)
                    # object near if at least two points close to each other
                    if np.any(np.array(dists) < self.near_thres):
                        near_objs.append(target_obj_id)


            # Mapping back to LLM IDs
            near_objs = [self.inv_object_id_map[obj] for obj in near_objs]            

            return near_objs
        
        except Exception as e:
            print(traceback.format_exc())


    class FindLeftSchema(BaseModel):
        """Find the IDs of the objects with name 'target_name' to the left of the object with ID 'anchor_id'. If multiple objects match this criterion, returns the list of all the objects that do. You may want to use this tool when you want to know, for example, which chair is to the left of the chair with ID 0, or which door is to the left of the night stand with ID 1.
        """

        target_name: str = Field(..., description="The name of the target object.")
        anchor_id: int = Field(..., description="The ID of the anchor object.")

    def find_left(self, target_name: str, anchor_id: int) -> list[int]:

        target_objs = [obj_id for obj_id, obj in self.object_dict.items() if target_name in obj["name"]]
        anchor_obj_id = self.object_id_map[anchor_id]

        if not len(target_objs):
            return f'{target_name} is not among the names in the list of objects. Please use a synonym that is found exactly in the list of objects.'

        left_objs = []
        for target_obj_id in target_objs:


            target_center = np.array(self.object_dict[target_obj_id]["centroid"])
            anchor_center = np.array(self.object_dict[anchor_obj_id]["centroid"])

            focal_point = (anchor_center + target_center) / 2
            anchor_point = self.freespace[np.argmin(np.linalg.norm(self.freespace - focal_point, axis=-1))]
            
            target_vec = (target_center - anchor_point)[:2]
            target_vec /= np.linalg.norm(target_vec)
            anchor_vec = (anchor_center - anchor_point)[:2]
            anchor_vec /= np.linalg.norm(anchor_vec)

            # < (anchor, target) > 0
            if np.cross(anchor_vec, target_vec) > 0:
                left_objs.append(target_obj_id)

        # Mapping back to LLM IDs
        left_objs = [self.inv_object_id_map[obj] for obj in left_objs]
        if len(left_objs) == 0:
            return f"No {target_name} found to the left of the object with ID {anchor_id}"
        elif len(left_objs) == 1:
            return f"One {target_name} found to the left of the object with ID {anchor_id}, the id is {left_objs[0]}"
        else:
            return f"Multiple {target_name} found to the left of the object with ID {anchor_id}, the ids are {left_objs}"    


    class FindRightSchema(BaseModel):
        """Find the ID of the object with name 'target_name' to the right of the object with ID 'anchor_id'. If multiple objects match this criterion, returns the list of all the objects that do. You may want to use this tool when you want to know, for example, which chair is to the right of the chair with ID 0, or which night stand is to the right of the bed with ID 1.
        """

        target_name: str = Field(..., description="The name of the target object.")
        anchor_id: int = Field(..., description="The ID of the anchor object.")

    
    def find_right(self, target_name: str, anchor_id: int) -> list[int]:
        
        target_objs = [obj_id for obj_id, obj in self.object_dict.items() if target_name in obj["name"]]
        anchor_obj_id = self.object_id_map[anchor_id]

        if not len(target_objs):
            return f'{target_name} is not among the names in the list of objects. Please use a synonym that is found exactly in the list of objects.'

        right_objs = []
        for target_obj_id in target_objs:


            target_center = np.array(self.object_dict[target_obj_id]["centroid"])
            anchor_center = np.array(self.object_dict[anchor_obj_id]["centroid"])

            focal_point = (anchor_center + target_center) / 2
            anchor_point = self.freespace[np.argmin(np.linalg.norm(self.freespace - focal_point, axis=-1))]
            
            target_vec = (target_center - anchor_point)[:2]
            target_vec /= np.linalg.norm(target_vec)
            anchor_vec = (anchor_center - anchor_point)[:2]
            anchor_vec /= np.linalg.norm(anchor_vec)

            # < (anchor, target) < 0
            if np.cross(anchor_vec, target_vec) < 0:
                right_objs.append(target_obj_id)

        # Mapping back to LLM IDs
        right_objs = [self.inv_object_id_map[obj] for obj in right_objs]  

        if len(right_objs) == 0:
            return f"No {target_name} found to the right of the object with ID {anchor_id}"
        elif len(right_objs) == 1:
            return f"One {target_name} found to the right of the object with ID {anchor_id}, the id is {right_objs[0]}"
        else:
            return f"Multiple {target_name} found to the right of the object with ID {anchor_id}, the ids are {right_objs}"


    class FindInFrontOfSchema(BaseModel):
        """Find the ID of the object with name 'target_name' near an object with name 'anchor_name'. If multiple objects match this criterion, returns the list of all the objects that do. You may want to use this tool when you want to know, for example, which chair is in front of the bed, or which night stand is in front of the bed.
        """

        target_name: str = Field(..., description="The name of the target object.")
        anchor_name: str = Field(..., description="The name of the anchor object.")

    
    def find_in_front_of(self, target_name: str, anchor_name: str) -> list[int]:
        pass


    class FindBehindSchema(BaseModel):
        """Find the ID of any objects with name 'target_name' that is behind an object with name 'anchor_name'. If multiple objects match this criterion, returns the list of all the objects that do.

        Example:
        Corresponding intermediate statement: 'the night stand close to the bed'.
        Object List: [
        [0, 'nightstand'],
        [1, 'nightstand'],
        [2, 'bed'],
        ]
        Python Script:
        object_id = find_near('nightstand', 'bed')
        # Returns [0], meaning nightstand 0 is the nightstand that is near the bed.

        Args:
            target_name (str): The name of the target object.
            anchor_name (str): The name of the anchor object.
        
        Returns:
            list[int]: A list of object IDs that match the criterion.
        """

        target_name: str = Field(..., description="The name of the target object.")
        anchor_name: str = Field(..., description="The name of the anchor object.")

    
    def find_behind(self, target_name: str, anchor_name: str) -> list[int]:
        pass


    class FindAboveSchema(BaseModel):
        """Find all IDs of the object with name 'target_name' above an object with name 'anchor_name'. You may want to use this tool when you want to know, which 'target_name' is above 'anchor_name'. For example, which towel is on the bed, or which cabinet is above the refrigerator.
        """

        target_name: str = Field(..., description="The name of the object that is 'above'.")
        anchor_name: str = Field(..., description="The name of the object that is 'below'.")

    
    def find_above(self, target_name: str, anchor_name: str) -> list[int]:

        target_objs = [obj_id for obj_id, obj in self.object_dict.items() if target_name in obj["name"]]
        anchor_objs = [obj_id for obj_id, obj in self.object_dict.items() if anchor_name in obj["name"]]

        if not len(target_objs):
            return f'{target_name} is not among the names in the list of objects. Please use a synonym that is found exactly in the list of objects.'
        if not len(anchor_objs):
            return f'{anchor_name} is not among the names in the list of objects. Please use a synonym that is found exactly in the list of objects.'

        above_objs = []
        for target_obj_id, anchor_obj_id in product(target_objs, anchor_objs):

            if target_obj_id == anchor_obj_id:
                continue


            target_center = np.array(self.object_dict[target_obj_id]["centroid"])
            target_bbox = np.array(get_bbox_coords_heading_xyzlwh(
                    target_center,
                    self.object_dict[target_obj_id]["dimensions"],
                    self.object_dict[target_obj_id]["heading"]))
            
            anchor_center = np.array(self.object_dict[anchor_obj_id]["centroid"])
            anchor_bbox = np.array(get_bbox_coords_heading_xyzlwh(
                    anchor_center,
                    self.object_dict[anchor_obj_id]["dimensions"],
                    self.object_dict[anchor_obj_id]["heading"]))

            # get lowest/highest points on objects
            max_z_target = max([pt[-1] for pt in target_bbox])
            min_z_target = min([pt[-1] for pt in target_bbox])
            min_z_anchor = min([pt[-1] for pt in anchor_bbox])
            max_z_anchor = max([pt[-1] for pt in anchor_bbox])



            if min_z_target >= min_z_anchor:

                center_dist = np.linalg.norm(np.array(anchor_center)[:2] - np.array(target_center)[:2])
                iom = calculate_iom_poly(target_bbox, anchor_bbox)

                # check if centers close
                if iom > self.vertical_iom:
                    above_objs.append(target_obj_id)
                else:
                    dists = []
                    # get distance between each pair of box coordinates
                    for p1 in anchor_bbox:
                        d = [np.linalg.norm(p1[:2]-p2[:2]) for p2 in target_bbox]
                        dists += d

                    # object near if at least two points close to each other
                    if np.any(np.array(dists) < self.near_thres):
                        above_objs.append(target_obj_id)

        # Mapping back to LLM IDs
        above_objs = [self.inv_object_id_map[obj] for obj in above_objs] 
        if len(above_objs) == 0:
            return f"No {target_name} found above {anchor_name}"
        elif len(above_objs) == 1:
            return f"Only one {target_name} found above {anchor_name}, the id is {above_objs[0]}"
        else:
            return f"Multiple {target_name} found above {anchor_name}, the ids are {above_objs}"


    class FindBelowSchema(BaseModel):
        """Find all IDs of the objects with name 'target_name' above an object with name 'anchor_name'. If multiple objects match this criterion, returns the list of all the objects that do. You may want to use this tool when you want to know, which 'target_name' is below 'anchor_name'. For example, which towel is under the bed, which cabinet is below the refrigerator, or which chair is under the desk.
        """

        target_name: str = Field(..., description="The name of the target object that is 'below'.")
        anchor_name: str = Field(..., description="The name of the anchor object that is 'above'.")

    
    def find_below(self, target_name: str, anchor_name: str) -> list[int]:

        target_objs = [obj_id for obj_id, obj in self.object_dict.items() if target_name in obj["name"]]
        anchor_objs = [obj_id for obj_id, obj in self.object_dict.items() if anchor_name in obj["name"]]

        if not len(target_objs):
            return f'{target_name} is not among the names in the list of objects. Please use a synonym that is found exactly in the list of objects.'
        if not len(anchor_objs):
            return f'{anchor_name} is not among the names in the list of objects. Please use a synonym that is found exactly in the list of objects.'

        below_objs = []
        for target_obj_id, anchor_obj_id in product(target_objs, anchor_objs):

            target_center = np.array(self.object_dict[target_obj_id]["centroid"])
            target_bbox = np.array(get_bbox_coords_heading_xyzlwh(
                    target_center,
                    self.object_dict[target_obj_id]["dimensions"],
                    self.object_dict[target_obj_id]["heading"]))
            
            anchor_center = np.array(self.object_dict[anchor_obj_id]["centroid"])
            anchor_bbox = np.array(get_bbox_coords_heading_xyzlwh(
                    anchor_center,
                    self.object_dict[anchor_obj_id]["dimensions"],
                    self.object_dict[anchor_obj_id]["heading"]))

            # get lowest/highest points on objects
            max_z_target = max([pt[-1] for pt in target_bbox])
            min_z_target = min([pt[-1] for pt in target_bbox])
            min_z_anchor = min([pt[-1] for pt in anchor_bbox])
            max_z_anchor = max([pt[-1] for pt in anchor_bbox])



            if max_z_target < max_z_anchor:

                center_dist = np.linalg.norm(np.array(anchor_center)[:2] - np.array(target_center)[:2])
                iom = calculate_iom_poly(target_bbox, anchor_bbox)

                if iom > self.vertical_iom:
                    below_objs.append(target_obj_id)
                else:
                    dists = []
                    # get distance between each pair of box coordinates
                    for p1 in anchor_bbox:
                        d = [np.linalg.norm(p1[:2]-p2[:2]) for p2 in target_bbox]
                        dists += d

                    # object near if at least two points close to each other
                    if np.any(np.array(dists) < self.near_thres):
                        below_objs.append(target_obj_id)

        # Mapping back to LLM IDs
        below_objs = [self.inv_object_id_map[obj] for obj in below_objs]  
        if len(below_objs) == 0:
            return f"No {target_name} found below {anchor_name}"
        elif len(below_objs) == 1:
            return f"Only one {target_name} found below {anchor_name}, the id is {below_objs[0]}"
        else:
            return f"Multiple {target_name} found below {anchor_name}, the ids are {below_objs}"
    
    
    class OrderFrontToBackSchema(BaseModel):
        """Order objects from front to back.
        
        Args:
            object_ids (list[int]): The list of object IDs to be ordered.
        """

        object_ids: list[int] = Field(..., description="The list of object IDs to be ordered.")

    def order_front_to_back(self, object_ids: list[int]) -> str:
        pass


    class OrderBottomToTopSchema(BaseModel):
        """Find objects with name 'target_name' ordered from low bottom to high top. You may want to use this tool to find the object that is the first/second/third/etc. lowest or highest. For example, you can use this tool to find which cabinet is the lowest/highest, or which pillow is the second lowest. You may also use the tool in combination with order_left_to_right to find the object that is the bottom left/top right/etc. For example, if order_bottom_to_top returns [0, 2, 1, 3], meaning cabinets 0 and 2 are likely to be on the bottom, and order_left_to_right returns [0, 1, 2, 3], meaning cabinets 0 and 1 are likely to be on the left, the bottom left cabinet is therefore most likely to be cabinet 0.
        """

        target_name: str = Field(..., description="The name of the target objects to be ordered from bottom to top.")
    
    def order_bottom_to_top(self, target_name: str) -> str:

        target_objs = [obj_id for obj_id, obj in self.object_dict.items() if target_name in obj["name"]]

        target_centers = np.array([self.object_dict[target_obj_id]["centroid"] for target_obj_id in target_objs])

        order_indices = np.argsort(target_centers[:, 2])

        sorted_target_ids = np.array(target_objs)[order_indices].tolist()

        # Mapping back to LLM IDs
        sorted_target_ids = [self.inv_object_id_map[obj] for obj in sorted_target_ids]

        print(sorted_target_ids)            
        if len(sorted_target_ids) == 0:
            return f"No {target_name} found in the object list!"
        elif len(sorted_target_ids) == 1:
            return f"There is only one {target_name} in the object list, and it has an ID of {sorted_target_ids[0]}!"
        elif len(sorted_target_ids) == 2:
            return f"The lowest {target_name} has an ID of {sorted_target_ids[0]}. The highest {target_name} has an ID of {sorted_target_ids[1]}."
        else:
            return f"{target_name}s, ordered from bottom to top, have IDs of {sorted_target_ids}."


    class OrderLeftToRightSchema(BaseModel):
        """Order the objects with name 'target_name' from left to right. You may want to use this tool to find the object that is the first/second/third/etc. from the left/right. For example, you can use this tool to find which stool is the second stool from the left, which chair is the third chair from the right, or which pillow is the first pillow from the right.
        """

        target_name: str = Field(..., description="The name of the target objects to be ordered from left to right.")

    def order_left_to_right(self, target_name: str) -> list[int]:

        target_objs = [obj_id for obj_id, obj in self.object_dict.items() if target_name in obj["name"]]

        target_centers = np.array([self.object_dict[target_obj_id]["centroid"] for target_obj_id in target_objs])

        focal_point = np.mean(target_centers, axis=0)
        anchor_point = self.freespace[np.argmin(np.linalg.norm(self.freespace - focal_point, axis=-1))]
        
        target_vec = (target_centers - anchor_point)[:, :2]
        target_vec /= np.linalg.norm(target_vec, axis=-1, keepdims=True)

        angles = np.arctan2(target_vec[:, 1], target_vec[:, 0])
        angles[angles<0] += 2 * np.pi
        order_indices = np.argsort(angles)[::-1]

        sorted_target_ids = np.array(target_objs)[order_indices].tolist()

        # Mapping back to LLM IDs
        sorted_target_ids = [self.inv_object_id_map[obj] for obj in sorted_target_ids]

        print(sorted_target_ids)   
        if len(sorted_target_ids) == 0:
            return f"No {target_name} found in the object list!"
        elif len(sorted_target_ids) == 1:
            return f"There is only one {target_name} in the object list, and it has an ID of {sorted_target_ids[0]}!"
        elif len(sorted_target_ids) == 2:
            return f"The leftmost {target_name} has an ID of {sorted_target_ids[0]}. The rightmost {target_name} has an ID of {sorted_target_ids[1]}."
        else:
            return f"{target_name}s, ordered from left to right, have IDs of {sorted_target_ids}."         


    class FindNearSchema(BaseModel):
        """Find objects with name 'target_name' ordered by distance from the object with ID 'anchor_id' (closest to furthest). You may want to use this function to find the objects with name 'target_name' that is closest/furthest/second closest/etc. to an object with ID 'anchor_id'. For example, you can use this function to find which night stand is closest to the bed with id 1, which pillow is furthest from the window with id 2, or which pillow is second closest to the window with id 0.
        """

        target_name: str = Field(..., description="The name of the target objects")
        anchor_id: int = Field(..., description="The ID of the anchor object.")

    def find_near(self, target_name: str, anchor_id: int) -> list[int]:
        try:
            print('Object ID Map:\n', self.object_id_map)
            target_objs = [obj_id for obj_id, obj in self.object_dict.items() if target_name in obj["name"] and self.inv_object_id_map[obj_id] != anchor_id]

            target_centers = [self.object_dict[target_obj_id]["centroid"] for target_obj_id in target_objs]
            anchor_center = self.object_dict[self.object_id_map[anchor_id]]["centroid"]

            target_dists = [np.linalg.norm(np.array(center) - np.array(anchor_center)) for center in target_centers]

            target_pairs = list(zip(target_objs, target_dists))

            target_pairs.sort(key = lambda x: x[1])

            sorted_target_ids = [pair[0] for pair in target_pairs]

            # Mapping back to LLM IDs
            sorted_target_ids = [self.inv_object_id_map[obj] for obj in sorted_target_ids]
            
            if len(sorted_target_ids) == 0:
                return f"No {target_name} found in the object list!"
            elif len(sorted_target_ids) == 1:
                return f"There is only one {target_name} in the object list, and it has an ID of {sorted_target_ids[0]}. It is both the closest and the furthest!"
            elif len(sorted_target_ids) == 2:             
                closest_target_id = sorted_target_ids[0]
                furthest_target_id = sorted_target_ids[-1]
                return f"The closest {target_name} to the object with ID {anchor_id} has an ID of {closest_target_id}. The furthest {target_name} to the object with ID {anchor_id} has an ID of {furthest_target_id}."
            else:
                return f"{target_name}s, ordered from closest to furthest from the object with ID {anchor_id}, have IDs of {sorted_target_ids}"
        
        except KeyError as e:
            print(traceback.format_exc())
            return "The index provided as an anchor_id is out of bounds! Provide a valid index for an object in the list."


    class OrderSmallestToLargestSchema(BaseModel):
        """Order the objects with name 'target_name' from smallest to largest. You may want to use this function to find the smallest/largest/second smallest/etc among all object with name 'target_name'. For example, you can use this function to find the smallest trash can, the largest window, or the second smallest chair.
        """

        target_name: str = Field(..., description="The name of the target objects to be ordered from smallest to largest.")

    def order_smallest_to_largest(self, target_name: str) -> list[int]:

        target_objs = [obj_id for obj_id, obj in self.object_dict.items() if target_name in obj["name"]]

        target_sizes = np.array([self.object_dict[target_obj_id]["largest_face"] for target_obj_id in target_objs])

        order_indices = np.argsort(target_sizes)

        sorted_target_ids = np.array(target_objs)[order_indices].tolist()

        # Mapping back to LLM IDs
        sorted_target_ids = [self.inv_object_id_map[obj] for obj in sorted_target_ids]            

        if len(sorted_target_ids) == 0:
            return f"No {target_name} found in the object list!"
        elif len(sorted_target_ids) == 1:
            return f"There is only one {target_name} in the object list, and it has an ID of {sorted_target_ids[0]}!"
        elif len(sorted_target_ids) == 2:
            return f"The smallest {target_name} has an ID of {sorted_target_ids[0]}. The largest {target_name} has an ID of {sorted_target_ids[1]}."
        else:
            return f"{target_name}s, ordered from smallest to largest, have IDs of {sorted_target_ids}."
    

    class FindObjectsNearRoomCornerSchema(BaseModel):
        """Finds the objects with the name "target_name" that are near a corner of the room. For example, you can use this tool to find which potted plant is most likely in the corner of the room, or is the chair in the object list likely to be in the corner of the room.
        """

        target_name: str = Field(..., description="The name of the target objects to be searched through to determine whether they are in the corner of the room.")

    def find_objects_near_room_corner(self, target_name: str) -> list[int]:

        target_objs = [obj_id for obj_id, obj in self.object_dict.items() if target_name in obj["name"]]

        target_centers = np.array([self.object_dict[target_obj_id]["centroid"] for target_obj_id in target_objs])
        target_bboxes = np.array([get_bbox_coords_heading_xyzlwh(
                target_center,
                self.object_dict[target_obj_id]["dimensions"],
                self.object_dict[target_obj_id]["heading"]) for target_obj_id, target_center in zip(target_objs, target_centers)])

        y_max = self.pcl[:, 1].max()
        y_min = self.pcl[:, 1].min()
        x_max = self.pcl[:, 0].max()
        x_min = self.pcl[:, 0].min()

        corners = np.array([
            [y_max, x_max],
            [y_max, x_min],
            [y_min, x_max],
            [y_min, x_min],
        ])

        target_ids = []

        for target_id, center, target_bbox in zip(target_objs, target_centers, target_bboxes):
            for corner in corners:
                if np.linalg.norm(center[:2] - corner) < self.near_thres:
                    target_ids.append(target_id)
                    break
                else:
                    # get distance between each pair of box coordinates
                    dists = np.array([np.linalg.norm(p[:2]-corner) for p in target_bbox])
                    dists.min(axis=0)

                    #print("dists", dists)
                    # object near if at least two points close to each other
                    if np.any(dists < self.near_thres):
                        print(dists[dists < self.near_thres])
                        target_ids.append(target_id)


        # Mapping back to LLM IDs
        target_ids = [self.inv_object_id_map[obj] for obj in target_ids]            

        if len(target_ids) == 0:
            return f"No {target_name} found near a corner of the room!"
        elif len(target_ids) == 1:
            return f"The only {target_name} found near a corner of the room has an ID of {target_ids[0]}."
        else:
            return f"Multiple {target_name}s found near a corner of the room. Their IDs are {target_ids}."


if __name__ == "__main__":
    toolbox = AgentToolbox()
    toolbox.object_dict = {
        0: {
            "name": {
                "string": "toilet", # To fit in with captioner backend
                "nyu_label": "toilet"
            },
            "image": {}, # To be filled in with captioning stuff
            "centroid": [0, 1, 1],
            "dimensions": [1, 1, 1],
            "heading": 0,
            "largest_face": 2
        },
        1: {
            "name": {
                "string": "window", # To fit in with captioner backend
                "nyu_label": "toilet"
            },
            "image": {}, # To be filled in with captioning stuff
            "centroid": [0, 1.07, 1],
            "dimensions": [1, 1, 1],
            "heading": 0,
            "largest_face": 2
        },
        2: {
            "name": {
                "string": "window", # To fit in with captioner backend
                "nyu_label": "toilet"
            },
            "image": {}, # To be filled in with captioning stuff
            "centroid": [0, 2, 1],
            "dimensions": [1, 1, 1],
            "heading": 0,
            "largest_face": 2
        },
    }
    print(toolbox.find_near("window", "toilet"))
    # print(toolbox.find_left("window", "toilet"))
    # print(toolbox.find_right("window", "toilet"))
    print(toolbox.find_between("toilet", "window", "window"))
    # ans = toolbox.calculate_distance((1, 2, 3), "office worker", [(2, 3, 4), (3, 4, 5)], ["chair", "coffee machine"])