import numpy as np
import os
import spacy
import json
#import open3d as o3d
import matplotlib.pyplot as plt
import re
import argparse

from language_planner.prompts import get_prompt
from language_planner.llm_backend.llm_query_langchain import LanguageModel, NavQueryRunMode, SystemMode
from language_planner.llm_backend.llm_query_langchain import LLMQueryHandler
from language_planner.mapping.grid_map import GridMapHandler
from language_planner.code_parsing.code_parser import CodeParser

class LanguagePlannerBackend:

    def __init__(
            self,
            model=LanguageModel.MISTRAL,
            run_mode=NavQueryRunMode.USE_TOOL_USE_GRAPH,
            system_mode=SystemMode.LIVE_NAVIGATION,
            log_info=print
            ):
        
        # Parameters

        self.log_info = log_info

        # Other

        self.nlp = spacy.load("en_core_web_sm")

        self.grid_map_handler = GridMapHandler()
        self.llm_handler = LLMQueryHandler(model=model, run_mode=run_mode, system_mode=system_mode) #entrypoint
        self.code_parser = CodeParser()
    

    def get_object_references(self, statement: str):
        """Extract object references from statement using LLM"""
        objects = self.llm_handler.extract_objects(statement)
        self.log_info(f'Extracted objects: {objects}')
        return objects
    
    def get_retrieved_objects(self, query_list, semantic_dict):
        """Retrieve relevant objects from object list given noun list"""
        filtered_objects = self.llm_handler.filter_objects(query_list, semantic_dict)
        self.log_info(f'Filtered objects: {filtered_objects}')
        return filtered_objects


    def parse_code(self, code, object_dict, object_id_map):
        return self.code_parser.parse_code(code, object_dict, object_id_map)


    def generate_plan(
            self,
            environment_name: str,
            input_statement: str,
            map_pcl: np.ndarray,
            freespace_pcl: np.ndarray,
            object_dict: dict,
            cur_pos: np.ndarray):

        if not object_dict:
            return [], [], [], ""


        # create 2D grid map of scene based on traversable space
        self.grid_map_handler.create_2d_map(
            map_pcl,
            freespace_pcl,
            0.1
        )

        # Simplify object IDs for potential ease of LLM tokenization
        object_id_map = {relative_id: absolute_id for relative_id, absolute_id in enumerate(object_dict.keys()) if absolute_id != -1} 
        object_id_map[-1] = -1

        self.log_info(f'Obj id map: {object_id_map}')

        # get xyz coords
        filtered_coords = [[float(obj["centroid"][0]), float(obj["centroid"][1]), float(obj["centroid"][2])] for obj in object_dict.values()]

        # convert object/robot coords to gridmap coords
        grid_coords = self.grid_map_handler.convert_global_to_grid(filtered_coords)

        robot_coords = self.grid_map_handler.convert_global_to_grid([cur_pos])

        # get info for relevant objects
        filtered_objects_out = [
            [
                absolute_id,
                relative_id,
                obj['name'], 
                obj['caption'],
                grid_coord[0],
                grid_coord[1],
                grid_coord[2],
                obj['largest_face']
                ] for (relative_id, absolute_id), obj, grid_coord in zip(object_id_map.items(), object_dict.values(), grid_coords)]

        filtered_objects_llm = [filtered_obj[1:] for filtered_obj in filtered_objects_out]
        

        # call LLM agent to process query
        output_code = self.llm_handler.generate_query(
            environment_name,
            self.grid_map_handler.grid_map.shape,
            robot_coords,
            filtered_objects_llm,
            input_statement,
            object_dict,
            map_pcl,
            freespace_pcl
        )
        

        # self.log_info(f'{output_code}')
        self.log_info("output code")
        self.log_info(f'{output_code}')

        waypoints, ids = self.parse_code(output_code, object_dict, object_id_map)

        return waypoints, ids, filtered_objects_out, output_code


