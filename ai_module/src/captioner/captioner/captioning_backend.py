import numpy as np
import os
import pandas as pd
import torch
from typing import Optional
import threading
from pathlib import Path

import json
from datetime import datetime
from enum import Enum
from scipy.spatial.transform import Rotation as R
from captioner.models.clip import OpenCLIP, SigLIPHF
from captioner.models.captioning import PaliGemmaHFBackend, QwenHFBackend
from captioner.image_utils.plotting import plot_captioned_images
from captioner.image_utils.image_conversion import binary_opening_torch
from captioner.image_utils.caption_postprocessing import postprocess_captions
from captioner.image_utils.save_semantic_dict import SemanticDictSaver


class CaptionGenerationOptions(Enum):
    ON_QUERY = "on_query" 
    ON_CLIP_UPDATE = "on_clip_update"


class CropUpdateSource(Enum):
    GROUND_TRUTH_SEMANTICS = "gt_semantics"
    SEMANTIC_MAPPING_MODULE = "semantic_mapping"


class Captioner:
    def __init__(
            self,
            semantic_dict: dict = {}, # Can initialize from previous map or initialize certain parameters
            save_semantic_dict = True,
            semantic_dict_load_path = None, # Load semantic dict from path
            num_crop_levels: int = 3,
            model_type = "clip",
            image_shape = (640, 1920, 3),
            min_pixel_threshold = (28, 28),
            max_pixel_threshold = (640, 960),
            crop_type = 'relative',
            crop_radius = 0.5,
            device = "cuda:0",
            log_info = print,
            load_captioner = True,
            crop_update_source = "gt_semantics",
            include_ego_robot_as_object = True,
            batch_size = 16,
    ):
        # Parameters

        self.num_crop_levels = num_crop_levels
        self.device = device
        self.model_type = model_type
        self.image_shape = image_shape
        self.min_pixel_threshold = min_pixel_threshold
        self.max_pixel_threshold = max_pixel_threshold 
        self.crop_radius_type = crop_type
        self.crop_radius = crop_radius
        self.log_info = log_info
        self.load_captioner = load_captioner
        self.include_ego_robot_as_object = include_ego_robot_as_object
        self.batch_size = batch_size
        self.save_semantic_dict = save_semantic_dict
        self.semantic_dict_load_path = semantic_dict_load_path
        # Options
    
        self.caption_generation_option = CaptionGenerationOptions.ON_QUERY
        self.crop_update_source = CropUpdateSource(crop_update_source)

        # CLIP Model

        if self.model_type == "clip":
            self.clip_model = OpenCLIP(device=self.device)
            self.clip_threshold = 3.5
        elif self.model_type == "siglip":
            self.clip_model = SigLIPHF(device=self.device)
            self.clip_threshold = -5
        
        # Captioning Model

        if self.load_captioner:
            self.captioning_model = QwenHFBackend(
                quantization='int4',
                batch_size=self.batch_size
            )

        # Variable Initialization

        self.semantic_id_set = set(semantic_dict.keys())
        self.semantic_dict = semantic_dict
        if self.semantic_dict_load_path is not None:
            self.semantic_dict = self.semantic_dict_saver.load_semantic_dict(self.semantic_dict_load_path)
        self.name_embeddings_dict = {}

        self.lock = threading.Lock()

        semantic_dict_save_path = str(Path(__file__).parent.resolve() / "crops" / f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        self.semantic_dict_saver = SemanticDictSaver(semantic_dict_save_path)

 
    def get_crop(self, image: torch.Tensor, crop_coords: np.ndarray, level: int):

        padded_crop_size = image.shape

        cur_crop_coords = np.array([
            [
                crop_coords[0, 0] * level // (self.num_crop_levels - 1), 
                crop_coords[0, 1] * level // (self.num_crop_levels - 1)
            ],
            [
                crop_coords[1, 0] + (padded_crop_size[0] - crop_coords[1, 0]) * 
                    (self.num_crop_levels - 1 - level) // (self.num_crop_levels - 1),
                crop_coords[1, 1] + (padded_crop_size[1] - crop_coords[1, 1]) * 
                    (self.num_crop_levels - 1 - level) // (self.num_crop_levels - 1),
            ]

        ])

        crop = image[cur_crop_coords[0, 0]:cur_crop_coords[1, 0]+1, cur_crop_coords[0, 1]:cur_crop_coords[1, 1]+1]

        return crop
    

    def get_crop_indices_from_semantic_ids(
            self, 
            semantic_id: int,
            semantic_id_set: set, 
            sem: torch.Tensor):

        if semantic_id not in semantic_id_set:
            return None
        crop_indices = (sem == semantic_id).squeeze(-1)
        try:
            crop_indices = binary_opening_torch(crop_indices, self.device).nonzero()
        except RuntimeError as e:
            return None
            
        if len(crop_indices) <= 2:
            return None
        return crop_indices

    
    # TODO: Move to utils
    def check_for_crop_update_naive(
            self, 
            semantic_dict: dict,
            semantic_id: int,
            crop: torch.Tensor,
            crop_coords: np.ndarray,
            crop_size: tuple,
            *args
            ):
        
        update_crop = np.prod(semantic_dict[semantic_id]["image"]["rgb"].shape[:-1]) < np.prod(crop.shape[:-1])

        # TODO: refactor
        if update_crop:
            semantic_dict[semantic_id]["image"]["clip"] = self.compute_clip_features(crop, crop_coords)

        return update_crop


    def check_for_crop_update_clip(
            self, 
            semantic_dict: dict,
            semantic_id: int,
            crop: torch.Tensor,
            crop_coords: np.ndarray,
            name: str,
            name_embeddings_dict: dict
            ):
        '''
        Updates crops by checking if new crop has higher clip similarity with closed vocab name
        '''
        if np.prod(semantic_dict[semantic_id]["image"]["rgb"].shape[:-1]) == np.prod(crop.shape[:-1]) \
            and (crop - semantic_dict[semantic_id]["image"]["rgb"]).float().norm() < 5: # To avoid repeated computations
            return False
        

        prev_similarity_scores = semantic_dict[semantic_id]["name"]["similarity_to_image"]
        if prev_similarity_scores is None:
            raise ValueError()

        name_embedding = name_embeddings_dict[name]
        new_features = self.compute_clip_features(crop, crop_coords)
        new_similarity_scores = [self.clip_model.get_similarity_score(name_embedding, image_feature) for image_feature in new_features]

        update_crop = max(new_similarity_scores) > max(prev_similarity_scores)

        semantic_dict[semantic_id]["name"]["similarity_to_image"] = new_similarity_scores if update_crop else prev_similarity_scores
    
        # TODO: Refactor
        if update_crop:
            semantic_dict[semantic_id]["image"]["clip"] = new_features

        return update_crop
        

    def check_for_crop_update_pcd_visibility(self):
        raise NotImplementedError()


    def merge_objects(
            self,
            semantic_id1: int,
            semantic_id2: int,
            centroid_3d: np.ndarray,
            bbox_3d: tuple[np.ndarray],
            semantic_dict: dict = None,
    ):
        '''
        Merges the second object id onto the first one.
        '''
        with self.lock:
            if semantic_dict is None:
                semantic_dict = self.semantic_dict
            
            self.update_crop(
                semantic_dict,
                semantic_id1,
                semantic_dict[semantic_id2]["image"]["rgb"],
                semantic_dict[semantic_id2]["image"]["crop_coords"],
                semantic_dict[semantic_id2]["name"]["string"],
                self.name_embeddings_dict,
                centroid_3d,
                bbox_3d
            )

            self.log_info(f'Merged dictionary {semantic_dict[semantic_id2]["name"]["string"]} ({semantic_id2}) into {semantic_dict[semantic_id1]["name"]["string"]} ({semantic_id1})')

            del semantic_dict[semantic_id2]
            assert semantic_id2 not in semantic_dict

    def remove_object(
            self,
            semantic_id: int,
            semantic_dict: dict = None,
    ):
        with self.lock:
            if semantic_dict is None:
                semantic_dict = self.semantic_dict

            self.log_info(f'Removed {semantic_dict[semantic_id]["name"]["string"]} ({semantic_id})')
            
            del semantic_dict[semantic_id]
            assert semantic_id not in semantic_dict

    def update_crop(
            self,
            semantic_dict: dict,
            semantic_id: int,
            crop: torch.Tensor,
            crop_coords: np.ndarray,
            name: str,
            name_embeddings_dict: dict,
            centroid_3d: np.ndarray,
            bbox_3d: tuple[np.ndarray]
            ):
        
        if semantic_dict[semantic_id]["image"]["rgb"] is None or \
            self.check_for_crop_update_clip(semantic_dict, semantic_id, crop, crop_coords, name, name_embeddings_dict):

            self.log_info(f'Updated {semantic_dict[semantic_id]["name"]["string"]} with ID {semantic_id}')

            # Compute CLIP Features
            if semantic_dict[semantic_id]["image"]["clip"] is None:
                semantic_dict[semantic_id]["image"]["clip"] = self.compute_clip_features(crop, crop_coords)
            # Compute ground truth label embeddings
            if name not in name_embeddings_dict:
                name_embeddings_dict[name] = self.clip_model.encode_text(name)
            # Compute similarity scores between crops and ground truth label embeddings
            if semantic_dict[semantic_id]["name"]["similarity_to_image"] is None:
                prev_features = semantic_dict[semantic_id]["image"]["clip"]
                name_embedding = name_embeddings_dict[name]
                semantic_dict[semantic_id]["name"]["similarity_to_image"] = [self.clip_model.get_similarity_score(name_embedding, image_feature) for image_feature in prev_features]

            semantic_dict[semantic_id]["image"]["rgb"] = crop
            semantic_dict[semantic_id]["image"]["crop_coords"] = crop_coords
            semantic_dict[semantic_id]["image"]["is_caption_generated"] = False

            # Update object attributes if using semantic mapping module

            if self.crop_update_source == CropUpdateSource.SEMANTIC_MAPPING_MODULE:

                center, extent, q = bbox_3d

                rpy = R.from_quat(q).as_euler('xyz')
                heading = rpy[2]

                l, w, h = extent.tolist()
                largest_face = max(
                    l * w,
                    w * h,
                    l * h)
                
                semantic_dict[semantic_id]["centroid"] = centroid_3d.tolist()
                semantic_dict[semantic_id]["dimensions"] = extent.tolist()
                semantic_dict[semantic_id]["heading"] = heading
                semantic_dict[semantic_id]["largest_face"] = largest_face


    def update_object_crops(
            self, 
            rgb: torch.Tensor,
            sem: torch.Tensor = None,
            bboxes_2d: list = None,
            obj_ids_global: list = None,
            centroids_3d: list[np.ndarray] = None,
            class_names: list = None,
            bboxes_3d: list[tuple[np.ndarray]] = None
            ):
        """
        Updates crops given an RGB image and:
        
        - A semantic image if ground truth semantics are used, set by self.crop_update_source == CropUpdateSource.GROUND_TRUTH_SEMANTICS
        - 2D bounding boxes, a global ist of object IDs, 3D centroids, class names, and 3D bounding boxes if the live semantic mapping module is used, set by self.crop_update_source == CropUpdateSource.SEMANTIC_MAPPING_MODULE
        """
        with self.lock:

            if self.crop_update_source == CropUpdateSource.GROUND_TRUTH_SEMANTICS:
                semantic_ids = [int(sem_id.cpu()) for sem_id in sem.unique()]
                object_attributes = semantic_ids
            else:
                object_attributes = zip(obj_ids_global, bboxes_2d, centroids_3d, class_names, bboxes_3d)

            for object_attribute in object_attributes:

                if self.crop_update_source == CropUpdateSource.GROUND_TRUTH_SEMANTICS:

                    semantic_id = object_attribute

                    crop_indices = self.get_crop_indices_from_semantic_ids(semantic_id, self.semantic_id_set, sem)
                    if crop_indices is None:
                        continue

                    crop_min_y = int(crop_indices[:, 0].min())
                    crop_max_y = int(crop_indices[:, 0].max())
                    crop_min_x = int(crop_indices[:, 1].min())
                    crop_max_x = int(crop_indices[:, 1].max())

                    centroid_3d, bbox_3d = None, None # These are preloaded into the initial dictionary

                else:

                    semantic_id, bbox_2d, centroid_3d, class_name, bbox_3d = object_attribute
                    
                    if semantic_id not in self.semantic_dict:
                        self.semantic_dict[semantic_id] = {
                        "image": {
                            "rgb": None,
                            "crop_coords": None,
                            "clip": None,
                            "caption": None,
                            "is_caption_generated": False
                            },
                        "centroid": None,
                        "dimensions": None,
                        "heading": None,
                        "largest_face": None,
                        "name": {
                            "string": class_name,
                            "similarity_to_image": None
                            }
                        }
                        self.log_info(f'Added {class_name} with ID {semantic_id}')

                    crop_min_x = int(bbox_2d[0])
                    crop_min_y = int(bbox_2d[1])
                    crop_max_x = int(bbox_2d[2])
                    crop_max_y = int(bbox_2d[3])

                crop_size = (crop_max_y - crop_min_y + 1, crop_max_x - crop_min_x + 1)
                if crop_size[0] < self.min_pixel_threshold[0] or crop_size[1] < self.min_pixel_threshold[1]:
                    continue 
                if crop_size[0] > self.max_pixel_threshold[0] or crop_size[1] > self.max_pixel_threshold[1]:
                    continue

                if self.crop_radius_type == 'relative':
                    crop_radius_x = round(crop_size[1] * self.crop_radius)
                    crop_radius_y = round(crop_size[0] * self.crop_radius)
                elif self.crop_radius_type == 'fixed':
                    crop_radius_x = self.crop_radius
                    crop_radius_y = self.crop_radius
                
                crop_min_y_padded = max(crop_min_y - crop_radius_y, 0)
                crop_max_y_padded = min(crop_max_y + crop_radius_y, self.image_shape[0] - 1)
                crop_min_x_padded = max(crop_min_x - crop_radius_x, 0)
                crop_max_x_padded = min(crop_max_x + crop_radius_x, self.image_shape[1] - 1)

                crop = rgb[crop_min_y_padded:crop_max_y_padded+1, crop_min_x_padded:crop_max_x_padded+1]

                crop_coords = np.array([
                    [crop_min_y - crop_min_y_padded, crop_min_x - crop_min_x_padded],
                    [crop_max_y - crop_min_y_padded, crop_max_x - crop_min_x_padded]])
                
                # TODO: add undistortion
                # crop = undistort_cylindrical_section(
                #     self.cur_rgb,
                #     crop_min_x,
                #     crop_max_x,
                #     crop_min_y,
                #     crop_max_y,
                #     crop_size
                # )

                self.update_crop(
                    self.semantic_dict, 
                    semantic_id, 
                    crop, 
                    crop_coords, 
                    self.semantic_dict[semantic_id]["name"]["string"],
                    self.name_embeddings_dict,
                    centroid_3d,
                    bbox_3d
                )
            

        if self.load_captioner and self.caption_generation_option == CaptionGenerationOptions.ON_CLIP_UPDATE:
            self.generate_captions(self.semantic_dict)


    def compute_clip_features(
            self,
            img: torch.Tensor,
            crop_coords: np.ndarray):

        clip_features = []

        for i in range(self.num_crop_levels):  # larger to smaller
            
            cur_crop = self.get_crop(img, crop_coords, i)
            clip_feature = self.clip_model.encode_image(cur_crop)
            clip_features.append(clip_feature)
        
        return clip_features
    

    def generate_captions(
            self,
            semantic_dict: dict,
            semantic_ids: Optional[list[int]] = None,
            crop_levels: Optional[list[int]] = None):
        
        with self.lock:


            if semantic_ids is None:
                semantic_ids = semantic_dict.keys()
            
            dict_items = [
                {
                    "image": semantic_dict[semantic_id]["image"]["rgb"],
                    "crop_coords": semantic_dict[semantic_id]["image"]["crop_coords"],
                    "name": semantic_dict[semantic_id]["name"]["string"],
                    "key": semantic_id
                } for semantic_id in semantic_ids if 
                    semantic_dict[semantic_id]["image"]["rgb"] is not None and 
                    not semantic_dict[semantic_id]["image"]["is_caption_generated"]]
            
            self.log_info(f'Generating captions for {len(dict_items)} objects')
            
            if crop_levels is None:
                # Generate captions for crops with highest similarity level to ground truth input
                crop_levels = [np.argmax(dict_entry["name"]["similarity_to_image"]) for dict_entry in self.semantic_dict.values() 
                                if dict_entry["name"]["similarity_to_image"] is not None and
                                not dict_entry["image"]["is_caption_generated"]]

            crops = [self.get_crop(
                dict_item["image"], 
                dict_item["crop_coords"], 
                crop_level) for dict_item, crop_level in zip(dict_items, crop_levels)]
            
            names = [dict_item["name"] for dict_item in dict_items]

            captions = self.captioning_model.generate_captions(crops, names)

            for dict_item, caption in zip(dict_items, captions):
                self.log_info(f'{semantic_dict[dict_item["key"]]["name"]}: {caption}')
                semantic_dict[dict_item["key"]]["image"]["caption"] = caption
                semantic_dict[dict_item["key"]]["image"]["is_caption_generated"] = True
            
            if self.save_semantic_dict:
                self.log_info(f'Saving semantic dict to {self.semantic_dict_saver.save_path}...')
                self.semantic_dict_saver.save_semantic_dict(semantic_dict)
                self.log_info(f'Saved semantic dict to {self.semantic_dict_saver.save_path}.')


    def generate_ego_robot_dict_entry(self, cur_pos: np.ndarray, cur_orient: np.ndarray):
        rpy = R.from_quat(cur_orient).as_euler('xyz')
        heading = rpy[2]

        l, w, h = 0.5, 0.5, 0.7 #TODO: Make parameter
        largest_face = max(
            l * w,
            w * h,
            l * h)

        return_dict = {
            "name": "egorobot",
            "centroid": cur_pos.tolist(),
            "caption": "The ego robot being navigated around the room. It has an ID of -1. When a language query requires a computation related to the robot's current position, use this. Example: \"go to the door closest to you\". In this case, you would call the tool find_near(target_name=\"door\", anchor_id=-1).",
            "dimensions": [l, w, h],
            "heading": heading,
            "largest_face": largest_face,
        }

        return return_dict

    
    def query_clip_features(self, query_list: list[str], cur_pos: np.ndarray = None, cur_orient: np.ndarray = None):

        if self.load_captioner and self.caption_generation_option == CaptionGenerationOptions.ON_QUERY:
            self.generate_captions(self.semantic_dict)
        
        if "GET_ALL" in query_list:
            return_dict = {
                "query": ["GET_ALL"],
                "response": {}
            }
            for obj_id, obj in self.semantic_dict.items():
                return_dict["response"][int(obj_id)] = {
                    "name": obj["name"]["string"],
                    "centroid": obj["centroid"],
                    "caption": obj["image"]["caption"],
                    "dimensions": obj["dimensions"],
                    "heading": obj["heading"],
                    "largest_face": obj["largest_face"]
                }
            if self.include_ego_robot_as_object:
                return_dict["response"][-1] = self.generate_ego_robot_dict_entry(cur_pos, cur_orient)
            return return_dict


        full_ordered_results = []

        for query in query_list:

            ordered_results = []
            text_feature = self.clip_model.encode_text(query)

            # Obtaining similarity scores
            for key, value in self.semantic_dict.items():
                if "clip" not in value["image"].keys():
                    continue
                image_features = value["image"]["clip"]
                if image_features is None:
                    continue
                similarity_scores = [self.clip_model.get_similarity_score(image_feature, text_feature) for image_feature in image_features]
                ordered_results.append((key, similarity_scores))

            ordered_results.sort(key=lambda x: max(x[1]), reverse=True)

            # Thresholding
            for i, score in enumerate([max(q[1]) for q in ordered_results]):
                if score < self.clip_threshold:
                    full_ordered_results.append(ordered_results[:i])
                    break

        return_dict = {
            "query": query_list,
            "response": {}
        }
        for ordered_results in full_ordered_results:
            for result in ordered_results:
                if result[0] not in return_dict["response"]:
                    return_dict["response"][int(result[0])] = {
                        "name": self.semantic_dict[result[0]]["name"]["string"],
                        "centroid": self.semantic_dict[result[0]]["centroid"],
                        "caption": self.semantic_dict[result[0]]["image"]["caption"],
                        "dimensions": self.semantic_dict[result[0]]["dimensions"],
                        "heading": self.semantic_dict[result[0]]["heading"],
                        "largest_face": self.semantic_dict[result[0]]["largest_face"]
                    }
        
        if self.include_ego_robot_as_object:
            return_dict["response"][-1] = self.generate_ego_robot_dict_entry(cur_pos, cur_orient)
        
        self.log_info(f'{return_dict}')
        return return_dict
            
