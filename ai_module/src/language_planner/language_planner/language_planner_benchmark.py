import numpy as np
import os
import sys
import json
import torch
from plyfile import PlyData
import random
import argparse
import csv
from tqdm import tqdm
from pathlib import Path
from time import time, sleep
import ast

captioner_src_path = Path(__file__).resolve().parents[3] / "src" / "captioner"
sys.path.append(str(captioner_src_path))

language_planner_src_path = Path(__file__).resolve().parents[1]
sys.path.append(str(language_planner_src_path))

from language_planner.llm_backend.llm_query_langchain import NavQueryRunMode, ObjectQueryType, LanguageModel, SystemMode
from language_planner.language_planner_backend import LanguagePlannerBackend
from captioner.captioning_backend import Captioner



class LanguagePlannerBenchmark:

    def __init__(
            self,
            sources=['Scannet'],
            scene_data_dir='',
            referential_data_dir='',
            caption_dir='',
            caption_file='',
            log_dir='logs',
            dataset ='vla3d',
            exp_name='',
            split='test',
            model='mistral',
            run_mode='use_tools',
            object_query_type='llm'
            ):
        
        # Parameters

        self.log_dir = log_dir
        self.exp_name = exp_name
        self.caption_file = caption_file

        self.object_query_type = object_query_type

        self.model = LanguageModel(model)
        self.run_mode = NavQueryRunMode(run_mode)
        self.object_query_type = ObjectQueryType(object_query_type)
        self.system_mode = SystemMode.BENCHMARK

        # Backends

        self.language_planner_backend = LanguagePlannerBackend(
            self.model, 
            self.run_mode,
            self.system_mode)
        self.captioner_backend = Captioner(
            load_captioner=False,
            include_ego_robot_as_object=False) # To use with querying

        # Scenes

        scene_file = os.path.join(referential_data_dir, split + '.txt')
        self.scene_list = []

        with open(scene_file) as f:
            for line in f:
                # skip empty lines
                if not line or line == '\n':
                    continue
                self.scene_list.append(line.rstrip())
        

        self.skipped_scenes = []
        self.scenes = self.load_scenes(scene_data_dir, sources)
        self.load_captions(self.scenes, caption_dir, sources)
        csv_path = os.path.join(referential_data_dir, f'{dataset}_test_200.csv')
        if dataset == 'vla3d':
            self.ref_statements = self.read_refer_csv(csv_path, sr3d=False, vla3d=True)
        elif dataset == 'nr3d':
            self.ref_statements = self.read_refer_csv(csv_path, sr3d=False)
        elif dataset == 'sr3d':
            self.ref_statements = self.read_refer_csv(csv_path, sr3d=True)
        print("Loaded data")


        # Variable Initialization

        self.cur_pos = np.array([0., 0., 4.]) # add a static height


    def _get_distractors(self, object_dict):
        for obj_id, obj_info in object_dict.items():
            target_label = obj_info['name']['nyu_label']
            distractor_ids = [o_id for o_id in object_dict.keys() if object_dict[o_id]['name']['nyu_label'] == target_label and o_id != obj_id]
            obj_info.update({"distractor_ids":distractor_ids})
        
        return object_dict


    def load_scenes(self, scene_data_dir, sources):
        scenes = {}
        for dataset in sources:
            dataset_folder = os.path.join(scene_data_dir, dataset)
            for scene in tqdm(os.listdir(dataset_folder)):
                scene_path = os.path.join(dataset_folder, scene)
                
                if os.path.isdir(scene_path) and scene in self.scene_list:
                    scene = Path(scene_path).parts[-1]
                    scene_dict = {}

                    # point cloud of scene
                    pc_path = os.path.join(scene_path, scene + '_pc_result.ply')
                    map_pcl: np.ndarray = self.load_pc(pc_path)
                    scene_dict.update({'map_pcl': map_pcl})

                    # freespace point cloud of scene
                    freespace_path = os.path.join(scene_path, scene + '_free_space_pc_result.ply')
                    
                    if not os.path.exists(freespace_path): # some scenes don't have free space, TODO: revisit this when evaling on nr3d
                        self.skipped_scenes.append(scene)
                        continue
                    
                    freespace_pcl: np.ndarray = self.load_pc(freespace_path, fs=True)
                    #print(self.freespace_pcl.shape)
                    scene_dict.update({'freespace_pcl': freespace_pcl})


                    # TODO: Refactor?
                    region_file = os.path.join(scene_path, scene + '_region_result.csv')
                    region_dict = {}
                    with open(region_file, encoding='utf-8') as csv_file:
                        csvReader = csv.DictReader(csv_file)

                        for row in csvReader:
                            region_id = int(row["region_id"])
                            region_label = row["region_label"]

                            region_entry = {"name": region_label}

                            region_dict[region_id] = region_entry
                    
                    scene_dict.update({'regions': region_dict})


                    # list of objects - get from instance segmentation module later
                    object_file = os.path.join(scene_path, scene + '_object_result.csv')
                    object_dict = {}
                    with open(object_file, encoding='utf-8') as csv_file:
                        csvReader = csv.DictReader(csv_file)

                        for row in csvReader:
                            obj_id = int(row["object_id"])
                            obj_label = row["raw_label"]
                            obj_nyu_label = row["nyu_label"]
                            obj_region_id = int(row["region_id"])
                            # TODO: check if these have to be in meters vs normalized values?
                            obj_cx, obj_cy, obj_cz = float(row["object_bbox_cx"]), float(row["object_bbox_cy"]), float(row["object_bbox_cz"])
                            l, w, h = float(row["object_bbox_xlength"]), float(row["object_bbox_ylength"]), float(row["object_bbox_zlength"])
                            heading = float(row["object_bbox_heading"])
                            largest_face = max(
                                l * w,
                                w * h,
                                l * h)

                            obj_entry = {
                                "region_id": region_id,
                                "name": {
                                    "string": obj_label, # To fit in with captioner backend
                                    "nyu_label": obj_nyu_label
                                },
                                "image": {}, # To be filled in with captioning stuff
                                "centroid": [obj_cx, obj_cy, obj_cz],
                                "dimensions": [l, w, h],
                                "heading": heading,
                                "largest_face": largest_face
                            }
                            
                            object_dict[obj_id] = obj_entry
                        
                        object_dict = self._get_distractors(object_dict)
                        #print("obj_dict", object_dict)

                        scene_dict.update({'object_dict': object_dict})
                    
                    scenes.update({scene: scene_dict})
        
        print("SKIPPED SCENES: ", self.skipped_scenes)
        self.scenes = scenes
        print("Total scenes: ", len(scenes))
        return scenes


    def load_captions(self, scenes: dict, caption_path: str, sources: list[str]):
        for dataset in sources:
            caption_folder = os.path.join(caption_path, dataset)
            for scene in tqdm(os.listdir(caption_folder)):
                scene_path = os.path.join(caption_folder, scene, 'instance_crops')
                if os.path.isdir(scene_path) and scene in self.scenes:

                    for object_id_name in os.listdir(scene_path):    
                        obj_id, obj_name = object_id_name.split('_')
                        obj_id = int(obj_id) - 1 # TODO: Since captions are currently 1-indexed for scannet, object csv is 0-indexed

                        object_path = os.path.join(scene_path, object_id_name)
                        with open(os.path.join(object_path, self.caption_file+'.txt'), 'r') as f: # CHANGE CAPTION PATH
                            caption = f.readlines()[0]
                        
                        if os.path.exists(os.path.join(object_path, 'clip.npy')):
                            clip = np.load(os.path.join(object_path, 'clip.npy'))
                            clip_ft = torch.from_numpy(clip).to(self.captioner_backend.device).to(torch.bfloat16)
                        else:
                            clip_ft = None

                        scenes[scene]['object_dict'][obj_id]['image'] = {
                            'caption': caption,
                            'clip': clip_ft,
                            'rgb': None
                        }

    def load_statements(self, data_path, sources):
        '''
        Load json annotation file with statements and bboxes
        :return:
        '''
        random_seed = 0
        random.seed(random_seed)

        referential_data = []
        max_distractors = 0
        for dataset in sources:
            dataset_folder = os.path.join(data_path, dataset)
            for scene in os.listdir(dataset_folder):
                if scene in self.skipped_scenes:
                    print(" in a skipped scene!!")
                    continue

                scene_path = os.path.join(dataset_folder, scene)
                
                if os.path.isdir(scene_path) and scene in self.scenes:
                    scene = Path(scene_path).parts[-1]
                    json_file = os.path.join(scene_path, scene + '_referential_statements.json')

                    # store in a scene name/region name dict?
                    with open(json_file) as f:
                        json_data = json.load(f)
                        num_skipped = 0
                        tot_refs = 0
                        for region, region_data in (json_data["regions"].items()):
                            for utt, data in region_data.items():
                                data = data[0]
                                # check for skipped obj references
                                if len(data) <= 1:
                                    continue
                                tot_refs += 1

                                if len(data["distractor_ids"]) > max_distractors:
                                    max_distractors = len(data["distractor_ids"])

                                ref_data = {
                                    "scene": scene,
                                    "region": region,
                                    "utterance": utt,
                                    "target_label": data["target_class"],
                                    #"target_color": data["target_color_used"],
                                    #"target_size": data["target_size_used"],
                                    "target_obj_id": data["target_index"],
                                    "distractor_ids": data["distractor_ids"],
                                    #"anchor_colors": [data["anchors"][key]["color_used"] for key in data["anchors"]],
                                    #"anchor_sizes": [data["anchors"][key]["size_used"] for key in data["anchors"]],
                                    "relation": data["relation"],
                                    "real": True
                                }
                                if data["relation_type"] == "ternary":
                                    ref_data.update({"anchor_obj_ids": [data["anchors"]["anchor_1"]["index"], data["anchors"]["anchor_2"]["index"]]})
                                else:
                                    ref_data.update({"anchor_obj_ids": [data["anchors"]["anchor_1"]["index"]]})
                                '''if data["relation_type"] == "ternary":
                                    ref_data.update({"anchors": [data["anchors"]["anchor_1"]["class"], data["anchors"]["anchor_2"]["class"]]})
                                else:
                                    ref_data.update({"anchors": [data["anchors"]["anchor_1"]["class"]]})'''
                                
                                referential_data.append(ref_data)

        return referential_data
    
    # from ReferIt3D dataloader
    def is_view_dependent(self, utt):
        target_words = ['front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost',
                        'looking', 'across']
        if any(word in utt for word in target_words):
            return True
        return False
    

    def read_refer_csv(self, csv_path, sr3d=True, vla3d=False):
        '''
        Load language data from referit3d csv
        '''
        refer_data = []
        with open(csv_path, encoding='utf-8') as csv_file:
            csvReader = csv.DictReader(csv_file)
            i = 0
            for row in csvReader:
                if row["scan_id"] in self.skipped_scenes:
                    print(" in a skipped scene!!")
                    continue

                # skip allocentric statements
                if row["scan_id"] in self.scenes: #and row["coarse_reference_type"] != "allocentric":
                    if sr3d:
                        hard = False
                        if len(ast.literal_eval(row["distractor_ids"])) > 2:
                            hard = True
                        # for sr3d
                        sample = {
                            "scene": row["scan_id"],
                            "region": "0",
                            "utterance": row["utterance"],
                            "target_label": row["instance_type"],
                            "target_obj_id": str(row["target_id"]),
                            "distractor_ids": ast.literal_eval(row["distractor_ids"]),
                            #"relation": row["reference_type"],
                            "anchor_obj_ids": ast.literal_eval(row["anchor_ids"]),
                            "hard": hard,
                            "view_dependent": self.is_view_dependent(row["utterance"])
                        }
                    elif vla3d:
                        hard = False
                        if len(ast.literal_eval(row["distractor_ids"])) > 2:
                            hard = True
                        sample = {
                            "scene": row["scan_id"],
                            "region": "0",
                            "utterance": row["utterance"],
                            "target_label": row["instance_type"],
                            "target_obj_id": str(row["target_id"]),
                            "distractor_ids": [int(i) for i in ast.literal_eval(row["distractor_ids"])],
                            "anchor_obj_ids": [int(i) for i in ast.literal_eval(row["anchor_ids"])],
                            "hard": hard,
                            "view_dependent": self.is_view_dependent(row["utterance"])
                        }
                    else:
                        # for nr3d
                        distractor_ids = self.scenes[row["scan_id"]]["object_dict"][int(row["target_id"])]["distractor_ids"]
                        hard = False
                        if len(distractor_ids) > 2:
                            hard = True
                        sample = {
                            "scene": row["scan_id"],
                            "region": "0",
                            "utterance": row["utterance"],
                            "target_label": row["instance_type"],
                            "target_obj_id": str(row["target_id"]),
                            "distractor_ids": distractor_ids, # not in data
                            "relation": "",
                            "anchor_obj_ids": [],
                            "hard": hard,
                            "view_dependent": self.is_view_dependent(row["utterance"])
                        }
                    #print(sample)
                    refer_data.append(sample)
                else:
                    print("Not in scenes: ", row["scan_id"])
                i += 1
        
        return refer_data


    def load_pc(self, pc_path, fs=False):
        plydata = PlyData.read(pc_path)
        #print(plydata["vertex"])

        xyz = np.vstack((np.asarray(plydata["vertex"]["x"]), np.asarray(plydata["vertex"]["y"]),
                         np.asarray(plydata["vertex"]["z"]))).transpose()
        xyz = torch.from_numpy(xyz)
        if not fs:
            colors = np.vstack((np.asarray(plydata["vertex"]["red"]), np.asarray(plydata["vertex"]["green"]),
                            np.asarray(plydata["vertex"]["blue"]))).transpose()
            colors = torch.from_numpy(colors).float()
            if torch.max(colors) > 1:
                colors = colors / 127.5 - 1
        
            pc = np.hstack([xyz, colors]) # whole scene
        else:
            pc = np.array(xyz)

        return pc


    def query_objects(self, query_list, semantic_dict):

        if self.object_query_type == ObjectQueryType.CLIP_BASED:
            return self.captioner_backend.query_clip_features(query_list)["response"]
        
        else:
                
            filtered_obj_ids = self.language_planner_backend.get_retrieved_objects(query_list, semantic_dict)
            filtered_obj_ids = [int(obj_id) for obj_id in filtered_obj_ids]

            filtered_objs = {}

            for obj_id in filtered_obj_ids:
                try:
                    filtered_objs[obj_id] = {
                        "name": semantic_dict[obj_id]["name"]["string"],
                        "centroid": semantic_dict[obj_id]["centroid"],
                        "caption": semantic_dict[obj_id]["image"]["caption"],
                        "dimensions": semantic_dict[obj_id]["dimensions"],
                        "heading": semantic_dict[obj_id]["heading"],
                        "largest_face": semantic_dict[obj_id]["largest_face"]
                        }
                except KeyError:
                    continue
            
            print(filtered_objs)
            return filtered_objs


    def handle_language_query(
            self, 
            scene: str, 
            query: str):

        sleep(2)
        # parse query for mentioned objects
        obj_query_list = self.language_planner_backend.get_object_references(query)

        # load scene info
        scene_dict = self.scenes[scene]
        semantic_dict = scene_dict['object_dict']
        freespace_pcl = scene_dict['freespace_pcl']
        map_pcl = scene_dict['map_pcl']
        environment_name = scene_dict['regions'][0]['name']
        cur_pos = self.cur_pos

        #self.captioner_backend.semantic_dict = semantic_dict
        sleep(2)
        # retrieve relevant objects
        object_dict = self.query_objects(obj_query_list, semantic_dict)
        #print(semantic_dict)
        sleep(2)
        waypoints, ids, filtered_objects_out, output_code = self.language_planner_backend.generate_plan(
            environment_name,
            query,
            map_pcl,
            freespace_pcl,
            object_dict,
            cur_pos
        )

        return ids, filtered_objects_out, output_code
    

    def eval(self):

        exp_name = self.exp_name
        log_dir = self.log_dir
        exp_path = os.path.join(log_dir, exp_name)

        times = []
        correct = 0
        hard_corr = 0
        easy_corr = 0
        num_easy = 0
        num_hard = 0
        num_viewdep = 0
        viewdep_corr = 0
        num_viewindep = 0
        viewindep_corr = 0

        i = 0

        full_response_dict = {
            "correct_responses": [],
            "incorrect_responses": []
        }
        
        for ref in tqdm(self.ref_statements):

            response_dict = {}

            scene = ref["scene"]
            utt = ref["utterance"]
            target = ref["target_obj_id"]

            print(scene)
            print(utt)

            if scene not in full_response_dict:
                full_response_dict[scene] = {
                    "correct_responses": [],
                    "incorrect_responses": []
                }

            if ref["hard"]:
                num_hard += 1
            else:
                num_easy += 1
            
            if ref["view_dependent"]:
                num_viewdep += 1
            else:
                num_viewindep += 1

            start = time()
            pred_objs, object_list, output_code = self.handle_language_query(
                scene, utt)
            # send to eval/comparison functions
            end = time()
        
            print("pred: ", pred_objs)
            print("target: ", target)
            print("inference time (s): ", end-start)
            times.append(end-start)
            i += 1
            
            # no prediction
            if not len(pred_objs):
                print("no prediction")
                print(scene, utt)
                response_dict = {
                    "idx": i,
                    "scene": scene,
                    "utterance": utt,
                    "target": target,
                    "pred_objs": None,
                    "pred_obj": None,
                    "filtered_object_list": [[str(i) for i in obj] for obj in object_list]
                }
                full_response_dict['incorrect_responses'].append(response_dict)
                continue

            response_dict = {
                "idx": i,
                "scene": scene,
                "utterance": utt,
                "target": target,
                "pred_objs": pred_objs,
                "pred_obj": None,
                "filtered_object_list": [[str(i) for i in obj] for obj in object_list],
                "llm_output": output_code.splitlines()
            }
            
            pred_obj = pred_objs[-1][0] # If multiple objects are output, picks last one
            response_dict["pred_obj"] = pred_obj
            if int(pred_obj) == int(target):
                correct += 1
                print("correct")
                full_response_dict['correct_responses'].append(response_dict)
                if ref["hard"]:
                    hard_corr += 1
                else:
                    easy_corr += 1
                
                if ref["view_dependent"]:
                    viewdep_corr += 1
                else:
                    viewindep_corr += 1
            else:
                print("incorrect")
                full_response_dict['incorrect_responses'].append(response_dict)
            
            print("Acc: ", correct/i)
            print("Total: ", i)
            
            print("\n")
 
            # Do it on the fly
            # add log for config file with params
            with open(os.path.join(exp_path, 'correct.json'), 'w') as f:
                json.dump(full_response_dict['correct_responses'], f, indent=4)
            with open(os.path.join(exp_path, 'incorrect.json'), 'w') as f:
                json.dump(full_response_dict['incorrect_responses'], f, indent=4)
            #break
        
            if i % 10 == 0:  
                print("Acc: ", correct/i)
                print("Total: ", i)
                print("Easy acc: ", 0. if num_easy==0 else easy_corr/num_easy)
                print("Hard acc: ", 0. if num_hard==0 else hard_corr/num_hard)
                print("VD acc: ", 0. if num_viewdep==0 else viewdep_corr/num_viewdep)
                print("VI acc: ", 0. if num_viewindep==0 else viewindep_corr/num_viewindep)
            
        acc = correct/len(self.ref_statements)
        print("ACCURACY: ", acc)
        print("TOTAL: ", len(self.ref_statements))
        print("Average Inference time: ", sum(times)/len(times))

        overall = {
            "Total_statements": len(self.ref_statements),
            "Accuracy": acc,
            "Easy_acc": easy_corr/num_easy,
            "Hard_acc": hard_corr/num_hard,
            "Viewdep_acc": viewdep_corr/num_viewdep,
            "View_indep_acc": viewindep_corr/num_viewindep,
            "Num_easy": num_easy,
            "Num_hard": num_hard,
            "Num_VD": num_viewdep,
            "Num_VI": num_viewindep
        }
        with open(os.path.join(exp_path, 'overall_stats.json'), 'w') as f:
            json.dump(overall, f, indent=4)

        return acc
    
        
def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_data_dir', default=str(Path(__file__).resolve().parents[4]/'data'/'IRef-VLA'))
    parser.add_argument('--referential_data_dir', default=str(Path(__file__).resolve().parents[4]/'data'/'referit3d'))
    parser.add_argument('--caption_dir', default=str(Path(__file__).resolve().parents[4]/'data'/'captions'))
    parser.add_argument('--caption_file', default='caption_qwen2')
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--dataset', default='nr3d')
    parser.add_argument('--split', default='referit3d_test_100')
    parser.add_argument('--model', default='mistral')
    parser.add_argument('--run_mode', default='use_tools')
    parser.add_argument('--object_query_type', default='llm')

    args = parser.parse_args()

    sources = ['Scannet'] # for referit3d

    if not args.exp_name:
    
        exp_nums = []
        for exp in os.listdir(args.log_dir):
            if exp.startswith('exp'):
                exp_num = exp[3:]
                try:
                    int(exp_num)
                    exp_nums.append(int(exp_num))
                except ValueError:
                    continue
        exp_nums.sort()
        i = 0
        for exp_num in exp_nums:
            if exp_num > i:
                break
            else:
                i += 1
        args.exp_name = f'exp{i:003}'

    exp_path = os.path.join(args.log_dir, args.exp_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    # save config file
    with open(os.path.join(exp_path, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    planner = LanguagePlannerBenchmark(**vars(args), sources=sources)

    planner.eval()


if __name__ == "__main__":
    main()
