import json
import os
import cv2
import numpy as np
import torch

class SemanticDictSaver:

    def __init__(self, save_path: str, device: str = "cuda:0"):
        self.save_path = save_path
        self.device = device


    def save_semantic_dict(self, semantic_dict: dict):

        os.makedirs(self.save_path, exist_ok=True)

        for obj_id, obj in semantic_dict.items():

            if obj["image"]["rgb"] is None:
                continue

            folder_name = f"{obj_id}_{obj['name']['string']}"
            folder_path = os.path.join(self.save_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            caption = obj["image"]["caption"]
            if caption is not None:
                with open(os.path.join(folder_path, "caption.txt"), "w") as f:
                    f.write(caption)
            
            clip = obj["image"]["clip"]
            if clip is not None:
                clip_np = np.array([feat.to(torch.float32).cpu().numpy() for feat in clip])
                np.save(os.path.join(folder_path, "clip.npy"), clip_np)

            img = obj["image"]["rgb"]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(folder_path, "crop.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            attributes = {
                "image": {
                    "crop_coords": obj["image"]["crop_coords"].tolist(),
                    "is_caption_generated": obj["image"]["is_caption_generated"]
                },
                "centroid": obj["centroid"],
                "dimensions": obj["dimensions"],
                "heading": obj["heading"],
                "largest_face": obj["largest_face"],
                "name": {
                    "string": obj["name"]["string"],
                    "similarity_to_image": obj["name"]["similarity_to_image"]
                }
            }
            with open(os.path.join(folder_path, "attributes.json"), "w") as f:
                json.dump(attributes, f, indent=4)
    
    def load_semantic_dict(self, path: str):
        semantic_dict = {}
        
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if not os.path.isdir(folder_path):
                continue
                
            obj_id = int(folder.split('_')[0])
            
            with open(os.path.join(folder_path, "caption.txt"), "r") as f:
                caption = f.read()
                
            crop_path = os.path.join(folder_path, "crop.png")
            if os.path.exists(crop_path):
                img = cv2.imread(crop_path)
                img = torch.from_numpy(img).to(torch.uint8)
            else:
                continue
        
            clip_path = os.path.join(folder_path, "clip.npy")
            if os.path.exists(clip_path):
                clip = np.load(clip_path)
                clip = [torch.from_numpy(feat).to(self.device).to(torch.bfloat16) for feat in clip]
            else:
                clip = None
                
            with open(os.path.join(folder_path, "attributes.json"), "r") as f:
                attributes = json.load(f)

            semantic_dict[obj_id] = {
                "image": {
                    "rgb": img,
                    "crop_coords": attributes["image"]["crop_coords"],
                    "clip": clip,
                    "caption": caption,
                    "is_caption_generated": attributes["image"]["is_caption_generated"]
                },
                "centroid": attributes["centroid"],
                "dimensions": attributes["dimensions"],
                "heading": attributes["heading"],
                "largest_face": attributes["largest_face"],
                "name": {
                    "string": attributes["name"]["string"],
                    "similarity_to_image": attributes["name"]["similarity_to_image"]
                }
            }
            
        return semantic_dict