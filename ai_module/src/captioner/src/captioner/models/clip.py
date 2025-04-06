import torch
import torchvision.transforms.v2 as tt
import open_clip
from transformers import (
    SiglipModel,
    SiglipProcessor,
    CLIPModel
)
from PIL import Image
import requests
import numpy as np

class BaseCLIP:

    def __init__(
            self,
            device = 'cuda:0'
            ):
        self.device = device

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def encode_text(self, text: str) -> torch.Tensor:
        pass

    @torch.no_grad()
    def get_similarity_score(
        self, 
        feature1: torch.Tensor, 
        feature2: torch.Tensor) -> float:
        logits = (feature1 @ feature2.T) * self.model.logit_scale.exp()
        if self.model.logit_bias is not None:
            logits += self.model.logit_bias
        logits = float(logits.cpu())
        return logits


class OpenCLIP(BaseCLIP):
    def __init__(
            self,
            model_id='hf-hub:apple/DFN5B-CLIP-ViT-H-14-378',
            device='cuda:0'
            ):
        super().__init__(device)
        self.model_id = model_id
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            self.model_id, 
            device=self.device, 
            precision='bf16'
            )
        self.tokenizer = open_clip.get_tokenizer(self.model_id)

        self.preprocess = tt.Compose(
            [lambda x: x.transpose(-1, 0)]
            + self.preprocess.transforms[:-3]
            + [tt.ToDtype(torch.bfloat16, scale=True)]
            + self.preprocess.transforms[-1:])
    
    @torch.no_grad()
    def encode_image(self, image):
        img_preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
        image_feature = self.model.encode_image(img_preprocessed)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        return image_feature
    
    @torch.no_grad()
    def encode_text(self, text):
        text_tokens = self.tokenizer([text]).to(self.device)
        text_feature = self.model.encode_text(text_tokens)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        return text_feature


class SigLIPHF(BaseCLIP):
    def __init__(
            self,
            model_id='google/siglip-so400m-patch14-384',
            device='cuda:0'
            ):
        super().__init__(device)
        self.model_id = model_id

        self.model = SiglipModel.from_pretrained(
            model_id,
            device_map=self.device,
            # torch_dtype=torch.bfloat16
            # attn_implementation="flash_attention_2"
            )
        self.processor = SiglipProcessor.from_pretrained(model_id)
    
    @torch.no_grad()
    def encode_image(self, image):
        image_processed = self.processor(images=image, return_tensors='pt', padding='max_length').to(self.device)
        features = self.model.get_image_features(**image_processed)
        features /= features.norm(dim=-1, keepdim=True)
        return features
    
    @torch.no_grad()
    def encode_text(self, text):
        text_processed = self.processor(text=text, return_tensors='pt', padding='max_length').to(self.device)
        features = self.model.get_text_features(**text_processed)
        features /= features.norm(dim=-1, keepdim=True)
        return features


    
if __name__=="__main__":

    # clip_model = SigLIPHF()
    clip_model = OpenCLIP()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # print(np.asarray(image))
    image = tt.functional.to_tensor(image).transpose(0, -1)

    candidate_labels = ["2 cats", "cats", "two cats", "two pets", "pets", "two animals", "animals"]
    texts = [f'This is a photo of {label}.' for label in candidate_labels]


    image_feature = clip_model.encode_image(image)
    text_features = [clip_model.encode_text(label) for label in candidate_labels]
    similarities = [clip_model.get_similarity_score(image_feature, text_feature) for text_feature in text_features]
    print(similarities)
    print('Sigmoid:', torch.sigmoid(torch.tensor(similarities)))
    print('Softmax:', torch.softmax(torch.tensor(similarities), dim=0))