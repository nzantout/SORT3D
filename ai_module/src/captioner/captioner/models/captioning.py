from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    PaliGemmaProcessor, 
    PaliGemmaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    GemmaTokenizerFast,
    SiglipVisionModel,
    SiglipImageProcessor,
    SiglipModel,
    SiglipTextModel,
    SiglipTokenizer,
    StopStringCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig
)
import torch
from PIL import Image
import requests
from typing import List, Optional
import cv2
import imageio.v3 as iio
import os
from time import time
try:
    from vllm import LLM, SamplingParams, RequestOutput
except ImportError as e:
    print("VLLM not installed, ignoring dependency.")


class CaptioningModel:
    def __init__(self):
        pass

    def generate_captions(self, images: list[torch.Tensor] | torch.Tensor) -> list[str]:
        pass

    @staticmethod
    def split_into_batches(lst, batch_size: int):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]


class PaliGemmaHFBackend(CaptioningModel):
    def __init__(
            self,
            model_id = "google/paligemma2-3b-ft-docci-448",
            quantization: Optional[str] = "int4"
            ):

        self.model_id = model_id
        # self.model_id = "google/paligemma2-3b-pt-448"

        if quantization == "int8":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif quantization is None:
            bnb_config = None
        else:
            raise ValueError('Invalid quantization. Must either be int8, int4, or None.')


        self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2"
            )

        self.stopping_criteria = StoppingCriteriaList([StopStringCriteria(self.processor.tokenizer, '.')])

        self.prompt = "<image>caption en "

    def generate_captions(self, images, batch_size = 24):

        captions = []

        for batch in self.split_into_batches(images, batch_size): 

            start_time = time()  

            inputs = self.processor(
                batch, 
                [self.prompt]*len(batch), 
                self.prompt,
                return_tensors="pt").to("cuda")


            output = self.model.generate(
                **inputs, 
                # max_new_tokens=200
                stopping_criteria=self.stopping_criteria
                )

            caption_batch = self.processor.batch_decode(output, skip_special_tokens=True)
            caption_batch = [caption[len("caption en \n"):] for caption in caption_batch]

            captions.extend(caption_batch)


        return captions


class QwenHFBackend(CaptioningModel):
    def __init__(
            self,
            model_id = "Qwen/Qwen2.5-VL-3B-Instruct",
            quantization: Optional[str] = "int4"
            ):

        self.model_id = model_id


        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_id, padding_side="left")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype="auto", 
            device_map="auto",
            # attn_implementation="flash_attention_2"
            )

        self.stopping_criteria = StoppingCriteriaList([StopStringCriteria(self.processor.tokenizer, '.')])

        self.prompt = "Describe the {obj} in this image, using properties like color, material, shape, affordances, and other meaningful attributes. Provide the response in this format: “The <object name> is <color>, <material>, <shape>."

    def generate_captions(self, images, names: list[str], batch_size = 4):

        messages = [[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": self.prompt.format(obj=name)},
                ],
            }
        ] for image, name in zip(images, names)]

        # Preparation for inference
        prompts = [self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        ) for message in messages]

        captions = []

        # prompts = [self.prompt.format(name) for name in names]

        for batch in self.split_into_batches(list(zip(images, prompts)), batch_size): 

            image_batch = [b[0] for b in batch]
            prompt_batch = [b[1] for b in batch]

            start_time = time()  

            inputs = self.processor(
                images=image_batch, 
                text=prompt_batch,
                padding=True, 
                return_tensors="pt").to("cuda")


            output = self.model.generate(
                **inputs, 
                max_new_tokens=200
                # stopping_criteria=self.stopping_criteria
                )

            caption_batch = self.processor.batch_decode(output, skip_special_tokens=True)
            caption_batch = [caption.split('\nassistant\n')[1] for caption in caption_batch]

            # print(caption_batch)

            captions.extend(caption_batch)


        return captions


class PaliGemmaVLLMBackend(CaptioningModel):
    def __init__(
            self,
            model_id="google/paligemma2-3b-ft-docci-448"
            ):
        self.model_id = model_id
        # model_id = "google/paligemma2-3b-pt-448"

        self.llm = LLM(
            model=model_id, 
            dtype=torch.bfloat16,
            cpu_offload_gb=16)


        self.prompt = "<image>caption en "

        self.sampling_params = self.llm.get_default_sampling_params()
        self.sampling_params.stop = '.'



    def generate_captions(self, images):

    
        inputs = [{
            "prompt": self.prompt,
            "multi_modal_data": {
                "image": image
            }
        } for image in images]

        outputs: List[RequestOutput] = self.llm.generate(inputs, sampling_params=self.sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text
            print(generated_text)



if __name__ == "__main__":

    crops_path = '/ocean/projects/cis220039p/nzantout/VLNav-Improved/VLA_Dataset_v3/Scannet/scene0000_00/instance_crops'
    
    # List to store images
    images = []
    names = []

    # Iterate over all subdirectories
    for subdir in sorted(os.listdir(crops_path)):  # Sorting ensures order
        subdir_path = os.path.join(crops_path, subdir)
        crop_name = subdir.split('_')[1]
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            img_path = os.path.join(subdir_path, "rgb.png")
            
            # Read the image if it exists
            if os.path.exists(img_path):
                img = iio.imread(img_path)
                if img is not None:
                    images.append(img)
                    names.append(crop_name)
    
    
    qwen = QwenHFBackend()

    captions = qwen.generate_captions(images, names)