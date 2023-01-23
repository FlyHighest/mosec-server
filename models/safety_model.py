from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
import  torch 

from PIL import Image 
import numpy as np

class SafetyModel:
    def __init__(self, device) -> None:
        self.device = device
        self.checker = StableDiffusionSafetyChecker.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder="safety_checker",torch_dtype=torch.float16)
        self.checker.to(device)
        self.featuer_extractor = CLIPFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")    

    def __call__(self, img:Image):
        safety_checker_input = self.featuer_extractor([img], return_tensors="pt").to(self.device)
        images=np.array(img)[None,...]
        image, has_nsfw_concept = self.checker(
                images=images, clip_input=safety_checker_input.pixel_values.to(torch.float16)
            )
        if has_nsfw_concept[0]:
            return "NSFW"
        else:
            return "OK"