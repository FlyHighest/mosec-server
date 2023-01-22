from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
import random ,torch 
import re 
from PIL import Image 
INT_RANGE = -2**31,2**31-1

class SafetyModel:
    def __init__(self, device) -> None:
        self.device = device
        self.checker = StableDiffusionSafetyChecker.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder="safety_checker",torch_dtype=torch.float16)
        self.checker.to(device)
        self.featuer_extractor = CLIPFeatureExtractor.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder="feature_extractor")    

    def __call__(self, img:Image):
        safety_checker_input = self.featuer_extractor([img], return_tensors="pt").to(self.device)
        image, has_nsfw_concept = self.checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(torch.float16)
            )
        if has_nsfw_concept[0]:
            return "NSFW"
        else:
            return "OK"