import os
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from params.constants import MODEL_CACHE,MODELS

SD_2_1 = "stabilityai/stable-diffusion-2-1"

StableDiffusionPipeline.from_pretrained(
    MODEL_CACHE+"/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
