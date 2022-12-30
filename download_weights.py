import os
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

MODEL_CACHE = "models-cache"

SD_2_1 = "stabilityai/stable-diffusion-2-1"

os.makedirs(MODEL_CACHE, exist_ok=True)

pipe_sd_2_1 = StableDiffusionPipeline.from_pretrained(
    SD_2_1,
    cache_dir=MODEL_CACHE,
    revision="fp16",
    torch_dtype=torch.float16
)

