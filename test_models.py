import os
import torch
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer
from params.constants import MODELS,MODEL_CACHE
for model_name in MODELS.keys():
    p = DiffusionPipeline.from_pretrained(
                                 MODELS[model_name],
                                 torch_dtype=torch.float16,
                                 cache_dir=MODEL_CACHE
                             )
       
    if model_name == "Chinese-style-sd-2-v0.1":
        p.tokenizer = AutoTokenizer.from_pretrained("lyua1225/clip-huge-zh-75k-steps-bs4096",cache_dir=MODEL_CACHE, trust_remote_code=True)
exit(0)