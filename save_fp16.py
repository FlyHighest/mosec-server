import os
import torch
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer
from params.constants import MODELS,MODEL_CACHE, MODELS_FP16
from typing import Dict
from models . text2image  import Text2ImageModel
if __name__=="__main__":
    m = Text2ImageModel(torch.device("cuda"),0, MODELS)
    for model_name, save_dir in MODELS_FP16.items():
        m.models[model_name].save_pretrained(save_dir)