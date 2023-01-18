import os
import torch
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer
from params.constants import MODELS,MODEL_CACHE,MODELS_FP16
from typing import Dict
from models . text2image  import Text2ImageModel
if __name__=="__main__":
    m = Text2ImageModel(torch.device("cuda"),0, MODELS_FP16)
