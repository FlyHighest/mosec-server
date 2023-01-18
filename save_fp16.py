import os
import torch
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer
from params.constants import MODELS,MODEL_CACHE, MODELS_FP16
from typing import Dict
from models . text2image  import Text2ImageModel
if __name__=="__main__":
    MODELS = {
        "Protogen-x5.8":  "darkstorm2150/Protogen_x5.8_Official_Release"
    }

    MODELS_FP16 = {
        "Protogen-x5.8":  MODEL_CACHE+"/fp16--Protogen-x5.8"
    }
    for model_name in MODELS_FP16.keys():
        p = DiffusionPipeline.from_pretrained(
                                    MODELS[model_name],
                                    custom_pipeline="lpw_stable_diffusion",
                                    torch_dtype=torch.float16,
                                ).to("cuda")
        p.save_pretrained(MODELS_FP16[model_name], safe_serialization=True)