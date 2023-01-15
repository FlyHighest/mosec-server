from diffusers import DiffusionPipeline
from params.constants import MODELS,MODEL_CACHE
import torch
from .scheduler import make_scheduler
import traceback
from transformers import AutoTokenizer
from typing import Dict
class Text2ImageModel:
    def __init__(self, device, worker_id) -> None:
        self.models:Dict[str,DiffusionPipeline] = dict()

        for model_name in MODELS.keys():
            print("Load model",model_name)
            if model_name == "Chinese-style-sd-2-v0.1":
                tokenizer = AutoTokenizer.from_pretrained(
                    "lyua1225/clip-huge-zh-75k-steps-bs4096",torch_dtype=torch.float16,
                    cache_dir=MODEL_CACHE, trust_remote_code=True)
                
                self.models[model_name]= DiffusionPipeline.from_pretrained(
                                    MODELS[model_name],
                                    torch_dtype=torch.float16,
                                    cache_dir=MODEL_CACHE,
                                    tokenizer=tokenizer
                                ).to(device)
            else:
                self.models[model_name]= DiffusionPipeline.from_pretrained(
                                    MODELS[model_name],
                                    torch_dtype=torch.float16,
                                    cache_dir=MODEL_CACHE,
                                ).to(device)
            self.models[model_name].enable_xformers_memory_efficient_attention()
            self.models[model_name].feature_extractor = self.models["Taiyi-Chinese-v0.1"].feature_extractor
            self.models[model_name].safety_checker =  self.models["Taiyi-Chinese-v0.1"].safety_checker
        
        self.models["Chinese-style-sd-2-v0.1"].tokenizer = AutoTokenizer.from_pretrained(
            "lyua1225/clip-huge-zh-75k-steps-bs4096",torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE, trust_remote_code=True)
        
        self.worker_id = worker_id
        self.device = device
        self.output_name = f"/tmp/output_t2i_id{self.worker_id}.webp"

    def __call__(self, model_name, scheduler_name,seed, pipeline_params: dict):
        try:
            pipeline = self.models[model_name]

            pipeline.scheduler = make_scheduler(scheduler_name,model_name)
            with torch.inference_mode():
                output = pipeline(generator=torch.Generator(device=self.device).manual_seed(seed),**pipeline_params)
                                    
            image = output.images[0]
            nsfw_detect = output.nsfw_content_detected[0]
            if nsfw_detect:
                return "NSFW"

            image.save(self.output_name, format='webp', quality=80)
            return self.output_name
        except:
            traceback.print_exc()
            print("Error while generating")
            return "Error"
