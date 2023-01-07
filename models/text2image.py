from diffusers import DiffusionPipeline
from params.constants import MODELS
import torch
from .scheduler import make_scheduler
import traceback

class Text2ImageModel:
    def __init__(self, device, worker_id) -> None:
        for model_name in MODELS.keys():
            print("Load model ",model_name)
            self.__setattr__(model_name,
                             DiffusionPipeline.from_pretrained(
                                 MODELS[model_name],
                                 torch_dtype=torch.float16
                             ).to(device))
            pipeline: DiffusionPipeline = self.__getattribute__(model_name)
            pipeline.safety_checker = self.AltDiffusion.safety_checker
            pipeline.enable_xformers_memory_efficient_attention()
        self.worker_id = worker_id
        self.device = device
        self.output_name = f"/tmp/output_t2i_id{self.worker_id}.webp"

    def __call__(self, model_name, scheduler_name,seed, pipeline_params: dict):
        try:
            pipeline = self.__getattribute__(model_name)

            pipeline.scheduler = make_scheduler(scheduler_name,model_name)
            with torch.inference_mode():
                if seed is None or seed==-1:
                    output = pipeline(**pipeline_params)
                else:
                    output = pipeline(generator=torch.Generator(device=self.device).manual_seed(seed),**pipeline_params)
                                    
            image = output.images[0]
            nsfw_detect = output.nsfw_content_detected[0]
            if nsfw_detect:
                return "NSFW"

            image.save(self.output_name, format='webp', quality=70)
            return self.output_name
        except:
            traceback.print_exc()
            print("Error while generating")
            return "Error"
