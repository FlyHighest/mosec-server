from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from params.constants import MODELS
import torch
from models.make_scheduler import make_scheduler


class Text2ImageModel:
    def __init__(self, device, worker_id) -> None:
        for model_name in MODELS.keys():
            self.__setattr__(model_name,
                             StableDiffusionPipeline.from_pretrained(
                                 MODELS[model_name],
                                 torch_dtype=torch.float16
                             ).to(device))
            pipeline: StableDiffusionPipeline = self.__getattribute__(model_name)
            pipeline.safety_checker = self.AltDiffusion.safety_checker
        self.worker_id = worker_id
        self.output_name = f"/tmp/output_id{self.worker_id}.jpg"

    def __call__(self, model_name, scheduler_name, pipeline_params: dict):
        try:
            pipeline = self.__getattribute__(model_name)

            pipeline.scheduler = make_scheduler(scheduler_name,model_name)

            image = pipeline(**pipeline_params).images[0]

            image.save(self.output_name, format='jpeg', quality=90)
            return self.output_name
        except:
            print("Error while generating")
            return None
