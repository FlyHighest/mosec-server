from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from params.constants import MODELS
import torch
from models.make_scheduler import make_scheduler
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
            pipeline: StableDiffusionPipeline = self.__getattribute__(model_name)
            pipeline.safety_checker = self.AltDiffusion.safety_checker
        self.worker_id = worker_id
        self.output_name = f"/tmp/output_id{self.worker_id}.jpg"

    def __call__(self, model_name, scheduler_name, pipeline_params: dict):
        try:
            pipeline = self.__getattribute__(model_name)

            pipeline.scheduler = make_scheduler(scheduler_name,model_name)

            output = pipeline(**pipeline_params)
            image = output.images[0]
            nsfw_detect = output.nsfw_content_detected[0]
            if nsfw_detect:
                return "NSFW"

            image.save(self.output_name, format='jpeg', quality=90)
            return self.output_name
        except:
            traceback.print_exc()
            print("Error while generating")
            return "Error"
