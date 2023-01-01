from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from params.constants import MODEL_CACHE, SD_2_1
import torch 
class StableDiffusionModel:
    def __init__(self, device) -> None:
        self.sd_2_1 = StableDiffusionPipeline.from_pretrained(
            SD_2_1,
            cache_dir=MODEL_CACHE,
            revision="fp16",
            torch_dtype=torch.float16
        )
        self.sd_2_1.to(device)
        self.sd_2_1.scheduler = DPMSolverMultistepScheduler.from_config(self.sd_2_1.scheduler.config)

    def __call__(self, model_name, prompt, **kargs):
        try:
            # TODO: choose pipeline according to model name:
            pipeline : StableDiffusionPipeline = self.sd_2_1
            # TODO: change scheduler

            image = pipeline(prompt=prompt, guidance_scale=7.0, num_inference_steps=20).images[0]

            # TODO: NSFW filter

            image.save("/tmp/output.jpg",format='jpeg', quality=90)
            return "/tmp/output.jpg"
        except:
            print("Error while generating")
            return None 
