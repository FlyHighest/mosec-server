from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler
)
from .constants import MODEL_CACHE


def make_scheduler(name, model, revision):
    return {
        "PNDM": PNDMScheduler.from_config(
            model,
            cache_dir=SD_MODEL_CACHE,
            local_files_only=True, 
            subfolder="scheduler",
            revision=revision or "main"
        ),
        "K_LMS": LMSDiscreteScheduler.from_config(
            model,
            cache_dir=SD_MODEL_CACHE,
            local_files_only=True,
            subfolder="scheduler",
            revision=revision or "main"
        ),
        "DDIM": DDIMScheduler.from_config(
            model,
            cache_dir=SD_MODEL_CACHE,
            local_files_only=True,
            subfolder="scheduler",
            revision=revision or "main"
        ),
        "K_EULER": EulerDiscreteScheduler.from_config(
            model,
            cache_dir=SD_MODEL_CACHE, 
            local_files_only=True, 
            subfolder="scheduler",
            revision=revision or "main"
        ),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(
            model,
            cache_dir=SD_MODEL_CACHE, 
            local_files_only=True,
            subfolder="scheduler",
            revision=revision or "main"
        ),
    }[name]
