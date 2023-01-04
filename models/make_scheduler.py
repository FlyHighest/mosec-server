from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler
)
from params.constants import MODELS


def make_scheduler(name, model_name):
    match name:
        case "PNDM": return PNDMScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler"
            )
        case "K_LMS": return LMSDiscreteScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler"
        )
        case "DDIM": return DDIMScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler"
        )
        case "K_EULER": return EulerDiscreteScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler"
        )
        case "K_EULER_ANCESTRAL": return EulerAncestralDiscreteScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler"
        )
        case "DPM": return DPMSolverMultistepScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler"
        )
