from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler
)
from params.constants import MODELS


def make_scheduler(name, model_name):
    match name:
        case "PNDM": return PNDMScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler"
            )
        case "LMS": return LMSDiscreteScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler"
        )
        case "DDIM": return DDIMScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler"
        )
        case "EULER": return EulerDiscreteScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler"
        )
        case "EULER_A": return EulerAncestralDiscreteScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler"
        )
        case "DPM": return DPMSolverMultistepScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler"
        )
