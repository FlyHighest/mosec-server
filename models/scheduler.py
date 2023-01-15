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
from params.constants import MODELS, MODEL_CACHE


def make_scheduler(name, model_name):
    match name:
        case "PNDM": return PNDMScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler",
            cache_dir=MODEL_CACHE
            )
        case "LMS": return LMSDiscreteScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler",
            cache_dir=MODEL_CACHE
        )
        case "DDIM": return DDIMScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler",
            cache_dir=MODEL_CACHE
        )
        case "Euler": return EulerDiscreteScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler",
            cache_dir=MODEL_CACHE
        )
        case "Euler_A": return EulerAncestralDiscreteScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler",
            cache_dir=MODEL_CACHE
        )
        case "DPMSolver": return DPMSolverMultistepScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler",
            cache_dir=MODEL_CACHE
        )
        case "Heun": return HeunDiscreteScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler",
            cache_dir=MODEL_CACHE
        )
        case "KDPM2_A": return KDPM2AncestralDiscreteScheduler.from_config(
            MODELS[model_name],
            subfolder="scheduler",
            cache_dir=MODEL_CACHE
        )
