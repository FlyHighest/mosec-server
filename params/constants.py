MODEL_CACHE = "models-cache"

# Models from huggingface
MODELS = {
    "Taiyi-Chinese-v0.1": "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1",
    "Taiyi-Chinese-Anime-v0.1": "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Anime-Chinese-v0.1",
    "Chinese-style-sd-2-v0.1":"Midu/chinese-style-stable-diffusion-2-v0.1",
    "Stable-Diffusion-2.1": "stabilityai/stable-diffusion-2-1",
    "OpenJourney": "prompthero/openjourney",
    "Anything-v4.5":  "andite/diffuser"
}

MODELS_FP16 = {
    "Taiyi-Chinese-v0.1": MODEL_CACHE+"fp16--Taiyi-Stable-Diffusion-1B-Chinese-v0.1",
    "Taiyi-Chinese-Anime-v0.1": MODEL_CACHE+"fp16--Taiyi-Stable-Diffusion-1B-Anime-Chinese-v0.1",
    "Chinese-style-sd-2-v0.1":MODEL_CACHE+"fp16--chinese-style-stable-diffusion-2-v0.1",
    "Stable-Diffusion-2.1": MODEL_CACHE+"fp16--stable-diffusion-2-1",
    "OpenJourney": MODEL_CACHE+"fp16--openjourney",
    "Anything-v4.5":  MODEL_CACHE+"fp16--anything-v4.5"
}