# MODEL_URL ={
#     0: {
#     "Anything-v3":           "http://localhost:6002/sdapi/v1/txt2img",
#     "ACertainThing":         "http://localhost:6003/sdapi/v1/txt2img",
#     "OpenJourney":           "http://localhost:6004/sdapi/v1/txt2img",
#     "Protogen-x5.8":         "http://localhost:6005/sdapi/v1/txt2img",
#     "Stable-Diffusion-v1.5": "http://localhost:6001/sdapi/v1/txt2img",
#     "RealisticVision-V1.3": "http://localhost:6000/sdapi/v1/txt2img"
#    }
# }
MODEL_URL = {
  0:"http://localhost:8000/sdapi/v1/txt2img",
}
MODEL_PORT={
    0:8000,
}
EXTRA_MODEL_LORA = {
  "LORA-KoreanDollLikeness":",<lora:koreandollv10:0.66>",
  "LORA-国风汉服少女":"<lora:hanfugirlv15:0.66>",
  "LORA-国风汉服少女仿明风格":"<lora:hanfu2ming:0.66>",
  "LORA-国风汉服少女仿宋风格":"<lora:hanfu2song:0.66>",
}