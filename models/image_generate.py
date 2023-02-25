import traceback
from params.constants import MODEL_URL,MODEL_PORT
import json ,httpx
from PIL import Image 
import base64
import io
import webuiapi
from collections import defaultdict

class ImageGenerationModel:
   
    def __init__(self, worker_id) -> None:
        self.worker_id = worker_id
        #self.model_url = MODEL_URL[worker_id]
        self.api = webuiapi.WebUIApi(port=MODEL_PORT[worker_id])
        self.output_name = f"/tmp/yunjing_id{self.worker_id}.jpeg"
        self.cn = webuiapi.ControlNetInterface(self.api)
        print(f"build txt2img api in worker {worker_id}, port={MODEL_PORT[worker_id]}")

        self.extra_options = defaultdict(dict)
        self.extra_options["ACertainThing"]= {
                     "override_settings":{"CLIP_stop_at_last_layers":2}
                }
        self.extra_options["YunJingAnime-v1"]= {
                     "enable_hr": True,
                     "hr_scale":  2,
                     "hr_upscaler": webuiapi.HiResUpscaler.Latent,
                     "hr_second_pass_steps": 20,
                     "denoising_strength":0.6,
                     "override_settings":{'sd_vae': 'vae-ft-mse-840000-ema-pruned.safetensors'}
                }
        self.extra_options["Counterfeit-V2.5"]={ 
            "override_settings":{'sd_vae': 'vae-ft-mse-840000-ema-pruned.safetensors'}
                }


    def __call__(self, model_name,  pipeline_params: dict):
        pass 

    def image2image(self, params):
        try:
            json_data = {
                "prompt" : params["prompt"],
                "seed" : params["seed"],
                "sampler_name" : params["scheduler_name"],
                "steps" : params["num_inference_steps"],
                "cfg_scale" : params["guidance_scale"],
                "width" : params["width"],
                "height" : params["height"],
                "restore_faces" : True,
                "negative_prompt" : params["negative_prompt"],
                "images":[params["image"]],
                "denoising_strength":params["i2i_denoising_strength"]
            }
            model_name = params["model_name"]
            
            if model_name=="YunJingAnime-v1":
                json_data["width"]= int(json_data['width'])//2,
                json_data["height"]= int(json_data['height'])//2,
                
                
            json_data.update(self.extra_options[model_name])

            self.api.util_set_model(model_name)

            result = self.api.img2img(**json_data)
            # print(json_data)
            image = result.image
            image.save(self.output_name, format='jpeg', quality=90)
            return self.output_name, image

        except:
            traceback.print_exc()
            print("Error while generating with model "+model_name)
            return "Error",None


    def text2image(self,params):
        try:
            json_data = {
                "prompt" : params["prompt"],
                "seed" : params["seed"],
                "sampler_name" : params["scheduler_name"],
                "steps" : params["num_inference_steps"],
                "cfg_scale" : params["guidance_scale"],
                "width" : params["width"],
                "height" : params["height"],
                "restore_faces" : True,
                "negative_prompt" : params["negative_prompt"]   
            }
            model_name = params["model_name"]
            extra_options = {}
            
            if model_name=="YunJingAnime-v1":
                json_data["width"]= int(json_data['width'])//2,
                json_data["height"]= int(json_data['height'])//2,
                
                
            json_data.update(self.extra_options[model_name])

            self.api.util_set_model(model_name)

            result = self.api.txt2img(**json_data)
            # print(json_data)
            image = result.image
            image.save(self.output_name, format='jpeg', quality=90)
            return self.output_name, image

        except:
            traceback.print_exc()
            print("Error while generating with model "+model_name)
            return "Error",None
