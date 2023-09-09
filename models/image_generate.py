import traceback
from params.constants import MODEL_URL,MODEL_PORT
import json ,httpx
from PIL import Image 
import base64
import io
import webuiapi
from collections import defaultdict
import nanoid ,string
import time

def retry_on_error(func):
    def wrapper(*args, **kwargs):
        i = 0
        while i<8:
            i += 1
            try:
                result = func(*args, **kwargs)
                if result[0] == "Error":
                    raise Exception("Function failed")
                return result
            except:
                print(f"retry on error {i}")
                
                time.sleep(0.1)  
        return "Error",None
    return wrapper

class ImageGenerationModel:
   
    def __init__(self, worker_id) -> None:
        self.worker_id = worker_id
        #self.model_url = MODEL_URL[worker_id]
        self.api = webuiapi.WebUIApi(port=MODEL_PORT[worker_id])
        # print("start api connection on port",MODEL_PORT[worker_id])
        self.output_name_webp = "/tmp/yunjing_id"+str(worker_id)+"_{}.webp"
        self.cn = webuiapi.ControlNetInterface(self.api)
        print(f"build txt2img api in worker {worker_id}, port={MODEL_PORT[worker_id]}")
        extra_options_vae_ft_mse = [
            'A-ZovyaRPGArtistTools-v3',
            "ACertainThing",
            "YunJingAnime-v1",
             'GuoFeng-v2',
             'GuoFeng-v3.3',
             'GuoFengRealMix',
            "BreakDomainRealistic"
        ]
        extra_options_vae_kl = [
            "Counterfeit-v2.5",
            "Counterfeit-v3",
            "GhostMix-v2",
            "ReVAnimated-v1.2.2"
        ]
        extra_options_clip_skip2 = [
            "ACertainThing",
            'MeinaMix-v10',
            'MeinaMix-v8',
            'DreamShaper-v4',
            'DreamShaper-v6',
            "YunJingAnime-v1",
            "Counterfeit-v2.5",
            "Counterfeit-v3",
            'NeverEndingDream-v1.22'
        ]
        self.extra_options = defaultdict(dict)
        for model_name in extra_options_vae_ft_mse:
            self.extra_options[model_name] = {
                 "override_settings":{
                     'sd_vae': 'vae-ft-mse-840000-ema-pruned.safetensors'
                 }
            }
        for model_name in extra_options_vae_kl:
            self.extra_options[model_name] = {
                 "override_settings":{
                     'sd_vae': 'klF8Anime2VAE.ckpt'
                 }
            }
        for model_name in extra_options_clip_skip2:
            if model_name in self.extra_options:
                self.extra_options[model_name]["override_settings"]['CLIP_stop_at_last_layers'] =2 
            else:
                self.extra_options[model_name]= {
                     "override_settings":{"CLIP_stop_at_last_layers":2}
                }
        
        
        self.i2i_preprocess_map = {
            "原图":"none",
            "边缘提取(Canny)":"canny",
            "边缘提取(HED)":"hed",
            "线段提取":"mlsd",
            "草图提取":"scribble_hed",
            "人体姿态估计":"openpose_full",
            "语义分割":"segmentation",
            "深度估计":"depth",
            "法线贴图估计":"normal_map",
            "线稿提取":"lineart",
            "线稿提取(动漫)":"lineart_anime",
        }
        self.i2i_model_map = {
            "原模型":"None",
            "ControlNet-Canny": 'control_v11p_sd15_canny [d14c016b]',
            "ControlNet-深度图":'control_v11f1p_sd15_depth [cfd03158]',
            "ControlNet-软边缘":'control_v11p_sd15_softedge [a8575a2a]',
            "ControlNet-线段":'control_v11p_sd15_mlsd [aca30ff0]',
            "ControlNet-法线贴图":'control_v11p_sd15_normalbae [316696f1]',
            "ControlNet-人体姿态":'control_v11p_sd15_openpose [cab727d4]',
            "ControlNet-草图":'control_v11p_sd15_scribble [d4ba51ff]',
            "ControlNet-语义分割":'control_v11p_sd15_seg [e1f51eb9]',
            "ControlNet-Tile": 'control_v11f1e_sd15_tile [a371b31b]',
            "ControlNet-线稿": 'control_v11p_sd15_lineart [43d4be0d]',
            "ControlNet-线稿(动漫)": 'control_v11p_sd15s2_lineart_anime [3825e83e]',
        }
#



    def __call__(self, model_name,  pipeline_params: dict):
        pass 
    
    @retry_on_error
    def image2image_controlnet(self, params):
        try:
            json_data = {
                "prompt" : params["prompt"],
                "seed" : params["seed"],
                "sampler_index" : params["scheduler_name"],
                "steps" : params["num_inference_steps"],
                "cfg_scale" : params["guidance_scale"],
                "width" : params["width"],
                "height" : params["height"],
                "restore_faces" : False,
                "negative_prompt" : params["negative_prompt"],
            }
            control_unit_data = {
                "input_image":params["image"],
                "weight":float(params["i2i_guidance_strength"]),
                "module": self.i2i_preprocess_map[params['i2i_preprocess']],
                "model": self.i2i_model_map[params['i2i_model']],
                "control_mode": 0
            }
            model_name = params["model_name"]
    
            if params["hiresfix"]=="On":
                json_data["width"]= int(json_data['width']//2)
                json_data["height"]= int(json_data['height']//2)
                json_data.update({
                     "enable_hr": True,
                     "hr_scale":  2,
                     "hr_upscaler": webuiapi.HiResUpscaler.Latent,
                    "hr_second_pass_steps": 10,
                })
                
                
            json_data.update(self.extra_options[model_name])

            self.api.util_set_model(model_name)
            unit1 = webuiapi.ControlNetUnit(**control_unit_data)

            result = self.api.txt2img(controlnet_units=[unit1],**json_data)
            
            image = result.image
            output_name = self.output_name_webp.format(nanoid.generate(string.ascii_lowercase,6))
            image.save(output_name,format="webp",quality=90)

            return output_name, image

        except:
            traceback.print_exc()
            return "Error",None


    @retry_on_error
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
                "restore_faces" : False,
                "negative_prompt" : params["negative_prompt"],
                "images":[params["image"]],
                "denoising_strength": 1 - float(params["i2i_guidance_strength"]),
                "initial_noise_multiplier": 1.0,
            }
            model_name = params["model_name"]
                
            json_data.update(self.extra_options[model_name])

            self.api.util_set_model(model_name)

            result = self.api.img2img(**json_data)
            # print(json_data)
            image = result.image
            output_name = self.output_name_webp.format(nanoid.generate(string.ascii_lowercase,6))
            image.save(output_name,format="webp",quality=90)

            return output_name, image

        except:
            return "Error",None

    @retry_on_error
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
                "restore_faces" : False,
                "negative_prompt" : params["negative_prompt"]   
            }
            model_name = params["model_name"]
            
            if params["hiresfix"]=="On":
                json_data["width"]= int(json_data['width']//2)
                json_data["height"]= int(json_data['height']//2)
                json_data.update({
                     "enable_hr": True,
                     "hr_scale":  2,
                     "hr_upscaler": webuiapi.HiResUpscaler.Latent,
                     "hr_second_pass_steps": 10,
                })
                
                
                
            json_data.update(self.extra_options[model_name])
            print(model_name,self.extra_options[model_name])
            
            self.api.util_set_model(model_name)
            self.api.has_controlnet=True
            result = self.api.txt2img(**json_data)
            # print(json_data)
            image = result.image
            output_name = self.output_name_webp.format(nanoid.generate(string.ascii_lowercase,6))
            image.save(output_name,format="webp",quality=90)

            return output_name, image
        except:
            return "Error",None
