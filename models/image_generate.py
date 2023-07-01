import traceback
from params.constants import MODEL_URL,MODEL_PORT
import json ,httpx
from PIL import Image 
import base64
import io
import webuiapi
from collections import defaultdict
import nanoid ,string


def retry_on_error(func):
    def wrapper(*args, **kwargs):
        i = 0
        while i<5:
            i += 1
            try:
                result = func(*args, **kwargs)
                if result[0] == "Error":
                    raise Exception("Function failed")
                return result
            except:
                time.sleep(0.1)  

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
            "草图提取":"fake_scribble",
            "人体姿态估计":"openpose",
            "语义分割":"segmentation",
            "深度估计":"depth",
            "法线贴图估计":"normal_map"
        }
        self.i2i_model_map = {
            "原模型":"None",
            "ControlNet-Canny": 'control_sd15_canny [e3fe7712]',
            "ControlNet-深度图":'control_sd15_depth [400750f6]',
            "ControlNet-HED":'control_sd15_hed [13fee50b]',
            "ControlNet-线段":'control_sd15_mlsd [e3705cfa]',
            "ControlNet-法线贴图":'control_sd15_normal [63f96f7c]',
            "ControlNet-人体姿态":'control_sd15_openpose [9ca67cc5]',
            "ControlNet-草图":'control_sd15_scribble [c508311e]',
            "ControlNet-语义分割":'control_sd15_seg [b9c1cc12]'
        }


    def __call__(self, model_name,  pipeline_params: dict):
        pass 
    
    @retry_on_error
    def image2image_controlnet(self, params):
        # 实际上调用的是txt2img接口，把引导图看成是图到图
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

                "controlnet_input_image":[params["image"]],
                "controlnet_weight":params["i2i_guidance_strength"],
                "controlnet_module": self.i2i_preprocess_map[params['i2i_preprocess']],
                "controlnet_model": self.i2i_model_map[params['i2i_model']],
                "guess_mode":False
            }
            model_name = params["model_name"]
    
            if params["hiresfix"]=="On":
                json_data["width"]= int(json_data['width']//2)
                json_data["height"]= int(json_data['height']//2)
                json_data.update({
                     "enable_hr": True,
                     "hr_scale":  2,
                     "hr_upscale": webuiapi.HiResUpscaler.Latent,
                     "denoising_strength":0.6
                })
                
                
            json_data.update(self.extra_options[model_name])

            self.api.util_set_model(model_name)

            # print(json_data)
            result = self.cn.txt2img(**json_data)
            
            image = result.image
            output_name = self.output_name_webp.format(nanoid.generate(string.ascii_lowercase,6))
            image.save(output_name,format="webp",quality=90)

            return output_name, image

        except:
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
                     "denoising_strength":0.6
                })
                
                
                
            json_data.update(self.extra_options[model_name])
            print(model_name,self.extra_options[model_name])
            
            self.api.util_set_model(model_name)

            result = self.api.txt2img(**json_data)
            # print(json_data)
            image = result.image
            output_name = self.output_name_webp.format(nanoid.generate(string.ascii_lowercase,6))
            image.save(output_name,format="webp",quality=90)

            return output_name, image
        except:
            return "Error",None
