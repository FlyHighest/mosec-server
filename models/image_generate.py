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
        self.output_name_webp = f"/tmp/yunjing_id{self.worker_id}.webp"
        self.cn = webuiapi.ControlNetInterface(self.api)
        print(f"build txt2img api in worker {worker_id}, port={MODEL_PORT[worker_id]}")

        self.extra_options = defaultdict(dict)
        self.extra_options["ACertainThing"]= {
                     "override_settings":{"CLIP_stop_at_last_layers":2,
                                          'sd_vae': 'vae-ft-mse-840000-ema-pruned.safetensors'}
                }
        self.extra_options["MeinaMix"]= {
                     "override_settings":{"CLIP_stop_at_last_layers":2}
                }
        self.extra_options["DreamShaper"] = {
                     "override_settings":{"CLIP_stop_at_last_layers":2}
                }
        self.extra_options["YunJingAnime-v1"]= {
                     "override_settings":{'sd_vae': 'vae-ft-mse-840000-ema-pruned.safetensors',
                                         "CLIP_stop_at_last_layers":2}
                }
        self.extra_options["Counterfeit-V2.5"]={ 
            "override_settings":{'sd_vae': 'vae-ft-mse-840000-ema-pruned.safetensors',
                                "CLIP_stop_at_last_layers":2}
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

            print(json_data)
            result = self.cn.txt2img(**json_data)
            
            image = result.image
            image.save(self.output_name, format='jpeg', quality=90)
            image.save(self.output_name_webp,format="webp",quality=90)

            return self.output_name, image

        except:
            traceback.print_exc()
            print("Error while generating with model "+model_name)
            return "Error",None



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
                "denoising_strength": 1 - float(params["i2i_guidance_strength"]),
                "initial_noise_multiplier": 1.0,
            }
            model_name = params["model_name"]
                
            json_data.update(self.extra_options[model_name])

            self.api.util_set_model(model_name)

            result = self.api.img2img(**json_data)
            # print(json_data)
            image = result.image
            image.save(self.output_name, format='jpeg', quality=90)
            image.save(self.output_name_webp,format="webp",quality=90)
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

            self.api.util_set_model(model_name)

            result = self.api.txt2img(**json_data)
            # print(json_data)
            image = result.image
            image.save(self.output_name, format='jpeg', quality=90)
            image.save(self.output_name_webp,format="webp",quality=90)
            return self.output_name, image

        except:
            traceback.print_exc()
            print("Error while generating with model "+model_name)
            print(json_data)
            return "Error",None
