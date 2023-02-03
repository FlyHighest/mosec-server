import traceback
from params.constants import MODEL_URL
import json ,httpx
from PIL import Image 
import base64
import io
class Text2ImageModel:
    '''
    webui json data
    {
  "enable_hr": false,
  "denoising_strength": 0,
  "firstphase_width": 0,
  "firstphase_height": 0,
  "hr_scale": 2,
  "hr_upscaler": "string",
  "hr_second_pass_steps": 0,
  "hr_resize_x": 0,
  "hr_resize_y": 0,
  "prompt": "",
  "styles": [
    "string"
  ],
  "seed": -1,
  "subseed": -1,
  "subseed_strength": 0,
  "seed_resize_from_h": -1,
  "seed_resize_from_w": -1,
  "sampler_name": "string",
  "batch_size": 1,
  "n_iter": 1,
  "steps": 50,
  "cfg_scale": 7,
  "width": 512,
  "height": 512,
  "restore_faces": false,
  "tiling": false,
  "negative_prompt": "string",
  "eta": 0,
  "s_churn": 0,
  "s_tmax": 0,
  "s_tmin": 0,
  "s_noise": 1,
  "override_settings": {},
  "override_settings_restore_afterwards": true,
  "script_args": [],
  "sampler_index": "Euler",
  "script_name": "string"
}
  -  we use:
    prompt
    seed
    sampler_name
    steps
    cfg_scale
    width
    height
    restore_faecs
    negative_prompt

  -  call pass pipeline params:
           
            
                "scheduler_name":      pin['scheduler_name'],
                "prompt":              pin['prompt'],
                "negative_prompt":     pin['negative_prompt'],
                "height":              int(pin['height']),
                "width":               int(pin['width']),
                "num_inference_steps": int(pin['num_inference_steps']),
                "guidance_scale":      int(pin['guidance_scale']),
                "seed":                seed
            }
    '''
    def __init__(self, worker_id) -> None:
        self.worker_id = worker_id
        self.model_url = MODEL_URL[worker_id]
        self.output_name = f"/tmp/output_t2i_id{self.worker_id}.jpeg"

    def __call__(self, model_name,  pipeline_params: dict):
        try:
            # prompt = pipeline_params["prompt"]
            # seed = pipeline_params["seed"]
            # sampler_name = pipeline_params["scheduler_name"]
            # steps = pipeline_params["num_inference_steps"]
            # cfg_scale = pipeline_params["guidance_scale"]
            # width = pipeline_params["width"]
            # height = pipeline_params["height"]
            # restore_faces = False
            # negative_prompt = pipeline_params["negative_prompt"]
            json_data = {
                "prompt" : pipeline_params["prompt"],
                "seed" : pipeline_params["seed"],
                "sampler_name" : pipeline_params["scheduler_name"],
                "steps" : pipeline_params["num_inference_steps"],
                "cfg_scale" : pipeline_params["guidance_scale"],
                "width" : pipeline_params["width"],
                "height" : pipeline_params["height"],
                "restore_faces" : True,
                "negative_prompt" : pipeline_params["negative_prompt"]   
            }
            post_data = json.dumps(json_data)
            prediction = httpx.post(
                self.model_url[model_name],
                data=post_data,
                timeout=40000
            )
            r = json.loads(prediction.content)
            images = []
            if 'images' in r.keys():
                images = [Image.open(io.BytesIO(base64.b64decode(i))) for i in r['images']]
            elif 'image' in r.keys():
                images = [Image.open(io.BytesIO(base64.b64decode(r['image'])))]
            image = images[0]
            image.save(self.output_name, format='jpeg', quality=90)
            return self.output_name

        except:
            traceback.print_exc()
            print("Error while generating")
            return "Error"
