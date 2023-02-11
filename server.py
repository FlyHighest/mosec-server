import logging
from io import BytesIO
from PIL import Image
import json
import torch  # type: ignore
import httpx
from mosec import Server, Worker
from mosec.errors import ValidationError
from models import Text2ImageModel,UpscaleModel,MagicPrompt,SafetyModel,Translator,AestheticModel
from storage.storage_tool import StorageTool
import nanoid 
import string 
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

INFERENCE_BATCH_SIZE = 1


class Preprocess(Worker):
    """Sample Preprocess worker"""
    '''
    Data structure of each type
    Type1: text2image
        - model_name
        - scheduler_name
        - prompt
        - negative_prompt
        - height
        - width
        - num_inference_steps
        - guidance_scale
    
    Type2: image2image

    Type3: upscale
    
    '''

    def __init__(self) -> None:
        super().__init__()

    def prompt_format(self, prompt_str):
        prompt_str = prompt_str.replace("，",",")
        prompt_str = prompt_str.replace("。",",")
        prompt_str = prompt_str.replace("{","(")
        prompt_str = prompt_str.replace("}",")")

        return prompt_str

    def forward(self, data: dict):
        try:
            match data['type']:
                case "text2image":
                    model_name = data['model_name']
                    data['prompt'] = self.prompt_format(data['prompt'])
                    data['negative_prompt'] = self.prompt_format(data['negative_prompt'])
                    image_gen_id = data['gen_id']
                    del data['model_name']
                    del data['type']
                    del data['gen_id']

                    ret = {
                                "type": "text2image",
                                "model_name":model_name, 
                                "pipeline_params": data,
                                "gen_id" : image_gen_id
                            }

                case "upscale":
                    img_bytes = httpx.get(data['img_url']).content
                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    ret = {
                        "type": "upscale",
                        "img": img
                    }
                case "enhanceprompt":
                    ret = {
                        "type": "enhanceprompt",
                        "starting_text": data["starting_text"]
                    }
                case "safety_check":
                    img_bytes = httpx.get(data['img_url']).content
                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    ret = {
                        "type":"safety_check",
                        "img": img
                    }

        except KeyError as err:
            raise ValidationError(f"cannot find key {err}") from err
        except Exception as err:
            raise ValidationError(
                f"error: {err}") from err

        return ret


class Inference(Worker):
    """Sample Inference worker"""

    def __init__(self):
        super().__init__()
        # initialization
        torch.backends.cudnn.benchmark = True
        worker_id = self.worker_id
        self.device = torch.device("cuda:"+str(worker_id))
        logger.info("using worker_id "+str(worker_id))

        # prepare models
        self.text2image_model = Text2ImageModel(worker_id)
        self.upscale_model = UpscaleModel(self.device, worker_id)
        self.prompt_enh_model = MagicPrompt(self.device)
        self.safety_checker = SafetyModel(self.device)
        self.translator = Translator(self.device)
        self.aesthetic_model = AestheticModel(self.device)

    def forward(self, preprocess_data: dict):
        match preprocess_data["type"]:
            case "text2image":
                del preprocess_data["type"]
                preprocess_data["pipeline_params"]['prompt'], preprocess_data["pipeline_params"]['negative_prompt'] = \
                    self.translator.prompt_handle(
                        preprocess_data["pipeline_params"]['prompt'], 
                        preprocess_data["pipeline_params"]['negative_prompt'] 
                    )
                if preprocess_data['model_name']=="OpenJourney" and not preprocess_data["pipeline_params"]['prompt'].startswith("mdjrny-v4 style"):
                    preprocess_data["pipeline_params"]['prompt'] = "mdjrny-v4 style, " + preprocess_data["pipeline_params"]['prompt']
                
                generated_img_path, generated_image = self.text2image_model(preprocess_data['model_name'],preprocess_data["pipeline_params"] )
                
                if self.safety_checker.has_nsfw(generated_image):
                    nsfw = True 
                else:
                    nsfw = False

                score = self.aesthetic_model.get_score(generated_image)
                ret = {
                    "type": "text2image",
                    "img_path" : generated_img_path,
                    "gen_id": preprocess_data['gen_id'],
                    "nsfw": nsfw,
                    "score": score
                }

                
            case "upscale":
                del preprocess_data["type"]
                ret = {
                    "type":"upscale", 
                    "img_path" : self.upscale_model(**preprocess_data)
                }
            case "enhanceprompt":
                starting_text = preprocess_data["starting_text"]
                if self.translator.detect(starting_text) != self.translator.target_flores:
                    enhanced = starting_text
                else:
                    enhanced = self.prompt_enh_model(starting_text=starting_text)
                ret = {
                    "type": "enhanceprompt",
                    "enhanced_text": enhanced
                }
            case "safety_check":
                ret = {
                    "type" : "safety_check",
                    "result" : self.safety_checker(img=preprocess_data['img'])
                }
        return ret


class Postprocess(Worker):
    """Sample Postprocess worker"""

    def __init__(self):
        super().__init__()
        self.storage_tool = StorageTool()

    def forward(self, inference_data):
        match inference_data["type"]:
            case "text2image" :
                img_path = inference_data["img_path"]
                
                if img_path == "Error":
                    return {
                        "img_url": "Error",
                    }
                
                else: 
                    if inference_data['nsfw']:
                        expire = "PT5M"
                    else:
                        expire = None
                    img_url = self.storage_tool.upload(img_path,expire)
                    return {
                        "img_url": img_url,
                        "score":inference_data['score'],
                        "nsfw":inference_data['nsfw']
                    }
            case "upscale":
                img_path = inference_data["img_path"]
                
                if img_path == "Error":
                    return {
                        "img_url": "Error",
                    }
                else: 
                    img_url = self.storage_tool.upload(img_path)
                    return {
                        "img_url": img_url
                    }
            case "enhanceprompt":
                return {
                    "enhanced_text": inference_data["enhanced_text"]
                }
            case "safety_check":
                return {
                    "result": inference_data["result"]
                }

if __name__ == "__main__":
    from gpuinfo import GPUInfo
    num_gpus = len(GPUInfo.gpu_usage()[0])
    print("num gpus ",num_gpus)
    server = Server()
    server.append_worker(Preprocess, num=4)
    server.append_worker(Inference, num=num_gpus,
                         max_batch_size=INFERENCE_BATCH_SIZE)
    server.append_worker(Postprocess, num=4)
    server.run()
