import logging
from io import BytesIO
from PIL import Image

import torch  # type: ignore
import httpx
from mosec import Server, Worker
from mosec.errors import ValidationError
from models import Text2ImageModel,UpscaleModel,MagicPrompt
from storage.storage_tool import StorageTool

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
        return prompt_str

    def forward(self, data: dict):
        try:
            match data['type']:
                case "text2image":
                    model_name = data['model_name']
                    data['prompt'] = self.prompt_format(data['prompt'])
                    data['negative_prompt'] = self.prompt_format(data['negative_prompt'])

                    del data['model_name']
                    del data['type']

                    ret = {
                                "type": "text2image",
                                "model_name":model_name, 
                                "pipeline_params": data
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

    def forward(self, preprocess_data: dict):
        match preprocess_data["type"]:
            case "text2image":
                del preprocess_data["type"]
                ret = {
                    "img_path" : self.text2image_model(**preprocess_data)
                }
                
            case "upscale":
                del preprocess_data["type"]
                ret = {
                    "img_path" : self.upscale_model(**preprocess_data)
                }
            case "enhanceprompt":
                ret = {
                    "enhanced_text": self.prompt_enh_model(starting_text=preprocess_data["starting_text"])
                }
        return ret


class Postprocess(Worker):
    """Sample Postprocess worker"""

    def __init__(self):
        super().__init__()
        self.storage_tool = StorageTool()

    def forward(self, inference_data):
        if "img_path" in inference_data:
            img_path = inference_data["img_path"]
            if img_path == "Error":
                return {
                    "img_url": "Error",
                }
            elif img_path == "NSFW":
                return {
                    "img_url": "NSFW",
                }
            else: 
                img_url = self.storage_tool.upload(img_path)
                return {
                    "img_url": img_url,
                }
        else:
            return {
                "enhanced_text": inference_data["enhanced_text"]
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
