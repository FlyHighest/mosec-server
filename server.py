import logging
from io import BytesIO
from typing import List
from urllib.request import urlretrieve

import torch  # type: ignore

from mosec import Server, Worker
from mosec.errors import ValidationError

from models.text2image import Text2ImageModel
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

    def forward(self, data: dict):
        try:
            match data['type']:
                case "text2image":
                    model_name = data['model_name']
                    scheduler_name = data['scheduler_name']
                    seed = int(data['seed'])
                    data['prompt'] = data['prompt'].replace("，",",")
                    data['negative_prompt'] = data['negative_prompt'].replace("，",",")

                    del data['seed']
                    del data['model_name']
                    del data['scheduler_name']
                    del data['type']

        except KeyError as err:
            raise ValidationError(f"cannot find key {err}") from err
        except Exception as err:
            raise ValidationError(
                f"error: {err}") from err

        return {
            "model_name":model_name, 
            "scheduler_name":scheduler_name,
            "seed": seed,
            "pipeline_params": data
        }


class Inference(Worker):
    """Sample Inference worker"""

    def __init__(self):
        super().__init__()
        worker_id = self.worker_id - 1
        self.device = (
            torch.device(f"cuda:{worker_id}") if torch.cuda.is_available(
            ) else torch.device("cpu")
        )
        logger.info("using computing device: %s", self.device)
        self.text2image = Text2ImageModel(self.device, worker_id)

    def forward(self, preprocess_data: dict):

        img_path = self.text2image(**preprocess_data)
        return img_path


class Postprocess(Worker):
    """Sample Postprocess worker"""

    def __init__(self):
        super().__init__()
        self.storage_tool = StorageTool()

    def forward(self, img_path):
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
        


if __name__ == "__main__":
    from gpuinfo import GPUInfo
    num_gpus = len(GPUInfo.gpu_usage()[0])
    
    server = Server()
    server.append_worker(Preprocess, num=4)
    server.append_worker(Inference, num=num_gpus,
                         max_batch_size=INFERENCE_BATCH_SIZE)
    server.append_worker(Postprocess, num=4)
    server.run()
