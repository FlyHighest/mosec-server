import logging
from io import BytesIO
from typing import List
from urllib.request import urlretrieve

import numpy as np  # type: ignore
import torch  # type: ignore
import torchvision  # type: ignore
from PIL import Image  # type: ignore
from torchvision import transforms  # type: ignore

from mosec import Server, Worker
from mosec.errors import ValidationError
from mosec.mixin import MsgpackMixin

from models.stable_diffusion_model import StableDiffusionModel
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


class Preprocess(MsgpackMixin, Worker):
    """Sample Preprocess worker"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: dict):
        try:
            prompt = data["prompt"]
        except KeyError as err:
            raise ValidationError(f"cannot find key {err}") from err
        except Exception as err:
            raise ValidationError(f"cannot decode as image data: {err}") from err

        return {
            "prompt": prompt
        }


class Inference(Worker):
    """Sample Inference worker"""

    def __init__(self):
        super().__init__()
        worker_id = self.worker_id() - 1
        self.device = (
            torch.device(f"cuda:{worker_id}") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info("using computing device: %s", self.device)
        self.model = StableDiffusionModel(self.device)

    def forward(self, data:dict):
        logger.info("processing batch with size: %d", len(data))
        prompt = data["prompt"]
        # TODO: and other params
        img_path = self.model(prompt=prompt,model_name="")
        return img_path 


class Postprocess(MsgpackMixin, Worker):
    """Sample Postprocess worker"""

    def __init__(self):
        super().__init__()
        self.storage_tool = StorageTool()
        
    def forward(self, img_path):
        img_url = self.storage_tool.upload(img_path)
        return {
            "img_url": img_url,
        }

        

if __name__ == "__main__":
    server = Server()
    server.append_worker(Preprocess, num=4)
    server.append_worker(Inference, num=2, max_batch_size=INFERENCE_BATCH_SIZE)
    server.append_worker(Postprocess, num=1)
    server.run()
