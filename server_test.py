import os 
os.environ['TRANSFORMERS_CACHE'] = "/root/mosec-server/models-cache"
os.environ['HF_HOME'] = "/root/mosec-server/models-cache"
import logging
from io import BytesIO
from PIL import Image
import json,re
import torch  # type: ignore
import httpx
from mosec import Server, Worker
from mosec.errors import ValidationError


logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


class Preprocess(Worker):
    """Sample Preprocess worker"""

    def __init__(self) -> None:
        super().__init__()

   

    def forward(self, data: dict):
        data["stage"] = "infer"
        data["res"] = "good"
        raise ValidationError("1")
        return data 


class Inference(Worker):
    """Sample Inference worker"""

    def __init__(self):
        super().__init__()
      
        
    def forward(self, preprocess_data: dict):
        preprocess_data["stage"] = "infer"
        return preprocess_data 

if __name__ == "__main__":
    INFERENCE_BATCH_SIZE = 1

    server = Server()
    server.append_worker(Preprocess, num=4)
    server.append_worker(Inference, num=4,
                         max_batch_size=INFERENCE_BATCH_SIZE)
    
    server.run()
