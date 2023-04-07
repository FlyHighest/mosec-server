# pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
import traceback

class UpscaleModel:
    def __init__(self,device,worker_id) -> None:
        self.model = RealESRGAN(device, scale=2)
        self.model.load_weights('models-cache/RealESRGAN_x2.pth', download=True)

        self.worker_id = worker_id
        self.output_name = f"/tmp/output_ups_id{self.worker_id}.webp"

    def __call__(self, img) -> str:
        try:
            with torch.inference_mode():
                sr_image = self.model.predict(img)

            sr_image.save(self.output_name, format='webp', quality=90)
            return self.output_name
        except:
            traceback.print_exc()
            return "Error"

