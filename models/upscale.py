# pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
import torch
from PIL import Image
import numpy as np
import traceback
import webuiapi 
from params.constants import MODEL_PORT
import nanoid ,string

class UpscaleModel:
    def __init__(self,worker_id) -> None:
        self.worker_id = worker_id
        self.api = webuiapi.WebUIApi(port=MODEL_PORT[worker_id])
        self.output_name_webp = "/tmp/yunjing_id"+str(self.worker_id)+"_upscale_{}.webp"

    def __call__(self, img, type="general", factor=2.0):
        try:
            # 传进来一个PIL img，返回一个图像本地路径and PIL image
            if type=="general":
                res = self.api.extra_single_image(image=img,upscaler_1='R-ESRGAN 4x+', upscaling_resize=factor)
            elif type=="anime":
                res = self.api.extra_single_image(image=img,upscaler_1='R-ESRGAN 4x+ Anime6B', upscaling_resize=factor)
            image= res.images[0]
            output_name = self.output_name_webp.format(nanoid.generate(string.ascii_lowercase,6))
            image.save(output_name,format="webp",quality=90)

            return output_name, image
        except:
            traceback.print_exc()
            return "Error",None

