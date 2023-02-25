import logging
from io import BytesIO
from PIL import Image
import json,re
import torch  # type: ignore
import httpx
from mosec import Server, Worker
from mosec.errors import ValidationError
from models import ImageGenerationModel,UpscaleModel,MagicPrompt,Translator,AestheticSafetyModel,FaceDetector
from storage.storage_tool import StorageTool
from params.constants import EXTRA_MODEL_LORA
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

INFERENCE_BATCH_SIZE = 1


class Preprocess(Worker):
    """Sample Preprocess worker"""

    def __init__(self) -> None:
        super().__init__()

    def prompt_format(self, prompt_str):
        prompt_str = re.sub(r"[\u3000-\u303F\uFF00-\uFFEF]",",",prompt_str)
        prompt_str = prompt_str.replace("{","(")
        prompt_str = prompt_str.replace("}",")")

        return prompt_str

    def forward(self, data: dict):
        # data is text2image_data in Yunjing remote_tasks.py
        try:
            match data['type']:
                case "text2image"  | "image2image":

                    # prompt preprocess
                    data['prompt'] = self.prompt_format(data['prompt'])
                    data['negative_prompt'] = self.prompt_format(data['negative_prompt'])
                    
                    if 'extra_model_name' in data and data['extra_model_name'] in EXTRA_MODEL_LORA:
                        data['prompt'] += EXTRA_MODEL_LORA[data['extra_model_name']]
                       
                    if data['model_name']=="OpenJourney" and not data['prompt'].startswith("mdjrny-v4 style"):
                        data['prompt'] = "mdjrny-v4 style, " + data['prompt']

                    if 'i2i_url' in data:
                        img_bytes = httpx.get(data['i2i_url']).content
                        img = Image.open(BytesIO(img_bytes)).convert("RGB")
                        data["image"] = img

                    ret = {
                                "type": data['type'],
                                "pipeline_params": data,
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
                case "face_detect":
                    img_bytes = httpx.get(data['img_url']).content
                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    ret = {
                        "type":"face_detect",
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
        self.image_gen_model = ImageGenerationModel(worker_id)
        self.upscale_model = UpscaleModel(self.device, worker_id)
        self.prompt_enh_model = MagicPrompt(self.device)
        self.translator = Translator(self.device)
        self.aesthetic_model = AestheticSafetyModel(self.device)
        self.face_detector = FaceDetector(self.device)
        
    def forward(self, preprocess_data: dict):
        match preprocess_data["type"]:
            case "text2image"  | "image2image":
                # 翻译中文
                image_generation_data = preprocess_data["pipeline_params"]

                image_generation_data['prompt'], image_generation_data['negative_prompt'] = \
                    self.translator.prompt_handle(
                        image_generation_data['prompt'], 
                        image_generation_data['negative_prompt'] 
                    )
                
                # 文生图
                if preprocess_data["type"]=="text2image":
                    generated_img_path, generated_image = self.image_gen_model.text2image(image_generation_data )
                elif image_generation_data['i2i_model']=="原模型":
                    generated_img_path, generated_image = self.image_gen_model.image2image(image_generation_data )


                
                score,nsfw_prob = self.aesthetic_model.get_aes_and_nsfw(generated_image)
                if nsfw_prob > 0.6:
                    nsfw = True 
                else:
                    nsfw = False
                has_face = self.face_detector.detect(generated_image)
                ret = {
                    "type": "text2image",
                    "img_path" : generated_img_path,
                    "gen_id": image_generation_data['gen_id'],
                    "nsfw": nsfw,
                    "score": score,
                    "face":has_face
                }

                
            case "upscale":
                del preprocess_data["type"]
                ret = {
                    "type":"upscale", 
                    "img_path" : self.upscale_model(**preprocess_data)
                }
            case "enhanceprompt":
                starting_text = preprocess_data["starting_text"]
                starting_text = self.translator.translate_chinese(starting_text)
                enhanced = self.prompt_enh_model(starting_text=starting_text)
                ret = {
                    "type": "enhanceprompt",
                    "enhanced_text": enhanced
                }
            case "safety_check":
                score,nsfw_prob = self.aesthetic_model.get_aes_and_nsfw(preprocess_data['img'])
                if nsfw_prob > 0.6:
                    nsfw = True 
                else:
                    nsfw = False
                ret = {
                    "type" : "safety_check",
                    "result" : "NSFW" if nsfw else "OK"
                }
            case "face_detect":
                has_face = self.face_detector.detect(preprocess_data['img'])
                ret = {
                    "type":"face_detec",
                    "face":has_face
                }
        return ret


class Postprocess(Worker):
    """Sample Postprocess worker"""

    def __init__(self):
        super().__init__()
        self.storage_tool = StorageTool()

    def forward(self, inference_data):
        match inference_data["type"]:
            case "text2image"  | "image2image" :
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
                    ret = {
                        "img_url": img_url,
                        "score":inference_data['score'],
                        "nsfw":inference_data['nsfw'],
                        "face":inference_data['face']
                    }
                    return ret 
            case "upscale":
                img_path = inference_data["img_path"]
                
                if img_path == "Error":
                    return {
                        "img_url": "Error",
                    }
                else: 
                    img_url = self.storage_tool.upload(img_path,expire="PT5M")
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
            case "face_detect":
                return {
                    "face":inference_data["face"]
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
