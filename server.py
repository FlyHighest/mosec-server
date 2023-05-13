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
from models import ImageGenerationModel,UpscaleModel,PromptEnhancer,Translator,SafetyModel,ScoreModel
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


class Preprocess(Worker):
    """Sample Preprocess worker"""

    def __init__(self) -> None:
        super().__init__()
        self.translator = Translator()
        self.prompt_enh_model = PromptEnhancer()

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
                    data['prompt'], data['negative_prompt'] = \
                    self.translator.prompt_handle(
                        data['prompt'], 
                        data['negative_prompt'] 
                    )
                
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
                    starting_text =  data["starting_text"]
                    model_type = data["model_type"]
                    result_text = self.prompt_enh_model(starting_text,model_type)
                    raise ValidationError(result_text) # suggested by mosec maintainer: https://github.com/mosecorg/mosec/discussions/349
                    


        except KeyError as err:
            raise ValidationError(f"cannot find key {err}") from err

        return ret


class Inference(Worker):
    """Sample Inference worker"""

    def __init__(self):
        super().__init__()
        # initialization
        torch.backends.cudnn.benchmark = True
        worker_id = self.worker_id - 1
        self.device = torch.device("cuda:"+str(worker_id))
        logger.info("using worker_id "+str(worker_id))

        # prepare models
        self.image_gen_model = ImageGenerationModel(worker_id)
        # self.upscale_model = UpscaleModel(self.device, worker_id)
        
    def forward(self, preprocess_data: dict):
        match preprocess_data["type"]:
            case "text2image"  | "image2image":
                image_generation_data = preprocess_data["pipeline_params"]

                # 文生图
                if preprocess_data["type"]=="text2image":
                    print("t2i task")
                    generated_img_path, generated_image = self.image_gen_model.text2image(image_generation_data )
                elif image_generation_data['i2i_model']=="原模型":
                    print("i2i task")
                    generated_img_path, generated_image = self.image_gen_model.image2image(image_generation_data )
                elif image_generation_data['i2i_model'].startswith("ControlNet"):
                    print("i2i task controlnet")
                    generated_img_path, generated_image = self.image_gen_model.image2image_controlnet(image_generation_data )
                    
                if 'userid' in image_generation_data:
                    userid = image_generation_data['userid']
                else:
                    userid = "Default"
                    
                # score, nsfw_res, has_face = self.aesthetic_model.get_aes_nsfw_and_face(generated_image,userid)


                ret = {
                    "type": "text2image",
                    "img_path" : generated_img_path,
                    "gen_id": image_generation_data['gen_id'],
                    "prompt": image_generation_data['prompt'],
                    # "nsfw": nsfw_res,
                    # "score": score,
                    # "face":has_face,
                    "userid": userid
                }

                
            case "upscale":
                del preprocess_data["type"]
                ret = {
                    "type":"upscale", 
                    "img_path" : None # self.upscale_model(**preprocess_data)
                }
            case "enhanceprompt":
                starting_text = preprocess_data["starting_text"]
                starting_text = self.translator.translate_chinese(starting_text)
                enhanced = self.prompt_enh_model(starting_text=starting_text)
                ret = {
                    "type": "enhanceprompt",
                    "enhanced_text": enhanced
                }

        return ret


class Postprocess(Worker):
    """Sample Postprocess worker"""

    def __init__(self):
        super().__init__()
        self.storage_tool = StorageTool()
        self.safety_model = SafetyModel()
        self.score_model = ScoreModel(device="cpu")


    def forward(self, inference_data):
        match inference_data["type"]:
            case "text2image"  | "image2image" :
                img_path = inference_data["img_path"]
                userid = inference_data['userid']
                if img_path == "Error":
                    return {
                        "img_url": "Error",
                    }
                
                else: 
                    score = self.score_model.get_score(img_path,inference_data["prompt"])
                    nsfw_ilive_score, face = self.safety_model.get_nsfw_and_face(img_path,userid=inference_data["userid"])
                    if nsfw_ilive_score==2:
                        img_url = self.storage_tool.tencent_copy(img_url,"tmp")
                        nsfw = True
                    elif nsfw_ilive_score==1:
                        img_url = self.storage_tool.upload(img_path,"tmp")
                        nsfw = self.storage_tool.tencent_check_nsfw(img_url)
                        if not nsfw:
                            img_url = self.storage_tool.tencent_copy(img_url,userid)
                    
                    else:
                        nsfw = False
                        img_url = self.storage_tool.upload(img_path,userid)
                        
                    ret = {
                        "img_url": img_url,
                        "score":score,
                        "nsfw":nsfw,
                        "face":face,
                    }
                    return ret 
            case "upscale":
                img_path = "Error" #inference_data["img_path"]
                
                if img_path == "Error":
                    return {
                        "img_url": "Error",
                    }
                else: 
                    img_url = self.storage_tool.upload(img_path,userid="tmp")
                    return {
                        "img_url": img_url
                    }
            case "enhanceprompt":
                return {
                    "enhanced_text": inference_data["enhanced_text"]
                }


if __name__ == "__main__":
    INFERENCE_BATCH_SIZE = 1

    from gpuinfo import GPUInfo
    num_gpus = len(GPUInfo.gpu_usage()[0])
    print("num gpus ",num_gpus)
    server = Server()
    server.append_worker(Preprocess, num=2)
    server.append_worker(Inference, num=num_gpus,
                         max_batch_size=INFERENCE_BATCH_SIZE)
    server.append_worker(Postprocess, num=2)
    server.run()
