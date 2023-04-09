import torch.nn as nn 
import torch 
import clip 
import os , json
import onnxruntime
import numpy as np 
import datetime
import base64
import hmac
import json
from hashlib import sha256 as sha256
from urllib.request import Request, urlopen
from PIL import Image 
from io import BytesIO
from params.secret import ilivedata_pid,ilivedata_secret
def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    cache_folder = "models-cache"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

def load_safety_model():
    m = onnxruntime.InferenceSession("models-cache/clip_bin_nsfw.onnx",providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    return m
    
endpoint_host = 'isafe.ilivedata.com'
endpoint_path = '/api/v1/image/check'
endpoint_url = 'https://isafe.ilivedata.com/api/v1/image/check'

def img2base64(img):
    if type(img)==str:
        image = Image.open(img)
    else:
        image = img
    byte_data = BytesIO()# 创建一个字节流管道
    image.save(byte_data, format="webp")# 将图片数据存入字节流管道
    byte_data = byte_data.getvalue()# 从字节流管道中获取二进制
    base64_str = base64.b64encode(byte_data).decode("ascii")# 二进制转base64
    return base64_str


def check(image, type, user_id):
    now_date = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    params = {
        "type": type,
        "strategyId":"001",
        "userId": str(user_id),
        "image": image
    }
    query_body = json.dumps(params)

    parameter = "POST\n"
    parameter += endpoint_host + "\n"
    parameter += endpoint_path + '\n'
    parameter += sha256(query_body.encode('utf-8')).hexdigest() + "\n"
    parameter += "X-AppId:" + ilivedata_pid + "\n"
    parameter += "X-TimeStamp:" + now_date

    signature = base64.b64encode(
        hmac.new(ilivedata_secret, parameter.encode('utf-8'), digestmod=sha256).digest())
    return send(query_body, signature, now_date)


def send(querystring, signature, time_stamp):
    headers = {
        "X-AppId": ilivedata_pid,
        "X-TimeStamp": time_stamp,
        "Content-type": "application/json",
        "Authorization": signature,
        "Host": endpoint_host,
        "Connection": "keep-alive"
    }

    # querystring = parse.urlencode(params)
    req = Request(endpoint_url, querystring.encode(
        'utf-8'), headers=headers, method='POST')
    return urlopen(req).read().decode()


class AestheticSafetyModel:
    def __init__(self,device) -> None:
        self.device = device 
        self.clip_model,self.clip_preprocess = clip.load("ViT-L/14",device=device,download_root="models-cache")

        self.aesthetic_model = get_aesthetic_model()
        self.aesthetic_model.to(device)
        
        self.safety_model = load_safety_model()
        
        
    def get_aes_and_nsfw(self,img):
        img = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_features = self.clip_model.encode_image(img)
            img_features /= img_features.norm(dim=-1,keepdim=True)
            score = self.aesthetic_model(img_features.type(torch.cuda.FloatTensor))
            score = score.item()
        output = self.safety_model.run(None, {'input_1': img_features.cpu().numpy().astype(np.float64)})
        nsfw = output[0][0][0]
        return score , nsfw
    
    def get_aes_nsfw_and_face(self,image,userid="Default"):
        img = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_features = self.clip_model.encode_image(img)
            img_features /= img_features.norm(dim=-1,keepdim=True)
            score = self.aesthetic_model(img_features.type(torch.cuda.FloatTensor))
            score = score.item()
        image_base64 = img2base64(image) 
        response = check(image_base64, 2, userid)
        res = json.loads(response)
        nsfw_res = int(res['result'])
        
        # nsfw = True if int(res['result'])==2 else False
        face = True if int(res['extraInfo']['numFace'])>0 else False
        return score, nsfw_res, face

if __name__=="__main__":
    from PIL import Image 
    import glob
    import time 
    aesthetic_model = AestheticSafetyModel(torch.device("cuda:0"))
    start = time.time()
    for img_path in glob.glob("testimage/*g"):
        img = Image.open(img_path)
        print(img_path)
        print(aesthetic_model.get_aes_and_nsfw(img))
    end = time.time()
    print(end-start)


