import json
import datetime
import base64
import hmac
import json
from hashlib import sha256 as sha256
from urllib.request import Request, urlopen
from PIL import Image 
from io import BytesIO
from params.secret import ilivedata_pid,ilivedata_secret

    
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


class SafetyModel:
    def __init__(self) -> None:
        pass 

    def get_nsfw_and_face(self,image,userid="Default"):
        image_base64 = img2base64(image) 
        response = check(image_base64, 2, userid)
        res = json.loads(response)
        nsfw_res = int(res['result'])
        
        face = True if int(res['extraInfo']['numFace'])>0 else False
        return nsfw_res, face




