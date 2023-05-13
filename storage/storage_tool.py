import sys 
sys.path.append(".")
import httpx
from params.secret import  tencentcloud_secret_id, tencentcloud_secret_key
import nanoid
import traceback
import json 
import os 

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

def is_url_image(image_url):
    r = httpx.head(image_url)
    if  r.headers["content-type"].startswith("image"):
        return True
    return False

def get_domain_key_from_tencent_url(image_url):
    splits = image_url.split("/")
    return "/".join(splits[:3]), "/".join(splits[3:])

client = CosS3Client(CosConfig(Region="ap-shanghai", SecretId=tencentcloud_secret_id, SecretKey=tencentcloud_secret_key, Scheme="https"))
alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'

class StorageTool:


    def __init__(self) -> None:
        pass 
        
  
    def upload(self, img_path,userid=None):
        return self.upload_tencent(img_path,userid)
        
    def upload_tencent(self,img_path,userid):
        response = None
        if os.path.exists(img_path.replace(".jpeg",".webp")):
            img_path = img_path.replace(".jpeg",".webp")
            
        key = f'{userid}/{nanoid.generate(alphabet,8)}.webp'
        for i in range(0, 3): # 尝试三次最多
            try:
                
                response = client.upload_file(
                    Bucket='yunjing-images-1256692038',
                    Key=key,
                    LocalFilePath=img_path,
                    ContentDisposition="attachment"
                )
                break
            except:
                continue
         
        if response is None:
            return ""
        else:
            return client.get_object_url(
                        Bucket='yunjing-images-1256692038',
                        Key=key)
           

    @staticmethod
    def tencent_copy(image_url_temp,userid):
        domain, source_key = get_domain_key_from_tencent_url(image_url_temp)
        key = f'{userid}/{nanoid.generate(alphabet,8)}.webp'
        client.copy(
            Bucket='yunjing-images-1256692038',
            Key=key,
            CopySource={
                'Bucket': 'yunjing-images-1256692038', 
                'Key': source_key, 
                'Region': 'ap-shanghai'
            }
        )
        return os.path.join(domain, key)
    
    @staticmethod
    def tencent_check_nsfw(image_url):
        _ , key = get_domain_key_from_tencent_url(image_url)
        response = client.get_object_sensitive_content_recognition(
            Bucket='yunjing-images-1256692038',
            Key=key,
            BizType="7ae30966d9f89aa719fa2b5ed21074d7"
        )
        res = int(response["Result"])
        if res==0:
            return False
        else:
            return True 

