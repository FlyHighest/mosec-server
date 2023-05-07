import sys 
sys.path.append(".")
import httpx
from params.secret import upload_url,upload_key, tencentcloud_secret_id, tencentcloud_secret_key
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

# 1. 设置用户属性, 包括 secret_id, secret_key, region 等。Appid 已在 CosConfig 中移除，请在参数 Bucket 中带上 Appid。Bucket 由 BucketName-Appid 组成

class StorageTool:
    def __init__(self) -> None:
        self.header = {
                "X-API-Key":upload_key
        }
        region = None              # 通过自定义域名初始化不需要配置 region
        token = None               # 如果使用永久密钥不需要填入 token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见 https://cloud.tencent.com/document/product/436/14048
        scheme = 'https'           # 指定使用 http/https 协议来访问 COS，默认为 https，可不填

        domain = 'images.dong-liu.com' # 用户自定义域名，需要先开启桶的自定义域名，具体请参见 https://cloud.tencent.com/document/product/436/36638
        config = CosConfig(Region=region, SecretId=tencentcloud_secret_id, SecretKey=tencentcloud_secret_key, Token=token, Domain=domain, Scheme=scheme)
        self.client = CosS3Client(config)
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
        self.tencent_url = "https://images.dong-liu.com/"
        
  
    def upload(self, img_path,userid=None):
        return self.upload_tencent(img_path,userid)
        
    def upload_tencent(self,img_path,userid):
        response = None
        if os.path.exists(img_path.replace(".jpeg",".webp")):
            img_path = img_path.replace(".jpeg",".webp")
            
        key = f'{userid}/{nanoid.generate(self.alphabet,8)}.webp'
        for i in range(0, 10):
            try:
                
                response = self.client.upload_file(
                    Bucket='yunjing-images-1256692038',
                    Key=key,
                    LocalFilePath=img_path,
                    ContentDisposition="attachment"
                )
                response["image_url"] = self.tencent_url+key
                break
            except:
                continue
         
        if response is None:
            return ""
        else:
            return response["image_url"]
           

    def upload_self(self,img_path,expire=None):
        try:
            payload = {
                'format': 'json',
                'title': 'user generated image'
            }
            if expire is not None:
                payload['expiration']=expire

            files = [
                ('source', open(img_path,'rb'))
            ]

            res = httpx.post(upload_url,
                timeout=20,
                files=files,headers=self.header,data=payload)
            assert res.status_code==200
            ret = json.loads(res.content.decode('utf-8'))
            return ret['image']["url"]
        except:
            traceback.print_exc()
            print("Error while uploading")
            return ""

    def tencent_copy(self,image_url_temp,userid):
        source_key = image_url_temp[len(self.tencent_url):]
        key = f'{userid}/{nanoid.generate(self.alphabet,8)}.webp'
        self.client.copy(
            Bucket='yunjing-images-1256692038',
            Key=key,
            CopySource={
                'Bucket': 'yunjing-images-1256692038', 
                'Key': source_key, 
                'Region': 'ap-shanghai'
            }
        )
        return self.tencent_url + key
        
    def tencent_check_nsfw(self,image_url):
        key = image_url[len(self.tencent_url):]
        response = self.client.get_object_sensitive_content_recognition(
            Bucket='yunjing-images-1256692038',
            Key=key,
            BizType="7ae30966d9f89aa719fa2b5ed21074d7"
        )
        res = int(response["Result"])
        if res==0:
            return False
        else:
            self.client.delete_object(
                Bucket='yunjing-images-1256692038',
                Key=key
            )
            return True 

if __name__=="__main__":
    storage_tools = StorageTool()
    x = storage_tools.upload("nahida.jpg","PT5M")
    