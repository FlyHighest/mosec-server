import sys 
sys.path.append(".")
import httpx
from params.secret import upload_url,upload_key
import os 
import traceback
import json 

class StorageTool:
    def __init__(self) -> None:
        self.header = {
                "X-API-Key":upload_key
        }
        
  
    def upload(self, img_path):
        try:
            payload = {
                'format': 'json',
                'title': 'user generated image'
            }

            files = [
                ('source', open(img_path,'rb'))
            ]

            res = httpx.post(upload_url,
                timeout=20,
                files=files,headers=self.header,json=payload)
            assert res.status_code==200
            ret = json.loads(res.content.decode('utf-8'))
            return ret['image']["url"]
        except:
            traceback.print_exc()
            print("Error while uploading")
            return ""

if __name__=="__main__":
    storage_tools = StorageTool()
    x = storage_tools.upload("/Users/zhangtianyu/rabit-newyear.jpg","test2")
    print(os.path.join("https://storage.yunjing.gallery",x))
    print(x)
