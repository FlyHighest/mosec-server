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
        
  
    def upload(self, img_path,expire=None):
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

if __name__=="__main__":
    storage_tools = StorageTool()
    x = storage_tools.upload("nahida.jpg","PT5M")
    