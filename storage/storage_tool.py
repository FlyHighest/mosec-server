import sys 
sys.path.append(".")
from qiniu import Auth, put_file, etag, BucketManager
from params.secret import qiniu_access_key_id,qiniu_access_key_secret,qiniu_public_url
import nanoid 
import traceback,time

IMAGE_ID_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

class StorageTool:
    def __init__(self) -> None:
        self.q = Auth(qiniu_access_key_id,qiniu_access_key_secret)
        self.bucket = BucketManager(self.q)
  
    def upload(self,img_path):
        object_name = time.strftime("%Y-%m-%d")+"/"+nanoid.generate(IMAGE_ID_ALPHABET, size=25)+".webp"
        try:
            token = self.q.upload_token("imagedraw",object_name)
            ret, _ = put_file(token,object_name,img_path,version="v2")
            assert ret['hash'] == etag(img_path)
            self.bucket.delete_after_days("imagedraw",object_name, '7')
            return qiniu_public_url+object_name
        except:
            traceback.print_exc()
            print("Error while uploading")
            return ""

if __name__=="__main__":
    storage_tools = StorageTool()
    storage_tools.upload("/Users/zhangtianyu/rabit-newyear.jpg")
        
