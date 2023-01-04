import sys 
sys.path.append(".")
from qiniu import Auth, put_file, etag
from params.secret import qiniu_access_key_id,qiniu_access_key_secret,qiniu_public_url
import nanoid 
import traceback

class StorageTool:
    def __init__(self) -> None:
        self.q = Auth(qiniu_access_key_id,qiniu_access_key_secret)
        self.nsfw_warning_picture = qiniu_public_url+"rabit-newyear.jpg"
        self.server_error_picture = qiniu_public_url+"rabit-newyear.jpg"

    def upload(self,img_path):
        object_name = nanoid.generate(size=12)+".jpg"
        try:
            token = self.q.upload_token("imagedraw",object_name)
            ret, _ = put_file(token,object_name,img_path,version="v2")
            assert ret['hash'] == etag(img_path)
            return qiniu_public_url+object_name
        except:
            traceback.print_exc()
            print("Error while uploading")
            return ""

if __name__=="__main__":
    storage_tools = StorageTool()
    storage_tools.upload("/Users/zhangtianyu/rabit-newyear.jpg")
        
