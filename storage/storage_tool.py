import sys 
sys.path.append(".")
import boto3
from params.secret import r2_access_key_id,r2_access_key_secret,r2_account_id
import os 
import traceback


class StorageTool:
    def __init__(self) -> None:
        self.s3 = boto3.resource('s3',
            endpoint_url = f'https://{r2_account_id}.r2.cloudflarestorage.com',
            aws_access_key_id = r2_access_key_id,
            aws_secret_access_key = r2_access_key_secret
        )
  
    def upload(self,img_path,gen_id):
        object_name = gen_id+".webp"
        try:
            self.s3.Object("imagedraw",object_name).upload_file(img_path)
            return object_name
        except:
            traceback.print_exc()
            print("Error while uploading")
            return ""

if __name__=="__main__":
    storage_tools = StorageTool()
    x = storage_tools.upload("/Users/zhangtianyu/rabit-newyear.jpg","test2")
    print(os.path.join("https://storage.yunjing.gallery",x))
    print(x)
