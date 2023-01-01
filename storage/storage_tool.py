import sys 
sys.path.append(".")
import boto3
from params.secret import r2_access_key_id,r2_access_key_secret,r2_account_id,r2_public_url
import nanoid 
import traceback

class StorageTool:
    def __init__(self) -> None:
        self.s3 = boto3.resource('s3',
            endpoint_url = f'https://{r2_account_id}.r2.cloudflarestorage.com',
            aws_access_key_id = r2_access_key_id,
            aws_secret_access_key = r2_access_key_secret
        )

    def upload(self,img_path):
        object_name = nanoid.generate(size=12)+".jpg"
        try:
            self.s3.Object("imagedraw",object_name).upload_file(img_path)
            return r2_public_url+object_name
        except:
            traceback.print_exc()
            print("Error while uploading")
            return ""

if __name__=="__main__":
    storage_tools = StorageTool()
    storage_tools.upload("/Users/dxm/Downloads/nahida.jpg")
        
