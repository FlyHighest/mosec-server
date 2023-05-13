import sys 
sys.path.append(".")
import re 
from params.secret import tencentcloud_secret_id,tencentcloud_secret_key
import json
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tmt.v20180321 import tmt_client, models

def tencent_machine_translate(chinese_list):
    try:
        cred = credential.Credential(tencentcloud_secret_id, tencentcloud_secret_key)

        client = tmt_client.TmtClient(cred, "ap-shanghai")

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.TextTranslateBatchRequest()
        params = {
            "Source": "zh",
            "Target": "en",
            "ProjectId": 0,
            "SourceTextList": chinese_list
        }
        req.from_json_string(json.dumps(params))

        # 返回的resp是一个TextTranslateBatchResponse的实例，与请求对象对应
        resp = client.TextTranslateBatch(req)
        # 输出json格式的字符串回包
        result = json.loads(resp.to_json_string())
        return result["TargetTextList"]
    
    except TencentCloudSDKException as err:
        print(err)
        return chinese_list


def get_chinese(input_str):
    pattern = re.compile(r'[\u4e00-\u9fa5]+')
    string_list, start_list,end_list = [],[],[]
    for match in re.finditer(pattern, input_str):
        string_list.append(match.group())
        start_list.append(match.start())
        end_list.append(match.end())
    return string_list, start_list, end_list 

def replace_substrings(string, start_list,end_list, replacement_list):
    # 将原始字符串转换为列表
    string_list = list(string)
    # 遍历每个要替换的子串
    for i in range(len(start_list)):
        # 获取要替换的子串的起始位置和结束位置
        start = start_list[i]
        end = end_list[i]
        # 获取要替换的字符串
        replacement = replacement_list[i]
        # 将原始字符串中的子串替换为新字符串
        string_list[start:end] = [""]*(end-start)
        string_list[start] = replacement
    # 将列表转换为字符串并返回
    return ''.join(string_list)

class Translator:
    # other languages to english 
    def __init__(self) -> None:
        pass 
    
    def prompt_handle(self,prompt,negative_prompt):
        return self.translate_chinese(prompt),self.translate_chinese(negative_prompt)

    def translate_chinese(self,prompt):
        chinese_list, start_list,end_list = get_chinese(prompt)
        if len(chinese_list)==0:
            return prompt
        chinese_to_english_list = tencent_machine_translate(chinese_list)
        prompt = replace_substrings(prompt,start_list,end_list,chinese_to_english_list)
        return prompt

    

if __name__=="__main__":
    
    chinese_list = ["坏掉的手部","多余的四肢","获奖摄影"]
    tencent_machine_translate(chinese_list)