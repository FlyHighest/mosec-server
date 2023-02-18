from lingua import LanguageDetectorBuilder,Language
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import math ,torch
import re 

LANG_TO_FLORES = {
    "CHINESE": "zho_Hans",
    "ENGLISH": "eng_Latn"
}
TRANSLATOR_MODEL_ID = "Helsinki-NLP/opus-mt-zh-en"

target_lang_score_max = 0.9
target_lang = Language.ENGLISH
target_lang_flores = LANG_TO_FLORES[target_lang.name]

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
    def __init__(self,device) -> None:
        self.device = device
        self.detect_language = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
        self.translate_tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL_ID)
        self.translate_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL_ID)
        self.translate_pipeline = pipeline(
                'translation',
                model=self.translate_model,
                tokenizer=self.translate_tokenizer,
                torch_dtype=torch.float16,
                device=self.device
            )
        self.target_flores = target_lang_flores
    
    def prompt_handle(self,prompt,negative_prompt):
        return self.translate_chinese(prompt),self.translate_chinese(negative_prompt)

    def translate_chinese(self,prompt):
        chinese_list, start_list,end_list = get_chinese(prompt)
        if len(chinese_list)==0:
            return prompt
        chinese_to_english_list = self.translate_chinese_list(chinese_list)
        prompt = replace_substrings(prompt,start_list,end_list,chinese_to_english_list)
        return prompt

    
    def translate_chinese_list(self, string_list):
        translate_output = self.translate_pipeline(string_list)
        res = []
        for i in translate_output:
            res.append(i['translation_text'].replace(".",""))
        return res 

    

if __name__=="__main__":
    t = Translator(torch.device("cuda:0"))

    import time 

    s = time.time()
    p = "masterpiece,我爱北京，我爱中国,best quality, 你是聪明人，jpeg artifact，我要画画"
    np = "坏掉的手部，多余的四肢，bad quality"
    print(t.prompt_handle(p,np))
    print(time.time()-s)