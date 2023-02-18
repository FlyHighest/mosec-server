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
        chinese_list, start_list,end_list = get_chinese(prompt)
        chinese_to_english_list = self.translate_chinese_list(chinese_list)
        prompt = replace_substrings(prompt,start_list,end_list,chinese_to_english_list)

        chinese_list, start_list,end_list = get_chinese(negative_prompt)
        chinese_to_english_list = self.translate_chinese_list(chinese_list)
        negative_prompt = replace_substrings(negative_prompt,start_list,end_list,chinese_to_english_list)

        return prompt, negative_prompt

    def detect(self,text):
        confidence_values = self.detect_language.compute_language_confidence_values(text)
        target_lang_score = -math.inf
        detected_lang = None
        
        for index in range(len(confidence_values[:10])):
            curr = confidence_values[index]
            if index == 0:
                detected_lang = curr[0]
            if curr[0] == target_lang:
                target_lang_score = curr[1]
                break 
                
        if detected_lang != target_lang and (target_lang_score < target_lang_score_max) and LANG_TO_FLORES.get(detected_lang.name) is not None:
            text_lang_flores = LANG_TO_FLORES[detected_lang.name]
        else:
            text_lang_flores = target_lang_flores
        return text_lang_flores
    
    def translate_chinese_list(self, string_list):
        translate_output = self.translate_pipeline(string_list)
        res = []
        for i in translate_output:
            res.append(i['translation_text'])
        return res 

    def translate(self,text, text_lang_flores):
        if text_lang_flores == target_lang_flores:
            return text 
        else:
            translate_output = self.translate_pipeline(text)
            translated_text = translate_output[0]['translation_text']
            return translated_text
    

if __name__=="__main__":
    t = Translator(torch.device("cuda:0"))

    import time 

    s = time.time()
    text = "我爱北京，我爱中国"
    text_lang_flores = t.detect(text)
    text_output = t.translate(text,text_lang_flores)
    print(text_output)
    print(time.time()-s)

    s = time.time()
    text = "今天是植树节。小猴想去野外种树。它好不容易才从森林里找了一棵小苹果树苗。小猴别提有多高兴了。小猴开始挖坑了。它挖了很长时间，可还是不合适。因为它挖的坑不是太小了，就是太浅了。小猴很着急。"
    text_lang_flores = t.detect(text)
    text_output = t.translate(text,text_lang_flores)
    print(text_output)
    print(time.time()-s)