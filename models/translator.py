from lingua import LanguageDetectorBuilder,Language
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import math ,torch

LANG_TO_FLORES = {
    "ARABIC": "arb_Arab",
    "CHINESE": "zho_Hans",
    "ENGLISH": "eng_Latn",
    "FRENCH": "fra_Latn",
    "JAPANESE": "jpn_Jpan",
    "KOREAN": "kor_Hang",
    "RUSSIAN": "rus_Cyrl",
    "SPANISH": "spa_Latn",
}
TRANSLATOR_MODEL_ID = "facebook/nllb-200-distilled-600M"

target_lang_score_max = 0.9
target_lang = Language.ENGLISH
target_lang_flores = LANG_TO_FLORES[target_lang.name]

class Translator:
    # other languages to english 
    def __init__(self,device) -> None:
        self.device = device
        self.detect_language = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
        self.translate_tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL_ID)
        self.translate_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL_ID)
        self.target_flores = target_lang_flores
    
    def prompt_handle(self,prompt,negative_prompt):
        fl1 = self.detect(prompt)
        fl2 = self.detect(negative_prompt)
        res1 = self.translate(prompt, fl1)
        res2 = self.translate(negative_prompt, fl2)
        return res1, res2

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
    
    def translate(self,text, text_lang_flores):
        if text_lang_flores == target_lang_flores:
            return text 
        else:
            translate_pipeline = pipeline(
                'translation',
                model=self.translate_model,
                tokenizer=self.translate_tokenizer,
                torch_dtype=torch.float16,
                src_lang=text_lang_flores,
                tgt_lang=target_lang_flores,
                device=self.device
            )

        translate_output = translate_pipeline(text, max_length=500)
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