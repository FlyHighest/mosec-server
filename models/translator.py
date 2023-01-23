from lingua import LanguageDetectorBuilder,Language
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import math ,torch

LANG_TO_FLORES = {
    "AFRIKAANS": "afr_Latn",
    "ALBANIAN": "als_Latn",
    "ARABIC": "arb_Arab",
    "ARMENIAN": "hye_Armn",
    "AZERBAIJANI": "azj_Latn",
    "BASQUE": "eus_Latn",
    "BELARUSIAN": "bel_Cyrl",
    "BENGALI": "ben_Beng",
    "BOKMAL": "nob_Latn",
    "BOSNIAN": "bos_Latn",
    "CATALAN": "cat_Latn",
    "CHINESE": "zho_Hans",
    "CROATIAN": "hrv_Latn",
    "CZECH": "ces_Latn",
    "DANISH": "dan_Latn",
    "DUTCH": "nld_Latn",
    "ENGLISH": "eng_Latn",
    "ESPERANTO": "epo_Latn",
    "ESTONIAN": "est_Latn",
    "FINNISH": "fin_Latn",
    "FRENCH": "fra_Latn",
    "GANDA": "lug_Latn",
    "GEORGIAN": "kat_Geor",
    "GERMAN": "deu_Latn",
    "GREEK": "ell_Grek",
    "GUJARATI": "guj_Gujr",
    "HEBREW": "heb_Hebr",
    "HINDI": "hin_Deva",
    "HUNGARIAN": "hun_Latn",
    "ICELANDIC": "isl_Latn",
    "INDONESIAN": "ind_Latn",
    "IRISH": "gle_Latn",
    "ITALIAN": "ita_Latn",
    "JAPANESE": "jpn_Jpan",
    "KAZAKH": "kaz_Cyrl",
    "KOREAN": "kor_Hang",
    "LATVIAN": "lvs_Latn",
    "LITHUANIAN": "lit_Latn",
    "MACEDONIAN": "mkd_Cyrl",
    "MALAY": "zsm_Latn",
    "MAORI": "mri_Latn",
    "MARATHI": "mar_Deva",
    "MONGOLIAN": "khk_Cyrl",
    "NYNORSK": "nno_Latn",
    "PERSIAN": "pes_Arab",
    "POLISH": "pol_Latn",
    "PORTUGUESE": "por_Latn",
    "PUNJABI": "pan_Guru",
    "ROMANIAN": "ron_Latn",
    "RUSSIAN": "rus_Cyrl",
    "SERBIAN": "srp_Cyrl",
    "SHONA": "sna_Latn",
    "SLOVAK": "slk_Latn",
    "SLOVENE": "slv_Latn",
    "SOMALI": "som_Latn",
    "SOTHO": "nso_Latn",
    "SPANISH": "spa_Latn",
    "SWAHILI": "swh_Latn",
    "SWEDISH": "swe_Latn",
    "TAGALOG": "tgl_Latn",
    "TAMIL": "tam_Taml",
    "TELUGU": "tel_Telu",
    "THAI": "tha_Thai",
    "TSONGA": "tso_Latn",
    "TURKISH": "tur_Latn",
    "UKRAINIAN": "ukr_Cyrl",
    "URDU": "urd_Arab",
    "VIETNAMESE": "vie_Latn",
    "XHOSA": "xho_Latn",
    "YORUBA": "yor_Latn",
    "ZULU": "zul_Latn",
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

        translate_output = translate_pipeline(text, max_length=1000)
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
    text = """今天是植树节。小猴想去野外种树。它好不容易才从森林里找了一棵小苹果树苗。小猴别提有多高兴了。
小猴开始挖坑了。它挖了很长时间，可还是不合适。因为它挖的坑不是太小了，就是太浅了。小猴很着急。
这时候，小兔刚好从这里经过。小兔说：“我来帮你挖坑吧。”小兔三下五除二地就把坑给挖好了。小猴说：“现在让我来栽树苗吧。”
很快，树苗就栽好了。
小猴说：“还有一件事没有办到。”小兔说：“是什么事呀？”
小猴说：“是浇水呀。”
小猴和小兔开始找水了。它们好不容易才找到一条小河，但是，找不到盛水的东西，这可怎么办呢。突然，小猴和小兔听到大大的脚步声，它俩吓得撒腿就想跑。可是由于太紧张，跑了没多远，它俩都跑不动了。这时候，它俩用手遮住眼睛，爬在地上一动不动。
“你们两个小家伙这是在干嘛呢，跑什么呀？”小猴和小兔睁开眼睛一看，原来是大象伯伯在说话。它俩这才松了一口气。
小猴说：“我俩在种树，可是没水，这不，刚刚才找到了小河的水。”
大象伯伯说：“哪不是挺好吗？”
小兔说：“可我们没有盛水的东西。”
大象伯伯说：“没事，没事。你们带我去小河边吧。”小猴和小兔就带着大象伯伯到了小河的地方。大象伯伯伸出它的长鼻子，开始“咕咚咕咚”的吸水了。吸完水后，它们三个就来到了种小树苗的地方。
大象伯伯给小树苗浇了水之后，它们三个就各回各家了。
小树苗一天天长高了。
秋天到了，小猴、小兔和大象伯伯都想看看小树苗，哪可是它们三个团结合作的成果呀。
当它们来到种小树苗地方的时候，它们惊呆了，小树苗已经长成大大的苹果树，结满了又大又红的苹果。它们三个可高兴了。
小猴很快爬到树上，摘了两个苹果。一个给了小兔，一个给了大象伯伯。大象伯伯用长长的鼻子给小猴也摘了一个大苹果。
吃着又香又甜的大苹果，它们高兴地说：“明年的植树节，我们要种更多的苹果树，让所有的动物朋友们都尝一尝香香甜甜的大苹果。”说完，它们围着苹果树又唱又跳。
"""
    text_lang_flores = t.detect(text)
    text_output = t.translate(text,text_lang_flores)
    print(text_output)
    print(time.time()-s)