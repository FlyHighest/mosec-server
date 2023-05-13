from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

class PromptEnhancer:
    def __init__(self) -> None:
        prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.promptist_model, self.promptist_tokenizer = prompter_model, tokenizer 
    
        self.anything_promptgen_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        self.anything_promptgen_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.anything_promptgen_model = GPT2LMHeadModel.from_pretrained('FredZhang7/anime-anything-promptgen-v2')

        self.anything_promptgen_pipeline = pipeline('text-generation', model=self.anything_promptgen_model, tokenizer=self.anything_promptgen_tokenizer )


    def promptist_generate(self, plain_text):
        input_ids = self.promptist_tokenizer(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids
        eos_id = self.promptist_tokenizer.eos_token_id
        outputs = self.promptist_model.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
        output_texts = self.promptist_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
        return res
    
    def anything_promptgen_generate(self, plain_text):
        outs = self.anything_promptgen_pipeline(plain_text, max_length=76, num_return_sequences=1, do_sample=True, repetition_penalty=1.2, temperature=0.7, top_k=4, early_stopping=True)
        return str(outs[0]['generated_text']).replace('  ', '').rstrip(',')


    def __call__(self, text, model_type):
        if model_type=="universal":
            return self.promptist_generate(text)
        elif model_type=="anime":
            return self.anything_promptgen_generate(text)

if __name__=="__main__":
    import time 
    pe = PromptEnhancer()

    start = time.time()
    res1 = pe("a tree on the moon","universal")
    res2 = pe("1girl drinking water","anime")
    print(res1)
    print(res2)
    print(time.time()-start)