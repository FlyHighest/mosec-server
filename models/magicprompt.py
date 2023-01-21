from transformers import pipeline, set_seed
import random 
import re 
INT_RANGE = -2**31,2**31-1

class MagicPrompt:
    def __init__(self, device) -> None:
        
        self.gpt2_pipe = pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')
        self.gpt2_pipe.to(device)
        with open("../ideas.txt", "r") as f:
            self.line = f.readlines()


    def __call__(self, starting_text):
        seed = random.randint(INT_RANGE[0],INT_RANGE[1]) 
        set_seed(seed)

        if starting_text == "":
            starting_text: str = self.line[random.randrange(0, len(self.line))].replace("\n", "").lower().capitalize()
            starting_text: str = re.sub(r"[,:\-–.!;?_]", '', starting_text)

        response = self.gpt2_pipe(starting_text, max_length=(len(starting_text) + random.randint(60, 90)), num_return_sequences=1)[0]
        
        resp = response['generated_text'].strip()
        if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
            return resp 
        else:
            return starting_text
