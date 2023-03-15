import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

class PromptGenAutomaticLexart:
    def __init__(self,device=None) -> None:    
        self.tokenizer = AutoTokenizer.from_pretrained("AUTOMATIC/promptgen-lexart")
        self.model = AutoModelForCausalLM.from_pretrained("AUTOMATIC/promptgen-lexart")
        self.device=device or torch.device("cuda")
    
    def __call__(self, starting_text,batch_size=1):
        input_ids = self.tokenizer(starting_text, return_tensors="pt").input_ids
        if input_ids.shape[1] == 0:
            input_ids = torch.asarray([[self.tokenizer.bos_token_id]], dtype=torch.long)
        input_ids = input_ids.to(self.device)
        input_ids = input_ids.repeat((batch_size, 1))

        outputs = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=max(float(1), 1e-6),
            repetition_penalty=1,
            length_penalty=1,
            top_p=None,
            top_k=12,
            num_beams=int(1),
            min_length=20,
            max_length=150,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if batch_size==1:
            texts = texts[0]
        return texts

if __name__=="__main__":
    os.environ['TRANSFORMERS_CACHE'] = "/root/mosec-server/models-cache"

    prompt_gen = PromptGenAutomaticLexart(torch.device("cuda"))
    results= prompt_gen("McDonald's Church,")
    print(results)