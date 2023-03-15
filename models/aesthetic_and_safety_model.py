import torch.nn as nn 
import torch 
import clip 
import os 
import onnxruntime
import numpy as np 
def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    cache_folder = "models-cache"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

def load_safety_model():
    m = onnxruntime.InferenceSession("models-cache/clip_bin_nsfw.onnx",providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    return m
    

class AestheticSafetyModel:
    def __init__(self,device) -> None:
        self.device = device 
        self.clip_model,self.clip_preprocess = clip.load("ViT-L/14",device=device,download_root="models-cache")

        self.aesthetic_model = get_aesthetic_model()
        self.aesthetic_model.to(device)
        
        self.safety_model = load_safety_model()
        
        
    def get_aes_and_nsfw(self,img):
        img = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_features = self.clip_model.encode_image(img)
            img_features /= img_features.norm(dim=-1,keepdim=True)
            score = self.aesthetic_model(img_features.type(torch.cuda.FloatTensor))
            score = score.item()
        output = self.safety_model.run(None, {'input_1': img_features.cpu().numpy().astype(np.float64)})
        nsfw = output[0][0][0]
        return score , nsfw

if __name__=="__main__":
    from PIL import Image 
    import glob
    import time 
    aesthetic_model = AestheticSafetyModel(torch.device("cuda:0"))
    start = time.time()
    for img_path in glob.glob("testimage/*g"):
        img = Image.open(img_path)
        print(img_path)
        print(aesthetic_model.get_aes_and_nsfw(img))
    end = time.time()
    print(end-start)


