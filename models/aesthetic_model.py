import torch.nn as nn 
import torch 
import clip 
class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

class AestheticModel:
    def __init__(self,device) -> None:
        self.device = device 
        self.predictor = AestheticPredictor(768)
        pt_state = torch.load("models-cache/sac+logos+ava1-l14-linearMSE.pth",map_location=device)
        self.predictor.load_state_dict(pt_state)
        self.predictor.to(device)
        self.predictor.eval()
        self.clip_model,self.clip_preprocess = clip.load("ViT-L/14",device=device)
    
    def get_score(self,img):
        img = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_features = self.clip_model.encode_image(img)
            img_features /= img_features.norm(dim=-1,keepdim=True)
            score = self.predictor(img_features.type(torch.cuda.FloatTensor))
            score = score.item()
        return score 

if __name__=="__main__":
    from PIL import Image 
    aesthetic_model = AestheticModel(torch.device("cuda:0"))
    img = Image.open("test.png")
    print(aesthetic_model.get_score(img))


