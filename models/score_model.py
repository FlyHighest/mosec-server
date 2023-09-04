import torch.nn as nn 
import torch 
import os , json

import json
from hashlib import sha256 as sha256
import ImageReward as RM 

class ScoreModel:
    def __init__(self,device="cpu") -> None:
        #self.score_model = RM.load("ImageReward-v1.0",download_root="./models-cache",device=device )
        pass 
    
    def get_score(self, image_path, prompt):
        # return self.score_model.score(prompt,image_path)
        return 0
