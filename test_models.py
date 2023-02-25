import os
import torch
from models import AestheticModel
import pandas as pd
import httpx
from io import BytesIO
from PIL import Image
from tqdm import tqdm
if __name__=="__main__":
    m = AestheticModel(torch.device("cuda"))
    df = pd.read_csv("image.csv")
    out = open("imagescore.sql","a")
    for imgurl in tqdm(df.itertuples()):
        try:
            url = imgurl.imgurl
            r= httpx.get(url)
            i = Image.open(BytesIO(r.content))
            score = m.get_score(i)
            out.write(f"update image set score={score} where imgurl=\"{url}\"; \n")
            
        except:
            continue
        

