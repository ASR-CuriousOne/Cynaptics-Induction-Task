import torch
from torch import nn

import NeuralNetwork as net
from utils import *

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader,random_split
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from PIL import Image

import pandas as pd

data_transform = transforms.Compose([
    transforms.Resize(size=(512,512)),   
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))    
]
)

Data_Folder = "./Data/Temp/Test/"

classes = ["Real","AI"]

Resnet50 = net.ResNet(net.block,[1,1,1],3,2).to("cuda")

Resnet50.load_state_dict(torch.load("./Model/Resnet_reduced.pth",weights_only=True))

Resnet50.eval()

ImageName,Predictions = [],[]

with torch.inference_mode():
    for i in range(0,200):
        image_path = Data_Folder + "image_" + str(i) + ".jpg"
        img = Image.open(image_path)
        img = data_transform(img)
        ImageName.append("image_" + str(i))
        Predictions.append(classes[Resnet50(img.unsqueeze(0).to("cuda")).argmax(dim = 1)])

df = pd.DataFrame({
    "Id" : ImageName,
    "Label": Predictions
})
df.reset_index(drop=True, inplace=True)

df.to_csv("Result.csv",index=False,index_label=None)