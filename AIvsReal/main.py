from utils import *

print("Loading Module.....")
import torch
from torch import nn
from torch.utils.data import DataLoader,random_split
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import NeuralNetwork as net

DEVICE = "cuda"

SAVE_PATH = "./Model/"

print("Modules Loaded.")

print("\nLoading Data....")
TrainDir = "D:/Projects/AI/CynapticsInductionTask/New_Data/New_Data"
Test_dir = "D:/Projects/AI/CynapticsInductionTask/Data/Test_Labels"

data_transform = transforms.Compose([
    transforms.Resize(size=(512,512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    
]
)

data = datasets.ImageFolder(root=TrainDir,
                                  transform=data_transform)
(train_data,test_data) = random_split(data,[0.8,0.2])


print("Data Loaded.")



#Training Block 1

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_OF_EPOCHS = 100

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True,pin_memory=True)
test_data_loader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True,pin_memory=True)

Resnet50 = net.ResNet(net.block,[1,1,1],3,2).to(DEVICE)
#Resnet50.load_state_dict(torch.load("./Model/Resnet_reduced.pth",weights_only=True))

Resnet50.train()

loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(params=Resnet50.parameters(),lr=LEARNING_RATE,weight_decay=0.1,betas=(0.5,0.999))

for epoch in (range(NUM_OF_EPOCHS)):
    print(f"Epoch {epoch}\n-----")

    train_single_epoch(Resnet50,train_dataloader,loss_fn,optimizer,accuracy_fn,DEVICE)
    test_step(Resnet50,test_data_loader,loss_fn,accuracy_fn,DEVICE)

    torch.save(Resnet50.state_dict(),SAVE_PATH + "Resnet_reduced.pth")
    print("Model saved.")




    
