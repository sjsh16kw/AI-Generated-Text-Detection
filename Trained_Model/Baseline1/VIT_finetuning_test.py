from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms
from torch import tensor
import torchsummary
import torch
import numpy as np
import ast
import time

#GPU Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device : {device}")

hole_data_path = "Model_Training/Data/AIGT_data.csv"

class Hole_db_Train(Dataset):
    def __init__(self,file_path, mode, train_rate, transformers = None):
        hole_df = pd.read_csv(file_path)
        if mode == "Train":
            self.df = hole_df[0:int(len(hole_df)*train_rate)]
        elif mode == "Test":
            self.df = hole_df[int(len(hole_df)*train_rate):]
        return
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        #make same channel with model ->3 channel(R, G, B)
        p_img = ast.literal_eval(self.df.iloc[idx, 1])
        RGB_img = [p_img, p_img, p_img]
        res_img = tensor(RGB_img).cuda()
        resize = transforms.Resize(224)
        img_resized = resize(res_img)
        label1 = self.df.iloc[idx, 2]
        label2 = self.df.iloc[idx, 3]
        #return label2 instead of label1 if we need.
        return img_resized, label1
    
#Data tromsform

train_dataset = Hole_db_Train(hole_data_path, 'Train', 0.8)
test_dataset = Hole_db_Train(hole_data_path, 'Test', 0.8)

loader_train = DataLoader(dataset = train_dataset, batch_size=32, shuffle=True)
test_dataset = DataLoader(dataset= test_dataset, batch_size=32, shuffle= False)



# Load model directly from HuggingFace
from transformers import AutoImageProcessor, AutoModelForImageClassification
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

import torch.nn as nn
#Define new classifier
model.classifier = nn.Sequential(
    nn.Linear(in_features=768, out_features=100, bias= True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=100, out_features=2, bias=True)
)

#set Freezing
for param in model.vit.parameters():
    param.requires_grad = False

print(model)
model.to("cuda")
#Learning_Loop

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

time1 = time.time()
#Training_Loop : 
epochs = 10
for epoch in range(epochs):
    epoch_loss = 0 #initalize epoch loss
    cnt = 0
    for images, labels in loader_train:
        #Allocate mini batch of image, label to device
        cnt += 1
        images = images.to(device)
        labels = labels.to(device)

        #initialize gradient in optimizer
        optimizer.zero_grad()
        #forward method models always return tuples
        output = model(images)
        loss = criterion(output.logits, labels)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        if cnt % 100 == 0:
            print(f"epoch : [{epoch + 1}/{epochs}] : {cnt/len(loader_train)}, Training_time{time.time()-time1}")

    print(f'epoch : [{epoch + 1}/{epochs}] - loss : {epoch_loss/len(loader_train)}')

    #model_save format : "pt" 
    torch.save(model, f"Base_Model_Epoch{epoch}")