from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms
from torch import tensor
import torch
import numpy as np
import ast
import time

#GPU Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device : {device}")

hole_data_path = "Model_Training/Data/AIGT_24Token_Ver2_data.csv"
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
        return img_resized, (label1, label2)
    
#Data tromsform

train_dataset = Hole_db_Train(hole_data_path, 'Train', 0.8)
test_dataset = Hole_db_Train(hole_data_path, 'Test', 0.8)

loader_train = DataLoader(dataset = train_dataset, batch_size=32, shuffle=True)
test_dataset = DataLoader(dataset= test_dataset, batch_size=32, shuffle= False)



# Load model directly from HuggingFace
from transformers import AutoImageProcessor, AutoModelForImageClassification
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = torch.load("Model_Training/Models/Baseline6/Base_Model6_Epoch6.pt").vit

import torch.nn as nn
#Define new classifier
#print(model)

#set Freezing
for param in model.embeddings.parameters():
    param.requires_grad = False

for i in range(0, 11):
    for param in model.encoder.layer[i].parameters():
        param.requires_grad = False

model.to(device)
#set_dict
"""
class Embed_Model(nn.Module):
    def __init__(self):
        super(Embed_Model, self).__init__()
        self.embed  = nn.Embedding(4, embedding_dim=10)

    def forward(self, x):
        ipt_vector = torch.tensor([0]*4, dtype=torch.long).to(x.device)
        ipt_vector[x-1] += 1
        x = torch.tensor(ipt_vector)
        x = self.embed(x)
        return x
"""

class Final_Model(nn.Module):
    def __init__(self):
        super(Final_Model, self).__init__()
        self.base_vit = model
        self.embedding = nn.Embedding(4, embedding_dim=256)
        self.fc = nn.Linear(256+768, 2)

    def forward(self, P_IMG, Label):
        x1 = self.base_vit(P_IMG).last_hidden_state[:,0,:]
        embed_weight = self.embedding.weight
        x2 = []
        for i in range(len(Label)):
            x2.append(embed_weight[int(Label[i])-1])
        #여 기 서 부 터 고 치 자 . 
        x2 = torch.stack(x2)
        #print(x2)
        x = torch.cat((x1, x2), dim=1)
        #print(x)
        x = self.fc(x)

        return x
    
AIGT_Model = Final_Model()
AIGT_Model.to(device)

#Learning_Loop

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(AIGT_Model.parameters(), lr=0.1)

print(AIGT_Model)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)

time1 = time.time()
#Training_Loop : 
epochs = 100
for epoch in range(epochs):
    epoch_loss = 0 #initalize epoch loss
    cnt = 0
    acc_cnt = 0
    for images, labels in loader_train:
        #Allocate mini batch of image, label to device
        cnt += 1
        images = images.to(device)
        type = labels[1]
        is_AI = labels[0]
        type = type.to(device)
        is_AI = is_AI.to(device)

        #initialize gradient in optimizer
        optimizer.zero_grad()
        #forward method models always return tuples
        output = AIGT_Model(images, type)
        loss = criterion(output, is_AI)

        _, predicted = torch.max(output, 1)


        acc_cnt += (predicted==is_AI).sum().item()

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()



        if cnt % 100 == 0:
            print(f"epoch : [{epoch + 1}/{epochs}] : {cnt/len(loader_train)}, Training_time{time.time()-time1}")
    scheduler.step()

    print(f'epoch : [{epoch + 1}/{epochs}] - loss : {epoch_loss/len(loader_train)}')
    print(f"accuracy {acc_cnt/len(loader_train)}")

    #model_save format : "pt" 
    save_PATH = f"Model_Training/Models/Model1/Model1_Epoch{epoch}.pt"
    torch.save({"AIGT_Model_state" : AIGT_Model.state_dict(),
                "Base_Line_state" : model.state_dict()}, save_PATH)
