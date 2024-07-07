from transformers import ViTConfig, ViTModel
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms
from torch import tensor
import torch
import numpy as np
import ast
import time
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device : {device}")



hole_data_path = "Model_Training/Data/AIGT_24_Ver3.csv"
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
        ch1_img = [p_img]
        res_img = tensor(ch1_img).cuda()
        label1 = self.df.iloc[idx, 2]
        label2 = self.df.iloc[idx, 3]
        #return label2 instead of label1 if we need.
        return res_img, label1
    
train_dataset = Hole_db_Train(hole_data_path, 'Train', 0.8)
test_dataset = Hole_db_Train(hole_data_path, 'Test', 0.8)

loader_train = DataLoader(dataset = train_dataset, batch_size=256, shuffle=True)
loader_test = DataLoader(dataset= test_dataset, batch_size=256, shuffle= False)

config = ViTConfig(
    hidden_size=96, 
    num_hidden_layers=4,      
    num_attention_heads=6,  
    intermediate_size=384,
    image_size=64,  
    patch_size=16,  
    num_channels=1      
)

model = ViTModel(config=config)
configuration = model.config
model.to(device)
#total_params : 472416
#기존 Vit : 8천 5백만개 정도

#Hidden_size : input image의 각 patch를 96dimension-vector로 embedding
#intermediate_size = 4*D (4*Hidden_size)
#num_attention_head : 멀티헤드어텐션에서의 헤드 수 - 어텐션을 병렬로 수행
#Attension을 한번에 계산하지 않고 Head 수 만큼 나눠 계산. 
#Head는 hidden_size의 배수여야 할듯...  96/6 = 16



#___load Final Model____
class Final_Model(nn.Module):
    def __init__(self):
        super(Final_Model, self).__init__()
        self.base_vit = model
        self.embedding = nn.Embedding(4, embedding_dim=256)
        self.fc = nn.Linear(96, 2)

    def forward(self, P_IMG):
        x1 = self.base_vit(P_IMG).last_hidden_state[:,0,:]
        """
        embed_weight = self.embedding.weight
        x2 = []
        for i in range(len(Label)):
            x2.append(embed_weight[int(Label[i])-1])
        #여 기 서 부 터 고 치 자 . 
        x2 = torch.stack(x2)
        #print(x2)
        x = torch.cat((x1, x2), dim=1)
        #print(x)
        """
        x = self.fc(x1)
        return x

AIGT_Model = Final_Model()
AIGT_Model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(AIGT_Model.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)

print(f"total parameters : {sum(p.numel() for p in AIGT_Model.parameters() if p.requires_grad)}")
print(AIGT_Model)
time1 = time.time()
#Training_Loop : 
epochs = 20
for epoch in range(epochs):
    epoch_loss = 0 #initalize epoch loss
    cnt = 0
    acc_cnt = 0
    tot_cnt = 0
    for images, labels in loader_train:
        #Allocate mini batch of image, label to device
        cnt += 1
        images = images.to(device)
        labels = labels.to(device)
        #initialize gradient in optimizer
        optimizer.zero_grad()
        #forward method models always return tuples
        output = AIGT_Model(images)
        loss = criterion(output, labels)
        _, predicted = torch.max(output, 1)
        acc_cnt += (predicted==labels).sum().item()
        tot_cnt += labels.size(0)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        if cnt % 300 == 0:
            print(f"epoch : [{epoch + 1}/{epochs}] : {cnt/len(loader_train)}, Training_time{time.time()-time1}")

    scheduler.step()

    print(f'epoch : [{epoch + 1}/{epochs}] - loss : {epoch_loss/len(loader_train)}')
    print(f"accuracy {acc_cnt/tot_cnt}")

    #model_save format : "pt" 
    save_PATH = f"Model_Training/Models/AIGTD_ViT1/Model1_Epoch{epoch}.pt"
    torch.save({"AIGT_Model_state" : AIGT_Model.state_dict(),
                "ViT_Model_state" : model.state_dict()}, save_PATH)