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

PATH = "Model_Training/Models/Final_Type1/Model1_Epoch29.pt"
weights = torch.load(PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device : {device}")

hole_data_path = "BenchMark/bench_data.csv"
class Hole_db_Train(Dataset):
    def __init__(self,file_path, transformers = None):
        self.df = pd.read_csv(file_path)
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
        label1 = int(self.df.iloc[idx, 2])
        #return label2 instead of label1 if we need.
        return img_resized, label1
    
train_dataset = Hole_db_Train(hole_data_path)
test_dataset = Hole_db_Train(hole_data_path)

loader_train = DataLoader(dataset = train_dataset, batch_size=256, shuffle=True)
loader_test = DataLoader(dataset= test_dataset, batch_size=256, shuffle= False)


from transformers import AutoImageProcessor, AutoModelForImageClassification
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = torch.load("Model_Training/Models/Baseline7(Type1)/Base_Model7_Epoch7.pt").vit
#model.load_state_dict(weights['ViT_Model_state'])
model.eval()
model.to(device)


class Final_Model(nn.Module):
    def __init__(self):
        super(Final_Model, self).__init__()
        self.base_vit = model
        self.embedding = nn.Embedding(4, embedding_dim=256)
        self.fc = nn.Linear(768, 2)

    def forward(self, P_IMG):
        x1 = self.base_vit(P_IMG).last_hidden_state[:,0,:]
        x = self.fc(x1)
        return x
    
AIGT_Model = Final_Model()
AIGT_Model.load_state_dict(weights['AIGT_Model_state'])
AIGT_Model.eval()
AIGT_Model.to(device)

true_list = []
preds_list = []
totlal= 0
correct = 0

epoch_loss = 0
criterion = nn.BCEWithLogitsLoss()

#Type of text and accuracy
type_acc_list = [0, 0, 0, 0]

with torch.no_grad():
    for images, labels in loader_test:
        images = images.to(device)
        labels = labels.to(device)
        output = AIGT_Model(images)
        _, predicted = torch.max(output, 1)
        correct += (predicted==labels).sum().item()
        for i in range(256):
            try:
                if predicted[i] == 1 and labels[i] == 1:
                    type_acc_list[0] += 1
                elif predicted[i] == 1 and labels[i] == 0:
                    type_acc_list[1] += 1
                elif predicted[i] == 0 and labels[i] == 1:
                    type_acc_list[2] += 1
                else:
                    type_acc_list[3] += 1
            except:
                pass

print(correct, len(test_dataset))
print(f"accuracy {correct/len(test_dataset)}")

#11, 10, 01, 00
print(type_acc_list)