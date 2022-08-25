import h5py # .h5 파일을 읽기 위한 패키지
import random
import pandas as pd
import numpy as np
import os
import glob
import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb


from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings(action='ignore') 

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

time_now = dt.datetime.now()
run_id = time_now.strftime("%Y%m%d%H%M%S")
os.makedirs(f'./runs/{run_id}', exist_ok=True)

CFG = {
    'EPOCHS':100,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':64,
    'SEED':41
}



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

# all_df = pd.read_csv('./train.csv')
# all_points = h5py.File('./train.h5', 'r')
# all_points = [np.array(all_points[str(i)]) for i in tqdm(all_df["ID"])]


# train_df = all_df.iloc[:int(len(all_df)*0.8)]
# val_df = all_df.iloc[int(len(all_df)*0.8):]

class CustomDataset(Dataset):
    def __init__(self, id_list, label_list, point_list, mode):
        self.id_list = id_list
        self.label_list = label_list
        self.point_list = point_list
        self.mode = mode
        
    def __getitem__(self, index):
        image_id = self.id_list[index]
        
        if self.mode in ['train', 'val']:
            points = self.point_list[int(image_id)][:]
            points = self.randomRotate(points)
        elif self.mode == 'test':
            points = self.point_list[int(image_id)-50000][:]

        image = self.get_vector(points)
        
        if self.label_list is not None:
            label = self.label_list[index]
            return torch.Tensor(image).unsqueeze(0), label
        else:
            return torch.Tensor(image).unsqueeze(0)
    
    def get_vector(self, points, x_y_z=[16, 16, 16]):
        # 3D Points -> [16,16,16]
        xyzmin = np.min(points, axis=0) - 0.001
        xyzmax = np.max(points, axis=0) + 0.001

        diff = max(xyzmax-xyzmin) - (xyzmax-xyzmin)
        xyzmin = xyzmin - diff / 2
        xyzmax = xyzmax + diff / 2

        segments = []
        shape = []

        for i in range(3):
            # note the +1 in num 
            if type(x_y_z[i]) is not int:
                raise TypeError("x_y_z[{}] must be int".format(i))
            s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
            segments.append(s)
            shape.append(step)

        n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]
        n_x = x_y_z[0]
        n_y = x_y_z[1]
        n_z = x_y_z[2]

        structure = np.zeros((len(points), 4), dtype=int)
        structure[:,0] = np.searchsorted(segments[0], points[:,0]) - 1
        structure[:,1] = np.searchsorted(segments[1], points[:,1]) - 1
        structure[:,2] = np.searchsorted(segments[2], points[:,2]) - 1

        # i = ((y * n_x) + x) + (z * (n_x * n_y))
        structure[:,3] = ((structure[:,1] * n_x) + structure[:,0]) + (structure[:,2] * (n_x * n_y)) 

        vector = np.zeros(n_voxels)
        count = np.bincount(structure[:,3])
        vector[:len(count)] = count

        vector = vector.reshape(n_z, n_y, n_x)
        return vector
    
    def randomRotate(self, dots):
        a, b, c = np.random.randint(low=-32, high=33, size=3)
        if a != 0:
            a = np.pi/a
        if b != 0:
            b = np.pi/b
        if c != 0:
            c = np.pi/c
        mx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        my = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        mz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        m = np.dot(np.dot(mx,my),mz)
        dots = np.dot(dots, m.T)
        return dots

    def __len__(self):
        return len(self.id_list)


# train_dataset = CustomDataset(train_df['ID'].values, train_df['label'].values, all_points, 'train')
# train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

# val_dataset = CustomDataset(val_df['ID'].values, val_df['label'].values, all_points, 'val')
# val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv3d(1,8,3),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.Conv3d(8,32,3),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(4),
            nn.Conv3d(32,32,3),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32,10)

    def forward(self,x):
        x = self.feature_extract(x)
        x = x.view(x.size()[0],-1)
        x = self.classifier(x)
        return x


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    wandb.init(project="235951", entity="auroraveil")
    wandb.config = {
    "EPOCHS": CFG['EPOCHS'],
    "LEARNING_RATE": CFG['LEARNING_RATE'],
    "BATCH_SIZE": CFG['BATCH_SIZE'],
    "SEED" : CFG['SEED']
    }

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    best_score = 0
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for data, label in tqdm(iter(train_loader)):
            data, label = data.float().to(device), label.long().to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        if scheduler is not None:
            scheduler.step()
            
        val_loss, val_acc = validation(model, criterion, val_loader, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss)}] Val Loss : [{val_loss}] Val ACC : [{val_acc}]')
        wandb.log({'train_loss': np.mean(train_loss), 'val_loss': val_loss, 'val_acc': val_acc})
        torch.save(model.state_dict(), f'./runs/{run_id}/epoch{epoch}-val_loss{val_loss}-val_acc{val_acc}.pth')
        
        if best_score < val_acc:
            best_score = val_acc
            torch.save(model.state_dict(), f'./runs/{run_id}/best_model(epoch{epoch}).pth')


def validation(model, criterion, val_loader, device):
    model.eval()
    true_labels = []
    model_preds = []
    val_loss = []
    with torch.no_grad():
        for data, label in tqdm(iter(val_loader)):
            data, label = data.float().to(device), label.long().to(device)
            
            model_pred = model(data)
            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
    
    return np.mean(val_loss), accuracy_score(true_labels, model_preds)


model = BaseModel()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = None

# train(model, optimizer, train_loader, val_loader, scheduler, device)


test_df = pd.read_csv('./sample_submission.csv')
test_points = h5py.File('./test.h5', 'r')
test_points = [np.array(test_points[str(i)]) for i in tqdm(test_df["ID"])]


test_dataset = CustomDataset(test_df['ID'].values, None, test_points, 'test')
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

checkpoint = torch.load('./runs/20220815184415/epoch40-val_loss0.41216670228227686-val_acc0.8743.pth')
model = BaseModel()
model.load_state_dict(checkpoint)
model.eval()

def predict(model, test_loader, device):
    model.to(device)
    model.eval()
    model_preds = []
    with torch.no_grad():
        for data in tqdm(iter(test_loader)):
            data = data.float().to(device)
            
            batch_pred = model(data)
            
            model_preds += batch_pred.argmax(1).detach().cpu().numpy().tolist()
    
    return model_preds


preds = predict(model, test_loader, device)


test_df['label'] = preds
test_df.to_csv('./submit40.csv', index=False)
