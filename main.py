import os
import datetime as dt
import warnings
warnings.filterwarnings(action='ignore') 

import torch
import torch.optim as optim

from model.PointNet import PointNetCls
from utils.seed import seed_everything
from utils.train import train
from utils.test import predict
from utils.AdaPC_Augmentor import Augmentor
from dataset.loader import load_train_data, load_test_data


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
time_now = dt.datetime.now()
run_id = time_now.strftime("%Y%m%d%H%M%S")

CFG = {
    'DEVICE':device,
    'EPOCHS':200,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':32,
    'SEED':41,
    'RUNID':run_id,
    'TRAINCSV':'./data/train.csv',
    'TRAINH5':'./data/train.h5',
    'TESTCSV':'./data/sample_submission.csv',
    'TESTH5':'./data/test.h5',
    'CHECKPOINT':'./runs/20220823135032/best_model(epoch157).pt',
}

seed_everything(CFG['SEED']) 

def run_train():
    os.makedirs(f'./runs/{CFG["RUNID"]}', exist_ok=True)
    train_loader, val_loader = load_train_data(CFG)
    model = PointNetCls()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=50, eta_min=0)
    # scheduler = optim.lr_scheduler.StepLR(optimizer_C, step_size=20, gamma=0.5)
    train(model, train_loader, val_loader, optimizer, None, CFG)

def run_test():
    test_loader = load_test_data(CFG)
    model = PointNetCls()
    preds = predict(model, CFG, test_loader, CFG['CHECKPOINT'], 'PointNetAugmentor157notestaug')

run_train()
# run_test()
