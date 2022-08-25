import warnings
warnings.filterwarnings(action='ignore') 
import os

import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score


def feature_transform_regularizer(trans):
    D = trans.size()[1]
    I = torch.eye(D)[None, :, :]
    I = I.to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


def train(model, train_loader, val_loader, optimizer, scheduler, cfg):
    os.environ['WANDB_START_METHOD'] = 'thread'
    wandb.init(project="235951", entity="auroraveil")
    wandb.config = {
    "EPOCHS": cfg['EPOCHS'],
    "LEARNING_RATE": cfg['LEARNING_RATE'],
    "BATCH_SIZE": cfg['BATCH_SIZE'],
    "SEED" : cfg['SEED']
    }
    device = cfg['DEVICE']
    scaler = torch.cuda.amp.GradScaler()
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    best_score = 0
    for epoch in range(1, cfg['EPOCHS']+1):
        model.train()
        train_loss = []
        for data, label in tqdm(iter(train_loader)):
            data, label = data.float().to(device), label.long().to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                output, trans_feat = model(data)
            loss = criterion(output, label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            
            train_loss.append(loss.item())
        
        if scheduler is not None:
            scheduler.step()
            
        val_loss, val_acc = validation(model, criterion, val_loader, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss)}] Val Loss : [{val_loss}] Val ACC : [{val_acc}]')
        wandb.log({'train_loss': np.mean(train_loss), 'val_loss': val_loss, 'val_acc': val_acc})

        torch.save(model, f'./runs/{cfg["RUNID"]}/{epoch}-val_loss{val_loss}-val_acc{val_acc}.pt')
        
        if best_score < val_acc:
            best_score = val_acc
            torch.save(model, f'./runs/{cfg["RUNID"]}/best_model(epoch{epoch}).pt')


def validation(model, criterion, val_loader, device):
    model.eval()
    true_labels = []
    model_preds = []
    val_loss = []
    with torch.no_grad():
        for data, label in tqdm(iter(val_loader)):
            data, label = data.float().to(device), label.long().to(device)
            model_pred, trans_feat = model(data)
            loss = criterion(model_pred, label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
            val_loss.append(loss.item())

            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
    
    return np.mean(val_loss), accuracy_score(true_labels, model_preds)

