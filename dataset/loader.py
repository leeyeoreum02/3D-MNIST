import pandas as pd
import numpy as np
import h5py
from tqdm.auto import tqdm
from dataset.CustomDataset import CustomDataset, PointCloudDataset
from torch.utils.data import DataLoader


def load_train_data(cfg):
    all_df = pd.read_csv(cfg['TRAINCSV'])
    all_points = h5py.File(cfg['TRAINH5'], 'r')
    all_points = [np.array(all_points[str(i)]) for i in tqdm(all_df["ID"])]

    train_df = all_df.iloc[:int(len(all_df)*0.8)]
    val_df = all_df.iloc[int(len(all_df)*0.8):]

    train_dataset = PointCloudDataset(train_df['ID'].values, train_df['label'].values, all_points, 8600, 'train')
    train_loader = DataLoader(train_dataset, batch_size = cfg['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = PointCloudDataset(val_df['ID'].values, val_df['label'].values, all_points, 8600, 'val')
    val_loader = DataLoader(val_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def load_test_data(cfg):
    test_df = pd.read_csv(cfg['TESTCSV'])
    test_points = h5py.File(cfg['TESTH5'], 'r')
    test_points = [np.array(test_points[str(i)]) for i in tqdm(test_df["ID"])]

    test_dataset = PointCloudDataset(test_df['ID'].values, None, test_points, 8600, 'test')
    test_loader = DataLoader(test_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=0)

    return test_loader
