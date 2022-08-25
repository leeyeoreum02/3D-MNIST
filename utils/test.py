import torch
from tqdm.auto import tqdm
import pandas as pd


def predict(model, cfg, test_loader, checkpoint=None, submit_name='submit'):
    device = cfg['DEVICE']
    if checkpoint:
        model = torch.load(checkpoint)

    model.to(device)
    model.eval()
    model_preds = []
    with torch.no_grad():
        for data in tqdm(iter(test_loader)):
            data = data.float().to(device)
            batch_pred, trans_feat = model(data)
            model_preds += batch_pred.argmax(1).detach().cpu().numpy().tolist()

    sample_submission = pd.read_csv(cfg['TESTCSV'])
    sample_submission['label'] = model_preds
    sample_submission.to_csv(f'./output/{submit_name}.csv', index=False)

    return model_preds