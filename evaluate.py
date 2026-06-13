import os
import torch
import numpy as np
from tqdm import trange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_loader import FolderFewShotDataset, default_transform_eval, make_episode
from models.backbone import get_backbone
from models.proto_net import PrototypicalNetwork
from utils import compute_metrics, plot_confusion


def evaluate(cfg, checkpoint_path=None):
    # robust device selection
    dev = cfg.get('device', 'cpu')
    if dev == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif dev == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    ds = FolderFewShotDataset(cfg['data_root'], transform=default_transform_eval(cfg.get('img_size', 224)))
    train_dict, test_dict = ds.split_train_val(val_ratio=cfg.get('val_ratio', 0.2), seed=cfg.get('seed', 42))

    backbone = get_backbone(
        cfg['backbone'],
        embedding_dim=int(cfg.get('embedding_dim', cfg['embedding_dim'])),
        fine_tune_layers=int(cfg.get('fine_tune_layers', 2)),
        dropout_p=float(cfg.get('projection_dropout', 0.0)),
    )
    model = PrototypicalNetwork(
        backbone,
        distance=str(cfg.get('distance', 'euclidean')),
        temperature=float(cfg.get('temperature', 1.0)),
    ).to(device)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])

    model.eval()
    accs = []
    precs = []
    recs = []
    f1s = []
    aucs = []
    all_y_true = []
    all_y_pred = []

    N_way = int(cfg.get('N_way', cfg.get('N_way')))
    K_shot = int(cfg.get('K_shot', cfg.get('K_shot')))
    Q_query = int(cfg.get('Q_query', cfg.get('Q_query')))

    for _ in trange(int(cfg.get('test_episodes', 200)), desc='Eval episodes'):
        s_x, s_y, q_x, q_y = make_episode(
            test_dict, N_way, K_shot, Q_query, default_transform_eval(cfg.get('img_size', 224)), device
        )
        with torch.no_grad():
            log_p = model(s_x, s_y, q_x)
            preds = log_p.argmax(dim=1).cpu().numpy()
            probs = torch.exp(log_p)[:, 1].cpu().numpy() if log_p.shape[1] > 1 else torch.exp(log_p)[:, 0].cpu().numpy()
            y_true = q_y.cpu().numpy()

        accs.append(accuracy_score(y_true, preds))
        precs.append(precision_score(y_true, preds, average='binary', zero_division=0))
        recs.append(recall_score(y_true, preds, average='binary', zero_division=0))
        f1s.append(f1_score(y_true, preds, average='binary', zero_division=0))
        try:
            aucs.append(roc_auc_score(y_true, probs))
        except Exception:
            aucs.append(float('nan'))

        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(preds.tolist())

    res = dict(
        accuracy=np.nanmean(accs),
        precision=np.nanmean(precs),
        recall=np.nanmean(recs),
        f1=np.nanmean(f1s),
        auc=np.nanmean(aucs)
    )
    print('Evaluation results:', res)
    # confusion
    plot_confusion(all_y_true, all_y_pred, ds.classes[:cfg['N_way']], os.path.join(cfg.get('output_dir', 'outs'), 'confusion.png'))
    return res


if __name__ == '__main__':
    import yaml
    cfg = yaml.safe_load(open('config.yaml'))
    evaluate(cfg)
