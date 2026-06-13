import os
import time
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm

from data_loader import FolderFewShotDataset, default_transform_train, default_transform_eval, make_episode
from models.backbone import get_backbone
from models.proto_net import PrototypicalNetwork
from utils import save_checkpoint, set_logger


def train(cfg):
    set_logger(cfg.get('log_file', None))
    # robust device selection: prefer configured device if available
    dev = cfg.get('device', 'cpu')
    if dev == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif dev == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    ds = FolderFewShotDataset(cfg['data_root'], transform=default_transform_eval(cfg.get('img_size', 224)))
    train_dict, val_dict = ds.split_train_val(val_ratio=cfg.get('val_ratio', 0.2), seed=cfg.get('seed', 42))

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
    # coerce lr/weight_decay to float in case YAML or CLI provided a string
    lr = float(cfg.get('lr', 1e-4))
    weight_decay = float(cfg.get('weight_decay', 0.0))
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.NLLLoss()

    logs = {'loss': [], 'val_loss': []}

    num_episodes = int(cfg.get('num_episodes', cfg.get('num_episodes', 1000)))
    N_way = int(cfg.get('N_way', cfg.get('N_way')))
    K_shot = int(cfg.get('K_shot', cfg.get('K_shot')))
    Q_query = int(cfg.get('Q_query', cfg.get('Q_query')))

    best_val = None
    no_improve = 0
    patience = int(cfg.get('patience_val', 0))  # validations without improvement

    for episode in trange(num_episodes, desc='Episodes'):
        model.train()
        s_x, s_y, q_x, q_y = make_episode(
            train_dict, N_way, K_shot, Q_query, default_transform_train(cfg.get('img_size', 224)), device
        )
        log_p = model(s_x, s_y, q_x)
        loss = criterion(log_p, q_y)

        optimizer.zero_grad()
        loss.backward()
        # optional gradient clipping
        max_norm = float(cfg.get('grad_clip', 0.0))
        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        logs['loss'].append(loss.item())

        if (episode + 1) % int(cfg.get('val_every', 100)) == 0:
            # quick validation on val episodes
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(int(cfg.get('val_episodes', 50))):
                    vs_x, vs_y, vq_x, vq_y = make_episode(
                        val_dict, N_way, K_shot, Q_query, default_transform_eval(cfg.get('img_size', 224)), device
                    )
                    val_log_p = model(vs_x, vs_y, vq_x)
                    val_loss = criterion(val_log_p, vq_y)
                    val_losses.append(val_loss.item())
            avg_val = sum(val_losses) / len(val_losses)
            logs['val_loss'].append(avg_val)
            logging.info(f'Episode {episode+1}/{cfg["num_episodes"]} loss={loss.item():.4f} val_loss={avg_val:.4f}')
            # save checkpoint
            ckpt = {
                'episode': episode + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'cfg': cfg,
            }
            os.makedirs(cfg.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
            save_checkpoint(ckpt, os.path.join(cfg.get('checkpoint_dir', 'checkpoints'), f'ckpt_ep{episode+1}.pth'))

            # track best val
            if best_val is None or avg_val < best_val:
                best_val = avg_val
                no_improve = 0
                save_checkpoint(ckpt, os.path.join(cfg.get('checkpoint_dir', 'checkpoints'), 'best.pth'))
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    logging.info(f'Early stopping at episode {episode+1} (no improvement for {no_improve} validations)')
                    break

    # final save
    save_checkpoint({'model_state': model.state_dict(), 'cfg': cfg}, os.path.join(cfg.get('checkpoint_dir', 'checkpoints'), 'final.pth'))
    logging.info('Training finished')


if __name__ == '__main__':
    import yaml
    cfg = yaml.safe_load(open('config.yaml'))
    train(cfg)
