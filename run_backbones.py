import os
import csv
import yaml
import time

from train import train as proto_train
from evaluate import evaluate as proto_evaluate


BACKBONES = ["efficientnet_b0", "densenet121", "mobilenet_v3"]

def run_sweep(config_path='config.yaml', num_episodes=300, test_episodes=200):
    base_cfg = yaml.safe_load(open(config_path))
    results = []
    for backbone in BACKBONES:
        print(f"\n=== Running prototypical training for backbone: {backbone} ===")
        cfg = dict(base_cfg)
        cfg['backbone'] = backbone
        cfg['num_episodes'] = int(num_episodes)
        cfg['test_episodes'] = int(test_episodes)
        # use per-backbone checkpoint dir to avoid overwriting
        cfg['checkpoint_dir'] = os.path.join(base_cfg.get('checkpoint_dir', 'checkpoints'), f'proto_{backbone}')
        os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
        # unique log per backbone
        cfg['log_file'] = f'train_{backbone}.log'

        start = time.time()
        proto_train(cfg)
        elapsed = time.time() - start
        print(f"Training finished for {backbone} in {elapsed/60:.2f} minutes")

        ckpt_path = os.path.join(cfg['checkpoint_dir'], 'final.pth')
        print(f"Evaluating {backbone} using checkpoint: {ckpt_path}")
        res = proto_evaluate(cfg, checkpoint_path=ckpt_path)
        res_row = {'backbone': backbone, 'accuracy': float(res['accuracy']), 'f1': float(res['f1']), 'auc': float(res['auc']), 'train_time_min': elapsed/60}
        results.append(res_row)

        # write intermediate CSV
        csv_path = 'backbone_results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['backbone','accuracy','f1','auc','train_time_min'])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Saved intermediate results to {csv_path}")

    print('\nSweep completed. Results:')
    for r in results:
        print(r)


if __name__ == '__main__':
    run_sweep(num_episodes=300, test_episodes=200)
