"""Train or continue the original MLP cGAN / residual U-Net DDPM."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from training.session import run_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', choices=['cgan', 'diffusion'], required=True)
    parser.add_argument('--epochs', type=int, required=True, help='Total target epochs, including prior epochs on resume')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--limit-train', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--resume', type=Path, help='A matching training-state.pt, never a recovered inference checkpoint')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or (args.limit_train is not None and args.limit_train < 2):
        raise ValueError('Positive budget and at least two examples required')
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1':
        raise ValueError('Expected canonical Keras MNIST archive')
    with np.load(args.data, allow_pickle=False) as archive:
        images = archive['x_train'][:args.limit_train]
        labels = archive['y_train'][:args.limit_train]
    if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != np.uint8 or len(images) != len(labels) or not np.isin(labels, range(10)).all():
        raise ValueError('Expected canonical MNIST uint8 images and labels')
    dataset = TensorDataset(torch.from_numpy(images).unsqueeze(1).float()/127.5-1,
                            torch.from_numpy(labels).long())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), drop_last=True)
    if not len(loader):
        raise ValueError('Batch exceeds available training rows')
    config = {'seed': args.seed, 'batch_size': args.batch_size, 'limit_train': args.limit_train,
              'train_rows_available': len(dataset), 'drop_last': True, 'data_sha256': digest}
    history, contract = run_session(args.model, loader, config, args.epochs, args.output,
                                    args.device, resume=args.resume)
    result = {'model': args.model, 'architecture': 'original MLP cGAN / residual U-Net',
              'contract': contract, 'rows_per_epoch': len(loader)*args.batch_size,
              'test_used_for_training': False, 'completed_epochs': args.epochs,
              'resumed_from': str(args.resume) if args.resume else None, 'history': history}
    (args.output/'run.json').write_text(json.dumps(result, indent=2)+'\n')

if __name__ == '__main__':
    main()
