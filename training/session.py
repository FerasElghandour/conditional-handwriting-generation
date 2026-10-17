"""Fresh or state-restored original-architecture generator training.

The session state binds data/configuration and both optimizer and RNG states.
Published recovered inference weights are not continuation checkpoints.
"""
from pathlib import Path
import random
import numpy as np
import torch
from model.cgan import Generator, Discriminator
from model.diffusion import ConditionalUNet
from training.train_cgan import train_cgan
from training.train_diffusion import train_diffusion


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng(loader, device):
    numpy_state = np.random.get_state()
    result = {
        'python': random.getstate(),
        'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        'torch': torch.get_rng_state(),
        'loader': loader.generator.get_state(),
    }
    if torch.device(device).type == 'cuda':
        result['cuda'] = torch.cuda.get_rng_state_all()
    if torch.device(device).type == 'mps':
        result['mps'] = torch.mps.get_rng_state()
    return result


def restore_rng(state, loader, device):
    random.setstate(state['python'])
    name, values, position, gaussian, cached = state['numpy']
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state['torch'])
    loader.generator.set_state(state['loader'])
    if torch.device(device).type == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda'])
    if torch.device(device).type == 'mps':
        torch.mps.set_rng_state(state['mps'])


def run_session(kind, loader, config, epochs, output, device='cpu', resume=None):
    output = Path(output)
    if kind not in ('cgan', 'diffusion') or epochs < 1 or loader.generator is None:
        raise ValueError('Known model, positive epochs and explicit loader generator required')
    if torch.device(device).type not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Reproducible sessions currently support CPU, CUDA or MPS')
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; choose --device cpu explicitly')
    if torch.device(device).type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    seed_everything(config['seed'])
    if kind == 'cgan':
        models = {'generator': Generator().to(device), 'discriminator': Discriminator().to(device)}
        optimizers = {name: torch.optim.Adam(model.parameters(), lr=.0002, betas=(.5, .999)) for name, model in models.items()}
    else:
        models = {'unet': ConditionalUNet(base_channels=32).to(device)}
        optimizers = {'unet': torch.optim.Adam(models['unet'].parameters(), lr=1e-4)}
    contract = {**config, 'kind': kind, 'device': str(torch.device(device)), 'torch': str(torch.__version__)}
    start, history = 0, []
    if resume is not None:
        state = torch.load(resume, map_location='cpu', weights_only=True)
        if state.get('format') != 'generator-session-v1':
            raise ValueError('Use training-state.pt, not recovered inference weights or a model-only checkpoint')
        if state['contract'] != contract:
            raise ValueError('Resume requires identical data, model, batch, seed, device and runtime contract')
        start, history = state['completed_epochs'], state['history']
        if start >= epochs:
            raise ValueError('--epochs is the total target and must exceed the completed epoch')
        for name, model in models.items():
            model.load_state_dict(state['models'][name], strict=True)
            optimizers[name].load_state_dict(state['optimizers'][name])
        restore_rng(state['rng'], loader, device)
    output.mkdir(parents=True, exist_ok=True)

    def save_session(records):
        state = {
            'format': 'generator-session-v1', 'contract': contract,
            'completed_epochs': records[-1]['epoch'], 'history': list(records),
            'models': {name: model.state_dict() for name, model in models.items()},
            'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            'rng': capture_rng(loader, device),
        }
        temporary = output/'training-state.tmp'
        torch.save(state, temporary)
        temporary.replace(output/'training-state.pt')

    if kind == 'cgan':
        history = train_cgan(models['generator'], models['discriminator'], loader,
                             optimizers['generator'], optimizers['discriminator'],
                             torch.nn.BCELoss(), device, 100, 10, epochs, str(output),
                             start_epoch=start, history=history, checkpoint_callback=save_session, seed=config['seed'])
    else:
        history = train_diffusion(models['unet'], loader, device, 10, epochs=epochs,
                                  checkpoint_dir=str(output), optimizer=optimizers['unet'],
                                  start_epoch=start, history=history, checkpoint_callback=save_session, seed=config['seed'])
    return history, contract
