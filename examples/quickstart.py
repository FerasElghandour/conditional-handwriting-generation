"""Offline example using the public implementation; see README for evidence scope."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import torch
from digit_models import Generator, Discriminator
torch.set_num_threads(2)
torch.manual_seed(0)
labels=torch.tensor([1,2,3,4])
images=Generator()(torch.randn(4,64),labels)
scores=Discriminator()(images,labels)
print({'images':list(images.shape),'scores':list(scores.shape),'bounded':bool(images.abs().max()<=1)})
