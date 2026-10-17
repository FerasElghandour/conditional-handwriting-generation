"""Class-conditioned sampling from original-architecture trained checkpoints."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from model.cgan import Generator
from model.diffusion import ConditionalUNet
from training.train_diffusion import sample_images

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['cgan','diffusion'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--samples',type=int,default=100);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.samples<1:raise ValueError('Positive sample count required')
 if a.output.exists():raise ValueError('Use a new output directory')
 a.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(a.seed)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);meta=ck['metadata'];labels=torch.arange(a.samples)%10
 if a.model=='cgan':
  m=Generator(z_dim=meta.get('z_dim',100));m.load_state_dict(ck['model_state'],strict=True);m.eval()
  with torch.no_grad():images=m(torch.randn(a.samples,m.z_dim),labels)
 else:
  m=ConditionalUNet(base_channels=meta.get('base_channels',32),time_dim=meta.get('time_dim',128));m.load_state_dict(ck['model_state'],strict=True)
  images,labels=sample_images(m,'cpu',a.samples,class_labels=labels,timesteps=meta.get('timesteps',200),schedule=meta.get('schedule','cosine'),seed=a.seed)
 if not torch.isfinite(images).all():raise FloatingPointError('Nonfinite generated samples')
 pixels=images.detach().clamp(-1,1).add(1).mul(127.5).round().byte().numpy()[:,0]
 grid=np.full((((a.samples+9)//10)*28,280),255,dtype=np.uint8)
 for i,im in enumerate(pixels):grid[i//10*28:(i//10+1)*28,i%10*28:(i%10+1)*28]=im
 Image.fromarray(grid).save(a.output/'grid.png');np.savez_compressed(a.output/'samples.npz',images=images.detach().numpy(),labels=labels.numpy())
 (a.output/'sampling.json').write_text(json.dumps({'model':a.model,'samples':a.samples,'seed':a.seed,'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'shape':list(images.shape),'finite':bool(torch.isfinite(images).all()),'range':[float(images.min()),float(images.max())]},indent=2)+'\n')
if __name__=='__main__':main()
