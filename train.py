"""Train the recovered original MLP cGAN or residual U-Net DDPM on MNIST."""
import argparse,hashlib,json,torch
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
from model.cgan import Generator,Discriminator
from model.diffusion import ConditionalUNet
from training.train_cgan import train_cgan
from training.train_diffusion import train_diffusion

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--model',choices=['cgan','diffusion'],required=True);p.add_argument('--epochs',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--limit-train',type=int);p.add_argument('--device',default='cpu');a=p.parse_args()
 if a.epochs<1 or a.batch_size<2 or (a.limit_train is not None and a.limit_train<2):raise ValueError('Positive budget and at least two examples required')
 if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use a new output directory')
 a.output.mkdir(parents=True,exist_ok=True);torch.manual_seed(42);torch.set_num_threads(2)
 if hashlib.sha256(a.data.read_bytes()).hexdigest()!='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1':raise ValueError('Expected canonical Keras MNIST archive')
 with np.load(a.data,allow_pickle=False) as z:x=z['x_train'][:a.limit_train];y=z['y_train'][:a.limit_train]
 if x.ndim!=3 or x.shape[1:]!=(28,28) or x.dtype!=np.uint8 or len(x)!=len(y) or not np.isin(y,range(10)).all():raise ValueError('Expected canonical MNIST uint8 images and labels')
 ds=TensorDataset(torch.from_numpy(x).unsqueeze(1).float().div(127.5).sub(1),torch.from_numpy(y).long())
 loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(42),drop_last=True)
 if not len(loader):raise ValueError('Batch exceeds available training rows')
 if a.model=='cgan':
  g,d=Generator().to(a.device),Discriminator().to(a.device)
  history=train_cgan(g,d,loader,torch.optim.Adam(g.parameters(),lr=.0002,betas=(.5,.999)),torch.optim.Adam(d.parameters(),lr=.0002,betas=(.5,.999)),torch.nn.BCELoss(),a.device,100,10,a.epochs,str(a.output))
 else:
  m=ConditionalUNet(base_channels=32).to(a.device)
  history=train_diffusion(m,loader,a.device,10,epochs=a.epochs,checkpoint_dir=str(a.output))
 (a.output/'run.json').write_text(json.dumps({'model':a.model,'architecture':'original MLP cGAN / residual U-Net, not compact digit_models.py','seed':42,'train_rows_available':len(ds),'rows_per_epoch':len(loader)*a.batch_size,'drop_last':True,'test_used_for_training':False,'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'torch':torch.__version__,'history':history},indent=2)+'\n')
if __name__=='__main__':main()
