"""Train a classifier on saved synthetic samples; test once on real MNIST test."""
import argparse,json,hashlib
from pathlib import Path
import torch,numpy as np
from torch.utils.data import DataLoader,TensorDataset
from utils.metrics import train_classifier,test_accuracy,compute_fid

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--samples',type=Path,required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--output',type=Path,required=True);p.add_argument('--fid',action='store_true');a=p.parse_args()
 if a.epochs<1 or a.output.exists():raise ValueError('Positive epochs and new output directory required')
 a.output.mkdir(parents=True);torch.set_num_threads(2)
 with np.load(a.samples,allow_pickle=False) as z:
  raw_x,raw_y=z['images'],z['labels']
  if raw_y.ndim!=1 or not np.issubdtype(raw_y.dtype,np.integer) or len(raw_y)<2:raise ValueError('Nonempty integer label vector required')
  x=torch.from_numpy(raw_x).float();y=torch.from_numpy(raw_y).long()
 if hashlib.sha256(a.data.read_bytes()).hexdigest()!='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1':raise ValueError('Expected canonical Keras MNIST archive')
 with np.load(a.data,allow_pickle=False) as z:tx=torch.from_numpy(z['x_test']).unsqueeze(1).float()/127.5-1;ty=torch.from_numpy(z['y_test']).long()
 if x.ndim!=4 or x.shape[1:]!=(1,28,28) or len(x)!=len(y) or not torch.isfinite(x).all() or x.min() < -1 or x.max() > 1 or not torch.isin(y,torch.arange(10)).all():raise ValueError('Invalid synthetic samples')
 model,history=train_classifier(x,y,device='cpu',epochs=a.epochs)
 accuracy,correct,total,confusion=test_accuracy(model,DataLoader(TensorDataset(tx,ty),batch_size=128),'cpu')
 result={'protocol':'classifier trained only on synthetic samples; fixed real MNIST test split','synthetic_rows':len(y),'test_rows':total,'correct':correct,'accuracy':accuracy,'confusion_matrix':confusion.tolist(),'epochs':a.epochs,'classifier_seed':2026,'history':history,'samples_sha256':hashlib.sha256(a.samples.read_bytes()).hexdigest(),'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest()}
 if a.fid:
  if len(x)<2049:raise ValueError('FID2048 requires at least2049 synthetic samples in this protocol')
  result['fid_2048']=compute_fid(tx,x,'cpu')
 torch.save(model.state_dict(),a.output/'classifier.pt');(a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
