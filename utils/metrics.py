"""Standard Inception FID and classifiers trained exclusively on synthetic images."""
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

def compute_fid(real_images,fake_images,device='cuda',batch_size=64):
    from torchmetrics.image.fid import FrechetInceptionDistance
    metric=FrechetInceptionDistance(feature=2048,normalize=True).to(device)
    # TorchMetrics accumulates feature moments in float64 internally.
    for real,images in [(True,real_images),(False,fake_images)]:
        for start in range(0,len(images),batch_size):
            batch=images[start:start+batch_size].to(device).clamp(-1,1).add(1).div(2).repeat(1,3,1,1)
            metric.update(batch,real=real)
        print(f'FID processed {len(images)} {"real" if real else "synthetic"} images',flush=True)
    return float(metric.compute().cpu())

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                nn.Conv2d(64,128,3,padding=1),nn.ReLU())
        self.fc=nn.Sequential(nn.Flatten(),nn.Linear(128*7*7,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x): return self.fc(self.conv(x))

def train_classifier(images,labels,device='cuda',epochs=20,seed=2026):
    torch.manual_seed(seed)
    model=SimpleCNN().to(device)
    loader=DataLoader(TensorDataset(images,labels),batch_size=128,shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    history=[]
    for epoch in range(epochs):
        model.train(); total=count=0
        for batch,y in loader:
            batch,y=batch.to(device),y.to(device)
            loss=nn.functional.cross_entropy(model(batch),y)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total+=loss.item()*len(y); count+=len(y)
        history.append(total/count)
        print(f'Synthetic classifier {epoch+1}/{epochs}: mean CE={history[-1]:.5f}',flush=True)
    return model,history

@torch.no_grad()
def test_accuracy(model,loader,device='cuda'):
    model.eval(); correct=total=0; confusion=torch.zeros(10,10,dtype=torch.long)
    for images,labels in loader:
        predicted=model(images.to(device)).argmax(1).cpu()
        correct+=int((predicted==labels).sum()); total+=len(labels)
        confusion+=torch.bincount(labels*10+predicted,minlength=100).view(10,10)
    return correct/total,correct,total,confusion
