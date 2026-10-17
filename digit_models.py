"""Conditioned digit synthesis components; model weights are not bundled."""
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self,latent=64,classes=10):
        super().__init__();self.latent=latent
        self.labels=nn.Embedding(classes,16)
        self.project=nn.Linear(latent+16,64*7*7)
        self.image=nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),nn.ConvTranspose2d(32,1,4,2,1),nn.Tanh())
    def forward(self,z,labels):
        h=self.project(torch.cat((z,self.labels(labels)),dim=1))
        return self.image(h.reshape(-1,64,7,7))

class Discriminator(nn.Module):
    def __init__(self,classes=10):
        super().__init__();self.labels=nn.Embedding(classes,28*28)
        self.score=nn.Sequential(nn.Conv2d(2,32,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(32,64,4,2,1),nn.LeakyReLU(.2),nn.Flatten(),nn.Linear(64*7*7,1))
    def forward(self,image,labels):
        label_map=self.labels(labels).reshape(-1,1,28,28)
        return self.score(torch.cat((image,label_map),dim=1))

class ConditionalDenoiser(nn.Module):
    def __init__(self,width=32,classes=10,steps=200):
        super().__init__();self.steps=steps;self.label=nn.Embedding(classes,width)
        self.time=nn.Sequential(nn.Linear(1,width),nn.SiLU(),nn.Linear(width,width))
        self.input=nn.Conv2d(1,width,3,padding=1)
        self.output=nn.Sequential(nn.SiLU(),nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,1,3,padding=1))
    def forward(self,image,timesteps,labels):
        condition=self.label(labels)+self.time(timesteps.float().reshape(-1,1)/self.steps)
        return self.output(self.input(image)+condition[:,:,None,None])

def noise_prediction_loss(model,images,labels,steps=200):
    if steps<2:raise ValueError('At least two diffusion steps required')
    grid=torch.linspace(0,1,steps+1,device=images.device)
    cumulative=torch.cos((grid+.008)/1.008*torch.pi/2).square()
    cumulative=(cumulative/cumulative[0]).clamp(1e-5,1)
    t=torch.randint(1,steps+1,(len(images),),device=images.device)
    alpha=cumulative[t,None,None,None];noise=torch.randn_like(images)
    noisy=alpha.sqrt()*images+(1-alpha).sqrt()*noise
    return nn.functional.mse_loss(model(noisy,t,labels),noise)
