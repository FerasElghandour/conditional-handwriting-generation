"""Compatibility and finite gradient checks, not a generator quality benchmark."""
from pathlib import Path
import torch
from model.cgan import Generator,Discriminator
from model.diffusion import ConditionalUNet

torch.set_num_threads(2);torch.manual_seed(42)
labels=torch.tensor([0,9]);generator=Generator();discriminator=Discriminator();unet=ConditionalUNet(base_channels=32)
for name,model in [('cgan_generator',generator),('cgan_discriminator',discriminator),('diffusion_unet',unet)]:
 ck=torch.load(Path('checkpoints')/(name+'.pt'),map_location='cpu',weights_only=True)
 model.load_state_dict(ck['model_state'],strict=True)
noise=torch.randn(2,100);fake=generator(noise,labels)
assert fake.shape==(2,1,28,28) and torch.isfinite(fake).all()
loss=torch.nn.functional.binary_cross_entropy(discriminator(fake,labels),torch.ones(2,1));loss.backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in generator.parameters())
pred=unet(torch.randn(2,1,28,28),torch.tensor([0,199]),labels)
assert pred.shape==(2,1,28,28) and torch.isfinite(pred).all()
pred.square().mean().backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in unet.parameters())
print('Strict recovered architecture loads, conditional outputs and finite backward paths passed.')
