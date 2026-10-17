import torch
from digit_models import Generator,Discriminator,ConditionalDenoiser,noise_prediction_loss
torch.set_num_threads(2);labels=torch.tensor([1,2,3,4]);g=Generator();d=Discriminator()
images=g(torch.randn(4,64),labels)
assert images.shape==(4,1,28,28) and images.abs().max()<=1
scores=d(images,labels);assert scores.shape==(4,1)
torch.nn.functional.binary_cross_entropy_with_logits(scores,torch.ones_like(scores)).backward()
assert g.project.weight.grad.abs().sum()>0
model=ConditionalDenoiser();loss=noise_prediction_loss(model,torch.randn(4,1,28,28),labels)
loss.backward();assert torch.isfinite(loss) and model.input.weight.grad.abs().sum()>0
print('Conditioned image shapes, bounded generation, adversarial gradients, and diffusion gradients passed.')
