"""Conditional DDPM: direct forward noising, noise MSE, and full reverse sampling."""
import math
import time
import torch
import torch.nn.functional as F
from utils.checkpoint import save_checkpoint

def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, .02, timesteps)

def cosine_beta_schedule(timesteps, s=.008):
    # Nichol & Dhariwal (2021), alpha_bar(t) = cos^2((t/T+s)/(1+s)*pi/2).
    steps = torch.arange(timesteps+1, dtype=torch.float64) / timesteps
    abar = torch.cos((steps+s)/(1+s)*math.pi/2).square()
    abar = abar/abar[0]
    return (1-abar[1:]/abar[:-1]).clamp(max=.999).float()

def make_schedule(timesteps=200, schedule="cosine", device="cpu"):
    if schedule not in ("linear", "cosine"):
        raise ValueError("Schedule must be linear or cosine")
    betas = (cosine_beta_schedule(timesteps) if schedule == "cosine" else linear_beta_schedule(timesteps)).to(device)
    alphas = 1-betas; abar = torch.cumprod(alphas, 0)
    previous = torch.cat([torch.ones(1, device=device), abar[:-1]])
    variance = betas*(1-previous)/(1-abar)
    return betas, alphas, abar, previous, variance

def q_sample(images, t, noise, abar):
    selected = abar[t].view(-1,1,1,1)
    return selected.sqrt()*images + (1-selected).sqrt()*noise

def reverse_step(images, predicted_noise, index, coefficients, noise=None):
    betas, alphas, abar, previous, variance = coefficients
    # Predict bounded x0, then use q(x_{t-1}|x_t,x0); fixed-small DDPM variance.
    clean = ((images-(1-abar[index]).sqrt()*predicted_noise)/abar[index].sqrt()).clamp(-1,1)
    mean = (betas[index]*previous[index].sqrt()/(1-abar[index]))*clean
    mean += ((1-previous[index])*alphas[index].sqrt()/(1-abar[index]))*images
    if index > 0:
        noise = torch.randn_like(images) if noise is None else noise
        mean = mean + variance[index].sqrt()*noise
    return mean

@torch.no_grad()
def sample_images(model, device, num_samples=16, num_classes=10, img_size=(1,28,28),
                  timesteps=200, class_labels=None, schedule="cosine", seed=None):
    was_training = model.training; model.eval()
    # fork_rng preserves training RNG when grids are generated during experiments.
    devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            torch.manual_seed(seed)
        labels = (torch.arange(num_samples, device=device)%num_classes if class_labels is None
                  else class_labels.to(device=device, dtype=torch.long))
        if labels.shape != (num_samples,):
            raise ValueError("Exactly one class label per image is required")
        images = torch.randn(num_samples, *img_size, device=device)
        coefficients = make_schedule(timesteps, schedule, device)
        for index in reversed(range(timesteps)):
            t = torch.full((num_samples,), index, device=device, dtype=torch.long)
            images = reverse_step(images, model(images,t,labels), index, coefficients)
        if not torch.isfinite(images).all():
            raise FloatingPointError("Non-finite generated images")
    model.train(was_training)
    return images, labels

def train_diffusion(model, dataloader, device, num_classes, timesteps=200, epochs=40, lr=1e-4,
                    schedule="cosine", checkpoint_dir="checkpoints", optimizer=None, start_epoch=0, history=None):
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    abar = make_schedule(timesteps, schedule, device)[2]
    model.train(); history = list(history or [])
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter(); total = count = 0
        for batch_index, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            t = torch.randint(timesteps, (len(labels),), device=device)
            noise = torch.randn_like(images)
            noisy = q_sample(images,t,noise,abar)
            predicted = model(noisy,t,labels)
            loss = F.mse_loss(predicted,noise)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite diffusion loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            total += loss.item()*len(labels); count += len(labels)
            if (batch_index+1)%100 == 0:
                print(f"DDPM {epoch+1}/{epochs}, batch {batch_index+1}/{len(dataloader)}, mean MSE={total/count:.5f}",flush=True)
        record = {"epoch":epoch+1,"loss":total/count,"examples":count,"seconds":time.perf_counter()-started}
        history.append(record)
        print(f"DDPM epoch {epoch+1}: MSE={record['loss']:.6f}, n={count}, {record['seconds']:.1f}s",flush=True)
        metadata = {"completed_epochs":epoch+1,"timesteps":timesteps,"schedule":schedule,
                    "base_channels":model.down1.conv1.out_channels,"time_dim":model.time_mlp[0].in_features,
                    "num_classes":num_classes,"history":history,"seed":42}
        save_checkpoint(model,optimizer,"latest",record['loss'],"diffusion_unet",checkpoint_dir,metadata)
        if (epoch+1)%5 == 0:
            save_checkpoint(model,optimizer,epoch+1,record['loss'],"diffusion_unet",checkpoint_dir,metadata)
    save_checkpoint(model,optimizer,"final",history[-1]['loss'],"diffusion_unet",checkpoint_dir,metadata)
    return history
