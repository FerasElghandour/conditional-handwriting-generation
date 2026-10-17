"""Conditional GAN training with separate BCE updates and checkpoint files."""
import time
import torch
from utils.checkpoint import save_checkpoint

def train_cgan(generator, discriminator, dataloader, optimizer_G, optimizer_D,
               criterion, device, z_dim, num_classes, epochs=50, checkpoint_dir="checkpoints",
               start_epoch=0, history=None):
    generator.train(); discriminator.train()
    history = list(history or [])
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter(); total_g = total_d = count = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            batch = len(labels)
            valid = torch.ones(batch, 1, device=device)
            fake = torch.zeros(batch, 1, device=device)
            # D update: detached samples cannot update G.
            optimizer_D.zero_grad(set_to_none=True)
            optimizer_G.zero_grad(set_to_none=True)
            with torch.no_grad():
                generated_labels = torch.randint(num_classes, (batch,), device=device)
                generated = generator(torch.randn(batch, z_dim, device=device), generated_labels)
            real_loss = criterion(discriminator(imgs, labels), valid)
            fake_loss = criterion(discriminator(generated, generated_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward(); optimizer_D.step()
            optimizer_D.zero_grad(set_to_none=True)
            # G update: differentiate through D's input, but freeze D's parameters.
            discriminator.requires_grad_(False)
            optimizer_G.zero_grad(set_to_none=True)
            generated_labels = torch.randint(num_classes, (batch,), device=device)
            generated = generator(torch.randn(batch, z_dim, device=device), generated_labels)
            g_loss = criterion(discriminator(generated, generated_labels), valid)
            g_loss.backward(); optimizer_G.step()
            discriminator.requires_grad_(True)
            if not torch.isfinite(g_loss + d_loss):
                raise FloatingPointError("Non-finite GAN loss")
            total_g += g_loss.item() * batch; total_d += d_loss.item() * batch; count += batch
        record = {"epoch": epoch + 1, "g_loss": total_g/count, "d_loss": total_d/count,
                  "examples": count, "seconds": time.perf_counter()-started}
        history.append(record)
        print(f"cGAN {epoch+1}/{epochs}: D={record['d_loss']:.5f}, G={record['g_loss']:.5f}, n={count}, {record['seconds']:.1f}s", flush=True)
        metadata = {"z_dim": z_dim, "num_classes": num_classes, "history": history, "seed":42}
        # Latest checkpoints permit recovery; numbered snapshots every ten epochs.
        for model, optimizer, name, loss in [(generator,optimizer_G,"cgan_generator",record['g_loss']),
                                            (discriminator,optimizer_D,"cgan_discriminator",record['d_loss'])]:
            save_checkpoint(model, optimizer, "latest", loss, name, checkpoint_dir, metadata | {"completed_epochs":epoch+1})
            if (epoch+1) % 10 == 0:
                save_checkpoint(model, optimizer, epoch+1, loss, name, checkpoint_dir, metadata)
    save_checkpoint(generator, optimizer_G, "final", history[-1]['g_loss'], "cgan_generator", checkpoint_dir, metadata)
    save_checkpoint(discriminator, optimizer_D, "final", history[-1]['d_loss'], "cgan_discriminator", checkpoint_dir, metadata)
    return history
