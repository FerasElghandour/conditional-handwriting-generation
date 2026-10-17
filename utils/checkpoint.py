"""Atomic checkpoints with model/optimizer states and experiment metadata."""
from pathlib import Path
import torch

def save_checkpoint(model, optimizer=None, epoch=None, loss=None, name="model", path="checkpoints", metadata=None):
    directory = Path(path); directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{name}_epoch{epoch if epoch is not None else 'final'}.pt"
    state = {"model_state": model.state_dict(), "epoch": epoch, "loss": loss, "metadata": metadata or {}}
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    temporary = destination.with_suffix('.tmp')
    torch.save(state, temporary); temporary.replace(destination)
    print(f"Saved checkpoint: {destination}", flush=True)
    return destination

def load_checkpoint(model, optimizer=None, path=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model_state"], strict=True)
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    print(f"Loaded checkpoint: {path}")
    return model
