"""Lightweight DDP training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from datasets import DistributedBucketSampler, LMDBDataset
from engine import TrainerConfig, cleanup_ddp, is_master, run_training, setup_ddp
from engine.logging_utils import prepare_history_file, save_yaml_config
from models import MetaField


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_loader(dataset: LMDBDataset, sampler: DistributedBucketSampler, loader_cfg: dict) -> DataLoader:
    kwargs = {
        "batch_sampler": sampler,
        "num_workers": loader_cfg["num_workers"],
        "pin_memory": loader_cfg.get("pin_memory", True),
    }
    if loader_cfg["num_workers"] > 0:
        kwargs["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)
        kwargs["persistent_workers"] = loader_cfg.get("persistent_workers", False)
    return DataLoader(dataset, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/metafield_ddp.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    gpu, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    trainer_cfg = TrainerConfig(**config["train"])
    if is_master():
        Path(trainer_cfg.save_dir).mkdir(parents=True, exist_ok=True)
        save_yaml_config(config, Path(trainer_cfg.save_dir) / "config.yaml")
        prepare_history_file(Path(trainer_cfg.save_dir) / "history.csv")

    train_ds = LMDBDataset(config["data"]["lmdb_path"], config["data"]["split_json"], mode="train")
    val_ds = LMDBDataset(config["data"]["lmdb_path"], config["data"]["split_json"], mode="val")

    train_sampler = DistributedBucketSampler(train_ds, trainer_cfg.batch_size, num_replicas=world_size, rank=rank)
    val_sampler = DistributedBucketSampler(val_ds, trainer_cfg.batch_size, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = build_loader(train_ds, train_sampler, config["train_loader"])
    val_loader = build_loader(val_ds, val_sampler, config["val_loader"])

    model_cfg = config["model"]
    model = MetaField(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        cond_channels=model_cfg["cond_channels"],
        base_channels=model_cfg["base_channels"],
        cond_embed_dim=model_cfg["cond_embed_dim"],
        heads=model_cfg["heads"],
        max_dist=model_cfg["max_dist"],
        transformer_depth=model_cfg.get("transformer_depth", 8),
    ).to(device)

    if torch.cuda.is_available():
        model = DDP(model, device_ids=[gpu])

    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer_cfg.lr, weight_decay=config["optimizer"].get("weight_decay", 1e-4))
    steps_per_epoch = (len(train_loader) + trainer_cfg.grad_accum_steps - 1) // trainer_cfg.grad_accum_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=trainer_cfg.lr,
        total_steps=steps_per_epoch * trainer_cfg.epochs,
        pct_start=config["scheduler"].get("pct_start", 0.05),
        div_factor=config["scheduler"].get("div_factor", 25),
        final_div_factor=trainer_cfg.final_div_factor,
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    try:
        run_training(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            config=trainer_cfg,
            device=device,
            world_size=world_size,
        )
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
