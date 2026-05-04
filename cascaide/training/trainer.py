import json
import math
import time
import copy
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import numpy as np
from torch.utils.data import Subset

def stratified_energy_split(encoded_dataset,
                              val_fraction: float,
                              n_bins: int = 5,
                              seed: int = 42,
                              verbose: bool = True):

    n = len(encoded_dataset)

    energies = np.array([
        float(encoded_dataset[i]["energy"]) for i in range(n)
    ])

    edges = np.quantile(energies, np.linspace(0, 1, n_bins + 1))
    edges[-1] = edges[-1] + 1e-9
    bin_idx = np.digitize(energies, edges[1:-1])  # 0..n_bins-1

    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for b in range(n_bins):
        in_bin = np.where(bin_idx == b)[0]
        if len(in_bin) == 0:
            continue
        rng.shuffle(in_bin)
        n_val = max(1, int(round(len(in_bin) * val_fraction)))
        val_idx.extend(in_bin[:n_val].tolist())
        train_idx.extend(in_bin[n_val:].tolist())

    train_idx = sorted(train_idx)
    val_idx   = sorted(val_idx)

    if verbose:
        print(f"[split] {n} samples, {n_bins} energy bins, "
              f"val_fraction={val_fraction}")
        for b in range(n_bins):
            in_bin = np.where(bin_idx == b)[0]
            n_v = sum(1 for i in val_idx if bin_idx[i] == b)
            n_t = sum(1 for i in train_idx if bin_idx[i] == b)
            lo, hi = edges[b], edges[b + 1]
            print(f"  bin {b}: E∈[{lo:.4f}, {hi:.4f}]  "
                  f"total={len(in_bin):4d}  train={n_t:4d}  val={n_v:3d}")
        print(f"  → train={len(train_idx)}  val={len(val_idx)}")

    return Subset(encoded_dataset, train_idx), Subset(encoded_dataset, val_idx)

def build_encoder(cfg, centroid):
    name = cfg.encoder.name
    params = getattr(cfg.encoder, name).__dict__.copy()
    if name == "Base":
        from cascaide.encoding.base_image import BaseImageEncoder
        return BaseImageEncoder(centroid=centroid, **params)
    if name == "Hilbert4Ch":
        from cascaide.encoding.hilbert4ch_image import Hilbert4ChEncoder
        return Hilbert4ChEncoder(centroid=centroid, **params)
    if name == "Hilbert3Ch":
        from cascaide.encoding.hilbert3ch_image import Hilbert3ChEncoder
        return Hilbert3ChEncoder(centroid=centroid, **params)
    raise ValueError(f"Unknown encoder: {name}")


def build_diffusion(cfg):
    from cascaide.diffusion.schedules import (LinearSchedule,
                                                CosineSchedule, SigmoidSchedule)
    from cascaide.diffusion.core import GaussianDiffusion
    sched_map = {"linear": LinearSchedule, "cosine": CosineSchedule,
                 "sigmoid": SigmoidSchedule}
    schedule = sched_map[cfg.diffusion.schedule]()
    return GaussianDiffusion(T=cfg.diffusion.T, schedule=schedule,
                              parameterization=cfg.diffusion.parameterization)


def build_conditioner(cfg, encoder):
    name = cfg.conditioner.name
    if name == "none":
        return None
    if name == "energy_channel":
        from cascaide.diffusion.conditioner import EnergyChannelConditioner
        return EnergyChannelConditioner(n_channels=encoder.output_shape[0])
    if name == "energy_vector":
        from cascaide.diffusion.conditioner import EnergyVectorConditioner
        return EnergyVectorConditioner(embed_dim=cfg.conditioner.embed_dim)
    raise ValueError(f"Unknown conditioner: {name}")


def build_loss(cfg, diffusion, encoder):
    from cascaide.diffusion.loss import (DiffusionLoss, OccupancyBCELoss,
                                           CountLoss, ClassificationLoss,
                                           MultiAxisProjectionAuxLoss)
    aux = []
    a = cfg.loss.aux_losses

    if a.occupancy.enabled:
        aux.append(OccupancyBCELoss(weight=a.occupancy.weight,
                                      sharpness=a.occupancy.sharpness))
    if a.count.enabled:
        aux.append(CountLoss(weight=a.count.weight,
                              separate_classes=a.count.separate_classes))
    if a.classification.enabled:
        aux.append(ClassificationLoss(weight=a.classification.weight))
    if a.projection.enabled:
        aux.append(MultiAxisProjectionAuxLoss(
            weight=a.projection.weight,
            apply_every=a.projection.apply_every,
            projection_size=a.projection.projection_size,
            sigma=a.projection.sigma,
            start_epoch=getattr(a.projection, "start_epoch", 0),
            t_threshold_frac=getattr(a.projection, "t_threshold_frac", 1.0),
            diffusion_T=cfg.diffusion.T,
        ))

    return DiffusionLoss(diffusion=diffusion, encoder=encoder,
                          defect_weight=cfg.loss.defect_weight,
                          aux_losses=aux)

def build_sampler(cfg):
    from cascaide.diffusion.samplers import DDPMSampler, DDIMSampler
    if cfg.sampling.sampler == "ddim":
        return DDIMSampler(n_steps=cfg.sampling.ddim_steps,
                            eta=cfg.sampling.ddim_eta)
    return DDPMSampler()


def get_lr(step, cfg, total_steps):
    base = cfg.optimizer.lr
    warmup = cfg.scheduler.warmup_steps
    if step < warmup:
        return base * step / max(warmup, 1)
    if cfg.scheduler.type == "none":
        return base
    progress = (step - warmup) / max(total_steps - warmup, 1)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.scheduler.min_lr + (base - cfg.scheduler.min_lr) * cos


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)

class Trainer:
    def __init__(self, cfg, raw_cfg):
        from cascaide.data.dataset import (CascadeDataset, EncodedCascadeDataset,
                                            collate_fn)
        from cascaide.diffusion.model import UNet
        from cascaide.configs.config import set_seed, make_run_dir

        self.cfg = cfg
        self.raw_cfg = raw_cfg
        set_seed(cfg.experiment.seed)
        self.run_dir = make_run_dir(cfg.experiment.output_dir,
                                     cfg.experiment.name)
        self.device = torch.device(cfg.experiment.device
                                     if torch.cuda.is_available() else "cpu")

        # Save config snapshot
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(raw_cfg, f, indent=2)

        # ----- Data -----
        raw_ds = CascadeDataset(data_root=cfg.data.data_root,
                                 max_samples=cfg.data.max_samples)
        self.encoder = build_encoder(cfg, raw_ds.global_centroid)
        encoded = EncodedCascadeDataset(
            raw_ds, {self.encoder.name: self.encoder},
            energy_norm_factor=cfg.data.energy_norm_factor)

        split_strategy = getattr(cfg.data, "split_strategy", "random")
        if split_strategy == "stratified_energy":
            n_bins = getattr(cfg.data, "n_energy_bins", 5)
            self.train_set, self.val_set = stratified_energy_split(
                encoded,
                val_fraction=cfg.data.val_fraction,
                n_bins=n_bins,
                seed=cfg.experiment.seed,
                verbose=True,
            )
        else:
            n_val = max(1, int(len(encoded) * cfg.data.val_fraction))
            n_train = len(encoded) - n_val
            self.train_set, self.val_set = random_split(
                encoded, [n_train, n_val],
                generator=torch.Generator().manual_seed(cfg.experiment.seed))

        self.train_loader = DataLoader(
            self.train_set, batch_size=cfg.data.batch_size,
            shuffle=True, num_workers=cfg.data.num_workers,
            pin_memory=True, drop_last=True,
            collate_fn=collate_fn)  # ← added

        self.val_loader = DataLoader(
            self.val_set, batch_size=cfg.data.batch_size,
            shuffle=False, num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=collate_fn)
        # self.train_loader = DataLoader(
        #     self.train_set, batch_size=cfg.data.batch_size,
        #     shuffle=True, num_workers=cfg.data.num_workers,
        #     pin_memory=True, drop_last=True)
        # self.val_loader = DataLoader(
        #     self.val_set, batch_size=cfg.data.batch_size,
        #     shuffle=False, num_workers=cfg.data.num_workers,
        #     pin_memory=True)


        self.conditioner = build_conditioner(cfg, self.encoder)


        self.model = UNet(
            encoder_shape=self.encoder.output_shape,
            conditioner=self.conditioner,
            base_channels=cfg.model.base_channels,
            channel_mults=tuple(cfg.model.channel_mults),
            num_res_blocks=cfg.model.num_res_blocks,
            attn_levels=tuple(cfg.model.attn_levels),
            time_embed_dim=cfg.model.time_embed_dim,
            dropout=cfg.model.dropout,
        ).to(self.device)


        C, H, W = self.encoder.output_shape
        n_downs = len(cfg.model.channel_mults) - 1
        assert H % (2 ** n_downs) == 0, (
            f"Encoder H={H} not divisible by 2^{n_downs}. "
            f"Reduce channel_mults length in YAML.")


        self.diffusion = build_diffusion(cfg).to(self.device)
        self.loss_fn = build_loss(cfg, self.diffusion, self.encoder).to(self.device)
        self.sampler = build_sampler(cfg)


        self.opt = AdamW(self.model.parameters(),
                          lr=cfg.optimizer.lr,
                          betas=tuple(cfg.optimizer.betas),
                          weight_decay=cfg.optimizer.weight_decay)


        self.ema = EMA(self.model, decay=cfg.ema.decay) if cfg.ema.enabled else None

        self.epoch = 0
        self.global_step = 0
        self.total_steps = cfg.training.num_epochs * len(self.train_loader)

        self.best_metric_name = getattr(cfg.training, "best_metric", "val_total")
        self.best_mode        = getattr(cfg.training, "best_mode", "min")
        self.save_best        = getattr(cfg.training, "save_best", True)
        self.best_value = float("inf") if self.best_mode == "min" else float("-inf")
        self.best_epoch = -1

        if cfg.training.resume_from:
            self.load_checkpoint(cfg.training.resume_from)

        self.log_path = self.run_dir / "train_log.jsonl"

    def _to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, list) and len(v) and torch.is_tensor(v[0]):
                out[k] = [t.to(self.device, non_blocking=True) for t in v]
            else:
                out[k] = v
        return out

    def _x0_from_batch(self, batch):
        return batch[self.encoder.name]

    def _cond_from_batch(self, batch):
        return {"energy": batch["energy"].to(self.device).float()}

    def _is_better(self, value):
        if self.best_mode == "min":
            return value < self.best_value
        return value > self.best_value

    def train_step(self, batch):
        batch = self._to_device(batch)
        x0 = self._x0_from_batch(batch)
        cond = self._cond_from_batch(batch)

        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        loss, info = self.loss_fn(self.model, x0, cond=cond, batch=batch)
        loss.backward()

        if self.cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                              self.cfg.training.grad_clip)

        # LR schedule
        lr = get_lr(self.global_step, self.cfg, self.total_steps)
        for g in self.opt.param_groups:
            g["lr"] = lr

        self.opt.step()
        if self.ema is not None:
            self.ema.update(self.model)

        info["lr"] = lr
        return info

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        agg = {}
        n = 0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            x0 = self._x0_from_batch(batch)
            cond = self._cond_from_batch(batch)
            _, info = self.loss_fn(self.model, x0, cond=cond, batch=batch)
            for k, v in info.items():
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0.0) + v
            n += 1
        return {f"val_{k}": v / max(n, 1) for k, v in agg.items()}


    @torch.no_grad()
    def sample_and_save(self, tag):
        sample_model = self.ema.shadow if self.ema is not None else self.model
        sample_model.eval()

        n = self.cfg.sampling.n_samples
        energies = torch.linspace(0.1, 1.0, n).to(self.device)
        cond = {"energy": energies}
        shape = (n,) + self.encoder.output_shape

        imgs = self.sampler.sample(sample_model, self.diffusion, shape,
                                     cond=cond, device=self.device)
        out_path = self.run_dir / "samples" / f"samples_{tag}.pt"
        torch.save({"images": imgs.cpu(), "energies": energies.cpu()}, out_path)

    def _build_checkpoint_dict(self):
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict() if self.ema else None,
            "config": self.raw_cfg,
            "best_metric_name": self.best_metric_name,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
        }

    def save_checkpoint(self, tag):
        ckpt = self._build_checkpoint_dict()
        path = self.run_dir / "checkpoints" / f"ckpt_{tag}.pt"
        torch.save(ckpt, path)
        torch.save(ckpt, self.run_dir / "checkpoints" / "latest.pt")
        print(f"[ckpt] saved {path}")

    def maybe_save_best(self, val_info):
        """Save 'best.pt' if the tracked metric improved."""
        if not self.save_best:
            return False
        if self.best_metric_name not in val_info:
            print(f"[best] metric '{self.best_metric_name}' not in val_info; "
                  f"skipping (available: {list(val_info.keys())})")
            return False

        current = val_info[self.best_metric_name]
        if not self._is_better(current):
            return False

        prev = self.best_value
        self.best_value = current
        self.best_epoch = self.epoch

        ckpt = self._build_checkpoint_dict()
        path = self.run_dir / "checkpoints" / "best.pt"
        torch.save(ckpt, path)

        prev_str = "inf" if math.isinf(prev) else f"{prev:.4f}"
        print(f"[best] {self.best_metric_name}={current:.4f} "
              f"(prev {prev_str}) — saved {path}")

        self.log({"phase": "best",
                  "epoch": self.epoch,
                  "metric": self.best_metric_name,
                  "value": current,
                  "prev": prev if not math.isinf(prev) else None})
        return True

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        if self.ema is not None and ckpt.get("ema") is not None:
            self.ema.load_state_dict(ckpt["ema"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]

        # Restore best-tracker state if present
        if "best_value" in ckpt:
            saved_metric = ckpt.get("best_metric_name", self.best_metric_name)
            if saved_metric != self.best_metric_name:
                print(f"[ckpt] best metric changed: '{saved_metric}' -> "
                      f"'{self.best_metric_name}'; resetting tracker")
            else:
                self.best_value = ckpt["best_value"]
                self.best_epoch = ckpt.get("best_epoch", -1)
                print(f"[ckpt] resumed best {self.best_metric_name}="
                      f"{self.best_value:.4f} (epoch {self.best_epoch})")

        print(f"[ckpt] resumed from {path} (epoch {self.epoch})")

    def log(self, info):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(info) + "\n")

    def fit(self):
        cfg = self.cfg
        print(f"Training {cfg.experiment.name} on {self.device}")
        print(f"  Encoder: {self.encoder.name}, shape {self.encoder.output_shape}")
        print(f"  Train/val: {len(self.train_set)}/{len(self.val_set)}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Best-tracker: {self.best_metric_name} ({self.best_mode})")

        for epoch in range(self.epoch, cfg.training.num_epochs):
            self.epoch = epoch
            t0 = time.time()
            running = {}

            for aux in self.loss_fn.aux_losses:
                if hasattr(aux, "set_epoch"):
                    aux.set_epoch(epoch)

            for batch in self.train_loader:
                info = self.train_step(batch)
                self.global_step += 1

                for k, v in info.items():
                    if isinstance(v, (int, float)):
                        running[k] = running.get(k, 0.0) + v

                if self.global_step % cfg.training.log_every == 0:
                    avg = {k: v / cfg.training.log_every for k, v in running.items()}
                    avg["epoch"] = epoch
                    avg["step"] = self.global_step
                    print(f"[ep {epoch:3d} step {self.global_step:6d}] " +
                          " ".join(f"{k}={v:.4f}" for k, v in avg.items()
                                    if k not in ("epoch", "step")))
                    self.log({"phase": "train", **avg})
                    running = {}

            # Per-epoch validation
            if (epoch + 1) % cfg.training.val_every == 0:
                val_info = self.validate()
                val_info["epoch"] = epoch
                print(f"[val ep {epoch:3d}] " +
                      " ".join(f"{k}={v:.4f}" for k, v in val_info.items()
                                if k != "epoch"))
                self.log({"phase": "val", **val_info})

                # Best-model check happens immediately after val
                self.maybe_save_best(val_info)

            # Sampling
            if (cfg.sampling.enabled and
                    (epoch + 1) % cfg.sampling.every == 0):
                self.sample_and_save(f"ep{epoch:04d}")

            # Periodic checkpoint
            if (epoch + 1) % cfg.training.ckpt_every == 0:
                self.save_checkpoint(f"ep{epoch:04d}")

            print(f"  epoch {epoch} took {time.time()-t0:.1f}s")

        # Final checkpoint
        self.save_checkpoint("final")
        if self.save_best and self.best_epoch >= 0:
            print(f"Training complete. "
                  f"Best {self.best_metric_name}={self.best_value:.4f} "
                  f"at epoch {self.best_epoch}.")
        else:
            print("Training complete.")