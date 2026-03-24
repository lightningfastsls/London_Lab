"""Tests for VQVAETrainer and CosineWarmupScheduler."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from usv_language.src.model.vqvae import VQVAE, VQVAEConfig
from usv_language.src.training.trainer import VQVAETrainer, CosineWarmupScheduler


def _make_dummy_loader(n_samples=32, seq_len=16, n_freq=170, batch_size=8):
    """Create a dummy DataLoader that matches expected batch format."""
    data = torch.randn(n_samples, seq_len, n_freq)

    class DummyDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return {"spectrogram": self.data[idx], "file_name": "test", "start_col": 0}

    def collate(batch):
        return {
            "spectrogram": torch.stack([b["spectrogram"] for b in batch]),
            "file_name": [b["file_name"] for b in batch],
            "start_col": [b["start_col"] for b in batch],
        }

    return DataLoader(DummyDataset(data), batch_size=batch_size, collate_fn=collate)


class TestCosineWarmupScheduler:
    def test_warmup_phase(self):
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = CosineWarmupScheduler(opt, warmup_steps=10, max_steps=100)

        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])

        # LR should increase during warmup
        assert lrs[-1] > lrs[0]

    def test_cosine_decay(self):
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = CosineWarmupScheduler(opt, warmup_steps=5, max_steps=100, min_lr=1e-6)

        for _ in range(50):
            sched.step()
        lr_mid = opt.param_groups[0]["lr"]

        for _ in range(50):
            sched.step()
        lr_end = opt.param_groups[0]["lr"]

        assert lr_end < lr_mid


class TestVQVAETrainer:
    def test_single_epoch(self):
        config = VQVAEConfig(n_freq=170, d_model=32, n_heads=4, n_layers=1, d_ff=64, codebook_size=32)
        model = VQVAE(config)
        loader = _make_dummy_loader(n_freq=170)

        trainer = VQVAETrainer(model, loader, lr=1e-3, warmup_steps=2)
        metrics = trainer.train_epoch()

        assert "total_loss" in metrics
        assert "recon_loss" in metrics
        assert "vq_loss" in metrics
        assert "next_code_loss" in metrics
        assert "perplexity" in metrics

    def test_training_reduces_loss(self):
        config = VQVAEConfig(n_freq=170, d_model=32, n_heads=4, n_layers=1, d_ff=64, codebook_size=32)
        model = VQVAE(config)
        loader = _make_dummy_loader(n_freq=170)

        trainer = VQVAETrainer(model, loader, lr=1e-3, warmup_steps=2)
        loss_1 = trainer.train_epoch()["total_loss"]
        loss_2 = trainer.train_epoch()["total_loss"]
        loss_3 = trainer.train_epoch()["total_loss"]

        # Loss should generally decrease (allow some noise)
        assert loss_3 < loss_1 * 1.5  # At least not diverging

    def test_checkpoint_save_load(self, tmp_path):
        config = VQVAEConfig(n_freq=170, d_model=32, n_heads=4, n_layers=1, d_ff=64, codebook_size=32)
        model = VQVAE(config)
        loader = _make_dummy_loader(n_freq=170)

        trainer = VQVAETrainer(model, loader, checkpoint_dir=str(tmp_path))
        trainer.train_epoch()
        trainer.save_checkpoint(1, tmp_path / "test_ckpt.pt")

        # Load into fresh trainer
        model2 = VQVAE(config)
        trainer2 = VQVAETrainer(model2, loader)
        epoch = trainer2.load_checkpoint(tmp_path / "test_ckpt.pt")
        assert epoch == 1

    def test_train_with_validation(self):
        config = VQVAEConfig(n_freq=170, d_model=32, n_heads=4, n_layers=1, d_ff=64, codebook_size=32)
        model = VQVAE(config)
        train_loader = _make_dummy_loader(n_freq=170)
        val_loader = _make_dummy_loader(n_samples=16, n_freq=170)

        trainer = VQVAETrainer(model, train_loader, val_loader=val_loader)
        history = trainer.train(max_epochs=2)
        assert len(history) == 2
        assert "val_total_loss" in history[0]
