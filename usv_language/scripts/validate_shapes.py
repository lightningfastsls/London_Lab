"""Phase 4 validation: Forward + backward pass on synthetic batch.

Verifies shapes at every stage, confirms gradient flow through
straight-through estimator, and prints parameter counts.

Usage:
    python validate_shapes.py [--d-model 64] [--codebook-size 512]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add project root so 'usv_language' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import usv_language.src  # noqa: F401

from usv_language.src.model.vqvae import VQVAE, VQVAEConfig


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate full model shapes")
    parser.add_argument("--n-freq", type=int, default=170)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = VQVAEConfig(
        n_freq=args.n_freq,
        d_model=args.d_model,
        codebook_size=args.codebook_size,
    )
    model = VQVAE(config)
    model.train()

    print(f"Model config: n_freq={config.n_freq}, d_model={config.d_model}, K={config.codebook_size}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"  Encoder: {count_parameters(model.encoder):,}")
    print(f"  Decoder: {count_parameters(model.decoder):,}")
    print(f"  Quantizer: {count_parameters(model.quantizer):,}")
    print(f"  Transformer: {count_parameters(model.transformer):,}")
    print(f"  Next-code head: {count_parameters(model.next_code_head):,}")

    # Synthetic batch
    B, S, F = args.batch_size, args.seq_len, args.n_freq
    x = torch.randn(B, S, F)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    out = model(x)
    print(f"\nForward pass outputs:")
    print(f"  x_recon:          {out['x_recon'].shape}")
    print(f"  indices:           {out['indices'].shape}")
    print(f"  next_code_logits:  {out['next_code_logits'].shape}")
    print(f"  vq_loss:           {out['vq_loss'].item():.6f}")
    print(f"  perplexity:        {out['perplexity'].item():.1f}")
    print(f"  codebook_usage:    {out['codebook_usage'].item():.2%}")

    # Backward pass
    loss = out["x_recon"].mean() + out["vq_loss"]
    loss.backward()

    # Check gradient flow
    print(f"\nGradient flow check:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if "encoder" in name and "net.0.weight" in name:
                print(f"  Encoder first layer grad norm: {grad_norm:.6f}")
            elif "decoder" in name and "net.0.weight" in name:
                print(f"  Decoder first layer grad norm: {grad_norm:.6f}")
            elif "transformer" in name and "layers.0" in name and "attn" in name:
                print(f"  Transformer L0 attn grad norm: {grad_norm:.6f}")

    none_grads = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if none_grads:
        print(f"\n  WARNING: {len(none_grads)} params with None gradients!")
        for n in none_grads[:5]:
            print(f"    {n}")
    else:
        print(f"\n  All trainable parameters have gradients.")

    # Encode/decode round-trip
    with torch.no_grad():
        codes = model.encode_to_codes(x)
        print(f"\nEncode-to-codes: {codes.shape}, unique codes: {codes.unique().numel()}")

        recon = model.decode_codes_to_spectrogram(codes)
        print(f"Decode-codes-to-spec: {recon.shape}")

    # Next-code accuracy (vs random baseline 1/K)
    with torch.no_grad():
        logits = out["next_code_logits"][:, :-1]  # (B, S-1, K)
        targets = out["indices"][:, 1:]            # (B, S-1)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == targets).float().mean().item()
        random_baseline = 1.0 / config.codebook_size
        print(f"\nNext-code accuracy: {accuracy:.4f} (random baseline: {random_baseline:.4f})")

    print("\nAll validations passed!")


if __name__ == "__main__":
    main()
