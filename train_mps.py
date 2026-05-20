"""
HRM-RWKV-7 Training on MPS with checkpoint save/resume.

Usage:
    # Start training
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python train_mps.py

    # Resume from checkpoint
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python train_mps.py --resume checkpoints/hrm_rwkv_768m_train/ckpt_latest.pth

    # Custom config
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python train_mps.py \
        --data-path test_dataset \
        --lr 1e-4 \
        --epochs 4 \
        --batch-max-length 256 \
        --save-dir checkpoints/hrm_rwkv_768m_train \
        --save-every 50 \
        --log-every 1
"""

import argparse
import os
import signal
import sys
import time
import json
from pathlib import Path

import torch
from torch.optim import AdamW

from models.baselines.hrm_rwkv import HierarchicalReasoningModelRWKV
from models.lm_head import LMHead
from models.adam_atan2 import AdamATan2
from dataset_new import V1Dataset, V1DatasetConfig


# ── Config ──────────────────────────────────────────────────────────────────

MODEL_CFG = {
    'hidden_size': 768, 'vocab_size': 65536, 'intermediate_size': 2048,
    'n_layers': 32, 'num_hidden_layers': 128, 'num_physical_layers': 2,
    'num_hidden_stacks': 2, 'qk_dim': 64, 'num_attention_heads': 12,
    'h_cycles': 2, 'l_cycles': 3, 'bp_warmup_cycles': 1,
    'half_layers': True, 'max_seq_len': 8192, 'head_size': 64, 'use_rwkv': True,
    'expansion': 4, 'num_heads': 12, 'norm_eps': 1e-6,
    'bp_warmup_ratio': 0.2, 'bp_min_steps': 2, 'bp_max_steps': 5,
}

RADLADS_CHECKPOINT = 'checkpoints/hrm_rwkv_768m_init.pth'
DEVICE = 'mps'
DTYPE = torch.bfloat16


# ── Model creation ──────────────────────────────────────────────────────────

def create_model(radlads_path: str | None = None) -> LMHead:
    """Create HRM-RWKV-768 model, optionally load RADLADS weights."""
    # Create on CPU first (MPS doesn't support bf16 linalg during init)
    with torch.device('cpu'):
        base = HierarchicalReasoningModelRWKV(MODEL_CFG)
        model = LMHead(base, MODEL_CFG)

    # Convert to bf16 and move to MPS
    model = model.to(device=DEVICE, dtype=DTYPE)

    # Load RADLADS checkpoint
    if radlads_path is not None and os.path.exists(radlads_path):
        sd = torch.load(radlads_path, map_location='cpu', weights_only=True)
        base_sd = {k: v.to(dtype=DTYPE) for k, v in sd.items()
                   if k.startswith(('H_level.', 'L_level.', 'zL_init'))}
        head_sd = {k: v.to(dtype=DTYPE) for k, v in sd.items()
                   if k.startswith(('embed_tokens.', 'lm_head.'))}
        model.model.load_state_dict(base_sd, strict=False)
        model.load_state_dict(head_sd, strict=False)
        print(f'  Loaded RADLADS checkpoint: {radlads_path}')
        print(f'    Base keys: {len(base_sd)}, Head keys: {len(head_sd)}')

    params = sum(p.numel() for p in model.parameters())
    print(f'  Model: {params/1e6:.1f}M params')
    return model


# ── Checkpoint save/load ────────────────────────────────────────────────────

def save_checkpoint(
    save_dir: str,
    model: LMHead,
    optimizer,
    carry,
    step: int,
    epoch: int,
    loss: float,
    is_final: bool = False,
):
    """Save model, optimizer, carry, and training state."""
    os.makedirs(save_dir, exist_ok=True)

    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'carry': carry,
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'model_cfg': MODEL_CFG,
    }

    # Save numbered checkpoint
    ckpt_path = os.path.join(save_dir, f'ckpt_step_{step:06d}.pth')
    torch.save(ckpt, ckpt_path)

    # Update latest symlink
    latest_path = os.path.join(save_dir, 'ckpt_latest.pth')
    if os.path.islink(latest_path) or os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.abspath(ckpt_path), latest_path)

    tag = 'FINAL' if is_final else ''
    print(f'  Checkpoint saved: {ckpt_path} {tag}')


def load_checkpoint(
    ckpt_path: str,
    model: LMHead | None = None,
    optimizer = None,
) -> tuple[LMHead, any, any, int, int, float]:
    """Load training checkpoint. Returns (model, optimizer, carry, step, epoch, loss)."""
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if model is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'  Model state loaded')

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f'  Optimizer state loaded')

    carry = ckpt.get('carry')
    step = ckpt.get('step', 0)
    epoch = ckpt.get('epoch', 0)
    loss = ckpt.get('loss', 0.0)

    print(f'  Resuming from step={step}, epoch={epoch}, loss={loss:.4f}')
    return model, optimizer, carry, step, epoch, loss


# ── Signal handler for graceful interruption ────────────────────────────────

_interrupted = False
_save_dir = None
_model = None
_optimizer = None
_carry = None
_step = 0
_epoch = 0
_loss = 0.0


def _signal_handler(signum, frame):
    global _interrupted
    sig_name = signal.Signals(signum).name
    print(f'\nReceived {sig_name}, saving checkpoint...')
    _interrupted = True


def setup_signal_handler():
    """Register signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def save_on_interrupt():
    """Save checkpoint if interrupted."""
    global _interrupted
    if _interrupted and _save_dir is not None:
        save_checkpoint(
            _save_dir, _model, _optimizer, _carry,
            _step, _epoch, _loss, is_final=True,
        )
        print('Checkpoint saved. Exiting.')
        sys.exit(0)


# ── Main training loop ──────────────────────────────────────────────────────

def train(args):
    global _save_dir, _model, _optimizer, _carry, _step, _epoch, _loss

    setup_signal_handler()
    _save_dir = args.save_dir

    # ── Create model ──
    print('Creating model...')
    model = create_model(radlads_path=None if args.resume else RADLADS_CHECKPOINT)
    _model = model

    # ── Optimizer ──
    optimizer = AdamATan2(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    _optimizer = optimizer

    # ── Dataset ──
    print(f'Loading dataset: {args.data_path}')
    dataset = V1Dataset(V1DatasetConfig(
        seed=args.seed,
        dataset_path=args.data_path,
        drop_last_batch=False,
        target_only=True,
        batch_max_length=args.batch_max_length,
        rank=0,
        num_replicas=1,
    ))
    print(f'  Total tokens: {dataset.metadata.total_length}')

    # ── Resume or fresh start ──
    start_step = 0
    start_epoch = 0
    carry = model.model.initial_carry(1, dtype=DTYPE)

    if args.resume:
        model, optimizer, carry, start_step, start_epoch, _ = load_checkpoint(
            args.resume, model, optimizer,
        )
        if carry is None:
            carry = model.model.initial_carry(1, dtype=DTYPE)
            print('  No carry in checkpoint, using fresh carry')

    _carry = carry
    _step = start_step
    _epoch = start_epoch

    # ── Training ──
    print(f'\nStarting training: lr={args.lr}, epochs={args.epochs}')
    print(f'  Save dir: {args.save_dir}')
    print(f'  Save every: {args.save_every} steps')
    print(f'  Log every: {args.log_every} steps')
    print()

    t0 = time.time()
    total_steps = 0

    for epoch in range(start_epoch, args.epochs):
        if _interrupted:
            break

        # Re-create dataset iterator each epoch
        # Use seed=seed+epoch for different shuffling each epoch
        try:
            dataset = V1Dataset(V1DatasetConfig(
                seed=args.seed + epoch,
                dataset_path=args.data_path,
                drop_last_batch=False,
                target_only=True,
                batch_max_length=args.batch_max_length,
                rank=0,
                num_replicas=1,
            ))
        except FileNotFoundError:
            # Dataset only has epoch_0, reuse it
            dataset = V1Dataset(V1DatasetConfig(
                seed=args.seed,
                dataset_path=args.data_path,
                drop_last_batch=False,
                target_only=True,
                batch_max_length=args.batch_max_length,
                rank=0,
                num_replicas=1,
            ))

        for step, (batch, info) in enumerate(dataset):
            if _interrupted:
                break

            # Move batch to device
            batch = {k: v.to(device=DEVICE) for k, v in batch.items()}

            # Forward + backward
            optimizer.zero_grad()
            carry_out, loss, metrics = model(carry=carry, batch=batch, bp_steps=5)
            loss.backward()
            optimizer.step()

            # Free MPS memory
            torch.mps.empty_cache()

            # Check for NaN
            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            if has_nan:
                print(f'  WARNING: NaN detected at step {start_step + step}, skipping')
                continue

            # Update carry
            H_states, L_states = carry_out
            carry = ([s.detach() for s in H_states], [s.detach() for s in L_states])

            _carry = carry
            _step = start_step + step
            _epoch = epoch
            _loss = loss.item()
            total_steps += 1

            # Logging
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                tokens = info.get('total_seqlen', 0)
                print(f'Epoch {epoch} Step {step} (global {_step}): '
                      f'loss={loss.item():.4f} tokens={tokens} '
                      f'time={elapsed:.1f}s')

            # Periodic checkpoint
            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                save_checkpoint(
                    args.save_dir, model, optimizer, carry,
                    start_step + step, epoch, loss.item(),
                )

        if _interrupted:
            # Save checkpoint on interruption
            save_checkpoint(
                args.save_dir, model, optimizer, carry,
                start_step + step, epoch, loss.item(),
            )
            break
        else:
            # Epoch checkpoint
            save_checkpoint(
                args.save_dir, model, optimizer, carry,
                start_step + step, epoch + 1, loss.item(),
            )

    # Final checkpoint (if not already saved by interruption)
    if not _interrupted:
        save_checkpoint(
            args.save_dir, model, optimizer, carry,
            _step, _epoch + 1, _loss, is_final=True,
        )

    elapsed = time.time() - t0
    print(f'\nTraining complete: {total_steps} steps in {elapsed:.1f}s')


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='HRM-RWKV-7 MPS Training')
    parser.add_argument('--data-path', type=str, default='test_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of epochs')
    parser.add_argument('--batch-max-length', type=int, default=256,
                        help='Max tokens per batch')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str,
                        default='checkpoints/hrm_rwkv_768m_train',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save checkpoint every N steps (0 to disable)')
    parser.add_argument('--log-every', type=int, default=1,
                        help='Log every N steps')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
