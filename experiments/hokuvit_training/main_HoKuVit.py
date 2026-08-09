import argparse
import math
import os
import sys
from time import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from hokuvit.HoKuVit import ConvolutionalVisionTransformer
from hokuvit.hopfield_linear.Hopfield_linear import HopfieldMLP
from hokuvit.kuramoto_cnn.Oscillatory_convolutions import (KuramotoConv2d,
                                                           KuramotoTokenConv2d)
from utils.hokuvit.checkpointing import save_checkpoint
from utils.hokuvit.diagnose_memory import print_memory_summary
from utils.hokuvit.io_utils import Tee, load_yaml_config
from utils.hokuvit.model_analysis import count_parameters, calculate_hopfield_ratio
from utils.hokuvit.plotting import plot_training, plot_debug_data
from utils.hokuvit.set_seeds import set_seed



def train_for_epoch(device, train_loader, model, criterion, optimizer, scaler, mixup_fn, epoch,
                    log_interval=50, compute_grad_norms=False, gradient_accumulation_steps=1,
                    hopfield_l1_lambda=0.0):
    """
    OPTIMIZED training loop

    Key optimizations:
    - Reduced logging (only every log_interval batches)
    - Optional gradient norm calculation
    - Gradient accumulation support
    - Progress bar with tqdm
    - Reduced synchronous operations
    """
    model.train()

    train_losses = []
    grad_norms = []

    # Use tqdm for progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (batch, targets) in enumerate(pbar):
        batch = batch.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply Mixup+CutMix
        batch_mixed, targets_mixed = mixup_fn(batch, targets)

        with autocast():
            # Forward pass
            logits = model(batch_mixed)
            loss = criterion(logits, targets_mixed)

            # Explicit L1 penalty on Hopfield memory weights
            # Applied here rather than in the module so the Hopfield class stays unchanged
            if hopfield_l1_lambda > 0.0:
                l1_penalty = torch.tensor(0.0, device=device)
                for module in model.modules():
                    if isinstance(module, (HopfieldMLP)):
                        for attr in ('memory', 'memory_up', 'memory_down'):
                            if hasattr(module, attr):
                                l1_penalty = l1_penalty + getattr(module, attr).weight.abs().mean()
                loss = loss + hopfield_l1_lambda * l1_penalty

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Optional: Compute gradient norms (slow, for debugging)
            if compute_grad_norms:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                grad_norms.append(grad_norm.item())

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Logging (only every log_interval batches)
        train_losses.append(loss.item() * gradient_accumulation_steps)

        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'l1':   f'{(hopfield_l1_lambda * l1_penalty).item():.5f}' if hopfield_l1_lambda > 0.0 else '0',
            })

    return np.mean(train_losses), grad_norms


def validate(model, val_loader, criterion, device):
    """Validation loop"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    # DIAGNOSTIC: Check first batch
    first_batch = True

    with torch.no_grad():
        for batch, targets in tqdm(val_loader, desc='Validating'):
            batch = batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast():
                logits = model(batch)
                loss = criterion(logits, targets)

            # DIAGNOSTIC
            if first_batch:
                print(f"\n=== VALIDATION DIAGNOSTIC ===")
                print(f"Logits shape: {logits.shape}")
                print(f"Logits sample: {logits[0]}")
                print(f"Targets sample: {targets[:10]}")
                print(f"Predictions sample: {torch.max(logits, 1)[1][:10]}")
                print(f"Are all predictions the same? {torch.max(logits, 1)[1].unique()}")
                print(f"Logits std: {logits.std().item()}")
                first_batch = False

            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc


def train_model(first_epoch, num_epochs, train_loader, valid_loader, model, criterion,
                model_name, optimizer, scaler, scheduler, val_criterion, mixup_fn,
                validate_every=1, log_interval=50, compute_grad_norms=False,
                gradient_accumulation_steps=1, hopfield_l1_lambda=0.0):
    """
    Main training loop
    """
    device = next(model.parameters()).device

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    all_grad_norms = []

    best_val_acc = 0.0

    for epoch in range(first_epoch, num_epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 80}")

        # Training
        epoch_start = time()
        train_loss, grad_norms = train_for_epoch(
            device, train_loader, model, criterion, optimizer, scaler, mixup_fn, epoch + 1,
            log_interval=log_interval,
            compute_grad_norms=compute_grad_norms,
            gradient_accumulation_steps=gradient_accumulation_steps,
            hopfield_l1_lambda=hopfield_l1_lambda,
        )
        epoch_time = time() - epoch_start

        train_losses.append(train_loss)
        if grad_norms:
            all_grad_norms.extend(grad_norms)

        # Validation (only every validate_every epochs)
        if (epoch + 1) % validate_every == 0 or (epoch + 1) == num_epochs:
            val_loss, val_acc = validate(model, valid_loader, val_criterion, device)
            valid_losses.append(val_loss)
            valid_accuracies.append(val_acc)

            print(f'\nEpoch {epoch + 1} Summary:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val Acc: {val_acc:.2f}%')
            print(f'  Epoch Time: {epoch_time:.2f}s')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # Save best model
            if val_acc  > best_val_acc:
                best_val_acc = val_acc
                checkpoint_filename = f'checkpoints/{model_name}_{best_val_acc:.2f}.pkl'
                save_checkpoint(optimizer, model, epoch, checkpoint_filename)
                print(f"Saved new best model: {checkpoint_filename}")

        # Step scheduler
        scheduler.step()

    return train_losses, valid_losses, valid_accuracies, all_grad_norms


if __name__ == '__main__':
    start_time = time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.normpath(os.path.join(script_dir, '..', '..', 'configs', 'hokuvit.yaml'))

    parser = argparse.ArgumentParser(description='Train HoKuVit')
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to YAML file of hyperparameters (single source of truth). '
                             'Defaults to ../configs/hokuvit.yaml relative to this script.')
    config_arg = parser.parse_args()

    if not os.path.isfile(config_arg.config):
        parser.error(f"Config file not found: {config_arg.config} "
                     f"(pass --config path/to/file.yaml)")
    args = argparse.Namespace(**load_yaml_config(config_arg.config))

    # Set up logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # check
    print("CUDA available:", torch.cuda.is_available())
    print("Built with CUDA:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    log_file_path = os.path.join(log_dir, f"{args.model_name}_log.txt")
    log_file = open(log_file_path, "w")
    sys.stdout = Tee(sys.stdout, log_file)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Data setup with OPTIMIZATIONS
    cifar_mean = tuple(args.cifar_mean)
    cifar_std = tuple(args.cifar_std)

    train_transform = transforms.Compose([
        transforms.RandomCrop(args.random_crop_size, padding=args.random_crop_padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_set = CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    valid_set = CIFAR10(args.data_dir, train=False, download=True, transform=valid_transform)

    # OPTIMIZED DataLoader with pin_memory and prefetch_factor
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if args.num_workers > 0 else False
    )

    test_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Create CVT model
    print("\n" + "=" * 80)
    print("CREATING CVT MODEL WITH HOPFIELD LAYERS")
    print("=" * 80)

    model = ConvolutionalVisionTransformer(
        in_channel=args.in_channel,
        num_classes=args.num_classes,
        img_size=args.img_size,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        mlp_expansion=args.mlp_expansion,
        num_heads=args.num_heads,
        attn_kernel_size=args.attn_kernel_size,
        stride_q=args.stride_q,
        stride_kv=args.stride_kv,
        padding_q=args.padding_q,
        padding_kv=args.padding_kv,
        attn_drop_p=args.attn_drop_p,
        attn_proj_drop_p=args.attn_drop_p,
        drop_path_p=args.drop_path_p,
        drop_p=args.drop_p,
        update_steps=args.hopfield_update_steps,
        zoneout_prob=args.hopfield_zoneout_prob,
        kuramoto_steps=args.ONN_update_steps,
        dt=args.ONN_dt,
        min_omega=args.ONN_min_frequency,
        omega_init_mean=args.ONN_mean_init_frequency,
        capture_enabled=args.capture_enabled
    )
    model.to(device)

    # Optional: Compile with PyTorch 2.0+
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    print("\n=== MODEL CONFIGURATION ===")
    count_parameters(model)
    calculate_hopfield_ratio(model)

    # Loss functions and optimizer
    mixup_fn = Mixup(
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob,
        switch_prob=args.mixup_switch_prob,
        mode=args.mixup_mode,
        label_smoothing=args.smoothing,
        num_classes=args.num_classes,
    )

    criterion = SoftTargetCrossEntropy()
    val_criterion = nn.CrossEntropyLoss()

    effective_batch = args.batch_size * args.gradient_accumulation_steps

    scaled_lr = args.learning_rate * math.sqrt(effective_batch / 256)

    # Separate Kuramoto parameters into three groups:
    #   1. Kuramoto coupling weights  -> weight decay applied  (coupling_strength, conv weights)
    #   2. Kuramoto oscillator params -> no weight decay       (omega_0, phase_offset, mu, etc.)
    #   3. All other model params     -> standard weight decay

    kuramoto_coupling_ids = set()  # coupling / conv weights  -> get WD
    kuramoto_oscillator_ids = set()  # physics params           -> no WD
    hopfield_memory_ids = set()      # Hopfield memory weights  -> L1 only, L2 WD=0

    for module in model.modules():
        if isinstance(module, (KuramotoConv2d, KuramotoTokenConv2d)):
            for param_name, param in module.named_parameters():
                is_proj = 'proj.weight' in param_name or 'dim_projection' in param_name
                is_coupling = ('coupling_strength' in param_name or
                               ('weight' in param_name and not is_proj))
                if is_coupling:
                    kuramoto_coupling_ids.add(id(param))
                else:
                    kuramoto_oscillator_ids.add(id(param))
        elif isinstance(module, HopfieldMLP):
            for attr in ('memory', 'memory_up', 'memory_down'):
                if hasattr(module, attr):
                    for param in getattr(module, attr).parameters():
                        hopfield_memory_ids.add(id(param))

    all_kuramoto_ids = kuramoto_coupling_ids | kuramoto_oscillator_ids
    all_special_ids  = all_kuramoto_ids | hopfield_memory_ids

    kuramoto_coupling_params   = [p for p in model.parameters() if id(p) in kuramoto_coupling_ids]
    kuramoto_oscillator_params = [p for p in model.parameters() if id(p) in kuramoto_oscillator_ids]
    hopfield_memory_params     = [p for p in model.parameters() if id(p) in hopfield_memory_ids]
    other_params               = [p for p in model.parameters() if id(p) not in all_special_ids]

    param_groups = [
        {'params': other_params,               'weight_decay': 0.0,                        'name': 'other'},
        {'params': kuramoto_coupling_params,   'weight_decay': args.kuramoto_weight_decay,  'name': 'kuramoto_coupling'},
        {'params': kuramoto_oscillator_params, 'weight_decay': 0.0,                        'name': 'kuramoto_oscillator'},
        {'params': hopfield_memory_params,     'weight_decay': 0.0,                        'name': 'hopfield_memory'},
    ]
    print(f"\n=== OPTIMIZER PARAM GROUPS ===")
    print(f"  Other params                : {sum(p.numel() for p in other_params):,}  wd={args.weight_decay}")
    print(f"  Kuramoto coupling weights   : {sum(p.numel() for p in kuramoto_coupling_params):,}  wd={args.kuramoto_weight_decay}")
    print(f"  Kuramoto oscillator params  : {sum(p.numel() for p in kuramoto_oscillator_params):,}  wd=0.0")
    print(f"  Hopfield memory weights     : {sum(p.numel() for p in hopfield_memory_params):,}  wd=0.0  l1={args.hopfield_l1_lambda}")

    optimizer = torch.optim.NAdam(
        param_groups,
        lr=scaled_lr
    )

    scaler = GradScaler()

    # Learning rate scheduling
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6 / scaled_lr,
                                end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[args.warmup_epochs])

    # Memory summary for the model
    print_memory_summary()

    # Train
    print(f"\n{'=' * 80}")
    print(f"STARTING TRAINING")
    print(f"  Validate every: {args.validate_every} epochs")
    print(f"  Log interval: {args.log_interval} batches")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps} steps")
    print(f"  Compute grad norms: {args.compute_grad_norms}")
    print(f"  Torch compile: {args.compile}")
    print(f"{'=' * 80}\n")

    train_losses, valid_losses, valid_accuracies, all_grad_norms = train_model(
        first_epoch=0,
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        valid_loader=test_loader,
        model=model,
        criterion=criterion,
        model_name=args.model_name,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        val_criterion=val_criterion,
        mixup_fn=mixup_fn,
        validate_every=args.validate_every,
        log_interval=args.log_interval,
        compute_grad_norms=args.compute_grad_norms,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        hopfield_l1_lambda=args.hopfield_l1_lambda,
    )

    # Plotting
    plot_training(train_losses, valid_losses, valid_accuracies, model_name=args.model_name)
    plot_debug_data(model_name=args.model_name, all_grad_norms=all_grad_norms)

    end_time = time()
    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETE")
    print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"{'=' * 80}")

    if valid_accuracies:
        print(f"\nFinal Results:")
        print(f"Best Validation Accuracy: {max(valid_accuracies):.2f}%")

    log_file.close()