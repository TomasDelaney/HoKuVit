import argparse
import os
import sys
from time import time

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from baselines.CVT import ConvolutionalVisionTransformer
from utils.hokuvit.checkpointing import save_checkpoint
from utils.hokuvit.io_utils import Tee, load_yaml_config
from utils.hokuvit.model_analysis import count_parameters, calculate_hopfield_ratio
from utils.hokuvit.set_seeds import set_seed

def train_for_epoch(device, train_loader, model, criterion, optimizer, scaler,
                    mixup_fn, epoch, log_interval=50, compute_grad_norms=False,
                    gradient_accumulation_steps=1):
    """
    Standard training loop

    Key features:
    - Mixed precision training with autocast
    - Gradient accumulation support
    - Progress bar with tqdm
    - Optional gradient norm calculation
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

            # Compute loss
            loss = criterion(logits, targets_mixed)

            # Gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Optional: compute gradient norms
            if compute_grad_norms:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                grad_norms.append(total_norm)

        # Log metrics
        train_losses.append(loss.item() * gradient_accumulation_steps)

        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})

    return np.mean(train_losses), grad_norms


def validate(model, val_loader, criterion, device):
    """
    Standard validation loop
    """
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch, targets in val_loader:
            batch = batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            logits = model(batch)

            # Calculate loss
            loss = criterion(logits, targets)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    avg_loss = val_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train(first_epoch, num_epochs, train_loader, valid_loader, model, criterion,
          model_name, optimizer, scaler, scheduler, val_criterion, mixup_fn,
          validate_every=1, log_interval=50, compute_grad_norms=False,
          gradient_accumulation_steps=1):
    """
    Main training loop
    """
    device = next(model.parameters()).device

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    all_grad_norms = []

    best_acc = 0.0

    for epoch in range(first_epoch, num_epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'=' * 80}")

        # Training
        train_loss, grad_norms = train_for_epoch(
            device=device,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            mixup_fn=mixup_fn,
            epoch=epoch + 1,
            log_interval=log_interval,
            compute_grad_norms=compute_grad_norms,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        train_losses.append(train_loss)
        all_grad_norms.append(grad_norms)

        print(f"\nTraining Loss: {train_loss:.4f}")

        # Validation (every N epochs)
        if (epoch + 1) % validate_every == 0:
            print("\nValidating...")
            val_loss, val_acc = validate(
                model=model,
                val_loader=valid_loader,
                criterion=val_criterion,
                device=device
            )

            valid_losses.append(val_loss)
            valid_accuracies.append(val_acc)

            print(f"\nValidation Results:")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    filename=f"{model_name}_best.pkl"
                )
                print(f"  New best model saved! (Acc: {best_acc:.2f}%)")

        # Update learning rate
        scheduler.step()

    return train_losses, valid_losses, valid_accuracies, all_grad_norms


def plot_training(train_losses, valid_losses, valid_accuracies, model_name="cvt_baseline"):
    """Plot training metrics"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training loss
    axes[0].plot(train_losses, label='Training Loss', linewidth=2, color='blue')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Validation loss
    if valid_losses:
        axes[1].plot(valid_losses, label='Validation Loss', linewidth=2, color='red')
        axes[1].set_xlabel('Validation Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    # Validation accuracy
    if valid_accuracies:
        axes[2].plot(valid_accuracies, label='Validation Accuracy', linewidth=2,
                     marker='o', color='green')
        axes[2].set_xlabel('Validation Epoch', fontsize=12)
        axes[2].set_ylabel('Accuracy (%)', fontsize=12)
        axes[2].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name}_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to plots/{model_name}_training.png")


def plot_debug_data(model_name, all_grad_norms):
    """Plot debugging information"""
    if not any(all_grad_norms):  # Skip if no gradient norms computed
        return

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Gradient norms
    flat_grad_norms = [norm for epoch_norms in all_grad_norms for norm in epoch_norms]
    if flat_grad_norms:
        ax.plot(flat_grad_norms, linewidth=1, alpha=0.7)
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title('Gradient Norms', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name}_debug.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Debug plot saved to plots/{model_name}_debug.png")


if __name__ == "__main__":
    start_time = time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.normpath(os.path.join(script_dir, 'configs', 'cvt.yaml'))

    parser = argparse.ArgumentParser(description="CVT Baseline Training")
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to YAML file of hyperparameters (single source of truth). '
                             'Defaults to configs/cvt.yaml next to this script.')
    config_arg = parser.parse_args()

    if not os.path.isfile(config_arg.config):
        parser.error(f"Config file not found: {config_arg.config} "
                     f"(pass --config path/to/file.yaml)")
    args = argparse.Namespace(**load_yaml_config(config_arg.config))

    # Set up logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{args.model_name}_log.txt")
    log_file = open(log_file_path, "w")
    sys.stdout = Tee(sys.stdout, log_file)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data setup
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

    # Optimized DataLoader
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

    # Create student model (CVT)
    print("\n" + "=" * 80)
    print("CREATING CVT MODEL")
    print("=" * 80)

    student_model = ConvolutionalVisionTransformer(
        in_channel=3,
        num_classes=args.num_classes,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        mlp_expansion=args.mlp_expansion,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        attn_kernel_size=args.attn_kernel_size,
        stride_q=args.stride_q,
        stride_kv=args.stride_kv,  # could be 2 for downsampling in attention
        padding_q=args.padding_q,
        padding_kv=args.padding_kv,
        attn_drop_p=args.attn_drop_p,
        attn_proj_drop_p=args.attn_proj_drop_p,
        drop_path_p=args.drop_path_p,
        drop_p=args.drop_p,
    )
    student_model.to(device)

    # Optional: Compile with PyTorch 2.0+
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling student model with torch.compile()...")
        student_model = torch.compile(student_model)

    print("\n=== STUDENT MODEL ===")
    count_parameters(student_model)
    calculate_hopfield_ratio(student_model)  # baseline has no Hopfield/Kuramoto layers,
                                              # so this reports 0% neuromorphic params —
                                              # useful as a direct comparison point vs. HoKuVit

    # Loss functions and optimizer
    criterion = SoftTargetCrossEntropy()  # For training with mixup/cutmix
    val_criterion = nn.CrossEntropyLoss()  # For validation

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

    optimizer = torch.optim.NAdam(student_model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()

    # Learning rate scheduling
    warmup_scheduler = LinearLR(optimizer, start_factor=args.warmup_lr,
                                end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[args.warmup_epochs])

    # Training configuration
    print(f"\n{'=' * 80}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"  Model: CVT Baseline")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Validate every: {args.validate_every} epochs")
    print(f"  Log interval: {args.log_interval} batches")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps} steps")
    print(f"  Compute grad norms: {args.compute_grad_norms}")
    print(f"  Torch compile: {args.compile}")
    print(f"{'=' * 80}\n")

    # Train
    results = train(
        first_epoch=0,
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        valid_loader=test_loader,
        model=student_model,
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
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    train_losses, valid_losses, valid_accuracies, all_grad_norms = results

    # Plotting
    plot_training(
        train_losses, valid_losses, valid_accuracies,
        model_name=args.model_name
    )
    plot_debug_data(
        model_name=args.model_name,
        all_grad_norms=all_grad_norms
    )

    end_time = time()
    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETE")
    print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"{'=' * 80}")

    if valid_accuracies:
        print(f"\nFinal Results:")
        print(f"  Best Accuracy: {max(valid_accuracies):.2f}%")
        print(f"  Final Accuracy: {valid_accuracies[-1]:.2f}%")

    log_file.close()