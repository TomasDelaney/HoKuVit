import argparse
import numpy as np
import torch
from time import time
from torch.utils.data import DataLoader
from utils.hokuvit.set_seeds import set_seed
from utils.hokuvit.checkpointing import save_checkpoint
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import sys
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import mobilenet_v3_small
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm
from utils.hokuvit.diagnose_memory import print_memory_summary
from utils.hokuvit.io_utils import Tee, load_yaml_config
from utils.hokuvit.model_analysis import count_parameters


def train_model(first_epoch, num_epochs, train_loader, valid_loader, model, criterion,
                model_name, optimizer, scaler, scheduler, val_criterion, mixup_fn=None,
                validate_every=1, log_interval=50, compute_grad_norms=False,
                gradient_accumulation_steps=1):
    """
    Training loop with optimizations.
    """
    device = next(model.parameters()).device

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    all_grad_norms = [] if compute_grad_norms else None

    best_acc = 0.0

    for epoch in range(first_epoch, num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Apply mixup/cutmix
            if mixup_fn is not None:
                inputs, targets = mixup_fn(inputs, targets)

            # Forward pass with mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * gradient_accumulation_steps
            batch_count += 1

            # Update progress bar
            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss / batch_count:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })

        # Average training loss
        epoch_train_loss = running_loss / batch_count
        train_losses.append(epoch_train_loss)

        # Validation phase
        if (epoch + 1) % validate_every == 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    with autocast():
                        outputs = model(inputs)
                        loss = val_criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            val_loss /= len(valid_loader)
            val_acc = 100.0 * correct / total

            valid_losses.append(val_loss)
            valid_accuracies.append(val_acc)

            print(f'\nEpoch [{epoch + 1}/{num_epochs}]')
            print(f'  Train Loss: {epoch_train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val Acc: {val_acc:.2f}%')

            # Save best checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_filename = f'checkpoints/{model_name}_{best_acc:.2f}.pkl'
                save_checkpoint(optimizer, model, epoch, checkpoint_filename)
                print(f"Saved new best model: {checkpoint_filename}")

        # Step scheduler
        scheduler.step()

    return train_losses, valid_losses, valid_accuracies, all_grad_norms


def plot_training(train_losses, valid_losses, valid_accuracies, model_name):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    if valid_losses:
        ax1.plot(np.arange(0, len(train_losses), len(train_losses) // len(valid_losses)),
                 valid_losses, label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    if valid_accuracies:
        ax2.plot(np.arange(0, len(train_losses), len(train_losses) // len(valid_accuracies)),
                 valid_accuracies, label='Val Accuracy', marker='o', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training.png', dpi=150)
    plt.close()
    print(f"Saved training plot to plots/{model_name}_training.png")


def plot_debug_data(model_name, all_grad_norms=None):
    """Plot debugging information."""
    if all_grad_norms and len(all_grad_norms) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(all_grad_norms)
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms During Training')
        plt.grid(True)
        plt.savefig(f'plots/{model_name}_grad_norms.png', dpi=150)
        plt.close()
        print(f"Saved gradient norms plot to plots/{model_name}_grad_norms.png")


if __name__ == '__main__':
    start_time = time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.normpath(os.path.join(script_dir, 'configs', 'mobilenetv3.yaml'))

    parser = argparse.ArgumentParser(description='Train MobileNetV3-Small on CIFAR-10')
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to YAML file of hyperparameters (single source of truth). '
                             'Defaults to configs/mobilenetv3.yaml next to this script.')
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

    # Check CUDA
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

    # Create MobileNetV3-Small model
    print("\n" + "=" * 80)
    print("CREATING MOBILENETV3-SMALL MODEL")
    print("=" * 80)

    # Create model from scratch (no pretrained weights)
    model = mobilenet_v3_small(
        weights=None,  # Train from scratch
        num_classes=args.num_classes,
        dropout=args.drop_p,
        width_mult=args.width_mult,
    )

    # Adapt the first conv for CIFAR-10's smaller 32x32 images.
    model.features[0][0] = nn.Conv2d(
        3, args.stem_out_channels,
        kernel_size=args.stem_kernel_size,
        stride=args.stem_stride,
        padding=args.stem_padding,
        bias=False,
    )

    model.to(device)

    # Optional: Compile with PyTorch 2.0+
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    print("\n=== MODEL CONFIGURATION ===")
    count_parameters(model)

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
    scaled_lr = args.learning_rate * (effective_batch / 128)  # Scale for MobileNet

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=args.weight_decay
    )
    scaler = GradScaler()

    # Learning rate scheduling
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.warmup_lr / scaled_lr,
        end_factor=1.0,
        total_iters=args.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs - args.warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )

    # Memory summary
    print_memory_summary()

    # Train
    print(f"\n{'=' * 80}")
    print(f"STARTING TRAINING")
    print(f"  Total epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {scaled_lr:.6f}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Validate every: {args.validate_every} epochs")
    print(f"  Log interval: {args.log_interval} batches")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps} steps")
    print(f"  Mixup alpha: {args.mixup}")
    print(f"  Cutmix alpha: {args.cutmix}")
    print(f"  Label smoothing: {args.smoothing}")
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
        gradient_accumulation_steps=args.gradient_accumulation_steps
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