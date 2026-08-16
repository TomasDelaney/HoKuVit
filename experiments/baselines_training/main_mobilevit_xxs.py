import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.hokuvit.checkpointing import save_checkpoint
from utils.hokuvit.io_utils import Tee, load_yaml_config
from utils.hokuvit.model_analysis import count_parameters
from utils.hokuvit.set_seeds import set_seed


def build_model(timm_name, num_classes, img_size):
    """Build the backbone with a CIFAR-10 head, pretrained=False."""
    import timm
    try:
        return timm.create_model(timm_name, pretrained=False, num_classes=num_classes, img_size=img_size)
    except TypeError:
        # older timm versions don't accept img_size for every model
        return timm.create_model(timm_name, pretrained=False, num_classes=num_classes)


def build_dataloaders(args):
    mean = tuple(args.cifar_mean)
    std = tuple(args.cifar_std)

    train_transform = transforms.Compose([
        transforms.RandomCrop(args.random_crop_size, padding=args.random_crop_padding),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    valid_set = torchvision.datasets.CIFAR10(args.data_dir, train=False, download=True, transform=valid_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    return train_loader, valid_loader


def train_for_epoch(device, train_loader, model, criterion, optimizer, scaler, epoch,
                    log_interval=50, compute_grad_norms=False, gradient_accumulation_steps=1):
    model.train()

    running_loss, correct, total = 0.0, 0, 0
    grad_norms = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, targets) / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if compute_grad_norms:
                grad_norms.append(grad_norm.item())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * gradient_accumulation_steps * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)

        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': f'{running_loss / total:.4f}', 'acc': f'{100.0 * correct / total:.2f}%'})

    return running_loss / total, 100.0 * correct / total, grad_norms


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()

    val_loss, correct, total = 0.0, 0, 0
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)

    return val_loss / total, 100.0 * correct / total


def train_model(first_epoch, num_epochs, train_loader, valid_loader, model, criterion,
                model_name, optimizer, scaler, scheduler, val_criterion,
                validate_every=1, log_interval=50, compute_grad_norms=False,
                gradient_accumulation_steps=1):
    device = next(model.parameters()).device

    train_losses, valid_losses, valid_accuracies, all_grad_norms = [], [], [], []
    best_acc = 0.0

    start_time = time.time()

    for epoch in range(first_epoch, num_epochs):
        train_loss, train_acc, grad_norms = train_for_epoch(
            device, train_loader, model, criterion, optimizer, scaler, epoch + 1,
            log_interval=log_interval,
            compute_grad_norms=compute_grad_norms,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        train_losses.append(train_loss)
        if grad_norms:
            all_grad_norms.extend(grad_norms)

        if (epoch + 1) % validate_every == 0 or (epoch + 1) == num_epochs:
            val_loss, val_acc = validate(model, valid_loader, val_criterion, device)
            valid_losses.append(val_loss)
            valid_accuracies.append(val_acc)

            elapsed = time.time() - start_time
            eta_min = elapsed / (epoch + 1 - first_epoch) * (num_epochs - epoch - 1) / 60

            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print(f'  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%')
            print(f'  ETA: {eta_min:.1f} min')

            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_filename = f'checkpoints/{model_name}_{best_acc:.2f}.pkl'
                save_checkpoint(optimizer, model, epoch, checkpoint_filename)
                print(f"  Saved new best model: {checkpoint_filename}")

        scheduler.step()

    print(f"\nDone. Best validation accuracy: {best_acc:.2f}%  ({(time.time() - start_time) / 60:.1f} min)")
    return train_losses, valid_losses, valid_accuracies, all_grad_norms, best_acc


def plot_training(train_losses, valid_losses, valid_accuracies, model_name):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train Loss')
    if valid_losses:
        ax1.plot(range(len(valid_losses)), valid_losses, label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    if valid_accuracies:
        ax2.plot(range(len(valid_accuracies)), valid_accuracies, label='Val Accuracy', marker='o', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name}_training.png', dpi=150)
    plt.close()
    print(f"Saved training plot to plots/{model_name}_training.png")


def plot_debug_data(model_name, all_grad_norms=None):
    import matplotlib.pyplot as plt

    if not all_grad_norms:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(all_grad_norms)
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms During Training')
    plt.grid(True)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name}_grad_norms.png', dpi=150)
    plt.close()
    print(f"Saved gradient norms plot to plots/{model_name}_grad_norms.png")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.normpath(os.path.join(script_dir, 'configs', 'mobilevit_xxs.yaml'))

    parser = argparse.ArgumentParser(description='Train MobileViT-XXS on CIFAR-10')
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to YAML file of hyperparameters (single source of truth). '
                             'Defaults to configs/mobilevit_xxs.yaml next to this script.')
    config_arg = parser.parse_args()

    if not os.path.isfile(config_arg.config):
        parser.error(f"Config file not found: {config_arg.config} "
                     f"(pass --config path/to/file.yaml)")
    args = argparse.Namespace(**load_yaml_config(config_arg.config))

    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    log_file = open(os.path.join("logs", f"{args.model_name}_log.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_file)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "=" * 80)
    print("CREATING MOBILEVIT-XXS MODEL")
    print("=" * 80)

    model = build_model(args.timm_name, args.num_classes, args.random_crop_size).to(device)
    count_parameters(model)

    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    train_loader, valid_loader = build_dataloaders(args)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"\n{'=' * 80}")
    print("STARTING TRAINING")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Label smoothing: {args.smoothing}")
    print(f"{'=' * 80}\n")

    train_losses, valid_losses, valid_accuracies, all_grad_norms, best_acc = train_model(
        first_epoch=0,
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        criterion=criterion,
        model_name=args.model_name,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        val_criterion=val_criterion,
        log_interval=args.log_interval,
        compute_grad_norms=args.compute_grad_norms,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    plot_training(train_losses, valid_losses, valid_accuracies, model_name=args.model_name)
    plot_debug_data(model_name=args.model_name, all_grad_norms=all_grad_norms)

    log_file.close()