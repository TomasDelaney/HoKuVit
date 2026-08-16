"""Hardware-realism stress test for HoKuVit: how well does classification
survive quantization and write-noise of the Hopfield memory weights?

Maps to memristive-crossbar non-idealities:
  - Finite programming resolution -> uniform quantization to N bits
  - Write-to-write variability    -> additive Gaussian noise (sigma as a
                                      fraction of the weight dynamic range)

Two independent sweeps over the CIFAR-10 test set (mean +/- std over repeats):
  (a) Accuracy vs weight bit-depth    (FP32, 8, 6, 5, 4, 3, 2 bits)
  (b) Accuracy vs write-noise level sigma (0 - 50% of dynamic range)

By default perturbs only the Hopfield memory matrices (memory_up /
memory_down) across all HopfieldMLP layers -- these are the weights that
would live on the crossbar. Set all_weights=True in main() to perturb every
Conv/Linear weight instead.

Outputs (into figures_dir):
  figure_name.pdf / .svg
  figure_name_values.csv
"""

import os
import sys
import csv

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hokuvit.HoKuVit import ConvolutionalVisionTransformer
from hokuvit.hopfield_linear.Hopfield_linear import HopfieldMLP
from utils.hokuvit.io_utils import load_yaml_config

_BLUE = '#0072B2'
_TEAL = '#009E73'
_GREY = '#999999'

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def build_model_kwargs(cfg):
    """Map hokuvit.yaml's flattened keys onto ConvolutionalVisionTransformer's
    constructor kwargs — matches main_HoKuVit's own model-construction call
    so this script can't silently drift out of sync with what a checkpoint
    was actually trained with."""
    return dict(
        in_channel=cfg['in_channel'], num_classes=cfg['num_classes'], img_size=cfg['img_size'],
        emb_dim=cfg['emb_dim'], n_layers=cfg['n_layers'],
        kernel_size=cfg['kernel_size'], stride=cfg['stride'], padding=cfg['padding'],
        mlp_expansion=cfg['mlp_expansion'], num_heads=cfg['num_heads'],
        attn_kernel_size=cfg['attn_kernel_size'],
        stride_q=cfg['stride_q'], stride_kv=cfg['stride_kv'],
        padding_q=cfg['padding_q'], padding_kv=cfg['padding_kv'],
        attn_drop_p=cfg['attn_drop_p'], attn_proj_drop_p=cfg['attn_drop_p'],
        drop_path_p=cfg['drop_path_p'], drop_p=cfg['drop_p'],
        update_steps=cfg['hopfield_update_steps'], zoneout_prob=cfg['hopfield_zoneout_prob'],
        kuramoto_steps=cfg['ONN_update_steps'], dt=cfg['ONN_dt'],
        min_omega=cfg['ONN_min_frequency'], omega_init_mean=cfg['ONN_mean_init_frequency'],
        capture_enabled=cfg['capture_enabled'],
    )


def collect_target_weights(model, all_weights=False):
    """Return list of (name, parameter_tensor) to perturb.

    Default: the Hopfield memory matrices (memory_up / memory_down) of every
    HopfieldMLP — these are the crossbar-resident weights. These params are
    parametrized (block-diagonal symmetric), so we perturb the underlying
    `.original` parameter that the parametrization reads from.

    all_weights=True: every Conv/Linear weight in the model.
    """
    targets = []
    if all_weights:
        for name, p in model.named_parameters():
            if p.dim() >= 2 and 'weight' in name:
                targets.append((name, p))
        return targets

    for name, mod in model.named_modules():
        if isinstance(mod, HopfieldMLP):
            for sub_name in ('memory_up', 'memory_down'):
                sub = getattr(mod, sub_name)
                if hasattr(sub, 'parametrizations') and 'weight' in sub.parametrizations:
                    raw = sub.parametrizations.weight.original
                    targets.append((f'{name}.{sub_name}.weight(orig)', raw))
                else:
                    targets.append((f'{name}.{sub_name}.weight', sub.weight))
    return targets


def quantize_tensor(w, n_bits):
    """Uniform symmetric quantization to n_bits over the tensor's own range."""
    if n_bits >= 32:
        return w.clone()
    w_min, w_max = w.min(), w.max()
    if (w_max - w_min) < 1e-12:
        return w.clone()
    levels = 2 ** n_bits - 1
    scale = (w_max - w_min) / levels
    return torch.round((w - w_min) / scale) * scale + w_min


def add_write_noise(w, sigma_frac, generator):
    """Additive Gaussian noise with std = sigma_frac * dynamic_range(w).
    Models memristor write variability."""
    if sigma_frac <= 0.0:
        return w.clone()
    dyn = (w.max() - w.min()).item()
    std = sigma_frac * dyn
    noise = torch.randn(w.shape, generator=generator, device=w.device, dtype=w.dtype) * std
    return w + noise


def evaluate(model, loader, n_samples, device):
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if total >= n_samples:
                break
    return correct / total


def run_perturbed(model, targets, originals, perturb_fn, loader, n_samples, device):
    """Apply perturb_fn to every target weight, evaluate, then restore.
    perturb_fn(w) -> perturbed tensor."""
    with torch.no_grad():
        for (_, p), w0 in zip(targets, originals):
            p.copy_(perturb_fn(w0))
    acc = evaluate(model, loader, n_samples, device)
    with torch.no_grad():
        for (_, p), w0 in zip(targets, originals):
            p.copy_(w0)
    return acc


def _spine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def build_figure(bit_depths, bit_acc, noise_levels, noise_mean, noise_std, baseline_acc):
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.80))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.18, wspace=0.30)

    # Panel a: bit-depth
    ax = axes[0]
    xs = np.arange(len(bit_depths))
    ax.plot(xs, bit_acc, color=_BLUE, lw=1.2, marker='o', markersize=4, zorder=3)
    ax.axhline(baseline_acc, color=_GREY, lw=0.7, ls='--', zorder=2,
              label=f'FP baseline ({baseline_acc:.1%})')
    ax.set_xticks(xs)
    ax.set_xticklabels(['FP' if b >= 32 else str(b) for b in bit_depths])
    ax.set_xlabel('Weight bit-depth')
    ax.set_ylabel('Top-1 accuracy')
    ax.set_ylim(0.75, min(1.0, baseline_acc * 1.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.yaxis.grid(True, lw=0.3, color='#cccccc', zorder=0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#dddddd', handlelength=1.2, loc='lower left')
    _spine(ax)
    ax.text(-0.12, 1.10, 'd', transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='right')

    # Panel b: write-noise
    ax = axes[1]
    pct = [s * 100 for s in noise_levels]
    lo = [m - s for m, s in zip(noise_mean, noise_std)]
    hi = [m + s for m, s in zip(noise_mean, noise_std)]
    ax.fill_between(pct, lo, hi, color=_TEAL, alpha=0.15, linewidth=0, zorder=1)
    ax.plot(pct, noise_mean, color=_TEAL, lw=1.2, marker='s', markersize=3.5, zorder=3)
    ax.axhline(baseline_acc, color=_GREY, lw=0.7, ls='--', zorder=2,
              label=f'FP baseline ({baseline_acc:.1%})')
    ax.set_xlabel('Write-noise  sigma  (% of weight range)')
    ax.set_ylabel('Top-1 accuracy')
    ax.set_ylim(0.75, min(1.0, baseline_acc * 1.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.yaxis.grid(True, lw=0.3, color='#cccccc', zorder=0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#dddddd', handlelength=1.2, loc='lower left')
    _spine(ax)
    ax.text(-0.12, 1.10, 'e', transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='right')

    return fig


def save_figure(fig, base_path, title=''):
    svg_path = base_path + '.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f'  saved {svg_path}')

    pdf_path = base_path + '.pdf'
    with PdfPages(pdf_path) as p:
        p.savefig(fig, bbox_inches='tight', facecolor='white')
        p.infodict()['Title'] = title
    print(f'  saved {pdf_path}')

    plt.close(fig)


def main():
    # CONFIG — adjust these per run
    hokuvit_config = "../../configs/hokuvit.yaml"  # adjust if your layout differs
    checkpoint = "../../checkpoints/26_0503_fixed_pointwise_cait_79.78.pkl"
    data_root = "../../data"
    figures_dir = "figures"
    figure_name = "fig1_e_hopfield_weight_quantization"
    all_weights = False  # False: only Hopfield memory weights. True: every Conv/Linear weight.

    bit_depths = [32, 8, 6, 5, 4, 3, 2]  # 32 = full precision baseline
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]  # fraction of dynamic range

    n_samples = 10000
    batch_size = 256
    n_repeats = 3  # repeats for the stochastic noise sweep
    seed = 42

    os.makedirs(figures_dir, exist_ok=True)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    cfg = load_yaml_config(hokuvit_config)
    model_kwargs = build_model_kwargs(cfg)

    model = ConvolutionalVisionTransformer(**model_kwargs)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval().to(device)
    print('Model loaded')

    targets = collect_target_weights(model, all_weights=all_weights)
    target_label = 'all Conv/Linear weights' if all_weights else 'Hopfield memory weights'
    print(f'Perturbing {len(targets)} weight tensors ({target_label}):')
    for name, p in targets:
        print(f'  {name:<55s}  {tuple(p.shape)}')

    # Snapshot originals (on device) so each run restores exactly
    originals = [p.detach().clone() for _, p in targets]

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tuple(cfg['cifar_mean']), tuple(cfg['cifar_std'])),
    ])
    dataset = CIFAR10(root=data_root, train=False, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    baseline_acc = evaluate(model, loader, n_samples, device)
    print(f'\nFull-precision baseline accuracy: {baseline_acc:.2%}\n')

    print('Quantization sweep:')
    print(f'{"bits":>5}  {"acc":>8}')
    print('-' * 16)
    bit_acc = []
    for nb in bit_depths:
        acc = run_perturbed(
            model, targets, originals,
            perturb_fn=lambda w, nb=nb: quantize_tensor(w, nb),
            loader=loader, n_samples=n_samples, device=device,
        )
        bit_acc.append(acc)
        print(f'{("FP" if nb >= 32 else nb):>5}  {acc:>7.2%}')

    print('\nWrite-noise sweep:')
    print(f'{"sigma%":>7}  {"acc_mean":>9}  {"acc_std":>8}')
    print('-' * 28)
    noise_mean, noise_std = [], []
    for sigma in noise_levels:
        accs = []
        for rep in range(n_repeats):
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + rep * 997)
            acc = run_perturbed(
                model, targets, originals,
                perturb_fn=lambda w, s=sigma, g=gen: add_write_noise(w, s, g),
                loader=loader, n_samples=n_samples, device=device,
            )
            accs.append(acc)
        m, sd = float(np.mean(accs)), float(np.std(accs))
        noise_mean.append(m)
        noise_std.append(sd)
        print(f'{sigma * 100:>6.1f}%  {m:>8.2%}  +/-{sd:.2%}')

    csv_path = os.path.join(figures_dir, figure_name + '_values.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sweep', 'x', 'acc_mean', 'acc_std'])
        for nb, a in zip(bit_depths, bit_acc):
            writer.writerow(['bit_depth', nb, f'{a:.6f}', '0'])
        for s, m, sd in zip(noise_levels, noise_mean, noise_std):
            writer.writerow(['noise_frac', f'{s:.4f}', f'{m:.6f}', f'{sd:.6f}'])
    print(f'\n  saved {csv_path}')

    print('\nBuilding figure...')
    fig = build_figure(bit_depths, bit_acc, noise_levels, noise_mean, noise_std, baseline_acc)
    save_figure(fig, os.path.join(figures_dir, figure_name),
               title='HoKuVit weight quantization / noise robustness')
    print(f'\nDone. Baseline: {baseline_acc:.3%}')


if __name__ == '__main__':
    main()