"""Plots HopfieldMLP up/down projection energy over update steps for a trained
HoKuVit checkpoint, one panel per projection, coloured by stage/layer.
Model hyperparameters are pulled from hokuvit.yaml, the same config used for
training, so this always matches whatever checkpoint you point it at.
Edit the CONFIG block below and run.
"""

import os
import sys
import importlib
from collections import defaultdict

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hokuvit.hopfield_linear.Hopfield_linear import HopfieldMLP
from utils.hokuvit.io_utils import load_yaml_config


def build_model_kwargs(cfg):
    """Map hokuvit.yaml's flattened keys onto ConvolutionalVisionTransformer's
    constructor kwargs — several are renamed between the two (e.g.
    hopfield_update_steps -> update_steps), matching main_HoKuVit's own
    model-construction call so this script can't silently drift out of sync
    with what a checkpoint was actually trained with."""
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

# Nature UC style — up/down projections get separate colour families, shaded
# light-to-dark within each stage (S1/S2/S3). Wong colorblind-safe palette.
# Up -> blue family (S1 sky, S2 blue, S3 navy)
# Down -> orange family (S1 amber, S2 vermillion, S3 brown)
_UP_STAGES = ['#56B4E9', '#0072B2', '#00306e']
_DOWN_STAGES = ['#E69F00', '#D55E00', '#7f2800']
_STAGE_MARKERS = ['s', '^', 'D']  # square, triangle, diamond

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'lines.linewidth': 0.9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def _layer_colors(stage_groups, stage_anchors, min_alpha=0.55):
    """Returns {layer_name: rgb_tuple}. Within each stage, blends from a
    lightened anchor color (shallowest layer) to the full anchor color
    (deepest layer). min_alpha kept high so single-layer stages stay visible."""
    colour_map = {}
    all_stages = sorted(stage_groups.keys())
    for idx, stage in enumerate(all_stages):
        items = stage_groups[stage]
        n = max(len(items), 1)
        base = to_rgb(stage_anchors[idx % len(stage_anchors)])
        alphas = np.linspace(min_alpha, 1.0, n)
        for i, (name, *_) in enumerate(items):
            a = alphas[i]
            colour_map[name] = tuple(base[c] * a + (1 - a) for c in range(3))
    return colour_map


def _parse_stage(layer_name):
    """Return (stage_int, layer_int_within_stage) from a layer name."""
    parts = layer_name.split('.')
    stage = 0
    for p in parts:
        if p.startswith('transformer_block'):
            try:
                stage = int(p.replace('transformer_block', ''))
            except ValueError:
                pass
    layer = 0
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            try:
                layer = int(parts[i + 1])
            except ValueError:
                pass
    return stage, layer


def _short(layer_name):
    stage, layer = _parse_stage(layer_name)
    return f'S{stage}-L{layer}'


def _spine_cleanup(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def enable_all(model):
    layers, names = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, HopfieldMLP):
            mod.enable_capture()
            layers.append(mod)
            names.append(name)
    return layers, names


def disable_all(layers):
    for mod in layers:
        mod.disable_capture()


def collect_energies(layers, names):
    """Returns list of (name, up_energies, down_energies)."""
    results = []
    for mod, name in zip(layers, names):
        d = mod.extract_states()
        up_e = d['up'][1]     # list of floats
        down_e = d['down'][1]
        results.append((name, up_e, down_e))
    return results


def build_figure(energy_data):
    stage_groups = defaultdict(list)
    for item in energy_data:
        stage, _ = _parse_stage(item[0])
        stage_groups[stage].append(item)

    all_stages = sorted(stage_groups.keys())
    up_colours = _layer_colors(stage_groups, _UP_STAGES)
    down_colours = _layer_colors(stage_groups, _DOWN_STAGES)

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.60))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.18, wspace=0.32)

    for col, (key, colour_map, anchors, title, panel_letter) in enumerate([
        ('up', up_colours, _UP_STAGES, 'HopfieldMLP (Up)', 'a'),
        ('down', down_colours, _DOWN_STAGES, 'HopfieldMLP (Down)', ''),
    ]):
        ax = axes[col]

        for name, up_e, down_e in energy_data:
            e = up_e if key == 'up' else down_e
            if not e:
                continue
            stage, _ = _parse_stage(name)
            marker = _STAGE_MARKERS[(all_stages.index(stage)) % len(_STAGE_MARKERS)]
            ax.plot(np.arange(len(e)), e,
                   color=colour_map[name],
                   lw=0.9, marker=marker, markersize=2.5,
                   alpha=0.95, zorder=3)

        ax.set_xlabel('Update step')
        if col == 0:
            ax.set_ylabel('Energy')
        ax.set_title(title, pad=4)
        ax.yaxis.grid(True, lw=0.3, color='#cccccc', zorder=0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        _spine_cleanup(ax)
        ax.text(-0.10, 1.08, panel_letter, transform=ax.transAxes,
               fontsize=8, fontweight='bold', va='bottom', ha='right')

        # Per-panel legend, one entry per stage at full saturation.
        handles = []
        for i, stage in enumerate(all_stages):
            color = to_rgb(anchors[i % len(anchors)])
            marker = _STAGE_MARKERS[i % len(_STAGE_MARKERS)]
            handles.append(mlines.Line2D(
                [0], [0], color=color, lw=1.2,
                marker=marker, markersize=3.5,
                label=f'Stage {stage}',
            ))
        leg = ax.legend(
            handles=handles,
            loc='upper right',
            frameon=True, framealpha=0.9,
            edgecolor='#dddddd',
            fontsize=5.5,
            handlelength=1.4, handletextpad=0.4,
            labelspacing=0.3, borderpad=0.4,
        )
        leg.get_frame().set_linewidth(0.35)

    return fig


def save_figure(fig, base_path, title=''):
    svg = base_path + '.svg'
    fig.savefig(svg, format='svg', bbox_inches='tight', facecolor='white')
    print(f'  saved {svg}')

    pdf = base_path + '.pdf'
    with PdfPages(pdf) as p:
        p.savefig(fig, bbox_inches='tight', facecolor='white')
        p.infodict()['Title'] = title
    print(f'  saved {pdf}')

    plt.close(fig)


def main():
    # CONFIG — adjust these per run
    model_module = "hokuvit.HoKuVit"
    hokuvit_config = "../../configs/hokuvit.yaml"  # adjust if your layout differs
    checkpoint = "../../checkpoints/26_0503_fixed_pointwise_cait_79.78.pkl"
    data_root = "../../data"
    out_dir = "figures"
    out_base = "fig1_a_hopfield_energy_evolution"
    input_class = 3  # cat
    batch_size = 64
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    cfg = load_yaml_config(hokuvit_config)
    model_kwargs = build_model_kwargs(cfg)

    mod = importlib.import_module(model_module)
    model = mod.ConvolutionalVisionTransformer(**model_kwargs)
    model.to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print('Model loaded')

    layers, names = enable_all(model)
    print(f'Found {len(layers)} HopfieldMLP layers')

    # Find one sample from the target class
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tuple(cfg['cifar_mean']), tuple(cfg['cifar_std'])),
    ])
    ds = CIFAR10(root=data_root, train=False, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

    input_img = None
    for imgs, labels in loader:
        for i in range(len(imgs)):
            if labels[i].item() == input_class:
                input_img = imgs[i]
                break
        if input_img is not None:
            break

    class_name = cifar10_classes[input_class]
    print(f'Input: class {input_class} ({class_name})')

    with torch.no_grad():
        _ = model(input_img.unsqueeze(0).to(device))

    energy_data = collect_energies(layers, names)
    disable_all(layers)

    print('\nCollected energy trajectories:')
    for name, up_e, down_e in energy_data:
        print(f'  {_short(name):<12s}  up steps={len(up_e)}  '
              f'down steps={len(down_e)}  '
              f'dE_up={up_e[0] - up_e[-1]:.4f}  '
              f'dE_down={down_e[0] - down_e[-1]:.4f}')

    print('\nBuilding figure...')
    fig = build_figure(energy_data)

    out = os.path.join(out_dir, out_base)
    save_figure(fig, out, title=f'HoKuVit Hopfield energy - {class_name}')
    print(f'\nDone.\n  {out}.svg\n  {out}.pdf')


if __name__ == '__main__':
    main()