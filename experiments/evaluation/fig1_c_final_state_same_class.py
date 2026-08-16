import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hokuvit.HoKuVit import ConvolutionalVisionTransformer
from hokuvit.hopfield_linear.Hopfield_linear import HopfieldMLP
from utils.hokuvit.io_utils import load_yaml_config


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


class HopfieldMLPStateExtractor:
    """Extract only the final states from HopfieldMLP layers."""

    def __init__(self):
        self.mlp_layers = []
        self.layer_names = []

    def enable_capture(self, model):
        for name, module in model.named_modules():
            if isinstance(module, HopfieldMLP):
                module.enable_capture()
                self.mlp_layers.append(module)
                self.layer_names.append(name)
                print(f"  Found HopfieldMLP: {name}")

        print(f"\nFound {len(self.mlp_layers)} HopfieldMLP layers")
        if len(self.mlp_layers) > 0:
            print(f"  Last layer: {self.layer_names[-1]}")

    def disable_capture(self, model):
        for module in model.modules():
            if isinstance(module, HopfieldMLP):
                module.disable_capture()

    def get_last_mlp_final_state(self):
        """Get the final ('down' projection, last update step) state from
        the last HopfieldMLP layer."""
        if len(self.mlp_layers) == 0:
            return None, None

        last_mlp = self.mlp_layers[-1]
        last_name = self.layer_names[-1]

        states_energies = last_mlp.extract_states()
        if isinstance(states_energies, dict) and 'down' in states_energies:
            down_states = states_energies['down'][0]  # list of states
            if len(down_states) > 0:
                return last_name, down_states[-1]

        return last_name, None


def visualize_hopfield_states_same_class(states_list, images_list, class_name, cifar_mean, cifar_std, save_path):
    """5 samples from the same class: input image next to its final Hopfield
    state pattern (Sequence Position x Dimension)."""
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)
    cmap = sns.light_palette("steelblue", as_cmap=True)

    num_samples = len(states_list)
    mean = np.array(cifar_mean).reshape(3, 1, 1)
    std = np.array(cifar_std).reshape(3, 1, 1)

    fig = plt.figure(figsize=(8, 2.5 * num_samples))

    for sample_idx in range(num_samples):
        final_state = states_list[sample_idx]
        input_image = images_list[sample_idx]

        if final_state is None:
            continue

        final_state_np = final_state.cpu().numpy() if torch.is_tensor(final_state) else final_state
        print(f"Sample {sample_idx + 1}: shape = {final_state_np.shape}, "
              f"range = [{final_state_np.min():.3f}, {final_state_np.max():.3f}]")

        # Input image
        ax1 = plt.subplot(num_samples, 2, sample_idx * 2 + 1)
        img_np = input_image.cpu().numpy() if torch.is_tensor(input_image) else input_image
        img_denorm = np.clip(img_np * std + mean, 0, 1)
        ax1.imshow(np.transpose(img_denorm, (1, 2, 0)))
        ax1.set_title(f'{class_name} #{sample_idx + 1}', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Hopfield state pattern
        ax2 = plt.subplot(num_samples, 2, sample_idx * 2 + 2)

        if final_state_np.ndim == 4:
            B, C, H, W = final_state_np.shape
            pattern_2d = final_state_np[0].reshape(C, -1).T
        elif final_state_np.ndim == 3:
            pattern_2d = final_state_np[0]
        elif final_state_np.ndim == 2:
            pattern_2d = final_state_np
        else:
            flat = final_state_np.flatten()
            size = int(np.ceil(np.sqrt(len(flat))))
            pattern_2d = np.zeros((size, size))
            pattern_2d.flat[:len(flat)] = flat

        ax2.imshow(pattern_2d, cmap=cmap, aspect='auto', interpolation='nearest')
        ax2.set_title('Hopfield Final State', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Dimension', fontsize=10)
        ax2.set_ylabel('Sequence Position', fontsize=10)
        sns.despine(ax=ax2, top=True, right=True)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02)

    svg_path = save_path + '.svg'
    pdf_path = save_path + '.pdf'
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved {svg_path}")
    print(f"Saved {pdf_path}")

    sns.reset_defaults()


def main():
    # CONFIG — adjust these per run
    hokuvit_config = "../../configs/hokuvit.yaml"  # adjust if your layout differs
    checkpoint = "../../checkpoints/26_0503_fixed_pointwise_cait_79.78.pkl"
    data_root = "../../data"
    figures_dir = "figures"
    figure_name = "fig1_c_hopfield_final_states_same_class"
    selected_class_id = 5  # dog
    max_samples = 5
    batch_size = 100
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    print("=" * 80)
    print("HOPFIELD MLP FINAL STATE VISUALIZATION - SAME CLASS")
    print("=" * 80)

    selected_class_name = cifar10_classes[selected_class_id]
    print(f"\nSelected class: {selected_class_name} (ID: {selected_class_id})")
    print(f"Will collect {max_samples} samples from this class")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    cfg = load_yaml_config(hokuvit_config)
    model_kwargs = build_model_kwargs(cfg)

    print("\nLoading model...")
    model = ConvolutionalVisionTransformer(**model_kwargs)
    model.to(device)

    print("Loading checkpoint...")
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print("Model loaded")

    print("\nLoading CIFAR-10 test set...")
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tuple(cfg['cifar_mean']), tuple(cfg['cifar_std'])),
    ])
    test_dataset = CIFAR10(root=data_root, train=False, download=True, transform=tf)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Dataset loaded")

    print("\nSetting up HopfieldMLP extractor...")
    extractor = HopfieldMLPStateExtractor()
    extractor.enable_capture(model)

    print("\n" + "=" * 80)
    print(f"COLLECTING {max_samples} SAMPLES FROM CLASS: {selected_class_name}")
    print("=" * 80)

    states_list, images_list = [], []
    samples_collected = 0

    with torch.no_grad():
        for images, labels in test_loader:
            for i in range(len(images)):
                if labels[i].item() == selected_class_id and samples_collected < max_samples:
                    images_list.append(images[i])
                    _ = model(images[i:i + 1].to(device))

                    layer_name, final_state = extractor.get_last_mlp_final_state()
                    if final_state is not None:
                        states_list.append(final_state)
                        samples_collected += 1
                        print(f"  Sample {samples_collected}/{max_samples} collected")

            if samples_collected >= max_samples:
                print(f"\nCollected all {samples_collected} samples")
                break

    extractor.disable_capture(model)

    os.makedirs(figures_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)

    out_path = os.path.join(figures_dir, f'{figure_name}_{selected_class_name}')
    visualize_hopfield_states_same_class(
        states_list, images_list, selected_class_name,
        cfg['cifar_mean'], cfg['cifar_std'],
        save_path=out_path,
    )

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\n  {out_path}.svg")
    print(f"  {out_path}.pdf")
    print(f"\nThis shows {max_samples} different {selected_class_name} samples:")
    print("  Column 1: Input images")
    print("  Column 2: Final converged states [Sequence Position x Dimension]")
    print("\nTo visualize a different class, change 'selected_class_id' in main()")


if __name__ == '__main__':
    main()