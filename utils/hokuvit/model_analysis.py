"""Parameter accounting for the CVT model: how many params live in plain
layers vs. the Hopfield and Kuramoto (ONN) neuromorphic components.

``analyze_model_parameters`` does the counting and returns a stats dict.
``print_parameter_report`` renders that dict as a human-readable report.
``calculate_hopfield_ratio`` is a thin convenience wrapper that does both,
matching the original script's behavior.
"""
import torch.nn as nn

from hokuvit.hopfield_linear.Hopfield_linear import HopfieldMLP
from hokuvit.kuramoto_cnn.Oscillatory_convolutions import (
    KuramotoConv2d, KuramotoPointwiseConv2d, KuramotoTokenConv2d,
)


def count_parameters(model) -> int:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The model has: ", n, " number of parameters")
    return n


def analyze_model_parameters(model) -> dict:
    """Walk the model and bucket every parameter by layer type / neuromorphic
    component. Pure computation — no printing. Returns the same stats dict
    shape `print_parameter_report` expects."""
    total_params_raw = sum(p.numel() for p in model.parameters())

    layernorm_params = layernorm_count = 0
    batchnorm_params = batchnorm_count = 0
    conv2d_params = conv2d_count = 0
    linear_params = linear_count = 0
    embedding_params = embedding_count = 0

    hopfield_memory_params = 0
    hopfield_projection_params = 0
    hopfield_norm_params = 0
    hopfield_other_params = 0
    hopfield_layer_count = 0
    zero_block_overhead = 0

    hopfield_breakdown = {
        'dual_memory': {'count': 0, 'memory_params': 0, 'projection_params': 0, 'norm_params': 0, 'other_params': 0},
    }

    onn_total_params = 0
    onn_layer_count = 0
    all_onn_types = (KuramotoConv2d, KuramotoPointwiseConv2d, KuramotoTokenConv2d)
    onn_breakdown = {cls.__name__: {'count': 0, 'conv_params': 0, 'oscillator_params': 0} for cls in all_onn_types}

    conv2d_breakdown = {
        'depthwise': {'count': 0, 'params': 0},
        'pointwise': {'count': 0, 'params': 0},
        'standard': {'count': 0, 'params': 0},
        'token_embedding': {'count': 0, 'params': 0},
    }

    dual_memory_types = (HopfieldMLP,)
    all_hopfield_types = dual_memory_types
    all_onn_types = (KuramotoConv2d, KuramotoPointwiseConv2d, KuramotoTokenConv2d)

    # Track which Conv2d modules live inside an ONN module to avoid double counting
    onn_conv_modules = set()

    for name, module in model.named_modules():
        module_params = sum(p.numel() for p in module.parameters(recurse=False))

        if isinstance(module, nn.LayerNorm):
            layernorm_count += 1
            layernorm_params += module_params

        # ONN check happens BEFORE Conv2d since ONNs contain Conv2d internally
        if isinstance(module, all_onn_types):
            onn_layer_count += 1
            onn_type = type(module).__name__

            for child_name, child_module in module.named_modules():
                if isinstance(child_module, nn.Conv2d):
                    onn_conv_modules.add(id(child_module))

            conv_params = 0
            oscillator_params = 0
            proj_params = 0  # non-ONN projection parameters, kept for readability

            for param_name, param in module.named_parameters():
                if 'proj.weight' in param_name or 'dim_projection' in param_name:
                    # Projection weights (dimension matching, not the oscillatory computation)
                    proj_params += param.numel()
                    conv2d_params += param.numel()
                    conv2d_count += 1
                elif 'coupling_strength' in param_name or ('weight' in param_name and 'proj' not in param_name and 'dim_projection' not in param_name):
                    conv_params += param.numel()
                else:
                    # Oscillator params: omega_0, phase_offset, mu, etc.
                    oscillator_params += param.numel()

            onn_total_params += (conv_params + oscillator_params)

            if onn_type in onn_breakdown:
                onn_breakdown[onn_type]['count'] += 1
                onn_breakdown[onn_type]['conv_params'] += conv_params
                onn_breakdown[onn_type]['oscillator_params'] += oscillator_params

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            batchnorm_count += 1
            batchnorm_params += module_params

        elif isinstance(module, nn.Conv2d) and id(module) not in onn_conv_modules:
            conv2d_count += 1
            conv2d_params += module_params

            if module.groups == module.in_channels and module.in_channels > 1:
                conv2d_breakdown['depthwise']['count'] += 1
                conv2d_breakdown['depthwise']['params'] += module_params
            elif module.kernel_size == (1, 1) or module.kernel_size == 1:
                conv2d_breakdown['pointwise']['count'] += 1
                conv2d_breakdown['pointwise']['params'] += module_params
            elif 'conv_token_emb' in name:
                conv2d_breakdown['token_embedding']['count'] += 1
                conv2d_breakdown['token_embedding']['params'] += module_params
            else:
                conv2d_breakdown['standard']['count'] += 1
                conv2d_breakdown['standard']['params'] += module_params

        elif isinstance(module, nn.Linear):
            linear_count += 1
            linear_params += module_params

        elif isinstance(module, nn.Embedding):
            embedding_count += 1
            embedding_params += module_params

        elif isinstance(module, all_hopfield_types):
            hopfield_layer_count += 1

            memory_params = projection_params = norm_params = other_params = 0

            if isinstance(module, dual_memory_types):
                breakdown_key = 'dual_memory'

                for up_down in ('memory_up', 'memory_down'):
                    if hasattr(module, up_down):
                        sub = getattr(module, up_down)
                        if hasattr(module, 'ratio') and hasattr(module, 'hidden_size'):
                            h, ratio = module.hidden_size, module.ratio
                            effective_params = ratio * h * h
                            actual_params = sum(p.numel() for p in sub.parameters())
                            zero_block_overhead += (actual_params - effective_params)
                            memory_params += effective_params
                        else:
                            memory_params += sum(p.numel() for p in sub.parameters())

                for attr in ('proj_in_up', 'proj_out_up', 'proj_in_down', 'proj_out_down'):
                    if hasattr(module, attr):
                        projection_params += sum(p.numel() for p in getattr(module, attr).parameters())

                for attr in ('norm_up', 'norm_down', 'norm_final'):
                    if hasattr(module, attr):
                        norm_params += sum(p.numel() for p in getattr(module, attr).parameters())

                # Remaining direct params (alpha, mix, etc.) not already counted above
                for param_name, param in module.named_parameters(recurse=False):
                    if not any(sub in param_name for sub in
                               ['memory_up.', 'memory_down.', 'proj_', 'norm_up', 'norm_down']):
                        other_params += param.numel()

            hopfield_memory_params += memory_params
            hopfield_projection_params += projection_params
            hopfield_norm_params += norm_params
            hopfield_other_params += other_params

            hopfield_breakdown[breakdown_key]['count'] += 1
            hopfield_breakdown[breakdown_key]['memory_params'] += memory_params
            hopfield_breakdown[breakdown_key]['projection_params'] += projection_params
            hopfield_breakdown[breakdown_key]['norm_params'] += norm_params
            hopfield_breakdown[breakdown_key]['other_params'] += other_params

    total_params = total_params_raw - zero_block_overhead
    hopfield_total = hopfield_memory_params + hopfield_projection_params + hopfield_norm_params + hopfield_other_params
    neuromorphic_total = hopfield_total + onn_total_params

    return {
        "total_params_raw": total_params_raw,
        "zero_block_overhead": zero_block_overhead,
        "total_params": total_params,
        "layernorm": {"count": layernorm_count, "params": layernorm_params},
        "batchnorm": {"count": batchnorm_count, "params": batchnorm_params},
        "conv2d": {"count": conv2d_count, "params": conv2d_params, "breakdown": conv2d_breakdown},
        "linear": {"count": linear_count, "params": linear_params},
        "embedding": {"count": embedding_count, "params": embedding_params},
        "onn": {"count": onn_layer_count, "total_params": onn_total_params, "breakdown": onn_breakdown},
        "hopfield": {
            "count": hopfield_layer_count,
            "memory_params": hopfield_memory_params,
            "projection_params": hopfield_projection_params,
            "norm_params": hopfield_norm_params,
            "other_params": hopfield_other_params,
            "total_params": hopfield_total,
            "breakdown": hopfield_breakdown,
        },
        "neuromorphic": {
            "total_params": neuromorphic_total,
            "percentage": neuromorphic_total / total_params * 100,
            "hopfield_params": hopfield_total,
            "onn_params": onn_total_params,
        },
    }


def print_parameter_report(stats: dict) -> None:
    """Render the stats dict from `analyze_model_parameters` as a report."""
    total = stats["total_params"]
    conv2d = stats["conv2d"]
    onn = stats["onn"]
    hopfield = stats["hopfield"]
    neuro = stats["neuromorphic"]

    def pct(n):
        return n / total * 100

    print("=" * 100)
    print("COMPREHENSIVE PARAMETER BREAKDOWN")
    print("=" * 100)
    print(f"\n{'TOTAL PARAMETERS':.<60} {stats['total_params_raw']:>15,}")
    if stats["zero_block_overhead"] > 0:
        print(f"{'Zero Block Overhead':.<60} {stats['zero_block_overhead']:>15,}")
        print(f"{'Effective Total':.<60} {total:>15,}")

    print(f"\n{'-' * 100}")
    print(f"{'LAYER TYPE':<40} {'COUNT':>10} {'PARAMETERS':>15} {'% OF TOTAL':>15}")
    print(f"{'-' * 100}")
    print(f"{'LayerNorm':<40} {stats['layernorm']['count']:>10} {stats['layernorm']['params']:>15,} {pct(stats['layernorm']['params']):>14.2f}%")
    print(f"{'BatchNorm (all types)':<40} {stats['batchnorm']['count']:>10} {stats['batchnorm']['params']:>15,} {pct(stats['batchnorm']['params']):>14.2f}%")
    print(f"{'Conv2d (all types)':<40} {conv2d['count']:>10} {conv2d['params']:>15,} {pct(conv2d['params']):>14.2f}%")
    for label, key in [('   Depthwise Conv', 'depthwise'), ('   Pointwise (1x1) Conv', 'pointwise'),
                        ('   Token Embedding Conv', 'token_embedding'), ('   Standard Conv', 'standard')]:
        c = conv2d['breakdown'][key]
        print(f"{label:<40} {c['count']:>10} {c['params']:>15,} {pct(c['params']):>14.2f}%")
    print(f"{'Linear (non-Hopfield)':<40} {stats['linear']['count']:>10} {stats['linear']['params']:>15,} {pct(stats['linear']['params']):>14.2f}%")
    if stats['embedding']['count'] > 0:
        print(f"{'Embedding':<40} {stats['embedding']['count']:>10} {stats['embedding']['params']:>15,} {pct(stats['embedding']['params']):>14.2f}%")

    if onn['count'] > 0:
        print(f"\n{'-' * 100}")
        print(f"{'OSCILLATORY NEURAL NETWORKS (ONN)':<40} {onn['count']:>10} {onn['total_params']:>15,} {pct(onn['total_params']):>14.2f}%")
        print(f"{'-' * 100}")
        for onn_type, data in onn['breakdown'].items():
            if data['count'] > 0:
                total_onn_type = data['conv_params'] + data['oscillator_params']
                print(f"{'  ' + onn_type:<40} {data['count']:>10} {total_onn_type:>15,} {pct(total_onn_type):>14.2f}%")
                print(f"{'    Conv weights (incl. coupling)':<40} {'-':>10} {data['conv_params']:>15,} {pct(data['conv_params']):>14.2f}%")
                print(f"{'    Oscillator params':<40} {'-':>10} {data['oscillator_params']:>15,} {pct(data['oscillator_params']):>14.2f}%")

    print(f"\n{'-' * 100}")
    print(f"{'HOPFIELD LAYERS (TOTAL)':<40} {hopfield['count']:>10} {hopfield['total_params']:>15,} {pct(hopfield['total_params']):>14.2f}%")
    print(f"{'-' * 100}")
    print(f"{'Hopfield Component Breakdown:':.<60}")
    print(f"{'  Memory Matrices':<40} {'-':>10} {hopfield['memory_params']:>15,} {pct(hopfield['memory_params']):>14.2f}%")
    print(f"{'  Projection Layers':<40} {'-':>10} {hopfield['projection_params']:>15,} {pct(hopfield['projection_params']):>14.2f}%")
    print(f"{'  Normalization Layers':<40} {'-':>10} {hopfield['norm_params']:>15,} {pct(hopfield['norm_params']):>14.2f}%")
    print(f"{'  Other (scaling, bias, etc.)':<40} {'-':>10} {hopfield['other_params']:>15,} {pct(hopfield['other_params']):>14.2f}%")

    print(f"\n{'-' * 100}")
    print(f"{'HOPFIELD LAYER TYPES':.<60}")
    print(f"{'-' * 100}")
    for layer_type, s in hopfield['breakdown'].items():
        if s['count'] > 0:
            total_type_params = s['memory_params'] + s['projection_params'] + s['norm_params'] + s['other_params']
            print(f"\n{layer_type.replace('_', ' ').title()}: {s['count']} layers, {total_type_params:,} parameters ({pct(total_type_params):.2f}%)")
            print(f"{'   Memory':<40} {s['memory_params']:>15,} {pct(s['memory_params']):>14.2f}%")
            print(f"{'   Projections':<40} {s['projection_params']:>15,} {pct(s['projection_params']):>14.2f}%")
            print(f"{'   Normalization':<40} {s['norm_params']:>15,} {pct(s['norm_params']):>14.2f}%")
            print(f"{'   Other':<40} {s['other_params']:>15,} {pct(s['other_params']):>14.2f}%")

    print(f"\n{'=' * 100}")
    print(f"{'NEUROMORPHIC PARAMETERS (HOPFIELD + ONN)':.<60}")
    print(f"{'=' * 100}")
    print(f"{'Hopfield Networks:':<40} {hopfield['count']:>10} {hopfield['total_params']:>15,} {pct(hopfield['total_params']):>14.2f}%")
    print(f"{'Oscillatory Neural Networks:':<40} {onn['count']:>10} {onn['total_params']:>15,} {pct(onn['total_params']):>14.2f}%")
    print(f"{'-' * 100}")
    print(f"{'TOTAL NEUROMORPHIC:':<40} {hopfield['count'] + onn['count']:>10} {neuro['total_params']:>15,} {pct(neuro['total_params']):>14.2f}%")
    print(f"{'Conventional Parameters:':<40} {'-':>10} {total - neuro['total_params']:>15,} {pct(total - neuro['total_params']):>14.2f}%")

    total_layers = (stats['layernorm']['count'] + stats['batchnorm']['count'] + conv2d['count']
                     + stats['linear']['count'] + hopfield['count'] + stats['embedding']['count'] + onn['count'])
    print(f"\n{'=' * 100}")
    print(f"{'SUMMARY STATISTICS':.<60}")
    print(f"{'=' * 100}")
    print(f"{'Total Layers Analyzed:':<60} {total_layers:>15,}")
    print(f"{'Hopfield Memory as % of Total:':<60} {pct(hopfield['memory_params']):>14.2f}%")
    print(f"{'All Hopfield Components as % of Total:':<60} {pct(hopfield['total_params']):>14.2f}%")
    print(f"{'All ONN Components as % of Total:':<60} {pct(onn['total_params']):>14.2f}%")
    print(f"{'All Neuromorphic Parameters as % of Total:':<60} {neuro['percentage']:>14.2f}%")
    print(f"{'Conventional Parameters as % of Total:':<60} {pct(total - neuro['total_params']):>14.2f}%")
    print("=" * 100)


def calculate_hopfield_ratio(model) -> dict:
    """Compute the parameter breakdown and print the full report (matches
    the original script's combined analyze+print behavior)."""
    stats = analyze_model_parameters(model)
    print_parameter_report(stats)
    return stats
