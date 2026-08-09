import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Kuramoto Oscillators with Input as Driving Force

Key change: Input acts as theta_0 the initial condition to the ODE.

The dynamics are:
    dθ/dt = ω₀ + (1/N) * Σⱼ w_ij * sin(θⱼ - θᵢ)

where θ(0) = x  (input seeds the initial phase configuration)
and N = kernel_size² normalises coupling so K is kernel-size-independent.

Where K is a set of learnable convolutional filters.
"""


class KuramotoConv2d(nn.Module):
    """
    Kuramoto oscillator-based depthwise convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, dt=0.1, num_steps=5, capture_enabled=False,
                 min_omega=0.3, omega_init_mean=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dt = dt
        self.num_steps = num_steps
        self.min_omega = min_omega

        # Dimension projection (only if channels differ)
        if in_channels != out_channels:
            self.dim_projection = nn.Conv2d(in_channels,
                                            out_channels - in_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            groups=groups,
                                            bias=False)
        else:
            self.dim_projection = None

        # CRITICAL: Depthwise coupling (within-channel only)
        self.coupling_strength = nn.Conv2d(
            out_channels, out_channels, kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=out_channels,
            bias=False
        )

        # Natural frequencies (learnable)
        magnitude = torch.randn(out_channels).abs() * 0.5 + omega_init_mean  # guaranteed > min_omega
        sign = torch.sign(torch.randn(out_channels))
        self.omega_0 = nn.Parameter(magnitude * sign)

        # Phase offsets (learnable)
        self.phase_offset = nn.Parameter(torch.zeros(out_channels))

        # Metrics storage (not saved in state_dict)
        self.capture_enabled = capture_enabled
        self._order_params = []
        self._final_phases = None
        self._output_stats = {}
        self._convergence_metrics = {}

        self.reset_parameters()

    def reset_parameters(self):
        if self.dim_projection is not None:
            nn.init.kaiming_uniform_(self.dim_projection.weight, a=math.sqrt(5))
        nn.init.orthogonal_(self.coupling_strength.weight.view(self.coupling_strength.weight.shape[0], -1))

    def _compute_order_parameter(self, phases):
        """Compute Kuramoto order parameter R"""
        complex_phases = torch.exp(1j * phases.to(torch.complex64))
        mean_complex = complex_phases.mean()
        return torch.abs(mean_complex).item()

    def decode_from_oscillators(self, theta):
        # Memory-efficient mean-field pairwise readout.

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        mean_sin  = sin_theta.mean(dim=1, keepdim=True)
        mean_cos  = cos_theta.mean(dim=1, keepdim=True)
        output    = sin_theta * mean_cos - cos_theta * mean_sin
        phase_offset_exp = self.phase_offset.view(1, -1, 1, 1)
        return output + phase_offset_exp

    def forward(self, x):
        # Use input directly as driving force (with optional dimension projection)
        if self.dim_projection is not None:
            # Learn additional channels (with stride/padding)
            additional = self.dim_projection(x)

            # Downsample input to match spatial size
            x_downsampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)

            theta = torch.cat([x_downsampled, additional], dim=1)
        else:
            theta = x

        # Track convergence
        prev_theta = theta.clone() if self.capture_enabled else None
        phase_changes = [] if self.capture_enabled else None

        # Kuramoto dynamics with input as driving force
        effective_omega = (self.omega_0.sign() * self.omega_0.abs().clamp(min=self.min_omega)).view(1, -1, 1, 1)
        for step in range(self.num_steps):
            # Track order parameter
            if self.capture_enabled:
                order = self._compute_order_parameter(theta)
                self._order_params.append(order)

            # Coupling term (depthwise - within channel only)
            coupling_term = (self.coupling_strength(torch.sin(theta)) * torch.cos(theta) -
                             self.coupling_strength(torch.cos(theta)) * torch.sin(theta))

            # Euler integration
            dtheta = effective_omega + coupling_term
            theta = theta + self.dt * dtheta

            # Track phase change magnitude
            if self.capture_enabled and prev_theta is not None:
                phase_change = torch.abs(theta - prev_theta).mean().item()
                phase_changes.append(phase_change)
                prev_theta = theta.clone()

        # Store final metrics
        if self.capture_enabled:
            final_order = self._compute_order_parameter(theta)
            self._order_params.append(final_order)
            self._final_phases = (theta % (2 * math.pi)).detach()

            # Convergence diagnostics
            self._convergence_metrics = {
                'phase_changes': phase_changes,
                'total_evolution': sum(phase_changes) if phase_changes else 0.0,
                'final_change': phase_changes[-1] if phase_changes else 0.0,
                'converged': phase_changes[-1] < 0.01 if phase_changes else False,
            }

        # Decode to output
        output = self.decode_from_oscillators(theta)

        # Track output stats
        if self.capture_enabled:
            self._output_stats = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'omega_0_mean': self.omega_0.mean().item(),
                'omega_0_std': self.omega_0.std().item(),
                'coupling_weight_std': self.coupling_strength.weight.std().item(),
            }

        return output

    def get_metrics(self):
        """Get collected metrics"""
        return {
            'order_parameters': self._order_params,
            'final_order': self._order_params[-1] if self._order_params else 0.0,
            'final_phases': self._final_phases,
            'output_stats': self._output_stats,
            'convergence_metrics': self._convergence_metrics,
            'num_steps': self.num_steps,
        }

    def clear_metrics(self):
        """Clear stored metrics"""
        self._order_params = []
        self._final_phases = None
        self._output_stats = {}
        self._convergence_metrics = {}


class KuramotoPointwiseConv2d(nn.Module):
    """
    Kuramoto oscillator-based pointwise convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, dt=0.1, num_steps=5, capture_enabled=False,
                 min_omega=0.3, omega_init_mean=1, spatial_size=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dt = dt
        self.num_steps = num_steps
        self.min_omega = min_omega

        # Dimension projection (only if channels differ)
        if in_channels != out_channels:
            self.dim_projection = nn.Conv2d(in_channels,
                                            out_channels - in_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            groups=groups,
                                            bias=False)
        else:
            self.dim_projection = None

        # Pointwise coupling: groups=1, so each x,y location couples across all feature channels
        self.coupling_strength = nn.Conv2d(
            out_channels, out_channels, kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=1,  # Pointwise: each x,y location couples across channels
            bias=False
        )

        # Natural frequencies (learnable)
        # Shape [1, 1, H, W] broadcasts to [B, C, H, W]
        magnitude = torch.randn(1, 1, spatial_size, spatial_size).abs() * 0.5 + omega_init_mean
        sign = torch.sign(torch.randn(1, 1, spatial_size, spatial_size))
        self.omega_0 = nn.Parameter(magnitude * sign)

        # Phase offsets (learnable)
        self.phase_offset = nn.Parameter(torch.zeros(out_channels))

        # Metrics storage (not saved in state_dict)
        self.capture_enabled = capture_enabled
        self._order_params = []
        self._final_phases = None
        self._output_stats = {}
        self._convergence_metrics = {}

        self.reset_parameters()

    def reset_parameters(self):
        if self.dim_projection is not None:
            nn.init.kaiming_uniform_(self.dim_projection.weight, a=math.sqrt(5))
        nn.init.orthogonal_(self.coupling_strength.weight.view(self.coupling_strength.weight.shape[0], -1))

    def _compute_order_parameter(self, phases):
        """Compute Kuramoto order parameter R"""
        complex_phases = torch.exp(1j * phases.to(torch.complex64))
        mean_complex = complex_phases.mean()
        return torch.abs(mean_complex).item()

    def decode_from_oscillators(self, theta):
        # Memory-efficient mean-field pairwise readout.

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        mean_sin  = sin_theta.mean(dim=1, keepdim=True)
        mean_cos  = cos_theta.mean(dim=1, keepdim=True)
        output    = sin_theta * mean_cos - cos_theta * mean_sin
        phase_offset_exp = self.phase_offset.view(1, -1, 1, 1)
        return output + phase_offset_exp

    def forward(self, x):
        # Use input directly as driving force (with optional dimension projection)
        if self.dim_projection is not None:
            # Learn additional channels (with stride/padding)
            additional = self.dim_projection(x)

            # Downsample input to match spatial size
            x_downsampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)

            theta = torch.cat([x_downsampled, additional], dim=1)
        else:
            theta = x

        # Track convergence
        prev_theta = theta.clone() if self.capture_enabled else None
        phase_changes = [] if self.capture_enabled else None

        # Kuramoto dynamics with input as driving force
        effective_omega = self.omega_0.sign() * self.omega_0.abs().clamp(min=self.min_omega)
        for step in range(self.num_steps):
            # Track order parameter
            if self.capture_enabled:
                order = self._compute_order_parameter(theta)
                self._order_params.append(order)

            # coupling_strength is applied identically at every (x,y) location
            # [B, C, H, W] → [B, C, H, W] via 1x1 grouped conv
            sin_t = torch.sin(theta)
            cos_t = torch.cos(theta)
            coupling_term = (self.coupling_strength(sin_t) * cos_t -
                             self.coupling_strength(cos_t) * sin_t)

            # Euler integration
            dtheta = effective_omega + coupling_term
            theta = theta + self.dt * dtheta

            # Track phase change magnitude
            if self.capture_enabled and prev_theta is not None:
                phase_change = torch.abs(theta - prev_theta).mean().item()
                phase_changes.append(phase_change)
                prev_theta = theta.clone()

        # Store final metrics
        if self.capture_enabled:
            final_order = self._compute_order_parameter(theta)
            self._order_params.append(final_order)
            self._final_phases = (theta % (2 * math.pi)).detach()

            # Convergence diagnostics
            self._convergence_metrics = {
                'phase_changes': phase_changes,
                'total_evolution': sum(phase_changes) if phase_changes else 0.0,
                'final_change': phase_changes[-1] if phase_changes else 0.0,
                'converged': phase_changes[-1] < 0.01 if phase_changes else False,
            }

        # Decode to output
        output = self.decode_from_oscillators(theta)

        # Track output stats
        if self.capture_enabled:
            self._output_stats = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'omega_0_mean': self.omega_0.mean().item(),
                'omega_0_std': self.omega_0.std().item(),
                'coupling_weight_std': self.coupling_strength.weight.std().item(),
            }

        return output

    def get_metrics(self):
        """Get collected metrics"""
        return {
            'order_parameters': self._order_params,
            'final_order': self._order_params[-1] if self._order_params else 0.0,
            'final_phases': self._final_phases,
            'output_stats': self._output_stats,
            'convergence_metrics': self._convergence_metrics,
            'num_steps': self.num_steps,
        }

    def clear_metrics(self):
        """Clear stored metrics"""
        self._order_params = []
        self._final_phases = None
        self._output_stats = {}
        self._convergence_metrics = {}


class KuramotoTokenConv2d(nn.Module):
    """
    Kuramoto for token embeddings with depthwise convolution. To match newer number of feature channels
    traditional depthwise convolutions are used.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, dt=0.1, num_steps=5, capture_enabled=False,
                 min_omega=0.3, omega_init_mean=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dt = dt
        self.num_steps = num_steps
        self.min_omega = min_omega

        # Dimension projection (only if channels differ)
        if in_channels != out_channels:
            self.dim_projection = nn.Conv2d(in_channels,
                                            out_channels - in_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            groups=groups,
                                            bias=False)
        else:
            self.dim_projection = None

        # Depthwise coupling (within-channel only)
        self.coupling_strength = nn.Conv2d(
            out_channels, out_channels, kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=out_channels,
            bias=False
        )

        # Natural frequencies
        magnitude = torch.randn(out_channels).abs() * 0.5 + omega_init_mean
        sign = torch.sign(torch.randn(out_channels))
        self.omega_0 = nn.Parameter(magnitude * sign)

        # Phase offsets
        self.phase_offset = nn.Parameter(torch.zeros(out_channels))

        # Metrics storage
        self.capture_enabled = capture_enabled
        self._order_params = []
        self._final_phases = None
        self._output_stats = {}
        self._convergence_metrics = {}

        self.reset_parameters()

    def reset_parameters(self):
        if self.dim_projection is not None:
            nn.init.kaiming_uniform_(self.dim_projection.weight, a=math.sqrt(5))
        nn.init.orthogonal_(self.coupling_strength.weight.view(self.coupling_strength.weight.shape[0], -1))

    def _compute_order_parameter(self, phases):
        complex_phases = torch.exp(1j * phases.to(torch.complex64))
        mean_complex = complex_phases.mean()
        return torch.abs(mean_complex).item()

    def decode_from_oscillators(self, theta):
        # Memory-efficient mean-field pairwise readout — same as KuramotoConv2d.
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        mean_sin  = sin_theta.mean(dim=1, keepdim=True)
        mean_cos  = cos_theta.mean(dim=1, keepdim=True)
        output    = sin_theta * mean_cos - cos_theta * mean_sin
        phase_offset_exp = self.phase_offset.view(1, -1, 1, 1)
        return output + phase_offset_exp

    def forward(self, x):
        # Use input directly as driving force (with optional dimension projection)
        if self.dim_projection is not None:
            # Learn additional channels (with stride/padding)
            additional = self.dim_projection(x)

            # Downsample input to match spatial size
            x_downsampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)

            theta = torch.cat([x_downsampled, additional], dim=1)
        else:
            theta = x

        # Track convergence
        prev_theta = theta.clone() if self.capture_enabled else None
        phase_changes = [] if self.capture_enabled else None

        # Kuramoto dynamics with input as driving force
        effective_omega = (self.omega_0.sign() * self.omega_0.abs().clamp(min=self.min_omega)).view(1, -1, 1, 1)
        for step in range(self.num_steps):
            if self.capture_enabled:
                order = self._compute_order_parameter(theta)
                self._order_params.append(order)

            # Coupling term (depthwise - within channel only)
            coupling_term = (self.coupling_strength(torch.sin(theta)) * torch.cos(theta) -
                             self.coupling_strength(torch.cos(theta)) * torch.sin(theta))

            # Euler integration
            dtheta = effective_omega + coupling_term
            theta = theta + self.dt * dtheta

            # Track phase change magnitude
            if self.capture_enabled and prev_theta is not None:
                phase_change = torch.abs(theta - prev_theta).mean().item()
                phase_changes.append(phase_change)
                prev_theta = theta.clone()

        if self.capture_enabled:
            final_order = self._compute_order_parameter(theta)
            self._order_params.append(final_order)
            self._final_phases = (theta % (2 * math.pi)).detach()

            # Convergence diagnostics
            self._convergence_metrics = {
                'phase_changes': phase_changes,
                'total_evolution': sum(phase_changes) if phase_changes else 0.0,
                'final_change': phase_changes[-1] if phase_changes else 0.0,
                'converged': phase_changes[-1] < 0.01 if phase_changes else False,
            }

        output = self.decode_from_oscillators(theta)

        if self.capture_enabled:
            self._output_stats = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'dim_proj_weight_std': self.dim_projection.weight.std().item() if self.dim_projection else 0.0,
                'omega_0_mean': self.omega_0.mean().item(),
                'omega_0_std': self.omega_0.std().item(),
                'coupling_weight_std': self.coupling_strength.weight.std().item(),
            }

        return output

    def get_metrics(self):
        return {
            'order_parameters': self._order_params,
            'final_order': self._order_params[-1] if self._order_params else 0.0,
            'final_phases': self._final_phases,
            'output_stats': self._output_stats,
            'convergence_metrics': self._convergence_metrics,
            'num_steps': self.num_steps,
        }

    def clear_metrics(self):
        self._order_params = []
        self._final_phases = None
        self._output_stats = {}
        self._convergence_metrics = {}