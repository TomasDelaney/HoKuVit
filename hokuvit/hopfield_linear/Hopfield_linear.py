import torch
import torch.nn as nn
from torch.nn.utils import parametrize
import math
from utils.hokuvit.Zoneout import Zoneout


class BinarizeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sigma=1.0):
        ctx.save_for_backward(x)
        ctx.sigma = sigma
        return (x + 1e-9).sign()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sigma = ctx.sigma
        grad = torch.exp(-0.5 * (x / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        return grad_output * grad, None


binarize = BinarizeF.apply


class BlockDiagonalSymmetricZeroDiag(nn.Module):
    def __init__(self, num_blocks, block_size):
        super().__init__()
        self.num_blocks = num_blocks
        # Register mask as a buffer
        mask = torch.block_diag(*[torch.ones(block_size, block_size) for _ in range(num_blocks)])
        self.register_buffer('mask', mask)

    def forward(self, W):
        W_blocked = W * self.mask
        W_sym = 0.5 * (W_blocked + W_blocked.T)
        W_sym.fill_diagonal_(0)
        return W_sym


class HopfieldMLP(nn.Module):
    """
    FastHopfieldMLP with dropout-based updates and built-in evaluation hooks.
    Tracks UP and DOWN projections separately.
    """

    def __init__(self, config, capture_enabled: bool = False):
        super().__init__()
        h = config["hidden_size"]
        i = config["intermediate_size"]

        if i % h != 0:
            raise ValueError(f"intermediate_size ({i}) must be a multiple of hidden_size ({h})")

        self.ratio = i // h
        self.hidden_size = h
        self.update_steps = config["hopfield_update_steps"]

        # Dropout configuration
        self.zoneout_prob = config.get("zoneout_prob", 0.2)

        # Block-diagonal weight matrices
        self.memory_up = nn.Linear(self.ratio * h, self.ratio * h, bias=False)
        parametrize.register_parametrization(
            self.memory_up, "weight",
            BlockDiagonalSymmetricZeroDiag(self.ratio, h)
        )

        self.memory_down = nn.Linear(self.ratio * h, self.ratio * h, bias=False)
        parametrize.register_parametrization(
            self.memory_down, "weight",
            BlockDiagonalSymmetricZeroDiag(self.ratio, h)
        )

        # Learnable scaling factors
        self.alpha_down = nn.Parameter(torch.full((self.ratio,), 0.5))

        # Layer norms
        self.norm_down = nn.ModuleList([nn.LayerNorm(h) for _ in range(self.ratio)])

        # Final mixing
        self.block_importance = nn.Parameter(torch.ones(self.ratio) / self.ratio)
        self.norm_final = nn.LayerNorm(h)

        self.reset_parameters()

        print(f"HopfieldMLP: ratio={self.ratio}, h={h}, "
              f"dropout_rate={self.zoneout_prob}, update_steps={self.update_steps}")

        # Evaluation hooks - separate for UP and DOWN
        self.capture_enabled = capture_enabled
        self.up_states = []
        self.up_energies = []
        self.down_states = []
        self.down_energies = []

        # zoneouts
        self.zoneout_up = Zoneout(p=self.zoneout_prob)
        self.zoneout_down = Zoneout(p=self.zoneout_prob)

    def reset_parameters(self):
        # Orthogonal initialization for better stability and convergence
        nn.init.orthogonal_(self.memory_up.weight)
        nn.init.orthogonal_(self.memory_down.weight)

    def _compute_energy_block_diagonal(self, state, W, num_blocks):
        """Compute energy for block-diagonal weight matrix"""
        block_size = W.shape[0] // num_blocks
        total_energy = torch.zeros(state.shape[0], device=state.device)

        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size if i < num_blocks - 1 else W.shape[0]

            # Extract diagonal block only
            W_block = W[start_idx:end_idx, start_idx:end_idx]
            state_block = state[:, :, start_idx:end_idx]

            # Energy for this block
            energy_per_position = -0.5 * torch.sum((state_block @ W_block) * state_block, dim=-1)
            total_energy += energy_per_position.mean(dim=-1)

        return total_energy.mean().item()

    def _dropout_update(self, x, W, zoneout):
        """Simplified dropout-based update"""
        # Compute activations
        h = torch.nn.functional.linear(x, W)

        # Binarize
        updates = binarize(h).to(x.dtype)

        # Apply Zoneout
        x = zoneout(x_old=x, x_new=updates)

        return x


    def forward(self, x):
        B, S, h = x.shape

        # --- UP-PROJECTION with vectorized block-async updates ---
        x_up = x.repeat(1, 1, self.ratio)  # x_expanded

        # Capture initial state for UP if enabled
        if self.capture_enabled:
            self.up_states.append(x_up.detach().cpu())
            self.up_energies.append(self._compute_energy_block_diagonal(x_up, self.memory_up.weight, self.ratio))

        for step in range(self.update_steps):
            x_up = self._dropout_update(x_up, self.memory_up.weight, self.zoneout_up)

            # Capture state after each update if enabled
            if self.capture_enabled:
                self.up_states.append(x_up.detach().cpu())
                self.up_energies.append(self._compute_energy_block_diagonal(x_up, self.memory_up.weight, self.ratio))

        # --- DOWN-PROJECTION with vectorized block-async updates ---
        x_base_down = x_up
        W_down = self.memory_down.weight

        x_down = x_up

        # Capture initial state for DOWN if enabled
        if self.capture_enabled:
            self.down_states.append(x_down.detach().cpu())
            self.down_energies.append(self._compute_energy_block_diagonal(x_down, W_down, self.ratio))

        for step in range(self.update_steps):
            x_down = self._dropout_update(x_down, W_down, self.zoneout_down)

            # Capture state after each update if enabled
            if self.capture_enabled:
                self.down_states.append(x_down.detach().cpu())
                self.down_energies.append(self._compute_energy_block_diagonal(x_down, W_down, self.ratio))

        # Apply block-wise normalization and residual
        x_down_list = []
        for i in range(self.ratio):
            block = x_down[:, :, i * h:(i + 1) * h]
            block_normed = self.norm_down[i](block)
            base_block = x_base_down[:, :, i * h:(i + 1) * h]
            alpha = torch.sigmoid(self.alpha_down[i])
            x_down_list.append(base_block + alpha * block_normed)

        # Stack for mixing
        downs_stacked = torch.stack(x_down_list, dim=0)

        # Optimized mixing
        weights = torch.softmax(self.block_importance, dim=0)
        out = torch.einsum("rbsh,r->bsh", downs_stacked, weights)

        return self.norm_final(out)

    def enable_capture(self):
        """Enable state/energy capture for evaluation"""
        self.capture_enabled = True
        self.up_states = []
        self.up_energies = []
        self.down_states = []
        self.down_energies = []

    def disable_capture(self):
        """Disable capture"""
        self.capture_enabled = False

    def extract_states(self):
        """Extract captured states and energies, then clear history"""
        up_states = self.up_states.copy()
        up_energies = self.up_energies.copy()
        down_states = self.down_states.copy()
        down_energies = self.down_energies.copy()

        self.up_states = []
        self.up_energies = []
        self.down_states = []
        self.down_energies = []

        return {
            'up': (up_states, up_energies),
            'down': (down_states, down_energies)
        }