import torch
import torch.nn.functional as F
from torch import nn
from hokuvit.hopfield_linear.Hopfield_linear import HopfieldMLP
from timm.layers import DropPath
from hokuvit.kuramoto_cnn.Oscillatory_convolutions import (KuramotoConv2d, KuramotoPointwiseConv2d,
                                                           KuramotoTokenConv2d)

class ConvolutionalTokenEmbedding(nn.Module):
    """Convolutional Token Embedding layer for CvT.

        Args:
            in_channel (int): Number of input channels (e.g., 3 for RGB images).
            emb_dim (int): Embedding dimension for each patch.
            kernel_size (int): Size of convolution kernel.
            stride (int): Stride for convolution.
            padding (int): Padding for convolution.
        """

    def __init__(self, in_channel, emb_dim, kernel_size, stride, padding, kuramoto_steps=20, dt=0.1,
                 min_omega=0.1, omega_init_mean=0.3):
        super().__init__()
        self.proj = KuramotoTokenConv2d(in_channels=in_channel,
                              out_channels=emb_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              num_steps=kuramoto_steps,
                              dt=dt,
                              min_omega=min_omega,
                              omega_init_mean=omega_init_mean)
        self.LN = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)
        x = x.permute(0, 2, 1)
        x = self.LN(x)
        return x, H, W


class Attention(nn.Module):
    """
        Convolutional self-attention module with depth-wise separable convolutional projections for Q, K, and V.

        Args:
            cls_token (Tensor or None): Class token for global context.
            in_dim (int): Input embedding dimension.
            num_heads (int): Number of attention heads.
            kernel_size (int): Kernel size for convolutional projections.
            stride_q (int): Stride for query convolution.
            stride_kv (int): Stride for key and value convolutions.
            padding_q (int): Padding for query convolution.
            padding_kv (int): Padding for key and value convolutions.
            attn_drop_p (float): Dropout probability for attention weights.
            attn_proj_drop_p (float): Dropout probability for attention output projection.
            update_steps (int): Number of update steps for Hopfield layers.
        """

    def __init__(self, cls_token, in_dim, num_heads, kernel_size, stride_q, stride_kv, padding_q, padding_kv,
                 attn_drop_p, attn_proj_drop_p, update_steps=10, capture_enabled=False, num_steps=5, dt=0.1,
                 min_omega=0.1, omega_init_mean=0.3, spatial_size=16):
        super().__init__()
        self.cls_token = cls_token
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = in_dim // num_heads
        self.scale = head_dim ** (-0.5)
        # Learnable temperature for cosine attention (Swin v2 style).
        # Initialized to match the default dot-product scale.
        self.logit_scale = nn.Parameter(torch.log(torch.ones(num_heads, 1, 1) * self.scale))

        # Convolutional Projection for Attention (Equation (2) in the paper.)
        # Depthwise Seperable Convolution (DepthwiseConv + BatchNorm + PointwiseConv)
        self.conv_proj_q = nn.Sequential(KuramotoConv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=kernel_size,
                                                        stride=stride_q, padding=padding_q, groups=in_dim, num_steps=num_steps, dt=dt,
                                                        min_omega=min_omega, omega_init_mean=omega_init_mean),
                                         nn.BatchNorm2d(num_features=in_dim),
                                         KuramotoPointwiseConv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1,
                                                        num_steps=num_steps, dt=dt, min_omega=min_omega,
                                                        omega_init_mean=omega_init_mean, spatial_size=spatial_size),
                                         )
        self.conv_proj_k = nn.Sequential(KuramotoConv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=kernel_size,
                                                        stride=stride_kv, padding=padding_kv, groups=in_dim, num_steps=num_steps, dt=dt,
                                                        min_omega=min_omega, omega_init_mean=omega_init_mean),
                                         nn.BatchNorm2d(num_features=in_dim),
                                         KuramotoPointwiseConv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1,
                                                        num_steps=num_steps, dt=dt, min_omega=min_omega,
                                                        omega_init_mean=omega_init_mean, spatial_size=spatial_size),
                                         )
        self.conv_proj_v = nn.Sequential(KuramotoConv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=kernel_size,
                                                        stride=stride_kv, padding=padding_kv, groups=in_dim, num_steps=num_steps, dt=dt,
                                                        min_omega=min_omega, omega_init_mean=omega_init_mean),
                                         nn.BatchNorm2d(num_features=in_dim),
                                         KuramotoPointwiseConv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1,
                                                        num_steps=num_steps, dt=dt, min_omega=min_omega,
                                                        omega_init_mean=omega_init_mean, spatial_size=spatial_size),
                                         )
        self.attn_drop = nn.Dropout(p=attn_drop_p)
        self.attn_proj_drop = nn.Dropout(p=attn_proj_drop_p)

        self.proj = nn.Linear(in_features=in_dim, out_features=in_dim)

    def forward(self, x, cls_token):
        # x: batch size, channel, height, width
        B, C, _, _ = x.shape
        q = self.conv_proj_q(x).reshape(B, C, -1).permute(0, 2, 1)  # batch size, height * width, channel
        k = self.conv_proj_k(x).reshape(B, C, -1).permute(0, 2, 1)
        v = self.conv_proj_v(x).reshape(B, C, -1).permute(0, 2, 1)

        if cls_token is not None:
            q = torch.cat([cls_token, q], dim=1)
            k = torch.cat([cls_token, k], dim=1)
            v = torch.cat([cls_token, v], dim=1)

        q = q.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Cosine attention — normalize Q and K to unit vectors so similarity
        # is purely angular, removing magnitude sensitivity from oscillator readout.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(100., device=self.logit_scale.device))).exp()
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k * scale,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, -1, C)
        attn_out = self.attn_proj_drop(self.proj(attn_out))
        return attn_out


class SingleTransformerLayer(nn.Module):
    """
        A single transformer layer of CvT.

        Args:
            cls_token (Tensor or None): Class token for global context.
            emb_dim (int): Embedding dimension.
            mlp_expansion (int): Expansion factor for MLP layers.
            num_heads (int): Number of attention heads.
            kernel_size (int): Kernel size for convolutional projections.
            stride_q (int): Stride for query convolution.
            stride_kv (int): Stride for key and value convolutions.
            padding_q (int): Padding for query convolution.
            padding_kv (int): Padding for key and value convolutions.
            attn_drop_p (float): Dropout probability for attention weights.
            attn_proj_drop_p (float): Dropout probability after the attention projection.
            drop_path_p (float): Stochastic depth probability.
            drop_p (float): Dropout probability for MLP layers.
            update_steps (int): Number of update steps for Hopfield layers.
        """

    def __init__(self, cls_token, emb_dim, mlp_expansion, num_heads, kernel_size, stride_q, stride_kv, padding_q,
                 padding_kv, attn_drop_p, attn_proj_drop_p, drop_path_p, drop_p, update_steps=10, capture_enabled=False,
                 num_steps=5, dt=0.1, min_omega=0.1, omega_init_mean=0.3, zoneout_prob=0.2, spatial_size=16,
                 layer_scale_init=1e-4):
        super().__init__()
        self.cls_token = cls_token
        self.LN1 = nn.LayerNorm(emb_dim)
        self.attn = Attention(cls_token, emb_dim, num_heads, kernel_size, stride_q, stride_kv, padding_q, padding_kv,
                              attn_drop_p, attn_proj_drop_p, update_steps=update_steps, capture_enabled=capture_enabled,
                              num_steps=num_steps, dt=dt, min_omega=min_omega, omega_init_mean=omega_init_mean,
                              spatial_size=spatial_size)
        self.LN2 = nn.LayerNorm(emb_dim)

        # Hopfield MLP configs
        mlp_config = {
            "hidden_size": emb_dim,
            "intermediate_size": emb_dim * mlp_expansion,
            "hopfield_update_steps": update_steps,
            "zoneout_prob": zoneout_prob,
        }
        self.MLP = HopfieldMLP(mlp_config, capture_enabled=capture_enabled)

        self.drop_path = DropPath(drop_prob=drop_path_p) if drop_path_p > 0.0 else nn.Identity()  # Stochastic depth

        # CaiT-style LayerScale (Touvron et al., 2021)
        if layer_scale_init is not None and layer_scale_init > 0:
            self.ls1 = nn.Parameter(layer_scale_init * torch.ones(emb_dim))
            self.ls2 = nn.Parameter(layer_scale_init * torch.ones(emb_dim))
            self.use_layer_scale = True
        else:
            self.register_parameter('ls1', None)
            self.register_parameter('ls2', None)
            self.use_layer_scale = False

    def forward(self, x, H, W):
        B, N, C = x.shape

        res = x
        x = self.LN1(x)
        if self.cls_token is not None:
            cls_token, x = torch.split(x, [1, H * W], dim=1)
        else:
            cls_token = None
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.attn(x, cls_token)

        # LayerScale on attention branch: broadcast [C] over [B, N, C]
        if self.use_layer_scale:
            x = x * self.ls1
        x = res + self.drop_path(x)

        res = x
        x = self.LN2(x)
        x = self.MLP(x)

        # LayerScale on MLP branch
        if self.use_layer_scale:
            x = x * self.ls2
        x = res + self.drop_path(x)

        return x


class TransformerBlock(nn.Module):
    """
        Transformer Block consisting of multiple SingleTransformerLayers.

        Args:
            cls_token (Tensor or None): Class token for global context.
            n_layers (int): Number of transformer layers.
            emb_dim (int): Embedding dimension throughout the block.
            mlp_expansion (int): Expansion factor for MLP layers.
            num_heads (int): Number of attention heads.
            kernel_size (int): Kernel size for convolutional projections.
            stride_q (int): Stride for query convolution.
            stride_kv (int): Stride for key and value convolutions.
            padding_q (int): Padding for query convolution.
            padding_kv (int): Padding for key and value convolutions.
            attn_drop_p (float): Dropout probability for attention weights.
            attn_proj_drop_p (float): Dropout probability after the attention projection.
            drop_path_p (float): Stochastic depth probability.
            drop_p (float): Dropout probability for MLP layers.
            update_steps (int): Number of update steps for Hopfield layers.
        """

    def __init__(self, cls_token, n_layers, emb_dim, mlp_expansion, num_heads, kernel_size, stride_q, stride_kv,
                 padding_q, padding_kv, attn_drop_p, attn_proj_drop_p, drop_path_p, drop_p, update_steps=10, capture_enabled=False,
                 num_steps=5, dt=0.1, min_omega=0.1, omega_init_mean=0.3, zoneout_prob=0.2, spatial_size=16):
        super().__init__()

        drop_path_rates = torch.linspace(0, drop_path_p, n_layers).tolist()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = SingleTransformerLayer(cls_token, emb_dim, mlp_expansion, num_heads, kernel_size, stride_q,
                                           stride_kv, padding_q, padding_kv, attn_drop_p, attn_proj_drop_p,
                                           drop_path_rates[i], drop_p, update_steps=update_steps, capture_enabled=capture_enabled,
                                           num_steps=num_steps, dt=dt, min_omega=min_omega, omega_init_mean=omega_init_mean,
                                           zoneout_prob=zoneout_prob, spatial_size=spatial_size)
            self.layers.append(layer)

    def forward(self, x, H, W):
        for layer in self.layers:
            x = layer(x, H, W)
        return x


class ConvolutionalVisionTransformer(nn.Module):
    """
        Implementation of the Convolutional Vision Transformer (CvT) with Hopfield layers.

        Args:
            in_channel (int): Number of input channels (e.g., 3 for RGB images).
            num_classes (int): Number of output classes for classification.
            emb_dim (list of int): Embedding dimensions for each stage.
            n_layers (list of int): Number of transformer layers in each stage.
            kernel_size (list of int): Kernel size for convolutional token embeddings at each stage.
            stride (list of int): Stride for token embeddings.
            padding (list of int): Padding for token embeddings.
            mlp_expansion (int): Expansion factor for MLP layers.
            num_heads (list of int): Number of attention heads per stage.
            attn_kernel_size (int): Kernel size for attention convolution projections.
            stride_q (list of int): Stride for query convolutions per stage.
            stride_kv (list of int): Stride for key and value convolutions per stage.
            padding_q (list of int): Padding for query convolutions per stage.
            padding_kv (list of int): Padding for key and value convolutions per stage.
            attn_drop_p (float): Dropout probability for attention weights.
            attn_proj_drop_p (float): Dropout probability after the attention projection.
            drop_path_p (float): Stochastic depth probability.
            drop_p (float): Dropout probability for MLP layers.
            update_steps (int): Number of update steps for Hopfield layers.
        """

    def __init__(self, in_channel: int=3, num_classes: int=1000, img_size: int=32,
                 emb_dim=(64, 128, 192), n_layers=(1, 2, 6),
                 kernel_size=(7, 3, 3), stride=(4, 2, 2), padding=(2, 1, 1), mlp_expansion=4, num_heads=(1, 3, 6),
                 attn_kernel_size=3, stride_q=(1, 1, 1), stride_kv=(2, 2, 2), padding_q=(1, 1, 1), padding_kv=(1, 1, 1),
                 attn_drop_p=0.0, attn_proj_drop_p=0.0, drop_path_p=0.1, drop_p=0.0, update_steps=10, capture_enabled=False,
                 kuramoto_steps=5, dt=0.1, min_omega=0.1, omega_init_mean=0.3, zoneout_prob=0.2):
        super().__init__()

        self.cls_token = nn.Parameter(data=torch.zeros(size=(1, 1, emb_dim[2])), requires_grad=True)

        # Spatial size after each stage's token-embedding conv, derived from
        # img_size and that stage's kernel/stride/padding — NOT hardcoded, so
        # changing `stride`/`kernel_size`/`img_size` can't silently desync
        # KuramotoPointwiseConv2d's omega_0 shape from the actual feature map.
        spatial_size1 = self._conv_out_size(img_size, kernel_size[0], stride[0], padding[0])
        spatial_size2 = self._conv_out_size(spatial_size1, kernel_size[1], stride[1], padding[1])
        spatial_size3 = self._conv_out_size(spatial_size2, kernel_size[2], stride[2], padding[2])

        ### STAGE 1 ###
        self.conv_token_emb1 = ConvolutionalTokenEmbedding(in_channel, emb_dim[0], kernel_size[0], stride[0],
                                                           padding[0], kuramoto_steps=kuramoto_steps, dt=dt,
                                                           min_omega=min_omega,
                                                           omega_init_mean=omega_init_mean
                                                           )
        self.transformer_block1 = TransformerBlock(None, n_layers[0], emb_dim[0], mlp_expansion, num_heads[0],
                                                   attn_kernel_size, stride_q[0], stride_kv[0], padding_q[0],
                                                   padding_kv[0], attn_drop_p, attn_proj_drop_p, drop_path_p, drop_p,
                                                   update_steps=update_steps, capture_enabled=capture_enabled,
                                                   num_steps=kuramoto_steps, dt=dt, min_omega=min_omega,
                                                   omega_init_mean=omega_init_mean, zoneout_prob=zoneout_prob,
                                                   spatial_size=spatial_size1)
        ### STAGE 2 ###
        self.conv_token_emb2 = ConvolutionalTokenEmbedding(emb_dim[0], emb_dim[1], kernel_size[1], stride[1],
                                                           padding[1], kuramoto_steps=kuramoto_steps, dt=dt,
                                                           min_omega=min_omega,
                                                           omega_init_mean=omega_init_mean
                                                           )
        self.transformer_block2 = TransformerBlock(None, n_layers[1], emb_dim[1], mlp_expansion, num_heads[1],
                                                   attn_kernel_size, stride_q[1], stride_kv[1], padding_q[1],
                                                   padding_kv[1], attn_drop_p, attn_proj_drop_p, drop_path_p, drop_p,
                                                   update_steps=update_steps, capture_enabled=capture_enabled,
                                                   num_steps=kuramoto_steps, dt=dt, min_omega=min_omega,
                                                   omega_init_mean=omega_init_mean, zoneout_prob=zoneout_prob,
                                                   spatial_size=spatial_size2)
        ### STAGE 3 ###
        self.conv_token_emb3 = ConvolutionalTokenEmbedding(emb_dim[1], emb_dim[2], kernel_size[2], stride[2],
                                                           padding[2], kuramoto_steps=kuramoto_steps, dt=dt,
                                                           min_omega=min_omega,
                                                           omega_init_mean=omega_init_mean
                                                           )
        self.transformer_block3 = TransformerBlock(self.cls_token, n_layers[2], emb_dim[2], mlp_expansion, num_heads[2],
                                                   attn_kernel_size, stride_q[2], stride_kv[2], padding_q[2],
                                                   padding_kv[2], attn_drop_p, attn_proj_drop_p, drop_path_p, drop_p,
                                                   update_steps=update_steps, capture_enabled=capture_enabled,
                                                   num_steps=kuramoto_steps, dt=dt, min_omega=min_omega,
                                                   omega_init_mean=omega_init_mean, zoneout_prob=zoneout_prob,
                                                   spatial_size=spatial_size3)

        # Classification head
        self.head = nn.Linear(emb_dim[2], num_classes)

        self.pos_drop = nn.Dropout(p=drop_p)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def _conv_out_size(in_size, kernel_size, stride, padding):
        """Standard Conv2d spatial-output-size formula (assumes dilation=1,
        square input/kernel), used to derive each stage's spatial_size from
        img_size instead of hardcoding it."""
        return (in_size + 2 * padding - kernel_size) // stride + 1

    def forward_features(self, x):
        ### STAGE 1 ###
        x, H, W = self.conv_token_emb1(x)
        x = self.pos_drop(x)
        x = self.transformer_block1(x, H, W)
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        ### STAGE 2 ###
        x, H, W = self.conv_token_emb2(x)
        x = self.pos_drop(x)
        x = self.transformer_block2(x, H, W)
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        ### STAGE 3 ###
        x, H, W = self.conv_token_emb3(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.pos_drop(x)
        x = self.transformer_block3(x, H, W)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        cls_output = x[:, 0]
        return self.head(cls_output)